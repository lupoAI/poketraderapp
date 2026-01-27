from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from model.config import STORAGE_DIR, DEFAULT_INDEX_DIR, DEFAULT_YOLO_SEG_WEIGHTS, CARDS_DIR
from model.embedding_backends import get_backend
from model.index_bundle import load_index_bundle, IndexVariant
from model.yolo_quad import quad_from_yolo_result
from model.adapter_indexing import load_adapter, pick_or_build_adapter_index

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_birdseye_view(image, corners):
    # corners should be 4 points: [top-left, top-right, bottom-right, bottom-left]
    width, height = 400, 550  # Standard aspect ratio for trading cards
    # width, height = 400, 400  # Standard aspect ratio for trading cards
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def _l2_normalize_query(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def get_classification_model(
    *,
    yolo_weights: Path | str = DEFAULT_YOLO_SEG_WEIGHTS,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    index_variant: str | None = None,
    top_k: int = 3,
    adapter_path: Path | str | None = None,
):

    yolo_weights = Path(yolo_weights)
    index_dir = Path(index_dir)

    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_dir}\n"
            f"Hint: expected under {DEFAULT_INDEX_DIR} or pass index_dir=..."
        )

    det_model = YOLO(str(yolo_weights))

    # If an adapter is requested, we must use RAW embeddings and an adapter-built index.
    adapter = None
    adapter_info = None
    if adapter_path is not None:
        if index_variant is not None and str(index_variant) != IndexVariant.RAW:
            raise ValueError(
                "adapter_path currently supports only index_variant='raw' (no normalization/centering)"
            )

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter_path not found: {adapter_path}")

        # indices_dir is the parent where base index lives (DEFAULT is storage/indices/...)
        indices_root = index_dir.parent
        adapter_index_dir = pick_or_build_adapter_index(
            indices_dir=indices_root,
            base_index_dir=index_dir,
            adapter_path=adapter_path,
            device=device,
        )
        bundle = load_index_bundle(adapter_index_dir, variant=IndexVariant.RAW)

        adapter, adapter_info = load_adapter(adapter_path, device=device)
    else:
        bundle = load_index_bundle(index_dir, variant=index_variant)

    search_index = bundle.index
    card_names = bundle.card_names

    backend = get_backend(bundle.model_type, device=device)

    def _transform_query(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if bundle.variant == IndexVariant.RAW:
            return q

        if bundle.variant == IndexVariant.NORMALIZED:
            return _l2_normalize_query(q)

        if bundle.variant == IndexVariant.RAW_CENTERED:
            if bundle.avg_embedding is None:
                raise ValueError("avg_embedding is required for raw_centered variant")
            return q - bundle.avg_embedding.reshape(1, -1)

        if bundle.variant == IndexVariant.CENTERED_NORMALIZED:
            if bundle.avg_embedding is None:
                raise ValueError("avg_embedding is required for centered_normalized variant")
            q = q - bundle.avg_embedding.reshape(1, -1)
            return _l2_normalize_query(q)

        raise ValueError(f"Unknown bundle.variant: {bundle.variant!r}")

    def identify_card(frame):
        """Identifica carte in un frame e ritorna info utili per visualizzazione.

        Return: List[dict] (una entry per istanza)
        Keys principali:
          - instance_index: int
          - contour_xy: np.ndarray | None   # (N,2) px
          - rect_xyxy: np.ndarray | None    # (4,) [x1,y1,x2,y2]
          - quad_xy: np.ndarray | None      # (4,2) px (TL,TR,BR,BL)
          - warped_bgr: np.ndarray | None   # (560,400,3)
          - top3: List[dict] con {card_id, similarity}
        """

        out = []

        # 1. Detect Card mask
        results = det_model(frame, verbose=False, conf=0.85)
        for r in results:
            # Ultralytics Results: r.masks.xy (list of Nx2), r.boxes.xyxy/conf
            masks_xy = None
            if r.masks is not None and r.masks.xy is not None:
                masks_xy = r.masks.xy

            boxes_xyxy = None
            if r.boxes is not None:
                boxes_xyxy = r.boxes.xyxy

            n = len(masks_xy) if masks_xy is not None else (len(boxes_xyxy) if boxes_xyxy is not None else 0)

            for i in range(n):
                contour_xy = np.asarray(masks_xy[i], dtype=np.float32) if masks_xy is not None else None

                rect_xyxy = None
                if boxes_xyxy is not None and i < len(boxes_xyxy):
                    rect_xyxy = np.asarray(boxes_xyxy[i].detach().cpu().numpy(), dtype=np.float32).reshape(4)
                quad = quad_from_yolo_result(r, instance_index=i)

                yolo_conf = float(r.boxes.conf[i].detach().cpu().numpy().reshape(-1)[0])
                warped_card = None
                top_preds = []

                if quad is not None:
                    # 2. Flatten the card
                    warped_card = get_birdseye_view(frame, quad)

                    # 3. Get Embedding
                    pil_img = Image.fromarray(cv2.cvtColor(warped_card, cv2.COLOR_BGR2RGB))

                    # Always compute RAW backend embedding.
                    query_vec = backend.encode(pil_img, normalize=False)

                    # Apply adapter (raw -> adapted raw) if enabled.
                    if adapter is not None:
                        with torch.no_grad():
                            q_t = torch.from_numpy(query_vec.astype(np.float32)).to(device)
                            query_vec = adapter(q_t).detach().cpu().numpy().astype(np.float32)

                    # Apply variant transform (note: adapter path forces RAW)
                    query_vec = _transform_query(query_vec)

                    # 4. Search Vector DB (top-k)
                    D, I = search_index.search(query_vec.astype("float32"), k=int(top_k))
                    top_preds = [
                        {"card_id": str(card_names[idx]), "similarity": float(D[0][j])}
                        for j, idx in enumerate(I[0])
                    ]

                out.append(
                    {
                        "instance_index": i,
                        "contour_xy": contour_xy,
                        "rect_xyxy": rect_xyxy,
                        "yolo_conf": yolo_conf,
                        "quad_xy": quad,
                        "warped_bgr": warped_card,
                        "top3": top_preds,
                    }
                )

        return out

    return identify_card



def _safe_imread(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _resize_keep_aspect(img: np.ndarray, *, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _tile_with_label(img: np.ndarray, label: str, *, tile_w: int, tile_h: int) -> np.ndarray:
    # fit into tile
    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    fitted = _resize_keep_aspect(img, max_w=tile_w, max_h=tile_h - 34)
    y0 = 34 + max(0, (tile_h - 34 - fitted.shape[0]) // 2)
    x0 = max(0, (tile_w - fitted.shape[1]) // 2)
    canvas[y0 : y0 + fitted.shape[0], x0 : x0 + fitted.shape[1]] = fitted

    cv2.rectangle(canvas, (0, 0), (tile_w, 34), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return canvas


def render_prediction_gallery(
    results: list[dict],
    *,
    max_instances: int = 3,
    top_k: int = 3,
    tile_w: int = 220,
    tile_h: int = 308,
    max_total_height: int | None = None,
    video_height: int | None = None,
    max_row_fraction: float = 0.25,
) -> np.ndarray | None:
    """Crea una gallery con le immagini delle predizioni.

    Layout: una riga per istanza (box) e `top_k` tile per riga.

    Modalità "video": se `video_height` è impostato, la dimensione dei tile viene scelta
    in modo che:
      - l'altezza per riga sia al massimo `video_height * max_row_fraction` (default 1/4)
      - se le righe sono molte, tutte si ridimensionano a `video_height / n_rows`
        così da riempire l'altezza.

    `max_total_height` resta supportato come limite "totale" (modalità legacy).
    """

    shown = results[:max_instances]
    # consider only instances that actually have predictions
    valid = [inst for inst in shown if (inst.get("top3") or [])[:top_k]]
    n_rows = len(valid)
    if n_rows == 0:
        return None

    # Determine tile size
    if video_height is not None and video_height > 0:
        # per-row height rule
        max_h_per_row = max(60, int(round(video_height * float(max_row_fraction))))
        target_h_per_row = max(60, int(video_height // max(1, n_rows)))
        tile_h = min(tile_h, max_h_per_row, target_h_per_row)
        # keep aspect similar to default
        tile_w = max(60, int(round(tile_w * (tile_h / 308.0))))
    elif max_total_height is not None and max_total_height > 0:
        max_h_per_row = max(80, int(max_total_height // max(1, n_rows)))
        if max_h_per_row < tile_h:
            scale = max_h_per_row / float(tile_h)
            tile_h = max_h_per_row
            tile_w = max(80, int(round(tile_w * scale)))

    rows: list[np.ndarray] = []

    for inst in valid:
        preds = inst.get("top3") or []
        tiles: list[np.ndarray] = []

        for pred in preds[:top_k]:
            card_id = str(pred.get("card_id", ""))
            sim = float(pred.get("similarity", 0.0))
            img_path = CARDS_DIR / f"{card_id}.jpg"
            img = _safe_imread(img_path)
            if img is None:
                missing = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                cv2.putText(missing, "missing", (10, max(20, tile_h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(missing, card_id[:16], (10, min(tile_h - 10, max(30, tile_h // 2 + 30))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tiles.append(_tile_with_label(missing, f"{card_id} ({sim:.2f})", tile_w=tile_w, tile_h=tile_h))
            else:
                tiles.append(_tile_with_label(img, f"{card_id} ({sim:.2f})", tile_w=tile_w, tile_h=tile_h))

        if tiles:
            row = cv2.hconcat(tiles) if len(tiles) > 1 else tiles[0]
            rows.append(row)

    if not rows:
        return None

    gallery = cv2.vconcat(rows) if len(rows) > 1 else rows[0]

    # In video mode, optionally pad to exactly video height when n_rows >= 4 (tile_h == H/n_rows)
    # We leave it as-is; the caller can place it in a right panel if desired.
    return gallery


def visualize_results_on_frame(
    frame_bgr: np.ndarray,
    results: list[dict],
    *,
    show_contour: bool = True,
    show_rect: bool = True,
    show_quad: bool = True,
) -> np.ndarray:
    """Ritorna una copia del frame con overlay di contorno, rettangolo YOLO, quad, conf YOLO e testo top-k."""
    out = frame_bgr.copy()

    for inst in results:
        idx = int(inst.get("instance_index", 0))

        # Colors per instance
        base = (37 * (idx + 1)) % 255
        col_contour = (int(80 + base) % 255, int(180 + base) % 255, int(60 + base) % 255)
        col_rect = (0, 0, 255)
        col_quad = (0, 255, 0)

        contour_xy = inst.get("contour_xy")
        rect_xyxy = inst.get("rect_xyxy")
        quad_xy = inst.get("quad_xy")
        yolo_conf = inst.get("yolo_conf")

        if show_contour and contour_xy is not None:
            pts = np.asarray(contour_xy, dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) >= 3:
                cv2.polylines(out, [pts], isClosed=True, color=col_contour, thickness=2, lineType=cv2.LINE_AA)

        if show_rect and rect_xyxy is not None:
            x1, y1, x2, y2 = [int(round(x)) for x in np.asarray(rect_xyxy).reshape(4)]
            cv2.rectangle(out, (x1, y1), (x2, y2), col_rect, thickness=2)

        if show_quad and quad_xy is not None:
            q = np.asarray(quad_xy, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [q], isClosed=True, color=col_quad, thickness=3, lineType=cv2.LINE_AA)
            # corner indices
            for j, (x, y) in enumerate(np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)):
                cv2.circle(out, (int(round(x)), int(round(y))), 5, (255, 255, 255), -1)
                cv2.putText(
                    out,
                    str(j),
                    (int(round(x)) + 6, int(round(y)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Text anchor near top-left corner of quad if available, else near rect
        anchor = None
        if quad_xy is not None:
            anchor = tuple(int(round(v)) for v in np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)[0])
        elif rect_xyxy is not None:
            x1, y1, _, _ = [int(round(x)) for x in np.asarray(rect_xyxy).reshape(4)]
            anchor = (x1, y1)

        # Write YOLO confidence (if available)
        if anchor is not None and yolo_conf is not None:
            ax, ay = anchor
            label = f"conf={float(yolo_conf):.2f}"
            cv2.putText(
                out,
                label,
                (ax, max(15, ay - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Top-k predictions
        preds = inst.get("top3") or []
        if anchor is not None and preds:
            ax, ay = anchor
            ay = max(20, ay - 8)
            for k, pred in enumerate(preds[:3]):
                card_id = str(pred.get("card_id", ""))
                sim = float(pred.get("similarity", 0.0))
                cv2.putText(
                    out,
                    f"{card_id} ({sim:.2f})",
                    (ax, ay + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    return out


if __name__ == "__main__":
    # Smoke test: run identify_card on the first image found under storage/test_data/images
    import os

    test_dir = STORAGE_DIR / "test_data" / "images"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    image_paths.sort(key=lambda p: str(p).lower())

    if not image_paths:
        raise SystemExit(f"No test images found under: {test_dir}")

    img_path = image_paths[0]
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read test image: {img_path}")

    identify = get_classification_model(top_k=3)
    out = identify(img)

    print(f"Test image: {os.path.basename(str(img_path))}")
    print(f"Instances: {len(out)}")

    for inst in out[:3]:
        contour_shape = None if inst["contour_xy"] is None else tuple(inst["contour_xy"].shape)
        quad_shape = None if inst["quad_xy"] is None else tuple(inst["quad_xy"].shape)
        print(
            f"- inst={inst['instance_index']} contour={contour_shape} rect={inst['rect_xyxy']} quad={quad_shape} top3={inst['top3']}"
        )

    # Render overlay + gallery and save under storage/vis
    overlay = visualize_results_on_frame(img, out)
    gallery = render_prediction_gallery(out, max_instances=3, top_k=3)

    if gallery is not None:
        # stack vertically with a small separator
        sep = np.full((12, max(overlay.shape[1], gallery.shape[1]), 3), 40, dtype=np.uint8)
        # pad to same width
        def pad_to_w(im, width):
            h, w = im.shape[:2]
            if w >= width:
                return im
            pad = np.zeros((h, width - w, 3), dtype=np.uint8)
            return np.hstack([im, pad])

        overlay_p = pad_to_w(overlay, max(overlay.shape[1], gallery.shape[1]))
        gallery_p = pad_to_w(gallery, max(overlay.shape[1], gallery.shape[1]))
        combined = np.vstack([overlay_p, sep, gallery_p])
    else:
        combined = overlay

    out_dir = STORAGE_DIR / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), combined)
    print(f"Saved visualization to: {out_path}")
