from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import os
import time

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from model.embedding_backends import get_backend
from model.classification_loop import get_birdseye_view, get_classification_model, render_prediction_gallery, visualize_results_on_frame
from model.index_bundle import IndexVariant
from model.config import STORAGE_DIR, DEFAULT_INDEX_DIR, DEFAULT_YOLO_SEG_WEIGHTS, CARDS_DIR

# --- INITIALIZATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer indices under storage/index, but allow running from anywhere.
# (Paths are centralized in model.config)


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    if pts.shape != (4, 2):
        raise ValueError(f"Expected (4,2) points, got {pts.shape}")

    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.stack([tl, tr, br, bl], axis=0)


def _try_approx_to_4(contour: np.ndarray) -> Optional[np.ndarray]:
    """Try approxPolyDP with multiple eps values until 4 vertices."""
    peri = float(cv2.arcLength(contour, True))
    if peri <= 0:
        return None

    for frac in (0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, frac * peri, True)
        if approx is not None and len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _quad_from_yolo_contour(contour_xy: np.ndarray) -> Optional[np.ndarray]:
    """Convert a YOLO mask contour (Nx2 px) to a 4-corner quad (float32, TL/TR/BR/BL)."""
    pts = np.asarray(contour_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None

    contour = pts.reshape((-1, 1, 2)).astype(np.float32)

    approx = _try_approx_to_4(contour)
    if approx is not None:
        return _order_points_clockwise(approx)

    # Fallback: convex hull then approx
    hull = cv2.convexHull(contour)
    approx = _try_approx_to_4(hull)
    if approx is not None:
        return _order_points_clockwise(approx)

    # Final fallback: minAreaRect (gives a rectangle, but always returns 4 corners)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    return _order_points_clockwise(box)


def _select_best_instance(result) -> Optional[int]:
    """Pick the most likely card instance index from a YOLO result."""
    if result is None or result.masks is None:
        return None

    n = len(result.masks)
    if n <= 0:
        return None

    # Prefer highest confidence box if available
    if getattr(result, "boxes", None) is not None and result.boxes is not None and len(result.boxes) == n:
        try:
            conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            return int(np.argmax(conf))
        except Exception:
            pass

    # Fallback: largest mask area
    try:
        if getattr(result.masks, "data", None) is not None and result.masks.data is not None:
            masks = result.masks.data.detach().cpu().numpy()
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            return int(np.argmax(areas))
    except Exception:
        pass

    return 0


def _find_card_corners_yolo(
    image_bgr: np.ndarray,
    *,
    yolo_model: YOLO,
    conf: float = 0.25,
    timer: Optional["_Timer"] = None,
    imgsz: Optional[int] = 640,
    half: bool = False,
) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Find 4 card corners using YOLO segmentation.

    Returns (corners, debug_dict). Corners are float32 (4,2) in TL,TR,BR,BL order.

    Notes:
      - imgsz controls the inference resolution (big speed lever, even on CPU).
      - half=True is only beneficial on CUDA; on CPU we force it off.
    """
    dbg: Dict[str, Any] = {}

    _contour_xy: Optional[np.ndarray] = None
    _quad: Optional[np.ndarray] = None

    results = None

    # fp16 is only useful on GPU; keep predictable behavior on CPU.
    half = bool(half and torch.cuda.is_available())

    # Ultralytics accepts numpy BGR images.
    if timer is None:
        results = yolo_model.predict(image_bgr, conf=conf, verbose=False, imgsz=imgsz, half=half)
    else:
        with timer.time_block("1_yolo_predict"):
            results = yolo_model.predict(image_bgr, conf=conf, verbose=False, imgsz=imgsz, half=half)

    if not results:
        return None, dbg

    result = results[0]
    if result.masks is None or getattr(result.masks, "xy", None) is None:
        dbg["yolo_no_masks"] = True
        return None, dbg

    idx = _select_best_instance(result)
    if idx is None:
        dbg["yolo_no_instances"] = True
        return None, dbg

    contours_xy = result.masks.xy  # list of N (Mi,2)
    if idx >= len(contours_xy):
        return None, dbg

    if timer is None:
        _contour_xy = np.asarray(contours_xy[idx], dtype=np.float32)
        _quad = _quad_from_yolo_contour(_contour_xy)
    else:
        with timer.time_block("2_yolo_to_quad"):
            _contour_xy = np.asarray(contours_xy[idx], dtype=np.float32)
            _quad = _quad_from_yolo_contour(_contour_xy)

    if _quad is None or _contour_xy is None:
        return None, dbg

    # Sanity: reject degenerate quads
    if abs(cv2.contourArea(_quad.reshape(-1, 1, 2).astype(np.int32))) < 1000:
        return None, dbg

    dbg["yolo_instance"] = idx
    dbg["yolo_contour_len"] = int(len(_contour_xy))
    return _quad, dbg


def _find_card_corners_opencv(
    image_bgr: np.ndarray,
    *,
    debug: bool = False,
) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Find 4 card corners using OpenCV only.

    (Kept as a fallback.)

    Returns (corners, debug_dict). Corners are float32 (4,2) in TL,TR,BR,BL order.
    """

    dbg: Dict[str, Any] = {}

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto-ish Canny thresholds based on median
    v = float(np.median(blur))
    lower = int(max(0.0, 0.66 * v))
    upper = int(min(255.0, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        dbg["edges"] = edges
        dbg["edges_closed"] = edges_closed

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, dbg

    img_area = float(h * w)

    best_quad: Optional[np.ndarray] = None
    best_score = -1.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 0.02 * img_area:  # too small
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)

            # rectangularity: contour area / minAreaRect area
            rect = cv2.minAreaRect(approx)
            (rw, rh) = rect[1]
            rect_area = float(rw * rh) if rw > 0 and rh > 0 else 0.0
            rectangularity = area / rect_area if rect_area > 0 else 0.0

            # Score favors big + rectangle-like shapes
            score = (area / img_area) * 1.5 + rectangularity

            if score > best_score:
                best_score = score
                best_quad = pts

    if best_quad is None:
        # Fallback: use minimum area rectangle around the largest contour
        largest = max(contours, key=cv2.contourArea)
        if float(cv2.contourArea(largest)) < 0.02 * img_area:
            return None, dbg
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)  # 4x2
        best_quad = box

    ordered = _order_points_clockwise(best_quad)

    # Sanity: reject degenerate quads (almost line)
    if abs(cv2.contourArea(ordered.reshape(-1, 1, 2).astype(np.int32))) < 1000:
        return None, dbg

    return ordered, dbg


def _load_index(index_dir: Path = DEFAULT_INDEX_DIR):
    index_dir = Path(index_dir)
    index_path = index_dir / "pokemon_cards.index"
    names_path = index_dir / "card_names.npy"

    if not index_path.exists() or not names_path.exists():
        raise FileNotFoundError(
            "Index not found. Expected:\n"
            f"- {index_path}\n"
            f"- {names_path}\n"
            "Build an index first (model/build_index.py) or pass index_dir=..."
        )
    search_index = faiss.read_index(str(index_path))
    card_names = np.load(str(names_path), allow_pickle=True)
    return search_index, card_names


def _safe_imread(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return img


def _make_grid(images_bgr: List[np.ndarray], *, cell_size: tuple[int, int]) -> np.ndarray:
    """Create a simple horizontal grid (1 row) of images resized to cell_size."""
    w, h = cell_size
    cells: List[np.ndarray] = []
    for im in images_bgr:
        resized = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cells.append(resized)
    return cv2.hconcat(cells) if len(cells) > 1 else cells[0]


def _render_topk_gallery(
    top_ids: List[str],
    top_scores: List[float],
    *,
    max_k: int,
    cell_width: int = 220,
    cell_height: int = 308,
) -> Optional[np.ndarray]:
    """Load top-k predicted card images and render them as a single-row grid.

    Expects images in storage/images/cards/<card_id>.jpg
    """

    imgs: List[np.ndarray] = []
    k = min(max_k, len(top_ids))
    for i in range(k):
        card_id = top_ids[i]
        img_path = CARDS_DIR / f"{card_id}.jpg"
        img = _safe_imread(img_path)
        if img is None:
            # Placeholder tile if missing
            tile = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            cv2.putText(tile, "missing", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(tile, card_id[:12], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            imgs.append(tile)
            continue

        # Add label overlay on the tile
        img = cv2.resize(img, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        label = f"{i+1}: {card_id} ({top_scores[i]:.2f})"
        cv2.rectangle(img, (0, 0), (cell_width, 34), (0, 0, 0), thickness=-1)
        cv2.putText(img, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        imgs.append(img)

    if not imgs:
        return None

    return _make_grid(imgs, cell_size=(cell_width, cell_height))


def _sync_if_cuda() -> None:
    """Sync CUDA to get accurate timings for GPU ops."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        # Best-effort sync; don't fail timing if CUDA isn't usable.
        pass


class _Timer:
    def __init__(self):
        self.ms: Dict[str, float] = {}

    def time_block(self, name: str):
        # tiny context manager
        class _Ctx:
            def __init__(self, outer: "_Timer", key: str):
                self.outer = outer
                self.key = key
                self.t0 = 0.0

            def __enter__(self):
                _sync_if_cuda()
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                _sync_if_cuda()
                dt = (time.perf_counter() - self.t0) * 1000.0
                self.outer.ms[self.key] = self.outer.ms.get(self.key, 0.0) + float(dt)

        return _Ctx(self, name)

# --- CLIP/CPU performance knobs (safe defaults) ---
try:
    if not torch.cuda.is_available():
        ncpu = os.cpu_count() or 4
        torch.set_num_threads(max(1, ncpu // 2))
        torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = True
except Exception:
    pass


def test_on_image(
    image_bgr: np.ndarray,
    *,
    search_index: "faiss.Index",
    card_names: np.ndarray,
    k: int = 3,
    show: bool = True,
    window_prefix: str = "Pokemon Card Scanner",
    debug_opencv: bool = True,
    yolo_model: Optional[YOLO] = None,
    use_yolo: bool = True,
    yolo_conf: float = 0.25,
    yolo_imgsz: int = 640,
    yolo_half: bool = False,
    profile: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run corners + warp + CLIP + Faiss on a single BGR image.

    CLIP speed knobs:
      - clip_fast_preprocess: avoids PIL and uses a torch-based preprocess.

    Prefer YOLO-seg for corners; fallback to OpenCV if needed.

    YOLO speed knobs:
      - yolo_imgsz: inference resolution (smaller = faster, less accurate).
      - yolo_half: fp16 inference (only used if CUDA is available).

    If profile=True, returns a timings dict (ms) with keys:
      - 1_yolo_predict
      - 2_yolo_to_quad
      - 2_warp_birdseye
      - 3_clip_preprocess
      - 3_clip_encode
      - 4_faiss_search
      - total
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    timer = _Timer() if profile else None

    with (timer.time_block("total") if timer is not None else _nullcontext()):
        dbg: Dict[str, Any] = {}

        corners: Optional[np.ndarray] = None
        if use_yolo:
            if yolo_model is None:
                raise ValueError("use_yolo=True but yolo_model is None")
            corners, dbg = _find_card_corners_yolo(
                image_bgr,
                yolo_model=yolo_model,
                conf=yolo_conf,
                timer=timer,
                imgsz=yolo_imgsz,
                half=yolo_half,
            )

        if corners is None:
            # OpenCV fallback (not part of the 4-step profiling request, but useful to know)
            if timer is None:
                corners, dbg_cv = _find_card_corners_opencv(image_bgr, debug=debug_opencv)
            else:
                with timer.time_block("2_opencv_fallback_corners"):
                    corners, dbg_cv = _find_card_corners_opencv(image_bgr, debug=debug_opencv)
            dbg.update({f"opencv_{k}": v for k, v in dbg_cv.items()})
            dbg["used_fallback_opencv"] = True

        if corners is None:
            if show:
                cv2.imshow(f"{window_prefix} - input", image_bgr)
                if debug_opencv and "opencv_edges_closed" in dbg:
                    cv2.imshow(f"{window_prefix} - edges", dbg["opencv_edges_closed"])
            return None

        if timer is None:
            warped = get_birdseye_view(image_bgr, corners)
        else:
            with timer.time_block("2_warp_birdseye"):
                warped = get_birdseye_view(image_bgr, corners)

        # BGR->RGB + CLIP embedding (via backend)
        backend = get_backend("clip")
        pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

        if timer is None:
            query_vec = backend.encode(pil_img, normalize=True)
        else:
            # Manteniamo la label timing legacy, anche se ora include preprocess+encode in un unico step.
            with timer.time_block("3_clip_encode"):
                query_vec = backend.encode(pil_img, normalize=True)

        if timer is None:
            D, I = cast(Any, search_index).search(query_vec.astype("float32"), k=k)
        else:
            with timer.time_block("4_faiss_search"):
                D, I = cast(Any, search_index).search(query_vec.astype("float32"), k=k)

        top_ids = [str(card_names[idx]) for idx in I[0]]
        top_scores = [float(x) for x in D[0]]

        detected: Dict[str, Any] = {
            "corners": corners,
            "warped": warped,
            "top_ids": top_ids,
            "top_scores": top_scores,
            "debug": dbg,
        }

        if timer is not None:
            detected["timings_ms"] = dict(timer.ms)

        if show:
            corners_np = cast(np.ndarray, corners)
            annotated = image_bgr.copy()
            pts_int = corners_np.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)

            lines = [f"{top_ids[i]} ({top_scores[i]:.2f})" for i in range(len(top_ids))]
            x0, y0 = int(corners_np[0][0]), max(20, int(corners_np[0][1]) - 10)
            for i, line in enumerate(lines[:5]):
                y = y0 + i * 22
                cv2.putText(annotated, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if "used_fallback_opencv" in dbg:
                cv2.putText(annotated, "(fallback: OpenCV)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(annotated, "(YOLO-seg)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if timer is not None:
                # small timing overlay (top-left)
                y = 60
                for key in (
                    "1_yolo_predict",
                    "2_yolo_to_quad",
                    "2_warp_birdseye",
                    "3_clip_preprocess",
                    "3_clip_encode",
                    "4_faiss_search",
                    "total",
                ):
                    if key in timer.ms:
                        cv2.putText(
                            annotated,
                            f"{key}: {timer.ms[key]:.1f} ms",
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        y += 22

            cv2.imshow(f"{window_prefix} - input", annotated)
            cv2.imshow(f"{window_prefix} - warped", warped)

            gallery = _render_topk_gallery(top_ids, top_scores, max_k=k)
            if gallery is not None:
                cv2.imshow(f"{window_prefix} - top{k}", gallery)

        return detected


# --- helpers ---


def _nullcontext():
    class _C:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _C()


def _iter_test_images(test_dir: Path, recursive: bool = True) -> List[Path]:
    """List common image files under a directory (used by __main__)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not test_dir.exists():
        return []

    if recursive:
        paths = [p for p in test_dir.rglob("*") if p.suffix.lower() in exts]
    else:
        paths = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    paths.sort(key=lambda p: str(p).lower())
    return paths


# --- MAIN ---
if __name__ == "__main__":
    # Use the migrated CLIP index dir.
    index_dir = STORAGE_DIR / "indices" / "model=clip__q=high"

    # Optional: enable CLIP adapter (raw-only) to better match query photos to clean index.
    adapter_path = STORAGE_DIR / "model_weights" / "clip_adapter.pt"
    use_adapter = False

    identify = get_classification_model(
        index_dir=index_dir,
        index_variant=(IndexVariant.RAW if use_adapter else IndexVariant.CENTERED_NORMALIZED),
        yolo_weights=DEFAULT_YOLO_SEG_WEIGHTS,
        adapter_path=(adapter_path if use_adapter else None),
    )

    print(
        f"Index: {index_dir} | "
        f"variant={'raw' if use_adapter else 'centered_normalized'} | "
        f"adapter={adapter_path if use_adapter else None}"
    )

    # test_dir = STORAGE_DIR / "test_data" / "images"
    test_dir = STORAGE_DIR / "debug_uploads"
    image_paths = _iter_test_images(test_dir, recursive=True)

    if not image_paths:
        print(f"No images found under: {test_dir}")
        raise SystemExit(0)

    print(f"Found {len(image_paths)} images under: {test_dir}")
    print("Keys: [n]/[space]=next, [q]/[esc]=quit")

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skipping unreadable image: {p}")
            continue

        results = identify(img)

        overlay = visualize_results_on_frame(img, results)
        gallery = render_prediction_gallery(results, max_instances=3, top_k=3)

        cv2.imshow(f"Pokemon Card Scanner - input ({p.name})", overlay)
        if gallery is not None:
            cv2.imshow("Pokemon Card Scanner - topk", gallery)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                raise SystemExit(0)
            if key in (ord("n"), ord(" ")):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
