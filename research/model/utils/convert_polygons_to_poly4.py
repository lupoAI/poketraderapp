"""Convertitore: label YOLO con poligoni -> poligoni con 4 vertici (non necessariamente rettangoli).

Obiettivo
---------
Se hai label come poligoni (class x1 y1 x2 y2 ... xN yN) e vuoi ridurle a 4 vertici
(una "quad"), questo script prova a mantenere la forma/prospettiva senza forzare un rettangolo.

Strategia (robusta, a step):
1) Per ogni poligono, costruisce una contour in pixel.
2) Prova a semplificare con cv2.approxPolyDP (Douglas–Peucker) finché ottiene 4 punti.
3) Se non riesce, fa fallback a: convex hull -> approxPolyDP.
4) Se ancora non riesce, fa fallback a una quad "best effort":
   - prende il quadrilatero con punti estremi (tl,tr,br,bl) da un set di punti (ruotato/convesso).

Output
------
Scrive label in formato YOLO-segmentation a 4 vertici:
    class x1 y1 x2 y2 x3 y3 x4 y4

Esempio (PowerShell)
-------------------
python -m model.convert_polygons_to_poly4 `
  --src storage\finetuning\poketraderfinetuning `
  --dst storage\finetuning\poketraderfinetuning_poly4 `
  --splits train valid test `
  --preview 12 --preview-split valid --preview-dir storage\vis_poly4

Nota importante
--------------
- Questo NON è OBB: è ancora un poligono (segmentazione) ma con 4 vertici.
- Se il tuo trainer supporta SOLO bbox/obb e non segmentazione, questa conversione non basta.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


def _read_data_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_data_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _copy_tree_images(src_root: Path, dst_root: Path, splits: Sequence[str]) -> None:
    for split in splits:
        src_images = src_root / split / "images"
        if not src_images.exists():
            continue
        dst_images = dst_root / split / "images"
        dst_images.parent.mkdir(parents=True, exist_ok=True)
        if dst_images.exists():
            shutil.rmtree(dst_images)
        shutil.copytree(src_images, dst_images)


def _make_dst_data_yaml(src_data_yaml: Path, dst_root: Path) -> None:
    data = _read_data_yaml(src_data_yaml)
    data["train"] = "../train/images"
    data["val"] = "../valid/images"
    data["test"] = "../test/images"
    _write_data_yaml(dst_root / "data.yaml", data)


def _parse_label_lines(path: Path) -> List[Tuple[int, List[float]]]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    out: List[Tuple[int, List[float]]] = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
        out.append((class_id, nums))
    return out


def _nums_to_pts_norm(nums: List[float]) -> List[Tuple[float, float]]:
    if len(nums) % 2 != 0:
        raise ValueError("Odd number of polygon coordinates")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _bbox_to_pts_norm(nums: List[float]) -> List[Tuple[float, float]]:
    xc, yc, w, h = nums
    return [(xc - w / 2.0, yc - h / 2.0), (xc + w / 2.0, yc - h / 2.0), (xc + w / 2.0, yc + h / 2.0), (xc - w / 2.0, yc + h / 2.0)]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points into (tl, tr, br, bl) in pixel space."""
    # pts shape (4,2)
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

    # try a range of eps fractions: small -> larger
    for frac in (0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
        eps = frac * peri
        approx = cv2.approxPolyDP(contour, eps, True)
        if approx is not None and len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _fallback_extreme_quad(points_xy: np.ndarray) -> np.ndarray:
    """Best-effort quad from a cloud of points.

    Uses extremes in sum/diff space (common heuristic for quadrilateral corners).
    """
    pts = points_xy.astype(np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    quad = np.stack([tl, tr, br, bl], axis=0)

    # de-duplicate if heuristic collapses points
    # if too many duplicates, jitter slightly (still deterministic-ish)
    uniq = np.unique(quad.round(decimals=2), axis=0)
    if len(uniq) < 4:
        quad = quad + np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return quad


def polygon_to_quad_px(pts_px: List[Tuple[float, float]]) -> np.ndarray:
    """Convert N-point polygon to 4-point quadrilateral in pixel space."""
    pts = np.array(pts_px, dtype=np.float32).reshape((-1, 2))
    if len(pts) < 4:
        # pad by repeating last point
        while len(pts) < 4:
            pts = np.vstack([pts, pts[-1:]])
        return _order_points_tl_tr_br_bl(pts[:4])

    contour = pts.reshape((-1, 1, 2))

    approx = _try_approx_to_4(contour)
    if approx is not None:
        return _order_points_tl_tr_br_bl(approx)

    hull = cv2.convexHull(contour)
    approx = _try_approx_to_4(hull)
    if approx is not None:
        return _order_points_tl_tr_br_bl(approx)

    # final fallback
    quad = _fallback_extreme_quad(pts)
    return _order_points_tl_tr_br_bl(quad)


def to_line_poly4(class_id: int, quad_norm: List[Tuple[float, float]]) -> str:
    flat: List[str] = []
    for x, y in quad_norm:
        flat.append(f"{_clamp01(x):.10f}")
        flat.append(f"{_clamp01(y):.10f}")
    return f"{class_id} " + " ".join(flat)


def _find_image_for_label(dataset_root: Path, split: str, label_file: Path) -> Optional[Path]:
    images_dir = dataset_root / split / "images"
    stem = label_file.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def convert_dataset(src_root: Path, dst_root: Path, splits: Sequence[str]) -> Tuple[int, int]:
    converted_files = 0
    skipped_no_img = 0

    for split in splits:
        src_labels_dir = src_root / split / "labels"
        if not src_labels_dir.exists():
            continue

        for src_label in sorted(src_labels_dir.glob("*.txt")):
            img_path = _find_image_for_label(src_root, split, src_label)
            if img_path is None:
                skipped_no_img += 1
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                skipped_no_img += 1
                continue
            h, w = img.shape[:2]

            out_lines: List[str] = []
            for class_id, nums in _parse_label_lines(src_label):
                if len(nums) == 4:
                    pts_norm = _bbox_to_pts_norm(nums)
                elif len(nums) >= 8 and len(nums) % 2 == 0:
                    pts_norm = _nums_to_pts_norm(nums)
                else:
                    continue

                pts_px = [(x * w, y * h) for x, y in pts_norm]
                quad_px = polygon_to_quad_px(pts_px)  # (4,2)
                quad_norm = [(float(x) / w, float(y) / h) for x, y in quad_px]
                out_lines.append(to_line_poly4(class_id, quad_norm))

            rel = src_label.relative_to(src_root)
            dst_label = dst_root / rel
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            converted_files += 1

    return converted_files, skipped_no_img


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Converte poligoni YOLO in poligoni a 4 vertici (quad).")
    parser.add_argument("--src", default=str(Path("storage") / "finetuning" / "poketraderfinetuning"))
    parser.add_argument("--dst", default=str(Path("storage") / "finetuning" / "poketraderfinetuning_poly4"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--preview-split", default="valid")
    parser.add_argument("--preview-dir", default=str(Path("storage") / "vis_poly4"))
    args = parser.parse_args(argv)

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    splits = [s.lower() for s in args.splits]

    src_data_yaml = src_root / "data.yaml"
    if not src_data_yaml.exists():
        raise FileNotFoundError(f"data.yaml non trovato in: {src_data_yaml}")

    dst_root.mkdir(parents=True, exist_ok=True)
    _copy_tree_images(src_root, dst_root, splits)

    converted, skipped = convert_dataset(src_root, dst_root, splits)
    _make_dst_data_yaml(src_data_yaml, dst_root)

    print(f"Done. Converted label files: {converted}. Skipped (no image): {skipped}.")
    print(f"POLY4 dataset root: {dst_root}")

    if args.preview and args.preview > 0:
        try:
            from model.visualize_yolo_dataset import main as vis_main  # type: ignore

            vis_main(
                [
                    "--data",
                    str(dst_root / "data.yaml"),
                    "--split",
                    args.preview_split,
                    "--max-samples",
                    str(args.preview),
                    "--save-dir",
                    str(args.preview_dir),
                    "--shuffle",
                ]
            )
        except Exception as e:
            print(f"Preview failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

