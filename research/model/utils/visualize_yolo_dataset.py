"""Visualizza coppie immagine/label in un dataset YOLO (v8/v11 style).

Supporta:
- BBox:   class x_center y_center width height (valori normalizzati 0..1)
- Poligono/OBB: class x1 y1 x2 y2 ... xN yN (normalizzati 0..1)

Esempio (PowerShell):
    python -m model.visualize_yolo_dataset --split valid --max-samples 12 --save-dir storage\\vis

Note:
- Per evitare finestre popup su Windows, di default salva su disco.
- Se usi --show prova ad aprire una finestra OpenCV (può non funzionare in ambienti headless).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Annotation:
    class_id: int
    points_px: List[Tuple[float, float]]  # polygon points in pixels


def _repo_root_from_this_file() -> Path:
    # model/visualize_yolo_dataset.py -> repo root
    return Path(__file__).resolve().parents[1]


def _resolve_existing_path(p: Path) -> Path:
    """Try to resolve a path that might be relative to CWD or repo root."""
    if p.is_absolute() and p.exists():
        return p

    # relative to current working directory
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # relative to repo root (useful when script is launched from elsewhere)
    repo_candidate = (_repo_root_from_this_file() / p).resolve()
    if repo_candidate.exists():
        return repo_candidate

    # return the most helpful candidate for error reporting
    return cwd_candidate


def _resolve_dataset_root(data_yaml: Path) -> Path:
    """Assume data.yaml is inside the dataset folder."""
    return data_yaml.parent


def _load_names_from_data_yaml(data_yaml: Path) -> List[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        # Ultralytics sometimes stores as {0: 'name', 1: 'name2'}
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return []


def _split_images_dir(dataset_root: Path, split: str) -> Path:
    split = split.lower()
    # Roboflow/Ultralytics common layout: <root>/<split>/images
    return dataset_root / split / "images"


def _labels_dir_for_images_dir(images_dir: Path) -> Path:
    # sibling folder: <split>/labels
    return images_dir.parent / "labels"


def iter_image_label_pairs(images_dir: Path) -> Iterable[Tuple[Path, Optional[Path]]]:
    """Yields (image_path, label_path_or_None)."""
    if not images_dir.exists():
        return

    labels_dir = _labels_dir_for_images_dir(images_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts):
        label_path = labels_dir / (img_path.stem + ".txt")
        yield img_path, (label_path if label_path.exists() else None)


def _parse_label_line_to_norm(class_and_nums: Sequence[str]) -> Tuple[int, List[float]]:
    if len(class_and_nums) < 6:
        raise ValueError("Label line too short")
    class_id = int(float(class_and_nums[0]))
    nums = [float(x) for x in class_and_nums[1:]]
    return class_id, nums


def parse_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> List[Annotation]:
    """Parse a YOLO label file into pixel-space polygon annotations."""
    annotations: List[Annotation] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return annotations

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        class_id, nums = _parse_label_line_to_norm(parts)

        # BBox: x y w h (4 numbers)
        if len(nums) == 4:
            xc, yc, w, h = nums
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            annotations.append(Annotation(class_id=class_id, points_px=pts))
            continue

        # Polygon/OBB: x1 y1 x2 y2 ... xN yN
        if len(nums) % 2 != 0 or len(nums) < 8:
            raise ValueError(
                f"Unsupported label format in {label_path.name}: expected 4 numbers (bbox) or even>=8 (polygon), got {len(nums)}"
            )

        pts: List[Tuple[float, float]] = []
        for i in range(0, len(nums), 2):
            x = nums[i] * img_w
            y = nums[i + 1] * img_h
            pts.append((x, y))
        annotations.append(Annotation(class_id=class_id, points_px=pts))

    return annotations


def _color_for_class(class_id: int) -> Tuple[int, int, int]:
    # Deterministic pseudo-random color
    rng = random.Random(class_id + 1337)
    return (rng.randrange(30, 255), rng.randrange(30, 255), rng.randrange(30, 255))


def _label_for_class(class_id: int, class_names: Sequence[str]) -> str:
    if 0 <= class_id < len(class_names):
        return f"{class_id}:{class_names[class_id]}"
    return str(class_id)


def _draw_polygon(img_bgr, pts_i: List[Tuple[int, int]], color: Tuple[int, int, int]) -> None:
    if len(pts_i) < 2:
        return
    contour = np.array(pts_i, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_bgr, [contour], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_annotations(image_bgr, annotations: Sequence[Annotation], class_names: Sequence[str] = ()):
    out = image_bgr.copy()

    for ann in annotations:
        color = _color_for_class(ann.class_id)
        pts_i = [(int(round(x)), int(round(y))) for x, y in ann.points_px]
        _draw_polygon(out, pts_i, color)

        # Label anchored to the first point (clamped)
        x0, y0 = pts_i[0]
        x0 = max(0, min(out.shape[1] - 1, x0))
        y0 = max(0, min(out.shape[0] - 1, y0))
        cv2.putText(
            out,
            _label_for_class(ann.class_id, class_names),
            (x0, max(0, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def annotate_image(image_path: Path, label_path: Optional[Path], class_names: Sequence[str]):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossibile leggere immagine: {image_path}")

    h, w = img.shape[:2]

    annotations: List[Annotation] = []
    if label_path is not None:
        annotations = parse_yolo_label_file(label_path, img_w=w, img_h=h)

    out = draw_annotations(img, annotations, class_names)
    return out, annotations


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualizza label YOLO su immagini (bbox/OBB/poligoni).")
    parser.add_argument("--data", default=str(Path("storage") / "finetuning" / "poketraderfinetuning" / "data.yaml"))
    parser.add_argument("--split", choices=["train", "valid", "val", "test"], default="valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save-dir", default=str(Path("storage") / "vis"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    random.seed(args.seed)

    data_yaml_arg = Path(args.data)
    data_yaml = _resolve_existing_path(data_yaml_arg)
    if not data_yaml.exists():
        repo_guess = (_repo_root_from_this_file() / data_yaml_arg).resolve()
        cwd_guess = (Path.cwd() / data_yaml_arg).resolve()
        raise FileNotFoundError(
            "data.yaml non trovato. Prova a lanciare lo script dalla root del repo oppure passa --data con un path assoluto.\n"
            f"Argomento: {data_yaml_arg}\n"
            f"Provati:  \n"
            f" - CWD : {cwd_guess}\n"
            f" - REPO: {repo_guess}\n"
        )

    dataset_root = _resolve_dataset_root(data_yaml)
    class_names = _load_names_from_data_yaml(data_yaml)

    split = "valid" if args.split == "val" else args.split
    images_dir = _split_images_dir(dataset_root, split)

    pairs = list(iter_image_label_pairs(images_dir))
    if args.shuffle:
        random.shuffle(pairs)

    pairs = pairs[: max(0, args.max_samples)]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, lbl_path) in enumerate(pairs, start=1):
        out_img, anns = annotate_image(img_path, lbl_path, class_names)
        out_path = save_dir / f"{split}_{i:03d}_{img_path.name}"
        cv2.imwrite(str(out_path), out_img)

        print(
            f"[{i}/{len(pairs)}] {img_path.name} -> {out_path.name} | label: {('NONE' if lbl_path is None else lbl_path.name)} | ann: {len(anns)}"
        )

        if args.show:
            cv2.imshow("yolo-dataset-preview", out_img)
            key = cv2.waitKey(0)
            if key in (27, ord("q")):
                break

    if args.show:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

