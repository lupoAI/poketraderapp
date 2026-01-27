"""Piccola UI per revisionare velocemente un dataset YOLO (bbox o segmentazioni/poligoni).

Features:
- Scorri immagini per split (train/valid/test)
- Overlay delle annotazioni (bbox o poligoni) con colori per classe
- Hotkey per marcare KEEP / TRASH (sposta in .trash, non cancella) + UNDO
- Stato persistente su disco (JSONL)

Uso (PowerShell, dalla root del repo):
    python scripts\review_yolo_dataset_ui.py --data "storage\finetuning\poketraderfinetuning - Copia\data.yaml"

Note:
- Funziona con layout Roboflow/Ultralytics: <dataset>/<split>/images e <dataset>/<split>/labels
- Supporta label YOLO:
  - bbox:    class xc yc w h
  - polygon: class x1 y1 x2 y2 ... xN yN
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageFont, ImageTk

from model.config import STORAGE_DIR

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Dipendenza mancante: PyYAML. Installala nell'environment (conda/pip) e riprova."
    ) from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Annotation:
    class_id: int
    points_px: List[Tuple[float, float]]  # polygon in pixel coords (also used for bbox as 4-pt poly)


def _load_data_yaml(data_yaml: Path) -> dict:
    return yaml.safe_load(data_yaml.read_text(encoding="utf-8"))


def _load_class_names(data: dict) -> List[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return []


def _split_images_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / "images"


def _labels_dir_for_images_dir(images_dir: Path) -> Path:
    return images_dir.parent / "labels"


def iter_image_label_pairs(images_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    if not images_dir.exists():
        return []

    labels_dir = _labels_dir_for_images_dir(images_dir)
    out: List[Tuple[Path, Optional[Path]]] = []
    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lbl = labels_dir / f"{img_path.stem}.txt"
        out.append((img_path, lbl if lbl.exists() else None))
    return out


def _parse_label_line(parts: Sequence[str]) -> Tuple[int, List[float]]:
    if len(parts) < 6:
        raise ValueError("Label line too short")
    class_id = int(float(parts[0]))
    nums = [float(x) for x in parts[1:]]
    return class_id, nums


def parse_yolo_label_file(label_path: Path, *, img_w: int, img_h: int) -> List[Annotation]:
    anns: List[Annotation] = []
    if label_path is None or not label_path.exists():
        return anns

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return anns

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        class_id, nums = _parse_label_line(parts)

        # BBox
        if len(nums) == 4:
            xc, yc, w, h = nums
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            anns.append(Annotation(class_id=class_id, points_px=pts))
            continue

        # Polygon
        if len(nums) % 2 != 0 or len(nums) < 8:
            raise ValueError(
                f"Unsupported label format in {label_path.name}: expected 4 numbers (bbox) or even>=8 (polygon), got {len(nums)}"
            )

        pts: List[Tuple[float, float]] = []
        for i in range(0, len(nums), 2):
            x = nums[i] * img_w
            y = nums[i + 1] * img_h
            pts.append((x, y))
        anns.append(Annotation(class_id=class_id, points_px=pts))

    return anns


def _color_for_class(class_id: int) -> Tuple[int, int, int]:
    rng = random.Random(1337 + class_id)
    return (rng.randrange(30, 255), rng.randrange(30, 255), rng.randrange(30, 255))


def _label_for_class(class_id: int, class_names: Sequence[str]) -> str:
    if 0 <= class_id < len(class_names):
        return f"{class_id}:{class_names[class_id]}"
    return str(class_id)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _draw_overlay(
    img: Image.Image,
    anns: Sequence[Annotation],
    *,
    class_names: Sequence[str],
    show_polygons: bool,
    fill_polygons: bool,
    show_labels: bool,
) -> Image.Image:
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = out.size

    # Basic font fallback
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for ann in anns:
        col = _color_for_class(ann.class_id)
        col_a = (col[0], col[1], col[2], 180)
        col_fill = (col[0], col[1], col[2], 70)

        pts = [(_clamp(x, 0, w - 1), _clamp(y, 0, h - 1)) for (x, y) in ann.points_px]
        if len(pts) < 2:
            continue

        if show_polygons:
            if fill_polygons and len(pts) >= 3:
                draw.polygon(pts, fill=col_fill)
            draw.line(pts + [pts[0]], fill=col_a, width=3, joint="curve")

        if show_labels:
            x0, y0 = pts[0]
            text = _label_for_class(ann.class_id, class_names)
            # small background
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            bx0, by0 = int(x0), int(max(0, y0 - th - 6))
            draw.rectangle([bx0, by0, bx0 + tw + 6, by0 + th + 4], fill=(0, 0, 0, 120))
            draw.text((bx0 + 3, by0 + 2), text, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(out, overlay).convert("RGB")
    return out


class ReviewStore:
    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.state_path = self.dataset_root / "review_state.json"
        self.log_path = self.dataset_root / "review_log.jsonl"
        self.trash_manifest_path = self.dataset_root / ".trash" / "trash_manifest.jsonl"

        self.state: Dict[str, dict] = {}
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                self.state = {}

    def save_state(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def get_decision(self, rel_img: str) -> str:
        return str(self.state.get(rel_img, {}).get("decision", ""))

    def set_decision(self, rel_img: str, *, decision: str, split: str, reason: str = "") -> None:
        ts = time.time()
        rec = {
            "ts": ts,
            "rel_image": rel_img,
            "decision": decision,
            "split": split,
            "reason": reason,
        }
        self.state[rel_img] = {"decision": decision, "split": split, "reason": reason, "ts": ts}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.save_state()

    def append_trash_manifest(self, rec: dict) -> None:
        self.trash_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trash_manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def last_trash_record(self) -> Optional[dict]:
        if not self.trash_manifest_path.exists():
            return None
        try:
            lines = self.trash_manifest_path.read_text(encoding="utf-8").splitlines()
            for raw in reversed(lines):
                raw = raw.strip()
                if raw:
                    return json.loads(raw)
            return None
        except Exception:
            return None


def _safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(dst)
    except Exception:
        # cross-device move fallback
        shutil.copy2(src, dst)
        src.unlink()


class YoloReviewUI:
    def __init__(
        self,
        root: tk.Tk,
        *,
        dataset_root: Path,
        class_names: Sequence[str],
        start_split: str,
        shuffle: bool,
        only_unreviewed: bool,
    ):
        self.root = root
        self.dataset_root = Path(dataset_root)
        self.class_names = list(class_names)

        self.store = ReviewStore(self.dataset_root)

        self.splits = ["train", "valid", "test"]
        self.split = start_split

        self.show_polygons = True
        self.fill_polygons = True
        self.show_labels = True

        self.only_unreviewed = bool(only_unreviewed)

        self.pairs_by_split: Dict[str, List[Tuple[Path, Optional[Path]]]] = {}
        for sp in self.splits:
            pairs = iter_image_label_pairs(_split_images_dir(self.dataset_root, sp))
            if shuffle:
                random.shuffle(pairs)
            self.pairs_by_split[sp] = pairs

        self.idx = 0
        self._apply_only_unreviewed_filter()

        # UI
        self.root.title("YOLO dataset review")
        self.root.geometry("1300x900")

        self._build_widgets()
        self._bind_keys()

        self._render_current()

    def _apply_only_unreviewed_filter(self) -> None:
        if not self.only_unreviewed:
            return
        for sp in list(self.pairs_by_split.keys()):
            filtered: List[Tuple[Path, Optional[Path]]] = []
            for img_path, lbl_path in self.pairs_by_split[sp]:
                rel = str(img_path.relative_to(self.dataset_root)).replace("\\", "/")
                if not self.store.get_decision(rel):
                    filtered.append((img_path, lbl_path))
            self.pairs_by_split[sp] = filtered

    def _build_widgets(self) -> None:
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, bg="#111")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.side = tk.Frame(self.frame, width=360)
        self.side.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_info = tk.Label(self.side, text="", justify=tk.LEFT, anchor="nw")
        self.lbl_info.pack(fill=tk.X, padx=10, pady=10)

        self.btns = tk.Frame(self.side)
        self.btns.pack(fill=tk.X, padx=10)

        tk.Button(self.btns, text="<< Prev (A)", command=self.prev).grid(row=0, column=0, sticky="ew")
        tk.Button(self.btns, text="Next (D) >>", command=self.next).grid(row=0, column=1, sticky="ew")
        self.btns.columnconfigure(0, weight=1)
        self.btns.columnconfigure(1, weight=1)

        self.btns2 = tk.Frame(self.side)
        self.btns2.pack(fill=tk.X, padx=10, pady=(8, 0))
        tk.Button(self.btns2, text="KEEP (K)", command=self.mark_keep).grid(row=0, column=0, sticky="ew")
        tk.Button(self.btns2, text="TRASH (X)", command=self.mark_trash_confirm).grid(row=0, column=1, sticky="ew")
        self.btns2.columnconfigure(0, weight=1)
        self.btns2.columnconfigure(1, weight=1)

        self.btns3 = tk.Frame(self.side)
        self.btns3.pack(fill=tk.X, padx=10, pady=(8, 0))
        tk.Button(self.btns3, text="UNDO (U)", command=self.undo_last_trash).grid(row=0, column=0, sticky="ew")
        tk.Button(self.btns3, text="Open folder (O)", command=self.open_in_explorer).grid(row=0, column=1, sticky="ew")
        self.btns3.columnconfigure(0, weight=1)
        self.btns3.columnconfigure(1, weight=1)

        self.chk_frame = tk.Frame(self.side)
        self.chk_frame.pack(fill=tk.X, padx=10, pady=10)

        self.var_poly = tk.IntVar(value=1)
        self.var_fill = tk.IntVar(value=1)
        self.var_lbl = tk.IntVar(value=1)

        tk.Checkbutton(self.chk_frame, text="Show polygons (P)", variable=self.var_poly, command=self._sync_opts).pack(anchor="w")
        tk.Checkbutton(self.chk_frame, text="Fill polygons (M)", variable=self.var_fill, command=self._sync_opts).pack(anchor="w")
        tk.Checkbutton(self.chk_frame, text="Show labels (L)", variable=self.var_lbl, command=self._sync_opts).pack(anchor="w")

        self.lbl_help = tk.Label(
            self.side,
            text=(
                "Hotkeys:\n"
                "  A/D prev/next\n"
                "  1/2/3 split train/valid/test\n"
                "  K keep\n"
                "  X trash (confirm)\n"
                "  U undo last trash\n"
                "  P polygons, M fill, L labels\n"
                "  O open folder\n"
                "  Q quit"
            ),
            justify=tk.LEFT,
            anchor="nw",
            fg="#444",
        )
        self.lbl_help.pack(fill=tk.X, padx=10, pady=(10, 0))

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-a>", lambda e: self.prev())
        self.root.bind("<KeyPress-d>", lambda e: self.next())
        self.root.bind("<KeyPress-k>", lambda e: self.mark_keep())
        self.root.bind("<KeyPress-x>", lambda e: self.mark_trash_confirm())
        self.root.bind("<KeyPress-u>", lambda e: self.undo_last_trash())
        self.root.bind("<KeyPress-p>", lambda e: self.toggle_poly())
        self.root.bind("<KeyPress-m>", lambda e: self.toggle_fill())
        self.root.bind("<KeyPress-l>", lambda e: self.toggle_labels())
        self.root.bind("<KeyPress-o>", lambda e: self.open_in_explorer())
        self.root.bind("<KeyPress-1>", lambda e: self.set_split("train"))
        self.root.bind("<KeyPress-2>", lambda e: self.set_split("valid"))
        self.root.bind("<KeyPress-3>", lambda e: self.set_split("test"))
        self.root.bind("<KeyPress-q>", lambda e: self.root.destroy())

        # re-render on resize
        self.canvas.bind("<Configure>", lambda e: self._render_current())

    def _sync_opts(self) -> None:
        self.show_polygons = bool(self.var_poly.get())
        self.fill_polygons = bool(self.var_fill.get())
        self.show_labels = bool(self.var_lbl.get())
        self._render_current()

    def toggle_poly(self) -> None:
        self.var_poly.set(0 if self.var_poly.get() else 1)
        self._sync_opts()

    def toggle_fill(self) -> None:
        self.var_fill.set(0 if self.var_fill.get() else 1)
        self._sync_opts()

    def toggle_labels(self) -> None:
        self.var_lbl.set(0 if self.var_lbl.get() else 1)
        self._sync_opts()

    def current_pairs(self) -> List[Tuple[Path, Optional[Path]]]:
        return self.pairs_by_split.get(self.split, [])

    def _current_item(self) -> Optional[Tuple[Path, Optional[Path]]]:
        pairs = self.current_pairs()
        if not pairs:
            return None
        self.idx = max(0, min(len(pairs) - 1, self.idx))
        return pairs[self.idx]

    def set_split(self, split: str) -> None:
        if split not in self.splits:
            return
        self.split = split
        self.idx = 0
        self._render_current()

    def prev(self) -> None:
        self.idx = max(0, self.idx - 1)
        self._render_current()

    def next(self) -> None:
        self.idx = self.idx + 1
        if self.idx >= len(self.current_pairs()):
            self.idx = max(0, len(self.current_pairs()) - 1)
        self._render_current()

    def mark_keep(self) -> None:
        item = self._current_item()
        if item is None:
            return
        img_path, _ = item
        rel = str(img_path.relative_to(self.dataset_root)).replace("\\", "/")
        self.store.set_decision(rel, decision="keep", split=self.split)
        self._render_current()
        self.next()

    def mark_trash_confirm(self) -> None:
        item = self._current_item()
        if item is None:
            return
        img_path, lbl_path = item

        rel = str(img_path.relative_to(self.dataset_root)).replace("\\", "/")
        msg = (
            "Spostare questa immagine (e la label, se presente) in .trash?\n\n"
            f"{rel}\n\n"
            "Puoi fare UNDO (U) dopo."
        )
        if not messagebox.askyesno("TRASH (sposta, non cancella)", msg):
            return

        token = time.strftime("%Y%m%d_%H%M%S")
        trash_root = self.dataset_root / ".trash" / token

        # destination paths mirror structure split/images and split/labels
        dst_img = trash_root / img_path.relative_to(self.dataset_root)
        _safe_move(img_path, dst_img)

        dst_lbl = None
        if lbl_path is not None and lbl_path.exists():
            dst_lbl = trash_root / lbl_path.relative_to(self.dataset_root)
            _safe_move(lbl_path, dst_lbl)

        rec = {
            "ts": time.time(),
            "token": token,
            "split": self.split,
            "src_image": rel,
            "dst_image": str(dst_img.relative_to(self.dataset_root)).replace("\\", "/"),
            "src_label": (str(lbl_path.relative_to(self.dataset_root)).replace("\\", "/") if lbl_path else None),
            "dst_label": (str(dst_lbl.relative_to(self.dataset_root)).replace("\\", "/") if dst_lbl else None),
        }
        self.store.append_trash_manifest(rec)
        self.store.set_decision(rel, decision="trash", split=self.split)

        # remove from list to avoid broken navigation
        pairs = self.current_pairs()
        try:
            pairs.pop(self.idx)
        except Exception:
            pass
        if self.idx >= len(pairs):
            self.idx = max(0, len(pairs) - 1)

        self._render_current()

    def undo_last_trash(self) -> None:
        rec = self.store.last_trash_record()
        if not rec:
            messagebox.showinfo("UNDO", "Nessuna operazione di trash trovata.")
            return

        src_img = self.dataset_root / rec["dst_image"]
        dst_img = self.dataset_root / rec["src_image"]
        if not src_img.exists():
            messagebox.showwarning("UNDO", f"File non trovato in trash: {src_img}")
            return

        if messagebox.askyesno("UNDO", f"Ripristinare?\n\n{rec['src_image']}"):
            _safe_move(src_img, dst_img)
            if rec.get("dst_label") and rec.get("src_label"):
                src_lbl = self.dataset_root / rec["dst_label"]
                dst_lbl = self.dataset_root / rec["src_label"]
                if src_lbl.exists():
                    _safe_move(src_lbl, dst_lbl)

            # mark decision cleared (empty)
            self.store.set_decision(rec["src_image"], decision="", split=rec.get("split", ""))
            messagebox.showinfo("UNDO", "Ripristinato. Nota: l'entry nel manifest resta per audit.")

            # reload split lists (cheap)
            for sp in self.splits:
                self.pairs_by_split[sp] = iter_image_label_pairs(_split_images_dir(self.dataset_root, sp))
            self._apply_only_unreviewed_filter()
            self._render_current()

    def open_in_explorer(self) -> None:
        item = self._current_item()
        if item is None:
            return
        img_path, _ = item
        folder = img_path.parent
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception:
            messagebox.showwarning("Open folder", f"Impossibile aprire: {folder}")

    def _render_current(self) -> None:
        self.canvas.delete("all")
        item = self._current_item()
        if item is None:
            self.lbl_info.config(text=f"Split: {self.split}\n(0 immagini)")
            return

        img_path, lbl_path = item

        # load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.lbl_info.config(text=f"Errore lettura immagine:\n{img_path}\n{e}")
            return

        w, h = img.size
        anns: List[Annotation] = []
        err = ""
        if lbl_path is not None and lbl_path.exists():
            try:
                anns = parse_yolo_label_file(lbl_path, img_w=w, img_h=h)
            except Exception as e:
                err = str(e)
                anns = []

        # overlay
        vis = _draw_overlay(
            img,
            anns,
            class_names=self.class_names,
            show_polygons=self.show_polygons,
            fill_polygons=self.fill_polygons,
            show_labels=self.show_labels,
        )

        # fit-to-canvas
        cw = max(1, int(self.canvas.winfo_width()))
        ch = max(1, int(self.canvas.winfo_height()))
        scale = min(cw / vis.width, ch / vis.height)
        new_w = max(1, int(vis.width * scale))
        new_h = max(1, int(vis.height * scale))
        vis2 = vis.resize((new_w, new_h), Image.BILINEAR)

        self._tk_img = ImageTk.PhotoImage(vis2)
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor=tk.CENTER)

        rel = str(img_path.relative_to(self.dataset_root)).replace("\\", "/")
        decision = self.store.get_decision(rel)
        info = [
            f"Split: {self.split}",
            f"Item: {self.idx + 1}/{len(self.current_pairs())}",
            "",
            f"Image: {img_path.name}",
            f"Label: {('NONE' if lbl_path is None else lbl_path.name)}",
            f"Instances: {len(anns)}",
            f"Decision: {decision or '(unreviewed)'}",
        ]
        if err:
            info += ["", "Label parse error:", err]
        self.lbl_info.config(text="\n".join(info))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="UI per review dataset YOLO (keep/trash).")
    parser.add_argument(
        "--data",
        default=str(STORAGE_DIR / "finetuning" / "poketraderfinetuning - Copia" / "data.yaml"),
        help="Path a data.yaml del dataset.",
    )
    parser.add_argument("--split", choices=["train", "valid", "val", "test"], default="train")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--only-unreviewed", action="store_true")
    args = parser.parse_args(argv)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml non trovato: {data_yaml}")

    data = _load_data_yaml(data_yaml)
    class_names = _load_class_names(data)
    dataset_root = data_yaml.parent

    start_split = "valid" if args.split == "val" else args.split

    root = tk.Tk()
    ui = YoloReviewUI(
        root,
        dataset_root=dataset_root,
        class_names=class_names,
        start_split=start_split,
        shuffle=args.shuffle,
        only_unreviewed=args.only_unreviewed,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

