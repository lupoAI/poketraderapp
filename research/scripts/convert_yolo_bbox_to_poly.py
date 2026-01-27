"""Converte label YOLO con bbox (xc yc w h) in poligoni a 4 lati (x1 y1 ... x4 y4).

Motivazione:
- In dataset YOLO segmentation/OBB può capitare di avere un mix di righe:
  - bbox:    class xc yc w h
  - polygon: class x1 y1 x2 y2 ... xN yN
- Questo script trova SOLO le righe bbox e le riscrive come poligono 4 punti.

Cosa fa:
- Scansiona una directory labels (ricorsiva) o un dataset root (train/valid/test)
- Per ogni file .txt:
  - per ogni riga:
    - se ha 4 numeri -> converte in 8 numeri (rettangolo)
    - altrimenti lascia invariato
- Opzionale backup dei file originali

Uso (PowerShell):
    python scripts\convert_yolo_bbox_to_poly.py --labels "storage\finetuning\poketraderfinetuning - Copia\train\labels" --backup

Oppure sul dataset root:
    python scripts\convert_yolo_bbox_to_poly.py --dataset "storage\finetuning\poketraderfinetuning - Copia" --backup

Dry-run (non scrive):
    python scripts\convert_yolo_bbox_to_poly.py --dataset "..." --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


IMG_LABEL_SUFFIX = ".txt"


@dataclass
class ConvertStats:
    files_seen: int = 0
    files_changed: int = 0
    lines_seen: int = 0
    bbox_lines_converted: int = 0
    parse_errors: int = 0


def _iter_label_files_under(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == IMG_LABEL_SUFFIX:
        yield path
        return
    if not path.exists():
        return
    for p in sorted(path.rglob("*.txt")):
        if p.is_file():
            yield p


def _format_nums(nums: List[float], *, precision: int) -> List[str]:
    # keep reasonably compact while stable
    fmt = f"{{:.{precision}f}}"
    return [fmt.format(x).rstrip("0").rstrip(".") if precision > 0 else str(x) for x in nums]


def _convert_bbox_nums_to_poly_nums(xc: float, yc: float, w: float, h: float) -> List[float]:
    # YOLO normalized bbox -> normalized polygon (rectangle) clockwise
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _try_parse_floats(parts: List[str]) -> Optional[List[float]]:
    try:
        return [float(x) for x in parts]
    except Exception:
        return None


def convert_file(label_path: Path, *, precision: int) -> Tuple[bool, str, ConvertStats]:
    """Returns (changed, new_text, stats_for_this_file)."""
    stats = ConvertStats(files_seen=1)

    raw = label_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    out_lines: List[str] = []
    changed = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            out_lines.append("")
            continue

        parts = line.split()
        if len(parts) < 5:
            # not a valid yolo line, keep as-is
            stats.lines_seen += 1
            out_lines.append(raw_line)
            continue

        class_token = parts[0]
        nums = _try_parse_floats(parts[1:])
        stats.lines_seen += 1
        if nums is None:
            stats.parse_errors += 1
            out_lines.append(raw_line)
            continue

        if len(nums) == 4:
            xc, yc, w, h = nums
            poly = _convert_bbox_nums_to_poly_nums(xc, yc, w, h)
            out_nums = _format_nums(poly, precision=precision)
            out_lines.append(" ".join([class_token] + out_nums))
            changed = True
            stats.bbox_lines_converted += 1
        else:
            out_lines.append(raw_line)

    new_text = "\n".join(out_lines) + ("\n" if raw.endswith("\n") else "")
    if changed:
        stats.files_changed = 1
    return changed, new_text, stats


def convert_labels(
    labels_path: Path,
    *,
    dry_run: bool,
    backup: bool,
    backup_suffix: str,
    precision: int,
) -> ConvertStats:
    stats_total = ConvertStats()

    for lbl in _iter_label_files_under(labels_path):
        changed, new_text, st = convert_file(lbl, precision=precision)
        stats_total.files_seen += st.files_seen
        stats_total.files_changed += st.files_changed
        stats_total.lines_seen += st.lines_seen
        stats_total.bbox_lines_converted += st.bbox_lines_converted
        stats_total.parse_errors += st.parse_errors

        if not changed:
            continue

        if dry_run:
            continue

        if backup:
            backup_path = lbl.with_suffix(lbl.suffix + backup_suffix)
            if not backup_path.exists():
                shutil.copy2(lbl, backup_path)

        lbl.write_text(new_text, encoding="utf-8")

    return stats_total


def _guess_label_dirs_from_dataset_root(dataset_root: Path) -> List[Path]:
    out: List[Path] = []
    for split in ("train", "valid", "val", "test"):
        d = dataset_root / split / "labels"
        if d.exists() and d.is_dir():
            out.append(d)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert YOLO bbox labels to 4-point polygons.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--labels", type=str, help="Path alla cartella labels (anche ricorsiva).")
    g.add_argument("--dataset", type=str, help="Path alla root dataset YOLO (train/valid/test).")

    p.add_argument("--dry-run", action="store_true", help="Non scrive file; stampa solo statistiche.")
    p.add_argument("--backup", action="store_true", help="Crea un backup affiancato (*.txt.bak).")
    p.add_argument("--backup-suffix", default=".bak", help="Suffix backup (default .bak).")
    p.add_argument("--precision", type=int, default=6, help="Decimali per output (default 6).")

    args = p.parse_args(argv)

    roots: List[Path]
    if args.labels:
        roots = [Path(args.labels)]
    else:
        dataset_root = Path(args.dataset)
        roots = _guess_label_dirs_from_dataset_root(dataset_root)
        if not roots:
            raise SystemExit(f"Nessuna cartella labels trovata sotto: {dataset_root}")

    total = ConvertStats()
    for r in roots:
        st = convert_labels(
            r,
            dry_run=bool(args.dry_run),
            backup=bool(args.backup),
            backup_suffix=str(args.backup_suffix),
            precision=int(args.precision),
        )
        total.files_seen += st.files_seen
        total.files_changed += st.files_changed
        total.lines_seen += st.lines_seen
        total.bbox_lines_converted += st.bbox_lines_converted
        total.parse_errors += st.parse_errors

    print(
        "\n".join(
            [
                "DONE.",
                f"Files visti:      {total.files_seen}",
                f"Files cambiati:   {total.files_changed}",
                f"Linee viste:      {total.lines_seen}",
                f"BBox convertite:  {total.bbox_lines_converted}",
                f"Parse errors:     {total.parse_errors}",
                f"Dry-run:          {bool(args.dry_run)}",
                f"Backup:           {bool(args.backup)}",
            ]
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

