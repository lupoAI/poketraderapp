"""Migrate legacy index directories to the new 3-variant paradigm.

This script converts legacy folders like:
  - model=clip__norm=0__q=high__fmt=jpg
  - model=dinov2-small__norm=0__q=high__fmt=jpg

into the new naming scheme without `norm=...`:
  - model=clip__q=high__fmt=jpg
  - model=dinov2-small__q=high__fmt=jpg

and ensures each migrated directory contains:
  - pokemon_cards.index                  (raw)
  - pokemon_cards_norm.index             (L2 normalized)
  - pokemon_cards_centered.index         (mean-centered, no normalization)
  - pokemon_cards_centered_norm.index    (mean-centered + L2 normalized)
  - avg_embedding.npy
  - meta.json with keys: model_type, source (kept if present), created_utc (kept if present),
    default_variant, variants, avg_embedding

It uses card_embeddings.npy (raw embeddings) to build the new indices.

Run:
  python scripts/migrate_indices_v2.py

By default modifies only the two known legacy dirs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INDICES_DIR = ROOT / "storage" / "indices"

LEGACY_DIRS = [
    "model=clip__norm=0__q=high__fmt=jpg",
    "model=dinov2-small__norm=0__q=high__fmt=jpg",
]


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _build_indices(out_dir: Path, embeddings: np.ndarray) -> Dict[str, str]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0:
        raise ValueError(f"Invalid embeddings shape: {embeddings.shape}")

    d = int(embeddings.shape[1])

    # raw
    idx_raw = faiss.IndexFlatIP(d)
    idx_raw.add(embeddings)
    raw_path = out_dir / "pokemon_cards.index"
    faiss.write_index(idx_raw, str(raw_path))

    # normalized
    emb_norm = _l2_normalize_rows(embeddings)
    idx_norm = faiss.IndexFlatIP(d)
    idx_norm.add(emb_norm)
    norm_path = out_dir / "pokemon_cards_norm.index"
    faiss.write_index(idx_norm, str(norm_path))

    # centered (no normalization)
    avg = embeddings.mean(axis=0).astype(np.float32)
    centered = embeddings - avg.reshape(1, -1)
    idx_cent = faiss.IndexFlatIP(d)
    idx_cent.add(centered)
    cent_path = out_dir / "pokemon_cards_centered.index"
    faiss.write_index(idx_cent, str(cent_path))

    # centered + normalized
    centered_norm = _l2_normalize_rows(centered)
    idx_cent_norm = faiss.IndexFlatIP(d)
    idx_cent_norm.add(centered_norm)
    cent_norm_path = out_dir / "pokemon_cards_centered_norm.index"
    faiss.write_index(idx_cent_norm, str(cent_norm_path))

    avg_path = out_dir / "avg_embedding.npy"
    np.save(str(avg_path), avg)

    return {
        "raw": raw_path.name,
        "normalized": norm_path.name,
        "raw_centered": cent_path.name,
        "centered_normalized": cent_norm_path.name,
        "avg_embedding": avg_path.name,
    }


def _parse_legacy_dirname(name: str) -> Tuple[str, Dict[str, str]]:
    # Example: model=clip__norm=0__q=high__fmt=jpg
    parts = name.split("__")
    kv: Dict[str, str] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip()

    model_type = kv.get("model", "clip")

    # Build new name WITHOUT norm
    new_parts = []
    if "model" in kv:
        new_parts.append(f"model={kv['model']}")
    if "arch" in kv:
        new_parts.append(f"arch={kv['arch']}")
    if "q" in kv:
        new_parts.append(f"q={kv['q']}")
    if "fmt" in kv:
        new_parts.append(f"fmt={kv['fmt']}")

    new_name = "__".join(new_parts) if new_parts else name.replace("__norm=0", "")
    return model_type, {"new_name": new_name, **kv}


def migrate_one(legacy_dir: Path) -> None:
    if not legacy_dir.exists():
        print(f"[SKIP] Missing: {legacy_dir}")
        return

    model_type, info = _parse_legacy_dirname(legacy_dir.name)
    new_dir = legacy_dir.parent / info["new_name"]

    if new_dir.exists() and new_dir.resolve() != legacy_dir.resolve():
        raise RuntimeError(f"Target already exists: {new_dir}")

    embeddings_path = legacy_dir / "card_embeddings.npy"
    names_path = legacy_dir / "card_names.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing card_embeddings.npy in {legacy_dir}")
    if not names_path.exists():
        raise FileNotFoundError(f"Missing card_names.npy in {legacy_dir}")

    embeddings = np.load(str(embeddings_path)).astype(np.float32)

    # Rename directory if needed
    if new_dir.resolve() != legacy_dir.resolve():
        print(f"[MOVE] {legacy_dir.name} -> {new_dir.name}")
        legacy_dir.rename(new_dir)
        legacy_dir = new_dir

    # Build/overwrite indices + avg
    print(f"[BUILD] indices in {legacy_dir}")
    variants = _build_indices(legacy_dir, embeddings)

    # Update meta
    meta_path = legacy_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    # Keep existing source/created_utc if present
    meta.setdefault("source", "migrated")
    meta.setdefault("created_utc", datetime.utcnow().isoformat() + "Z")

    meta["model_type"] = str(meta.get("model_type") or model_type)
    meta["default_variant"] = str(meta.get("default_variant") or "raw")
    meta["variants"] = {
        "raw": variants["raw"],
        "normalized": variants["normalized"],
        "raw_centered": variants["raw_centered"],
        "centered_normalized": variants["centered_normalized"],
    }
    meta["avg_embedding"] = variants["avg_embedding"]

    # Remove any old keys
    meta.pop("normalize", None)

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] migrated: {legacy_dir}")


def main() -> None:
    for d in LEGACY_DIRS:
        migrate_one(INDICES_DIR / d)


if __name__ == "__main__":
    main()

