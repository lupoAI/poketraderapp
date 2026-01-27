from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class IndexVariant:
    RAW = "raw"
    NORMALIZED = "normalized"
    RAW_CENTERED = "raw_centered"
    CENTERED_NORMALIZED = "centered_normalized"


@dataclass(frozen=True)
class IndexBundle:
    """Bundle coerente: (faiss index + card_names + meta).

    Il contratto dell'app ora e` passare solo `index_dir`.
    In questa directory ci aspettiamo sempre:
      - pokemon_cards.index (+ varianti)
      - card_names.npy
      - meta.json

    `normalize` e` derivato dalla `variant` (non dal meta).
    """

    index_dir: Path
    index: Any  # faiss.Index
    card_names: np.ndarray
    model_type: str
    normalize: bool
    variant: str
    avg_embedding: np.ndarray | None


_DEFAULT_INDEX_FILENAME = "pokemon_cards.index"
_DEFAULT_INDEX_NORM_FILENAME = "pokemon_cards_norm.index"
_DEFAULT_INDEX_CENTERED_FILENAME = "pokemon_cards_centered.index"
_DEFAULT_INDEX_CENTERED_NORM_FILENAME = "pokemon_cards_centered_norm.index"
_DEFAULT_NAMES_FILENAME = "card_names.npy"
_DEFAULT_AVG_FILENAME = "avg_embedding.npy"
_DEFAULT_META_FILENAME = "meta.json"


def load_index_bundle(index_dir: str | Path, *, variant: str | None = None) -> IndexBundle:
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"index_dir not found: {index_dir}")

    names_path = index_dir / _DEFAULT_NAMES_FILENAME
    meta_path = index_dir / _DEFAULT_META_FILENAME

    if not names_path.exists():
        raise FileNotFoundError(f"card_names.npy not found: {names_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_type = str(meta["model_type"]) if "model_type" in meta else "clip"

    # Determine variant
    if variant is None:
        variant = meta.get("default_variant")
        if variant is None:
            raise KeyError(f"meta.json missing required key: default_variant ({meta_path})")

    variant = str(variant)
    if variant not in {IndexVariant.RAW, IndexVariant.NORMALIZED, IndexVariant.RAW_CENTERED, IndexVariant.CENTERED_NORMALIZED}:
        raise ValueError(f"Unknown index variant: {variant!r}")

    # Pick index filename
    if variant == IndexVariant.RAW:
        index_path = index_dir / _DEFAULT_INDEX_FILENAME
    elif variant == IndexVariant.NORMALIZED:
        index_path = index_dir / _DEFAULT_INDEX_NORM_FILENAME
    elif variant == IndexVariant.RAW_CENTERED:
        index_path = index_dir / _DEFAULT_INDEX_CENTERED_FILENAME
    else:
        index_path = index_dir / _DEFAULT_INDEX_CENTERED_NORM_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found for variant={variant}: {index_path}")

    avg_embedding = None
    if variant in {IndexVariant.RAW_CENTERED, IndexVariant.CENTERED_NORMALIZED}:
        avg_path = index_dir / _DEFAULT_AVG_FILENAME
        if not avg_path.exists():
            raise FileNotFoundError(f"avg_embedding.npy required for variant={variant} but not found: {avg_path}")
        avg_embedding = np.load(str(avg_path)).astype(np.float32).reshape(-1)

    idx = faiss.read_index(str(index_path))
    card_names = np.load(str(names_path), allow_pickle=True)

    # normalize flag is now derived from variant
    normalize = variant in {IndexVariant.NORMALIZED, IndexVariant.CENTERED_NORMALIZED}

    return IndexBundle(
        index_dir=index_dir,
        index=idx,
        card_names=card_names,
        model_type=model_type,
        normalize=normalize,
        variant=variant,
        avg_embedding=avg_embedding,
    )
