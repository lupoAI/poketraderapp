from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from model.build_index import _write_meta, _write_variants
from model.embedding_adapter import PokemonClipAdapter


@dataclass(frozen=True)
class AdapterInfo:
    path: Path
    input_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float


def load_adapter(adapter_path: str | Path, *, device: str | None = None) -> tuple[PokemonClipAdapter, AdapterInfo]:
    """Load an adapter checkpoint saved by `train_adapter`.

    Returns (adapter_module, AdapterInfo). The adapter outputs L2-normalized embeddings.
    """
    p = Path(adapter_path)
    if not p.exists():
        raise FileNotFoundError(f"adapter not found: {p}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(p), map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid adapter checkpoint format: {p}")

    input_dim = int(ckpt.get("input_dim", 512))
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    output_dim = int(ckpt.get("output_dim", 256))
    dropout = float(ckpt.get("dropout", 0.0))

    adapter = PokemonClipAdapter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)

    adapter.load_state_dict(ckpt["state_dict"], strict=True)
    adapter.eval()

    info = AdapterInfo(path=p, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    return adapter, info


def apply_adapter_to_embeddings(
    embeddings: np.ndarray,
    *,
    adapter: PokemonClipAdapter,
    device: str | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Apply adapter to a [N,D] embedding matrix.

    Expects raw (not normalized) embeddings.
    Returns float32 [N, output_dim] normalized.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError(f"Invalid embeddings shape: {x.shape}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            chunk = torch.from_numpy(x[start : start + int(batch_size)]).to(device)
            y = adapter(chunk)
            out_chunks.append(y.detach().cpu().numpy().astype(np.float32))

    return np.vstack(out_chunks)


def build_adapter_index_from_raw_embeddings(
    *,
    base_index_dir: str | Path,
    adapter_path: str | Path,
    out_index_dir: str | Path,
    model_type: str,
    default_variant: str = "raw",
    device: str | None = None,
) -> Path:
    """Create a new FAISS index directory by applying an adapter to `card_embeddings.npy`.

    Contract:
      - reads card_names.npy + card_embeddings.npy from base_index_dir
      - applies adapter to raw embeddings
      - writes standard variant indices in out_index_dir
      - writes meta.json with `adapter` metadata

    Important: The adapter currently supports ONLY `raw` embeddings.
    """
    base_index_dir = Path(base_index_dir)
    out_index_dir = Path(out_index_dir)

    names_path = base_index_dir / "card_names.npy"
    emb_path = base_index_dir / "card_embeddings.npy"
    if not names_path.exists():
        raise FileNotFoundError(f"Missing card_names.npy: {names_path}")
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing card_embeddings.npy: {emb_path}\n"
            f"Hint: rebuild base index first using build_index.py"
        )

    card_names = np.load(str(names_path), allow_pickle=True)
    raw_emb = np.load(str(emb_path)).astype(np.float32)

    adapter, info = load_adapter(adapter_path, device=device)
    adapted = apply_adapter_to_embeddings(raw_emb, adapter=adapter, device=device)

    out_index_dir.mkdir(parents=True, exist_ok=True)

    # Save adapted raw embeddings as card_embeddings.npy (keeps the same consumer shape)
    np.save(str(out_index_dir / "card_embeddings.npy"), adapted.astype(np.float32))
    np.save(str(out_index_dir / "card_names.npy"), card_names)

    variants = _write_variants(str(out_index_dir), embeddings_np=adapted)

    _write_meta(
        str(out_index_dir),
        model_type=str(model_type),
        source="adapter_from_raw_embeddings",
        extra={
            "default_variant": str(default_variant),
            "base_index_dir": str(base_index_dir),
            "adapter": {
                "path": str(Path(adapter_path)),
                "input_dim": info.input_dim,
                "hidden_dim": info.hidden_dim,
                "output_dim": info.output_dim,
                "dropout": info.dropout,
            },
            "variants": {
                "raw": variants["raw"],
                "normalized": variants["normalized"],
                "raw_centered": variants["raw_centered"],
                "centered_normalized": variants["centered_normalized"],
            },
            "avg_embedding": variants["avg_embedding"],
        },
    )

    return out_index_dir


def pick_or_build_adapter_index(
    *,
    indices_dir: str | Path,
    base_index_dir: str | Path,
    adapter_path: str | Path,
    device: str | None = None,
) -> Path:
    """Find a compatible adapter index under `indices_dir` or build it from base embeddings.

    Strategy:
      - compute a deterministic out dir name based on base dir name and adapter filename
      - if it exists with pokemon_cards.index -> use it
      - else build it using card_embeddings.npy

    Returns the adapter index directory.
    """
    indices_dir = Path(indices_dir)
    base_index_dir = Path(base_index_dir)

    adapter_path = Path(adapter_path)
    safe_adapter = adapter_path.stem

    out_dir = indices_dir / f"{base_index_dir.name}__adapter={safe_adapter}"

    if (out_dir / "pokemon_cards.index").exists() and (out_dir / "meta.json").exists():
        return out_dir

    # infer model_type from base meta.json if present
    model_type = "clip"
    meta_path = base_index_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_type = str(meta.get("model_type", model_type))
        except Exception:
            pass

    return build_adapter_index_from_raw_embeddings(
        base_index_dir=base_index_dir,
        adapter_path=adapter_path,
        out_index_dir=out_dir,
        model_type=model_type,
        default_variant="raw",
        device=device,
    )
