from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image


@runtime_checkable
class ImageEmbeddingBackend(Protocol):
    """Backend che trasforma una PIL.Image in un embedding numpy (1, D) float32."""

    model_type: str
    device: str

    def encode(self, image: Image.Image, *, normalize: bool = True) -> np.ndarray: ...


@runtime_checkable
class ClipLikeBackend(ImageEmbeddingBackend, Protocol):
    """Estensione usata solo per compatibilita` col codice legacy (CLIP)."""

    @property
    def model(self) -> object: ...

    @property
    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]: ...


@dataclass(frozen=True)
class BackendOutputs:
    index_filename: str
    names_filename: str
    embeddings_filename: str


def build_index_subdir(
    *,
    model_type: str,
    images_quality: str | None = None,
    image_format: str | None = None,
    dino_model_name: str | None = None,
) -> str:
    """Crea un nome directory autosufficiente per descrivere come e` stato costruito l'indice.

    Esempi:
      - model=clip
      - model=dino__arch=facebook-dinov2-small__q=high__fmt=jpg

    Nota: i filename restano sempre gli stessi; cambia solo la directory.
    """
    mt = (model_type or "").strip().lower() or "model"
    parts: list[str] = [f"model={mt}"]

    if dino_model_name:
        safe_arch = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in dino_model_name)
        parts.insert(1, f"arch={safe_arch}")

    if images_quality:
        parts.append(f"q={images_quality}")
    if image_format:
        parts.append(f"fmt={image_format}")

    return "__".join(parts)


def outputs_for(model_type: str, *, normalize: bool, out_dir: str) -> BackendOutputs:
    """Determina i filename di output.

    Per evitare di rompere i consumer (classification_loop, test, ecc.)
    i nomi file sono sempre gli stessi a prescindere dal modello.

    La variabilita` (model_type/normalize/etc) deve stare nel nome della directory.
    """
    return BackendOutputs(
        index_filename="pokemon_cards.index",
        names_filename="card_names.npy",
        embeddings_filename="card_embeddings.npy",
    )


def _pick_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_backend(
    model_type: str = "clip",
    *,
    device: Optional[str] = None,
    dino_model_name: str = "facebook/dinov2-small",
) -> ImageEmbeddingBackend:
    from typing import cast

    model_type = (model_type or "").strip().lower()

    if model_type in {"clip", "openclip", "open_clip"}:
        return cast(ImageEmbeddingBackend, cast(object, _OpenClipBackend(device=_pick_device(device))))

    if model_type in {"dino", "dinov2", "dinov2-small", "dinov2_small"}:
        return cast(
            ImageEmbeddingBackend,
            cast(object, _DinoV2Backend(model_name=dino_model_name, device=_pick_device(device))),
        )

    raise ValueError(
        f"Unknown model_type={model_type!r}. Supported: clip, dino (dinov2)"
    )


class _OpenClipBackend:
    model_type: str = "clip"

    def __init__(self, *, device: str):
        import open_clip  # lazy import

        self.device: str = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", image_resize_mode="longest"
        )
        self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def preprocess(self) -> Callable[[Image.Image], torch.Tensor]:
        return self._preprocess

    @property
    def model(self) -> object:
        return self._model

    def encode(self, image: Image.Image, *, normalize: bool = True) -> np.ndarray:
        t = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(t)
            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)


class _DinoV2Backend:
    model_type: str = "dino"

    def __init__(self, *, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModel  # lazy import

        self.device: str = device
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device)
        self._model.eval()

    def encode(self, image: Image.Image, *, normalize: bool = True) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            # DINOv2: token [CLS] (idx 0) come rappresentazione globale
            emb = outputs.last_hidden_state[:, 0, :]
            if normalize:
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.detach().cpu().numpy().astype(np.float32)
