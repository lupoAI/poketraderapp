from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def _to_uint8_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _denormalize_tensor(x, mean, std):
    # x: torch tensor [C,H,W]
    import torch

    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = x * std_t + mean_t
    return torch.clamp(y, 0, 1)


def _tensor_to_pil(x) -> Image.Image:
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    # [C,H,W] -> uint8 HWC
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class ModeResult:
    mode: str
    preproc_pil: Image.Image
    final_pil: Image.Image


def _build_preproc_pil(
    img: Image.Image,
    *,
    image_size: int | tuple[int, int],
    resize_mode: str,
    interpolation: str = "bicubic",
    fill_color: int = 0,
) -> Image.Image:
    """Ricostruisce la parte *pre-Tensor* di open_clip.image_transform (is_train=False).

    Serve per visualizzare cosa succede geometricamente prima della normalizzazione.
    """

    from open_clip.transform import ResizeKeepRatio, CenterCropOrPad, _convert_to_rgb
    from torchvision.transforms import Resize, CenterCrop, InterpolationMode

    interpolation_mode = InterpolationMode.BILINEAR if interpolation == "bilinear" else InterpolationMode.BICUBIC

    rm = resize_mode
    if rm == "longest":
        # ResizeKeepRatio(longest=1) + CenterCropOrPad
        pre = ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1)
        pre2 = CenterCropOrPad(image_size, fill=fill_color)
        out = pre2(pre(img))
    elif rm == "squash":
        # direct resize to square/rect (aspect ratio NOT preserved)
        if isinstance(image_size, int):
            size2 = (image_size, image_size)
        else:
            size2 = tuple(image_size)
        out = Resize(size2, interpolation=interpolation_mode)(img)
    else:
        # shortest
        if not isinstance(image_size, (tuple, list)):
            size2 = (image_size, image_size)
        else:
            size2 = tuple(image_size)

        if size2[0] == size2[1]:
            # torchvision Resize(scalar) uses shortest-edge behavior
            out = Resize(size2[0], interpolation=interpolation_mode)(img)
        else:
            out = ResizeKeepRatio(size2)(img)

        out = CenterCrop(size2)(out)

    out = _convert_to_rgb(out)
    return out


def visualize_open_clip_resize_modes(
    image_path: str | Path,
    *,
    image_size: int | tuple[int, int] = 224,
    interpolation: str = "bicubic",
    fill_color: int = 0,
    out_path: str | Path | None = None,
):
    """Visualizza tutti i resize_mode di open_clip.transform.image_transform (inference).

    Mostra per ogni mode:
      - output PIL *prima* di ToTensor/Normalize (solo geometria)
      - output finale (denormalizzato) della Compose completa

    Modes:
      - shortest
      - longest
      - squash
    """

    import torch
    from open_clip.transform import image_transform, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")

    modes = ["shortest", "longest", "squash"]
    results: list[ModeResult] = []

    for mode in modes:
        # 1) geometry-only preview (PIL)
        pre = _build_preproc_pil(
            img,
            image_size=image_size,
            resize_mode=mode,
            interpolation=interpolation,
            fill_color=fill_color,
        )

        # 2) full open_clip pipeline (tensor normalized)
        tfm = image_transform(
            image_size=image_size,
            is_train=False,
            resize_mode=mode,
            interpolation=interpolation,
            fill_color=fill_color,
            mean=OPENAI_DATASET_MEAN,
            std=OPENAI_DATASET_STD,
        )
        x = tfm(img)
        x = _denormalize_tensor(x, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        out = _tensor_to_pil(x)

        results.append(ModeResult(mode=mode, preproc_pil=pre, final_pil=out))

    # plot
    fig, axs = plt.subplots(2, 1 + len(results), figsize=(4 * (1 + len(results)), 8))

    axs[0, 0].imshow(_to_uint8_rgb(img))
    axs[0, 0].set_title(f"Original\n{img.size[0]}x{img.size[1]}")
    axs[0, 0].axis("off")

    axs[1, 0].axis("off")
    axs[1, 0].text(
        0.5,
        0.5,
        "Row 1: pre-ToTensor\nRow 2: final (denorm)",
        ha="center",
        va="center",
        fontsize=12,
    )

    for j, r in enumerate(results, start=1):
        axs[0, j].imshow(_to_uint8_rgb(r.preproc_pil))
        axs[0, j].set_title(f"{r.mode} pre\n{r.preproc_pil.size[0]}x{r.preproc_pil.size[1]}")
        axs[0, j].axis("off")

        axs[1, j].imshow(_to_uint8_rgb(r.final_pil))
        axs[1, j].set_title(f"{r.mode} final\n{r.final_pil.size[0]}x{r.final_pil.size[1]}")
        axs[1, j].axis("off")

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)

    plt.show()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    default_img = repo_root / "storage" / "images" / "cards" / "A1-003.jpg"
    out = repo_root / "storage" / "vis" / "debug_openclip_resize_modes.jpg"

    visualize_open_clip_resize_modes(
        default_img,
        image_size=224,
        interpolation="bicubic",
        fill_color=0,
        out_path=out,
    )

