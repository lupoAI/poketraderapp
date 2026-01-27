from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import CenterCrop


def _to_uint8_rgb(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    return np.asarray(img)


def visualize_open_clip_resize_keep_ratio_center_crop(
    image_path: str | Path,
    *,
    image_size: int | tuple[int, int] = 224,
    longest: float = 0.0,
    out_path: str | Path | None = None,
):
    """Visualizza cosa fa: [open_clip.transform.ResizeKeepRatio(image_size), torchvision.CenterCrop(image_size)].

    - ResizeKeepRatio (open_clip) fa solo un resize che mantiene il ratio, scegliendo la scala in modo che
      l'immagine sia "almeno" della size target in un senso (dipende da `longest`).
    - CenterCrop poi taglia al centro esattamente a image_size.

    Nota: non fa padding. Se dopo il resize una dimensione e` piu` piccola del crop, CenterCrop taglierà
    comunque ma potresti perdere contenuto o ottenere un crop strano. Normalmente open_clip garantisce
    che il resize porti l'immagine >= target su entrambe le dimensioni quando usato correttamente.
    """

    from open_clip.transform import ResizeKeepRatio  # import locale per evitare dipendenza hard nei moduli core

    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")

    rk = ResizeKeepRatio(image_size, longest=longest)
    cc = CenterCrop(image_size)

    img_r = rk(img)
    img_c = cc(img_r)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    axs[0].imshow(_to_uint8_rgb(img))
    axs[0].set_title(f"Original\n{img.size[0]}x{img.size[1]}")
    axs[0].axis("off")

    axs[1].imshow(_to_uint8_rgb(img_r))
    axs[1].set_title(f"ResizeKeepRatio(longest={longest})\n{img_r.size[0]}x{img_r.size[1]}")
    axs[1].axis("off")

    axs[2].imshow(_to_uint8_rgb(img_c))
    if isinstance(image_size, int):
        t = f"CenterCrop({image_size})"
    else:
        t = f"CenterCrop({image_size[0]}x{image_size[1]})"
    axs[2].set_title(f"{t}\n{img_c.size[0]}x{img_c.size[1]}")
    axs[2].axis("off")

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)

    plt.show()


if __name__ == "__main__":
    # Default example image in this repo
    repo_root = Path(__file__).resolve().parents[2]
    default_img = repo_root / "storage" / "images" / "cards" / "A1-003.jpg"

    out = repo_root / "storage" / "vis" / "debug_openclip_resize_keep_ratio_center_crop.jpg"

    visualize_open_clip_resize_keep_ratio_center_crop(
        default_img,
        image_size=224,
        longest=0.0,
        out_path=out,
    )

