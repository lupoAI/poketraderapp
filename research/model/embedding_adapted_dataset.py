import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import random
from matplotlib import pyplot as plt
import numpy as np

from model.config import CARDS_DIR

IMG_SIZE = (337, 245)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _to_pil_safe(x):
    """Convert PIL or torch Tensor to PIL (best-effort) for visualization."""
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return transforms.ToPILImage()(torch.clamp(x, 0, 1))
    raise TypeError(f"Unsupported type for visualization: {type(x)!r}")


def _stats_str(x) -> str:
    if isinstance(x, Image.Image):
        arr = np.asarray(x)
        return f"PIL {x.size} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.1f}"
    if torch.is_tensor(x):
        t = x.detach().float()
        return (
            f"Tensor {tuple(t.shape)} min={t.min().item():.3f} max={t.max().item():.3f} "
            f"mean={t.mean().item():.3f} std={t.std().item():.3f}"
        )
    return str(type(x))


# --- 1. DEFINE THE AUGMENTATIONS ---

def get_clean_transform():
    """Standard preprocessing for the Anchor (Clean Index)"""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])


def get_noisy_transform(*, positive_strength: float = 1.0):
    """Transform per il Positive (noisy) con intensità scalabile.

    positive_strength:
      - 0.0 -> quasi nessun rumore/augment (baseline)
      - 1.0 -> piena intensità (ma con default già meno aggressivi rispetto a prima)

    Nota: qui scaliamo principalmente *le probabilità* (p) degli eventi più distruttivi.
    """

    s = _clamp01(positive_strength)

    # Base probabilities (già ridotte rispetto a prima)
    p_thumb = 0.05
    p_small_occ = 0.2
    p_blur = 0.2
    p_persp = 0.2

    # scala lineare con strength
    p_thumb *= s
    p_small_occ *= s
    p_blur *= s
    p_persp *= s

    return transforms.Compose([
        transforms.Resize(IMG_SIZE),

        # Occlusioni principali (pollice) - ora meno frequente
        RandomThumbOcclusion(p=p_thumb, scale=(0.08, 0.25)),

        # Occlusioni piccole (graffi/riflessi/finger)
        RandomSmallOcclusions(p=p_small_occ, n=(1, 5), size=(0.01, 0.05), shape="rect"),

        # Lighting & Quality
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        ], p=0.1 * s),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))
        ], p=p_blur),

        # Crop/scale directly from the original image
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),

        transforms.ToTensor(),

        # Geometry
        transforms.RandomPerspective(distortion_scale=0.3, p=p_persp),
        transforms.RandomRotation(degrees=15 * s),

    ])


# --- 2. THE TRIPLET DATASET ---

class CardTripletDataset(Dataset):
    def __init__(self, image_paths, *, positive_strength: float = 1.0):
        """
        image_paths: List of file paths to your CLEAN card scans.

        positive_strength: controlla quanto è aggressivo il positive (noisy transform).
        """
        self.image_paths = image_paths
        self.clean_transform = get_clean_transform()
        self.noisy_transform = get_noisy_transform(positive_strength=float(positive_strength))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load the Anchor (The specific card we are training on)
        anchor_path = self.image_paths[idx]
        image = Image.open(anchor_path).convert('RGB')

        # 2. Create the Anchor (Clean) and Positive (Noisy version of SAME card)
        anchor_img = self.clean_transform(image)
        positive_img = self.noisy_transform(image)  # Applies random augmentations

        # 3. Select a Negative (Any DIFFERENT card)
        # We keep picking a random index until it's not the current one
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.image_paths) - 1)

        neg_path = self.image_paths[neg_idx]
        neg_image = Image.open(neg_path).convert('RGB')

        # The Negative can be Clean or Noisy.
        # Usually, making the negative 'Clean' helps the model distinguish
        # the noisy input from the Clean Database entries.
        negative_img = self.clean_transform(neg_image)

        return anchor_img, positive_img, negative_img


class RandomThumbOcclusion(object):
    def __init__(self, p=0.2, scale=(0.05, 0.20)):
        self.p = p
        self.scale = scale
        self.skin_tones = [
            (255, 224, 189), (255, 205, 148), (224, 172, 105),
            (141, 85, 36), (198, 134, 66), (96, 56, 19)
        ]

    def __call__(self, img):
        # Respect probability (in training). For deterministic visualization, set p=1.0.
        if random.random() > float(self.p):
            return img

        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        thumb_area = random.uniform(self.scale[0], self.scale[1]) * (w * h)
        r = int(np.sqrt(thumb_area) / 2)
        edge = random.randint(0, 3)

        if edge == 0:
            center = (0, random.randint(0, h))  # Left
        elif edge == 1:
            center = (random.randint(0, w), 0)  # Top
        elif edge == 2:
            center = (w, random.randint(0, h))  # Right
        else:
            center = (random.randint(0, w), h)  # Bottom

        x0, y0, x1, y1 = center[0] - r, center[1] - r, center[0] + r, center[1] + r
        color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in random.choice(self.skin_tones))

        draw.ellipse([x0, y0, x1, y1], fill=color)
        return img


class RandomSmallOcclusions(object):
    """Draw small random occlusions on the image (PIL).

    Useful to simulate:
      - glare spots
      - scratches
      - partial finger/hand occlusions
      - dust

    Parameters:
      p: probability of applying the transform
      n: (min,max) number of occlusions
      size: (min,max) fraction of image area per occlusion
      shape: "rect" | "ellipse" | "mixed"
      color: "random" | "black" | "white" | "gray" | "skin"
      alpha: (min,max) transparency in [0,1]
    """

    def __init__(
            self,
            p: float = 0.3,
            n: tuple[int, int] = (1, 4),
            size: tuple[float, float] = (0.01, 0.05),
            shape: str = "mixed",
            color: str = "random",
            alpha: tuple[float, float] = (0.35, 0.9),
    ):
        self.p = float(p)
        self.n = n
        self.size = size
        self.shape = str(shape)
        self.color = str(color)
        self.alpha = alpha

        self._skin_tones = [
            (255, 224, 189), (255, 205, 148), (224, 172, 105),
            (141, 85, 36), (198, 134, 66), (96, 56, 19)
        ]

    def _pick_color(self) -> tuple[int, int, int]:
        if self.color == "black":
            return (0, 0, 0)
        if self.color == "white":
            return (255, 255, 255)
        if self.color == "gray":
            g = random.randint(40, 220)
            return (g, g, g)
        if self.color == "skin":
            base = random.choice(self._skin_tones)
            # tipizzazione esplicita (3 canali)
            r, g, b = (max(0, min(255, c + random.randint(-20, 20))) for c in base)
            return int(r), int(g), int(b)

        # random
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _pick_shape(self) -> str:
        if self.shape in {"rect", "ellipse"}:
            return self.shape
        return "rect" if random.random() < 0.5 else "ellipse"

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img = img.copy().convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        n_occ = random.randint(int(self.n[0]), int(self.n[1]))
        for _ in range(max(0, n_occ)):
            # area fraction -> pick approximate square size then random aspect ratio
            area = random.uniform(self.size[0], self.size[1]) * float(w * h)
            base = max(2.0, np.sqrt(area))

            # aspect ratio for rects
            ar = random.uniform(0.4, 2.5)
            occ_w = int(max(2, round(base * ar)))
            occ_h = int(max(2, round(base / ar)))

            x0 = random.randint(0, max(0, w - occ_w))
            y0 = random.randint(0, max(0, h - occ_h))
            x1 = x0 + occ_w
            y1 = y0 + occ_h

            col = self._pick_color()
            a = int(round(255 * random.uniform(self.alpha[0], self.alpha[1])))
            fill = (int(col[0]), int(col[1]), int(col[2]), int(a))

            shp = self._pick_shape()
            if shp == "ellipse":
                draw.ellipse([x0, y0, x1, y1], fill=fill)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=fill)

        out = Image.alpha_composite(img, overlay).convert("RGB")
        return out


def _apply_and_record(name: str, fn, x, notes: list[str]):
    """Apply transform and record a small note about inputs/outputs."""
    y = fn(x)
    notes.append(f"{name}: { _stats_str(x) } -> { _stats_str(y) }")
    return y


def _sample_affine_params(img: Image.Image, *, degrees=0, scale=(0.8, 1.2)):
    """Sample params compatibili con RandomAffine(degrees=0, scale=...)."""
    from torchvision.transforms import RandomAffine

    ra = RandomAffine(degrees=degrees, scale=scale)
    # get_params wants: degrees, translate, scale_ranges, shears, img_size
    angle, translations, sc, shear = ra.get_params(
        ra.degrees,  # type: ignore[arg-type]
        ra.translate,  # type: ignore[arg-type]
        ra.scale,  # type: ignore[arg-type]
        ra.shear,  # type: ignore[arg-type]
        list(img.size),
    )
    return angle, translations, sc, shear


def _apply_affine_with_params(img: Image.Image, *, angle, translations, scale, shear):
    from torchvision.transforms.functional import affine as F_affine

    return F_affine(
        _to_pil_safe(img),
        angle=float(angle),
        translate=[int(translations[0]), int(translations[1])],
        scale=float(scale),
        shear=shear,
    )


def _sample_perspective_params(t: torch.Tensor, *, distortion_scale=0.3):
    """Sample params for a perspective warp (stable across torchvision versions).

    Returns (startpoints, endpoints) like torchvision functional expects.
    """
    # torchvision expects a list of 4 points in TL, TR, BR, BL order.
    h = int(t.shape[-2])
    w = int(t.shape[-1])

    startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]

    half_w = w * float(distortion_scale) / 2.0
    half_h = h * float(distortion_scale) / 2.0

    def _jitter(x, half):
        return int(round(x + random.uniform(-half, half)))

    endpoints = [
        [_jitter(0, half_w), _jitter(0, half_h)],
        [_jitter(w - 1, half_w), _jitter(0, half_h)],
        [_jitter(w - 1, half_w), _jitter(h - 1, half_h)],
        [_jitter(0, half_w), _jitter(h - 1, half_h)],
    ]

    # Clamp to image bounds
    for p in endpoints:
        p[0] = max(0, min(w - 1, p[0]))
        p[1] = max(0, min(h - 1, p[1]))

    return startpoints, endpoints


def _sample_rotation_angle(*, degrees=15):
    from torchvision.transforms import RandomRotation

    rr = RandomRotation(degrees=degrees)
    angle = rr.get_params(rr.degrees)  # type: ignore[arg-type]
    return float(angle)


def _apply_perspective_with_params(t: torch.Tensor, *, startpoints, endpoints) -> torch.Tensor:
    from torchvision.transforms.functional import perspective as F_persp

    return F_persp(t, startpoints=startpoints, endpoints=endpoints)


def _apply_rotation(t: torch.Tensor, *, angle: float) -> torch.Tensor:
    from torchvision.transforms.functional import rotate as F_rotate

    return F_rotate(t, angle=float(angle))


def visualize_pipeline(
    image_path,
    seed: int = 42,
    *,
    positive_strength: float = 1.0,
    force_p: float | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Visualize `get_noisy_transform()` step-by-step with *real sampled params*.

    Mirrors the exact order in `get_noisy_transform()` and prints a debug report with sampled params.

    Args:
      image_path: path to a clean image.
      seed: RNG seed for reproducibility.
      positive_strength: scala l'intensità del positive (0..1).
      force_p: if set, overrides probability for thumb/small occlusions + RandomApply steps.
      save_path: if provided, saves the figure.
      show: if False, doesn't call plt.show().
    """

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    original = Image.open(image_path).convert("RGB")

    notes: list[str] = []
    imgs: list[tuple[str, Image.Image]] = [("Original", original)]

    s = _clamp01(positive_strength)

    # 0) Resize
    step0 = transforms.Resize(IMG_SIZE)
    img0 = _apply_and_record("0.Resize", step0, original, notes)
    imgs.append(("0.Resize", _to_pil_safe(img0)))

    # 1) Thumb occlusion
    p_thumb = (0.05 * s) if force_p is None else float(force_p)
    thumb = RandomThumbOcclusion(p=p_thumb, scale=(0.08, 0.25))
    img1 = _apply_and_record(f"1.Thumb(p={p_thumb})", thumb, img0, notes)
    imgs.append(("1.Thumb", _to_pil_safe(img1)))

    # 2) Small occlusions
    p_small = (0.2 * s) if force_p is None else float(force_p)
    small_occ = RandomSmallOcclusions(p=p_small, n=(1, 5), size=(0.01, 0.05), shape="rect")
    img2 = _apply_and_record(f"2.SmallOcc(p={p_small})", small_occ, img1, notes)
    imgs.append(("2.SmallOcc", _to_pil_safe(img2)))

    # 3) RandomApply(ColorJitter)
    p_cj = (0.5 * s) if force_p is None else float(force_p)
    cj = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    if random.random() < p_cj:
        img3 = _apply_and_record(f"3.ColorJitter(p={p_cj})", cj, img2, notes)
        cj_applied = True
    else:
        img3 = img2
        cj_applied = False
        notes.append(f"3.ColorJitter(p={p_cj}): skipped")
    imgs.append((f"3.ColorJitter\n(applied={cj_applied})", _to_pil_safe(img3)))

    # 4) RandomApply(Blur)
    p_blur = (0.2 * s) if force_p is None else float(force_p)
    blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))
    if random.random() < p_blur:
        img4 = _apply_and_record(f"4.Blur(p={p_blur})", blur, img3, notes)
        blur_applied = True
    else:
        img4 = img3
        blur_applied = False
        notes.append(f"4.Blur(p={p_blur}): skipped")
    imgs.append((f"4.Blur\n(applied={blur_applied})", _to_pil_safe(img4)))

    # 5) Affine
    angle, translations, sc, shear = _sample_affine_params(img4, degrees=0, scale=(0.8, 1.2))
    notes.append(
        f"5.Affine params: angle={float(angle):.2f} translate={tuple(int(x) for x in translations)} "
        f"scale={float(sc):.3f} shear={shear}"
    )
    img5 = _apply_and_record(
        "5.Affine",
        lambda im: _apply_affine_with_params(im, angle=angle, translations=translations, scale=sc, shear=shear),
        img4,
        notes,
    )
    imgs.append(("5.Affine", _to_pil_safe(img5)))

    # 6) ToTensor
    to_tensor = transforms.ToTensor()
    t6 = _apply_and_record("6.ToTensor", to_tensor, img5, notes)
    imgs.append(("6.ToTensor", _to_pil_safe(t6)))

    # 7) Perspective
    p_persp = (0.2 * s) if force_p is None else float(force_p)
    if random.random() < p_persp:
        startpoints, endpoints = _sample_perspective_params(t6, distortion_scale=0.3)
        notes.append(f"7.Perspective params:\n  start={startpoints}\n  end={endpoints}")
        t7 = _apply_and_record(
            f"7.Perspective(p={p_persp})",
            lambda tt: _apply_perspective_with_params(tt, startpoints=startpoints, endpoints=endpoints),
            t6,
            notes,
        )
        persp_applied = True
    else:
        t7 = t6
        persp_applied = False
        notes.append(f"7.Perspective(p={p_persp}): skipped")
    imgs.append((f"7.Perspective\n(applied={persp_applied})", _to_pil_safe(t7)))

    # 8) Rotation
    angle8 = _sample_rotation_angle(degrees=int(round(15 * s)))
    notes.append(f"8.Rotation angle={angle8:.2f} deg")
    t8 = _apply_and_record("8.Rotation", lambda tt: _apply_rotation(tt, angle=angle8), t7, notes)
    imgs.append(("8.Rotation (final)", _to_pil_safe(t8)))

    n = len(imgs)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 6, rows * 6))
    for i, (title, im) in enumerate(imgs, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"get_noisy_transform debug (seed={seed}, positive_strength={positive_strength}, force_p={force_p})",
        fontsize=14,
    )
    plt.tight_layout()

    print("\n--- get_noisy_transform debug report ---")
    print("\n".join(notes))

    if save_path:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure: {save_path}")

    if show:
        plt.show()

    return {"images": imgs, "notes": notes, "final_tensor": t8}


# --- 3. EXAMPLE USAGE ---
if __name__ == "__main__":
    visualize_pipeline(
        str(CARDS_DIR / 'A1-003.jpg'),
        seed=101,
        positive_strength=1.0,
        force_p=1.0,
        save_path=str((CARDS_DIR.parent / 'vis' / 'debug_noisy_transform.jpg')),
    )
