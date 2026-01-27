"""Benchmark rapido: OpenCLIP ViT-B-16 vs ViT-B-32.

Obiettivo:
- Stimare quanto e` piu' lento ViT-B-16 rispetto a ViT-B-32 nel TUO ambiente.

Uso (PowerShell):
    python scripts\benchmark_openclip_vit_b16_vs_b32.py --device cuda --iters 200

Note:
- Se --device non e` specificato, usa cuda se disponibile altrimenti cpu.
- Prova a usare lo stesso pretrained per entrambi; se non disponibile fa fallback.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _pick_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_any_image() -> Image.Image:
    # Prefer a test image from repo if available
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = repo_root / "storage" / "test_data" / "images"
    for p in sorted(test_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return Image.open(p).convert("RGB")
    # fallback: synthetic
    arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _bench_model(model_name: str, pretrained: str, *, device: str, iters: int, warmup: int) -> float:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, image_resize_mode="longest"
    )
    model = model.to(device)
    model.eval()

    img = _load_any_image()
    t = preprocess(img).unsqueeze(0).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.encode_image(t)
    if device == "cuda":
        torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model.encode_image(t)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / float(iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = ap.parse_args()

    device = _pick_device(args.device)

    # Try with requested pretrained; if not available for b16, fall back to default open_clip suggestion.
    pretrained = str(args.pretrained)

    print(f"Device: {device}")
    print(f"Iters: {args.iters} | Warmup: {args.warmup}")
    print(f"Pretrained requested: {pretrained}")

    try:
        s32 = _bench_model("ViT-B-32", pretrained, device=device, iters=args.iters, warmup=args.warmup)
    except Exception as e:
        raise SystemExit(f"Failed to benchmark ViT-B-32 with pretrained={pretrained}: {e}")

    try:
        s16 = _bench_model("ViT-B-16", pretrained, device=device, iters=args.iters, warmup=args.warmup)
    except Exception as e:
        # retry with a common fallback
        fallback = "openai"
        print(f"WARN: ViT-B-16 pretrained={pretrained} not available or failed: {e}")
        print(f"Retrying ViT-B-16 with pretrained={fallback}...")
        s16 = _bench_model("ViT-B-16", fallback, device=device, iters=args.iters, warmup=args.warmup)

    ms32 = 1000.0 * s32
    ms16 = 1000.0 * s16
    ratio = s16 / max(1e-12, s32)

    print("\nResults (lower is better):")
    print(f"  ViT-B-32: {ms32:.3f} ms/img")
    print(f"  ViT-B-16: {ms16:.3f} ms/img")
    print(f"  Slowdown: {ratio:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

