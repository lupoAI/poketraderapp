import glob
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.config import CARDS_DIR
from model.embedding_adapted_dataset import CardTripletDataset
from model.embedding_backends import get_backend


class PokemonClipAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256, dropout: float = 0.0):
        super(PokemonClipAdapter, self).__init__()

        dropout = float(dropout)
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        # A simple Multi-Layer Perceptron (MLP)
        # We compress to output_dim for faster search & smaller index
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # more stable than BatchNorm for small/noisy batches
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # 1. Pass through the MLP
        x = self.network(x)

        # 2. L2 Normalize!
        # Crucial for vector search (Cosine Similarity = Dot product of normalized vectors)
        return F.normalize(x, p=2, dim=1)


# --- TRAINING (simple, pragmatic) ---


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure 2D (B, D)."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def _checkpoint_paths(checkpoint_path: str | None) -> tuple[Path | None, Path | None]:
    """Return (checkpoint_dir, explicit_file).

    - Se checkpoint_path è None -> (None, None)
    - Se checkpoint_path punta a una directory -> (dir, None)
    - Se checkpoint_path punta a un file .pt/.pth -> (parent_dir, file)

    Per compatibilità: se l'utente passa un file, continuiamo a scrivere anche quel file,
    ma in più scriviamo snapshot versionati nella stessa directory.
    """
    if checkpoint_path is None:
        return None, None

    p = Path(checkpoint_path)
    if p.exists() and p.is_dir():
        return p, None

    # se termina con separatore o non ha suffix, trattalo come dir
    if str(checkpoint_path).endswith(("/", "\\")) or p.suffix == "":
        return p, None

    return p.parent, p


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    snapshot_name: str,
    ckpt_obj: dict,
    latest_name: str = "latest.pt",
    also_write: Path | None = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snap_path = checkpoint_dir / snapshot_name
    torch.save(ckpt_obj, snap_path)

    # latest (copy overwrite)
    latest_path = checkpoint_dir / latest_name
    try:
        torch.save(ckpt_obj, latest_path)
    except Exception:
        pass

    # compat: scrivi anche nel path esplicito (sovrascrive) se richiesto
    if also_write is not None:
        try:
            also_write.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_obj, also_write)
        except Exception:
            pass

    return snap_path


def train_adapter(
    *,
    images_dir: str | None = None,
    model_type: str = "clip",
    output_dim: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    lr: float = 1e-3,
    margin: float = 0.1,
    loss: str = "triplet",
    positive_strength: float = 1.0,
    batch_size: int = 32,
    epochs: int = 2,
    max_steps: int | None = None,
    num_workers: int = 0,
    device: str | None = None,
    log_every: int = 10,
    checkpoint_path: str | None = None,
    save_every_steps: int | None = None,
    seed: int = 42,
):
    """Train the adapter to map noisy/real-photo CLIP embeddings closer to clean ones.

    loss options:
      - triplet: TripletMarginLoss(margin=...)
      - cosine:  1 - cosine_similarity(anchor, positive) + max(0, cosine(anchor, negative) - cosine(anchor, positive) + margin)
                (a cosine-based triplet-like objective)

    Notes:
      - CardTripletDataset crea positive molto *aggressive* (thumb occlusion, blur, affine, perspective, rotation).
        In questo setting, dropout alto tende a peggiorare (stai gia' iniettando tanto rumore).
      - margin default piu' basso (0.1) per evitare triplet troppo difficili in uno spazio normalizzato.
    """

    loss = str(loss).strip().lower()
    if loss not in {"triplet", "cosine"}:
        raise ValueError(f"loss must be 'triplet' or 'cosine', got {loss!r}")

    # print all args
    print("Training PokemonClipAdapter with parameters:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    device = device or _default_device()

    # Reproducibility (best-effort)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if images_dir is None:
        images_dir = str(CARDS_DIR)

    img_dir = Path(images_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {img_dir}")

    # Common extensions
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(glob.glob(str(img_dir / ext)))

    if len(paths) < 2:
        raise ValueError(
            f"Need at least 2 images to sample negatives, got {len(paths)} in {img_dir}"
        )

    dataset = CardTripletDataset(paths, positive_strength=float(positive_strength))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    backend = get_backend(model_type=model_type, device=device)

    # Infer input dim from backend quickly (one sample)
    # We keep this tiny and safe.
    sample_img = dataset[0][0]  # anchor tensor
    # dataset returns torch.Tensor image; backend expects PIL.Image.
    # So we instead load an actual PIL for probing.
    from PIL import Image

    pil_probe = Image.open(paths[0]).convert("RGB")
    probe = backend.encode(pil_probe, normalize=False)
    input_dim = int(probe.shape[-1])

    adapter = PokemonClipAdapter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(adapter.parameters(), lr=lr)

    # Loss
    triplet_criterion = nn.TripletMarginLoss(margin=margin, p=2)

    def _cosine_triplet_loss(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # all are (B,D) and already normalized by adapter
        # maximize cos(a,p), minimize cos(a,n) by a margin
        sap = F.cosine_similarity(a, p, dim=1)  # (B,)
        san = F.cosine_similarity(a, n, dim=1)
        # hinge: want san <= sap - margin
        return torch.relu(san - sap + float(margin)).mean() + (1.0 - sap).mean()

    def _encode_batch(img_batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into embeddings (B, D) on the training device."""
        # backend preprocess is a callable on PIL; but our dataset returns tensors already.
        # To keep it dead simple: convert tensor->PIL and use backend.encode per image.
        # This is slower, but works and stays consistent with existing backend.
        to_pil = None
        try:
            from torchvision.transforms.functional import to_pil_image as _to_pil
            to_pil = _to_pil
        except Exception:
            to_pil = None

        embs = []
        for img_t in img_batch:
            if to_pil is None:
                raise RuntimeError("torchvision is required for training (to_pil_image missing)")
            pil = to_pil(img_t.cpu())
            e = backend.encode(pil, normalize=False)  # (1, D) numpy
            embs.append(torch.from_numpy(e[0]))
        return torch.stack(embs, dim=0).to(device)

    global_step = 0
    adapter.train()

    ckpt_dir, ckpt_file = _checkpoint_paths(checkpoint_path)

    for epoch in range(int(epochs)):
        running = 0.0
        epoch_running = 0.0
        epoch_batches = 0

        for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(loader):
            if max_steps is not None and global_step >= int(max_steps):
                break

            # Encode images -> embeddings (frozen backend)
            with torch.no_grad():
                anchor_emb = _encode_batch(anchor_img)
                positive_emb = _encode_batch(positive_img)
                negative_emb = _encode_batch(negative_img)

            # Train step
            optimizer.zero_grad(set_to_none=True)

            anchor_out = adapter(_as_batch(anchor_emb))
            positive_out = adapter(_as_batch(positive_emb))
            negative_out = adapter(_as_batch(negative_emb))

            if loss == "triplet":
                loss_t = triplet_criterion(anchor_out, positive_out, negative_out)
            else:
                loss_t = _cosine_triplet_loss(anchor_out, positive_out, negative_out)

            loss_t.backward()
            optimizer.step()

            loss_val = float(loss_t.detach().cpu().item())
            running += loss_val
            epoch_running += loss_val
            epoch_batches += 1
            global_step += 1

            # Batch-level logging (throttled)
            if log_every and global_step % int(log_every) == 0:
                avg = running / float(log_every)
                print(
                    f"epoch={epoch+1}/{epochs} batch={batch_idx+1} step={global_step} "
                    f"loss(batch)={loss_val:.4f} loss(avg@{log_every})={avg:.4f} "
                    f"(batch_size={batch_size} device={device} model_type={model_type} loss={loss})"
                )
                running = 0.0

            # Optional step checkpointing
            if ckpt_dir is not None and save_every_steps is not None and int(save_every_steps) > 0:
                if global_step % int(save_every_steps) == 0:
                    ckpt = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_type": model_type,
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "output_dim": output_dim,
                        "dropout": float(dropout),
                        "loss": str(loss),
                        "margin": float(margin),
                        "positive_strength": float(positive_strength),
                        "state_dict": adapter.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    snap = _save_checkpoint(
                        checkpoint_dir=ckpt_dir,
                        snapshot_name=f"step_{global_step:07d}.pt",
                        ckpt_obj=ckpt,
                        also_write=ckpt_file,
                    )
                    print(f"Saved checkpoint snapshot: {snap}")

        # Epoch-level logging
        if epoch_batches > 0:
            epoch_mean = epoch_running / float(epoch_batches)
            print(
                f"epoch_end {epoch+1}/{epochs}: mean_loss={epoch_mean:.4f} batches={epoch_batches} steps={global_step}"
            )

        # Save checkpoint at end of epoch (optional)
        if ckpt_dir is not None:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "dropout": float(dropout),
                "loss": str(loss),
                "margin": float(margin),
                "positive_strength": float(positive_strength),
                "state_dict": adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            snap = _save_checkpoint(
                checkpoint_dir=ckpt_dir,
                snapshot_name=f"epoch_{epoch+1:03d}_step_{global_step:07d}.pt",
                ckpt_obj=ckpt,
                also_write=ckpt_file,
            )
            print(f"Saved checkpoint snapshot: {snap}")

        if max_steps is not None and global_step >= int(max_steps):
            break

    return adapter


# --- HOW TO USE FOR INDEXING & SEARCH ---

def get_new_index_embedding(adapter: PokemonClipAdapter, clean_clip_embedding: torch.Tensor) -> torch.Tensor:
    """Run this on your entire Clean Database once."""
    adapter.eval()
    with torch.no_grad():
        return adapter(_as_batch(clean_clip_embedding))


def get_query_embedding(adapter: PokemonClipAdapter, noisy_real_photo_embedding: torch.Tensor) -> torch.Tensor:
    """Run this on the user's input image."""
    adapter.eval()
    with torch.no_grad():
        return adapter(_as_batch(noisy_real_photo_embedding))


if __name__ == "__main__":
    # Minimal CLI usage:
    #   python -m model.embedding_adapter --images_dir ... --checkpoint ...
    import argparse

    p = argparse.ArgumentParser(description="Train PokemonClipAdapter with a simple triplet loop")
    p.add_argument("--images_dir", type=str, default=None, help="Folder containing clean card images")
    p.add_argument("--model_type", type=str, default="clip", help="Embedding backend: clip | dino")
    p.add_argument("--output_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--loss", type=str, default="triplet", choices=["triplet", "cosine"])
    p.add_argument("--positive_strength", type=float, default=1.0, help="0..1: quanto è aggressivo il positive (noisy transform)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str((__import__("pathlib").Path(__file__).resolve().parents[1] / "storage" / "finetuning" / "adapter" / "adapter.pt")),
        help=(
            "Path file .pt oppure directory. Se directory, salva snapshot per-epoch (e latest.pt). "
            "Se file, scrive anche snapshot nella stessa directory."
        ),
    )
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Se >0, salva anche snapshot ogni N step (in checkpoint directory).",
    )
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    train_adapter(
        images_dir=args.images_dir,
        model_type=args.model_type,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        margin=args.margin,
        loss=args.loss,
        positive_strength=args.positive_strength,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        device=args.device,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint,
        save_every_steps=(None if int(args.save_every_steps) <= 0 else int(args.save_every_steps)),
        seed=args.seed,
    )
