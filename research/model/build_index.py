import os
import sqlite3
import json
from datetime import datetime

import faiss
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# NOTE: nessun export globale emb_model/preprocess.
# Tutto passa tramite embedding_backends.get_backend(...).
from model.embedding_backends import get_backend, outputs_for
from model.embedding_backends import build_index_subdir
from model.config import INDICES_DIR, CARDS_DIR


def _write_meta(index_dir: str, *, model_type: str, source: str, extra: dict | None = None) -> None:
    meta = {
        "model_type": str(model_type),
        "source": str(source),
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        meta.update(extra)
    meta_path = os.path.join(index_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _write_variants(
    out_dir: str,
    *,
    embeddings_np: np.ndarray,
) -> dict:
    """Scrive su disco gli indici (raw, normalized, centered, centered_normalized) + avg_embedding.npy.

    Usa sempre IndexFlatIP.
    Ritorna un dict con i filename creati.
    """
    embeddings_np = np.asarray(embeddings_np, dtype=np.float32)
    if embeddings_np.ndim != 2 or embeddings_np.shape[0] <= 0:
        raise ValueError(f"Invalid embeddings shape: {embeddings_np.shape}")

    d = int(embeddings_np.shape[1])

    # 1) raw
    raw_index = faiss.IndexFlatIP(d)
    raw_index.add(embeddings_np)
    raw_path = os.path.join(out_dir, "pokemon_cards.index")
    faiss.write_index(raw_index, raw_path)

    # 2) normalized
    emb_norm = _l2_normalize_rows(embeddings_np)
    norm_index = faiss.IndexFlatIP(d)
    norm_index.add(emb_norm)
    norm_path = os.path.join(out_dir, "pokemon_cards_norm.index")
    faiss.write_index(norm_index, norm_path)

    # 3) centered (NO normalization)
    avg = embeddings_np.mean(axis=0, keepdims=False).astype(np.float32)
    centered = embeddings_np - avg.reshape(1, -1)
    centered_index = faiss.IndexFlatIP(d)
    centered_index.add(centered)
    centered_path = os.path.join(out_dir, "pokemon_cards_centered.index")
    faiss.write_index(centered_index, centered_path)

    # 4) centered + normalized
    centered_norm = _l2_normalize_rows(centered)
    centered_norm_index = faiss.IndexFlatIP(d)
    centered_norm_index.add(centered_norm)
    centered_norm_path = os.path.join(out_dir, "pokemon_cards_centered_norm.index")
    faiss.write_index(centered_norm_index, centered_norm_path)

    avg_path = os.path.join(out_dir, "avg_embedding.npy")
    np.save(avg_path, avg)

    return {
        "raw": os.path.basename(raw_path),
        "normalized": os.path.basename(norm_path),
        "raw_centered": os.path.basename(centered_path),
        "centered_normalized": os.path.basename(centered_norm_path),
        "avg_embedding": os.path.basename(avg_path),
    }


def build_index(
        image_folder,
        *,
        model_type: str = "clip",
        out_dir=INDICES_DIR,
        default_variant: str = "raw",
):
    # Put model metadata in a dedicated subfolder; keep filenames stable.
    subdir = build_index_subdir(model_type=model_type)
    out_dir = os.path.join(str(out_dir), subdir)
    os.makedirs(out_dir, exist_ok=True)

    outs = outputs_for(model_type, normalize=False, out_dir=str(out_dir))
    names_out = os.path.join(out_dir, outs.names_filename)
    embeddings_out = os.path.join(out_dir, outs.embeddings_filename)

    backend = get_backend(model_type)

    embeddings = []
    card_names = []

    # 2. Process all images (tracked)
    filenames = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    for filename in tqdm(filenames, desc=f"Indexing images (local, {model_type})"):
        img_path = os.path.join(image_folder, filename)
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # IMPORTANT: salva embeddings raw; le varianti vengono create dopo.
            emb = backend.encode(img, normalize=False)
            embeddings.append(emb)
            card_names.append(os.path.splitext(filename)[0])
        except Exception as exc:
            print(f"Failed to process {filename}: {exc}")
            continue

    if not embeddings:
        raise ValueError("No embeddings generated (empty folder? unreadable images?)")

    embeddings_np = np.vstack(embeddings).astype("float32")

    # Save card names and raw embeddings
    np.save(names_out, card_names)
    np.save(embeddings_out, embeddings_np)

    variants = _write_variants(out_dir, embeddings_np=embeddings_np)

    _write_meta(
        out_dir,
        model_type=model_type,
        source="local_folder",
        extra={
            "image_folder": str(image_folder),
            "default_variant": str(default_variant),
            "variants": {
                "raw": variants["raw"],
                "normalized": variants["normalized"],
                "raw_centered": variants["raw_centered"],
                "centered_normalized": variants["centered_normalized"],
            },
            "avg_embedding": variants["avg_embedding"],
        },
    )
    print(
        "Database built successfully! Wrote:\n"
        f"- {os.path.join(out_dir, variants['raw'])}\n"
        f"- {os.path.join(out_dir, variants['normalized'])}\n"
        f"- {os.path.join(out_dir, variants['raw_centered'])}\n"
        f"- {os.path.join(out_dir, variants['centered_normalized'])}\n"
        f"- {names_out}\n"
        f"- {embeddings_out}\n"
        f"- {os.path.join(out_dir, variants['avg_embedding'])}\n"
        f"- {os.path.join(out_dir,'meta.json')}"
    )


def build_index_from_db(
        db_path: str,
        *,
        model_type: str = "clip",
        images_quality: str = "high",
        image_format: str = "jpg",
        out_dir=INDICES_DIR,
        timeout_s: float = 30.0,
        default_variant: str = "raw",
):
    # Put model + DB image parameters in subfolder; keep filenames stable.
    subdir = build_index_subdir(
        model_type=model_type,
        images_quality=images_quality,
        image_format=image_format,
    )
    out_dir = os.path.join(str(out_dir), subdir)
    os.makedirs(out_dir, exist_ok=True)

    outs = outputs_for(model_type, normalize=False, out_dir=str(out_dir))
    names_out = os.path.join(out_dir, outs.names_filename)
    embeddings_out = os.path.join(out_dir, outs.embeddings_filename)

    """
    Builds an index from high-definition images fetched from the base URL stored in the tcgdex_fetch DB
    (card_data.image_url).

    Does not save images to disk: download -> preprocess/encode -> discard bytes.
    """
    if images_quality not in {"low", "high"}:
        raise ValueError("images_quality must be one of: low, high")
    if image_format not in {"jpg", "png"}:
        raise ValueError("image_format must be: jpg or png")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT card_id, image_url FROM card_data WHERE image_url IS NOT NULL AND image_url != ''")
    rows = cur.fetchall()
    conn.close()

    backend = get_backend(model_type)

    embeddings = []
    card_names = []

    session = requests.Session()

    for card_id, base_url in tqdm(rows, desc=f"Indexing images (DB -> remote HD, {model_type})"):
        url = f"{base_url}/{images_quality}.{image_format}"

        try:
            resp = session.get(url, stream=True, timeout=timeout_s)
            resp.raise_for_status()
            img = Image.open(resp.raw)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as exc:
            print(f"Error downloading/decoding {card_id} {url}: {exc}")
            continue

        try:
            # IMPORTANT: salva embeddings raw; le varianti vengono create dopo.
            emb = backend.encode(img, normalize=False)
        except Exception as exc:
            print(f"Error encoding {card_id}: {exc}")
            continue

        embeddings.append(emb)
        card_names.append(card_id)

    if not embeddings:
        raise ValueError("No embeddings generated (empty DB? network issues? missing image_url?)")

    embeddings_np = np.vstack(embeddings).astype("float32")

    # Save card names and raw embeddings
    np.save(names_out, card_names)
    np.save(embeddings_out, embeddings_np)

    variants = _write_variants(out_dir, embeddings_np=embeddings_np)

    _write_meta(
        out_dir,
        model_type=model_type,
        source="tcgdex_db",
        extra={
            "db_path": str(db_path),
            "images_quality": str(images_quality),
            "image_format": str(image_format),
            "timeout_s": float(timeout_s),
            "default_variant": str(default_variant),
            "variants": {
                "raw": variants["raw"],
                "normalized": variants["normalized"],
                "raw_centered": variants["raw_centered"],
                "centered_normalized": variants["centered_normalized"],
            },
            "avg_embedding": variants["avg_embedding"],
        },
    )
    print(
        "Database built successfully! Wrote:\n"
        f"- {os.path.join(out_dir, variants['raw'])}\n"
        f"- {os.path.join(out_dir, variants['normalized'])}\n"
        f"- {os.path.join(out_dir, variants['raw_centered'])}\n"
        f"- {os.path.join(out_dir, variants['centered_normalized'])}\n"
        f"- {names_out}\n"
        f"- {embeddings_out}\n"
        f"- {os.path.join(out_dir, variants['avg_embedding'])}\n"
        f"- {os.path.join(out_dir,'meta.json')}"
    )


if __name__ == "__main__":
    # build_index(
    #     r"C:\Users\salom\workspace\poketrade\storage\images\cards",
    #     out_dir=r"C:\Users\salom\workspace\poketrade\storage\index_large",
    # )

    # build_index_from_db(
    #     db_path=r"C:\Users\salom\workspace\poketrade\storage\db\tcgdex_cards.db",
    #     model_type="clip",
    #     images_quality="high",
    #     image_format="jpg",
    #     default_variant="raw",
    #     out_dir=INDICES_DIR / "test",
    # )

    build_index(
        CARDS_DIR,
        model_type="clip",
        default_variant="raw",
    )

    # build_index_from_db(
    #     db_path=r"C:\Users\salom\workspace\poketrade\storage\db\tcgdex_cards.db",
    #     model_type="dinov2-small",
    #     images_quality="high",
    #     image_format="jpg",
    #     default_variant="raw",
    # )
