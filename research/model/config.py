"""Configurazione centralizzata di path e directory del progetto.

Obiettivo
---------
Evitare duplicazioni tra `model/classification_loop.py` e gli script in `tests/`.

Questo modulo non dovrebbe importare roba pesante (torch/ultralytics) per restare
leggero e sicuro da importare ovunque.
"""

from __future__ import annotations

from pathlib import Path


# Repo root = parent della cartella `model/`
REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = REPO_ROOT / "storage"

# Default bundle directory (faiss index + names + meta.json)
INDICES_DIR = STORAGE_DIR / "indices"
DEFAULT_INDEX_DIR = INDICES_DIR / "model=clip__q=high"

# YOLO detector weights (segmentation)
DEFAULT_YOLO_SEG_WEIGHTS = STORAGE_DIR / "model_weights" / "pokemon-yolo11n-seg-v3.pt"

# Card images directory (per la gallery)
CARDS_DIR = STORAGE_DIR / "images" / "cards"

# Test data directories
TEST_DATA_IMAGES_DIR = STORAGE_DIR / "test_data" / "images"
TEST_DATA_VIDEOS_DIR = STORAGE_DIR / "test_data" / "videos"

# Visualization output
VIS_DIR = STORAGE_DIR / "vis"
