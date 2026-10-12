"""Review Organizer source package.

Importable modules:
    preprocess  - text cleaning / normalisation
    features    - TF-IDF and sentence-embedding feature extraction
    cluster     - KMeans / Agglomerative / DBSCAN / LDA clustering + metrics
    classify    - supervised classifier training, tuning, evaluation
    evaluate    - shared metric helpers, coherence, sentiment, plots
    predict     - load saved artifacts and label new reviews
"""

from __future__ import annotations

import os
import random

import numpy as np

# --------------------------------------------------------------------------
# Project paths (single source of truth, relative to repo root)
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
CONFIG_DIR = os.path.join(ROOT, "config")

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG this project touches for reproducibility.

    Covers Python, NumPy and (if installed) PyTorch, so clustering,
    train/test splits and embedding-based models are deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch only needed for the optional DistilBERT step
