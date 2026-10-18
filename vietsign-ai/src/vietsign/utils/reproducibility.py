"""Reproducibility utilities."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds when available."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch  # type: ignore
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)

