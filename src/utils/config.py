"""Configuration and reproducibility utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    if not path.exists():
        # Try relative to repo root
        root = Path(__file__).resolve().parents[2]
        path = root / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 0) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

