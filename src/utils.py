"""
Small shared helpers used across the training, prediction, and decoding
scripts.
"""

import random
from glob import glob
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed python/numpy/torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth: randomly drops whole residual branches during training."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def get_session2idx(data_dir):
    """Scans the data directory to find all unique recording sessions
    (i.e. "days") chronologically and assigns each one a stable integer id.

    The model uses this id to select a per-day linear transform, which lets
    it adapt to day-to-day drift in neural recordings.
    """
    paths = glob(f'{data_dir}/**/data_*.hdf5', recursive=True)
    sessions = sorted(list(set([Path(p).parent.name for p in paths])))
    return {s: i for i, s in enumerate(sessions)}
