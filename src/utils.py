"""
src/utils.py — small cross-cutting helpers shared by every other module:
seeding, stochastic-depth, the session->day-index map, robust path resolution
for Kaggle mounts, and an AMP compatibility shim that avoids the deprecated
``torch.cuda.amp`` API.
"""

import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Stochastic depth (drop path)
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# ---------------------------------------------------------------------------
# Session -> day index (chronological)
# ---------------------------------------------------------------------------
def get_session2idx(data_dir):
    """Scan the data directory and map each unique session folder to a day index."""
    paths = glob(f'{data_dir}/**/data_*.hdf5', recursive=True)
    sessions = sorted(list(set(Path(p).parent.name for p in paths)))
    return {s: i for i, s in enumerate(sessions)}


# ---------------------------------------------------------------------------
# Path resolution (Kaggle mounts / kagglehub fallback)
# ---------------------------------------------------------------------------
def resolve_competition_path(local_path, competition_slug):
    """Return the local competition mount if present, else fetch via kagglehub."""
    if os.path.exists(local_path):
        return local_path
    try:
        import kagglehub
    except ImportError:
        kagglehub = None
    if kagglehub is not None:
        return os.path.join(
            kagglehub.competition_download(competition_slug),
            "t15_copyTask_neuralData/hdf5_data_final",
        )
    raise FileNotFoundError(f"{local_path} not found and kagglehub unavailable.")


def resolve_kaggle_dataset(slug, local_dirname=None):
    """Prefer an already-attached dataset under /kaggle/input (works in both
    interactive and non-interactive runs); fall back to kagglehub only if
    nothing is found locally."""
    if local_dirname is None:
        local_dirname = slug.split('/')[-1]
    owner = slug.split('/')[0] if '/' in slug else None

    candidates = [f'/kaggle/input/{local_dirname}']
    if owner:
        candidates.append(f'/kaggle/input/datasets/{owner}/{local_dirname}')

    if os.path.isdir('/kaggle/input'):
        for root, dirs, _ in os.walk('/kaggle/input'):
            for d in dirs:
                if local_dirname.lower() in d.lower() or d.lower() in local_dirname.lower():
                    candidates.append(os.path.join(root, d))

    for path in candidates:
        if os.path.exists(path):
            return path
    try:
        import kagglehub
        return kagglehub.dataset_download(slug)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find dataset '{slug}' under /kaggle/input (tried {candidates}) "
            f"and kagglehub.dataset_download() also failed: {e}\n"
            f"Fix: attach it via '+ Add Input' in the Kaggle editor, or pass an "
            f"explicit path on the command line."
        ) from e


def find_file(root_dir, filename_pattern):
    """Recursively find the first file under root_dir matching filename_pattern."""
    matches = sorted(glob(os.path.join(root_dir, '**', filename_pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matching '{filename_pattern}' found under {root_dir}")
    return matches[0]


# ---------------------------------------------------------------------------
# AMP compatibility shim (new torch.amp API, graceful fallback for old torch)
# ---------------------------------------------------------------------------
try:
    from torch.amp import autocast as _amp_autocast, GradScaler as _AmpGradScaler

    def amp_autocast(enabled=True):
        return _amp_autocast('cuda', enabled=enabled)

    def make_grad_scaler(enabled=True):
        return _AmpGradScaler('cuda', enabled=enabled)

except Exception:  # pragma: no cover - only on very old torch
    from torch.cuda.amp import autocast as _cuda_autocast, GradScaler as _CudaGradScaler

    def amp_autocast(enabled=True):
        return _cuda_autocast(enabled=enabled)

    def make_grad_scaler(enabled=True):
        return _CudaGradScaler(enabled=enabled)
