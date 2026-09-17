"""
src/utils.py — cross-cutting helpers shared by every other module: seeding,
robust path resolution for Kaggle mounts, scratch-disk selection, atomic saves,
memory probes, and the figure/table writers used by the reporting stages.

Nothing in here imports the model or the dataset, so the training worker can
pull it in without dragging in decoding dependencies (flashlight / kenlm /
transformers), which is exactly what OOM-killed the v6.0 workers.
"""

import contextlib
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Session -> day index (chronological)
# ---------------------------------------------------------------------------
def get_session2idx(data_dir):
    """Map each recording-session folder under data_dir to a day index."""
    paths = glob(f'{data_dir}/**/data_*.hdf5', recursive=True)
    sessions = sorted(set(Path(p).parent.name for p in paths))
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
            't15_copyTask_neuralData/hdf5_data_final',
        )
    raise FileNotFoundError(f'{local_path} not found and kagglehub unavailable.')


def resolve_kaggle_dataset(slug, local_dirname=None):
    """Prefer an already-attached dataset under /kaggle/input (works in both
    interactive and non-interactive runs); fall back to kagglehub only if
    nothing is found locally. A plain directory path is returned unchanged."""
    if slug is None:
        raise FileNotFoundError('no dataset slug given')
    if os.path.isdir(slug):
        return slug
    local_dirname = local_dirname or slug.split('/')[-1]
    owner = slug.split('/')[0] if '/' in slug else None

    candidates = ['/kaggle/input/' + local_dirname]
    if owner:
        candidates.append('/kaggle/input/datasets/' + owner + '/' + local_dirname)
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
            f'and kagglehub.dataset_download() also failed: {e}\n'
            f"Fix: attach it via '+ Add Input' in the Kaggle editor, or pass an "
            f'explicit path on the command line.'
        ) from e


def find_file(root_dir, filename_pattern):
    """Recursively find the first file under root_dir matching filename_pattern."""
    matches = sorted(glob(os.path.join(root_dir, '**', filename_pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matching '{filename_pattern}' found under {root_dir}")
    return matches[0]


def pick_cache_dir(candidates, need_gb=14.0, log=print):
    """First writable, NON-tmpfs directory with enough free disk.

    The trial cache is ~13 GB of float16. Putting it on a RAM-backed filesystem
    (/dev/shm, some /tmp mounts) silently spends host RAM that the training
    workers need, which is how v6.0 got OOM-killed at minute 27.
    """
    need_gb = float(os.environ.get('B2T_CACHE_NEED_GB', need_gb))
    tmpfs = set()
    try:
        for line in open('/proc/mounts'):
            parts = line.split()
            if len(parts) > 2 and parts[2] in ('tmpfs', 'ramfs'):
                tmpfs.add(parts[1])
    except Exception:
        pass
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, '.probe')
            open(probe, 'w').close()
            os.remove(probe)
            mount = c
            while not os.path.ismount(mount) and mount != '/':
                mount = os.path.dirname(mount)
            st = os.statvfs(c)
            free_gb = st.f_bavail * st.f_frsize / 1e9
            done = os.path.exists(os.path.join(c, 'test.done'))
            if mount in tmpfs:
                log(f'[cache] {c}: RAM-backed (tmpfs) -> skipped')
                continue
            if free_gb < need_gb and not done:
                log(f'[cache] {c}: only {free_gb:.1f} GB free -> skipped')
                continue
            log(f'[cache] using {c} (mount {mount}, {free_gb:.1f} GB free)')
            return c
        except Exception as e:
            log(f'[cache] {c}: not usable ({e!r})')
    raise RuntimeError(f'no usable cache directory among {candidates}')


# ---------------------------------------------------------------------------
# Checkpoints / memory
# ---------------------------------------------------------------------------
def atomic_save(obj, path):
    """Write to a temp file and rename: a checkpoint is never half-written, even
    if the process is killed mid-save."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rss_gb():
    """(process RSS, machine available) in GB; (0, inf) if psutil is missing."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9, psutil.virtual_memory().available / 1e9
    except Exception:
        return 0.0, 1e9


def mem_status():
    s = ''
    try:
        import psutil
        vm = psutil.virtual_memory()
        s += f'RAM avail {vm.available / 1e9:.1f}G rss {psutil.Process().memory_info().rss / 1e9:.1f}G'
    except Exception:
        pass
    if torch.cuda.is_available():
        s += (f' | GPU alloc {torch.cuda.memory_allocated() / 1e9:.1f}G '
              f'peak {torch.cuda.max_memory_allocated() / 1e9:.1f}G')
    return s


def human_time(s):
    s = int(max(s, 0))
    return f'{s // 3600:d}h{(s % 3600) // 60:02d}m'


def is_oom(e):
    msg = str(e)
    return (isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in msg.lower()
            or 'ALLOC_FAILED' in msg)


# ---------------------------------------------------------------------------
# Reporting outputs
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def figure_guard(name):
    """A plotting bug must never kill an 11-hour pipeline."""
    import matplotlib.pyplot as plt
    try:
        yield
    except Exception as e:
        print(f'  [figure_guard] {name} skipped: {e!r}')
    finally:
        plt.close('all')


def save_fig(fig, name, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figures_dir, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    print('  figure ->', os.path.join(figures_dir, name + '.png'))


def save_table(df, name, tables_dir):
    os.makedirs(tables_dir, exist_ok=True)
    p = os.path.join(tables_dir, name)
    df.to_csv(p, index=False)
    print('  table  ->', p)
    return p


def init_matplotlib():
    """Headless, paper-ish defaults. Called by the entry points, not on import."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt
