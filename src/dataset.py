"""
src/dataset.py — the data path: HDF5 -> flat float16 cache -> pread reader ->
length-bucketed batches, plus the cross-fit split and the GPU augmentations.

Why a cache and not h5py in the DataLoader
------------------------------------------
Two training workers on one machine must not each hold a private copy of ~13 GB
of neural data. Each split is written once as ONE flat float16 binary plus a
meta pickle; trials are then read with `os.pread`, so nothing is mapped into the
process address space, RSS stays flat, and the OS page cache (shared between
both workers) does the caching. v6.0 used h5py + DataLoader subprocesses +
pinned memory + memmap and was OOM-killed by the kernel after 27 minutes.
"""

import os
import pickle
import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cache build / read
# ---------------------------------------------------------------------------
def trial_key(session, block, trial):
    return f'{session}|{int(block)}|{int(trial)}'


def build_cache(data_dir, cache_dir, splits=('train', 'val', 'test'), log=print):
    """One float16 flat binary per split + a meta pickle. Idempotent: a split
    with a `.done` flag is skipped, so a relaunched session reuses the cache."""
    import h5py
    os.makedirs(cache_dir, exist_ok=True)
    for split in splits:
        done_flag = os.path.join(cache_dir, f'{split}.done')
        if os.path.exists(done_flag):
            log(f'[cache] {split}: already cached')
            continue
        files = sorted(glob(data_dir + f'/**/data_{split}.hdf5', recursive=True))
        bin_path = os.path.join(cache_dir, f'{split}_neural.f16')
        meta, offset, t0 = [], 0, time.time()
        with open(bin_path, 'wb') as fout:
            for fp in files:
                session = Path(fp).parent.name
                with h5py.File(fp, 'r') as f:
                    entries = []
                    for k in f.keys():
                        tr = f[k]
                        if 'input_features' not in tr:
                            continue
                        entries.append((int(tr.attrs.get('block_num', 0)),
                                        int(tr.attrs.get('trial_num', 0)), k))
                    entries.sort(key=lambda e: (e[0], e[1]))     # canonical trial order
                    for block, trial, k in entries:
                        tr = f[k]
                        n = int(tr.attrs['n_time_steps'])
                        x = np.asarray(tr['input_features'][:n], dtype=np.float16)
                        fout.write(np.ascontiguousarray(x).tobytes())
                        s = tr.attrs.get('sentence_label', '')
                        s = s.decode('utf-8') if isinstance(s, bytes) else (s or '')
                        if 'seq_class_ids' in tr:
                            plen = (int(tr.attrs['seq_len']) if 'seq_len' in tr.attrs
                                    else len(tr['seq_class_ids']))
                            ph = np.asarray(tr['seq_class_ids'][:plen], dtype=np.int16)
                        else:
                            ph = np.zeros(0, dtype=np.int16)
                        meta.append({'session': session, 'block': block, 'trial': trial,
                                     'n': n, 'offset': offset, 'sentence': s, 'phonemes': ph,
                                     'key': trial_key(session, block, trial)})
                        offset += n
        with open(os.path.join(cache_dir, f'{split}_meta.pkl'), 'wb') as f:
            pickle.dump({'meta': meta, 'n_frames': offset, 'n_feat': 512}, f)
        open(done_flag, 'w').close()
        log(f'[cache] {split}: {len(meta)} trials, {offset} frames, '
            f'{offset * 512 * 2 / 1e9:.2f} GB, {time.time() - t0:.0f}s')


class CacheReader:
    """Reads trials with os.pread. The file descriptor is re-opened per process
    (see __getstate__) so the reader can be pickled into a worker."""

    def __init__(self, cache_dir, split):
        with open(os.path.join(cache_dir, f'{split}_meta.pkl'), 'rb') as f:
            d = pickle.load(f)
        self.meta = d['meta']
        self.n_feat = d['n_feat']
        self.n_frames = d['n_frames']
        self.path = os.path.join(cache_dir, f'{split}_neural.f16')
        self._fd, self._pid = None, None

    def _fdesc(self):
        if self._fd is None or self._pid != os.getpid():
            self._fd, self._pid = os.open(self.path, os.O_RDONLY), os.getpid()
        return self._fd

    def __getstate__(self):
        st = dict(self.__dict__)
        st['_fd'] = None
        st['_pid'] = None
        return st

    def __len__(self):
        return len(self.meta)

    def read_frames(self, offset, n):
        nbytes = n * self.n_feat * 2
        buf = os.pread(self._fdesc(), nbytes, offset * self.n_feat * 2)
        return np.frombuffer(buf, dtype=np.float16).reshape(n, self.n_feat)

    def neural(self, i):
        m = self.meta[i]
        return self.read_frames(m['offset'], m['n'])


def compute_norm_stats(reader, chunk=200_000):
    """Per-channel mean/std over the whole train split, streamed in chunks.

    Computed once and then COPIED next to the checkpoints: decoding must use the
    exact statistics the model was trained under, and a warm-started run must
    reuse its source run's statistics or the day adapters no longer mean the
    same thing.
    """
    n = reader.n_frames
    s1 = np.zeros(reader.n_feat, np.float64)
    s2 = np.zeros(reader.n_feat, np.float64)
    for a in range(0, n, chunk):
        x = reader.read_frames(a, min(chunk, n - a)).astype(np.float32)
        s1 += x.sum(0, dtype=np.float64)
        s2 += (x * x).sum(0, dtype=np.float64)
        del x
    mean = s1 / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 0))
    std[std < 1e-6] = 1.0
    return {'mean': torch.tensor(mean, dtype=torch.float32),
            'std': torch.tensor(std, dtype=torch.float32)}


# ---------------------------------------------------------------------------
# cross-fit split
# ---------------------------------------------------------------------------
def make_crossfit_split(val_meta, pattern=('A', 'B', 'A', 'B', 'C'), seed=1337):
    """TRIAL-level split INSIDE every (session, block) of val.

    Trial-level and not block-level because the Kaggle test trials are
    interleaved with val trials inside the SAME blocks (e.g. t15.2025.01.10 has
    both a val block 8 and a test block 8). A model trained on part of a block
    and scored on the rest of that block is therefore the honest proxy for
    'trained on val, scored on test'; holding out whole blocks would understate
    how much the day adapters help.
    """
    rng = random.Random(seed)
    groups = {}
    for m in val_meta:
        groups.setdefault((m['session'], m['block']), []).append(m)
    labels = {}
    for g in sorted(groups):
        items = sorted(groups[g], key=lambda m: m['trial'])
        off = rng.randrange(len(pattern))
        for j, m in enumerate(items):
            labels[m['key']] = pattern[(j + off) % len(pattern)]
    return labels


# ---------------------------------------------------------------------------
# datasets / batching
# ---------------------------------------------------------------------------
class TrialSet(torch.utils.data.Dataset):
    """items = list of (reader_name, index) pairs, so one dataset can mix trials
    from the train and val caches (which is exactly what a cross-fit run does)."""

    def __init__(self, readers, items, session2idx):
        self.readers = readers
        self.items = items
        self.session2idx = session2idx
        self.lengths = [readers[r].meta[i]['n'] for r, i in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        r, i = self.items[j]
        rd = self.readers[r]
        m = rd.meta[i]
        return {'neural': np.array(rd.neural(i), dtype=np.float16, copy=True),
                'target': m['phonemes'].astype(np.int64),
                'day': self.session2idx[m['session']], 'j': j}


def collate_trials(batch):
    B = len(batch)
    T = max(b['neural'].shape[0] for b in batch)
    L = max(max(len(b['target']) for b in batch), 1)
    x = torch.zeros(B, T, batch[0]['neural'].shape[1], dtype=torch.float16)
    tg = torch.zeros(B, L, dtype=torch.long)
    lens, tlens, days, js = [], [], [], []
    for k, b in enumerate(batch):
        n = b['neural'].shape[0]
        x[k, :n] = torch.from_numpy(b['neural'])
        t = b['target']
        tg[k, :len(t)] = torch.from_numpy(t)
        lens.append(n)
        tlens.append(len(t))
        days.append(b['day'])
        js.append(b['j'])
    return {'neural': x, 'target': tg, 'lengths': torch.tensor(lens),
            'target_lengths': torch.tensor(tlens), 'day_idx': torch.tensor(days),
            'j': torch.tensor(js)}


class BucketBatchSampler(torch.utils.data.Sampler):
    """Length-bucketed batches capped by BOTH #trials and #bins.

    The bin cap is what keeps GPU memory bounded: without it one batch of long
    trials can be 5x the activation footprint of an average batch, and on a 15 GB
    T4 running at ~7 GB peak that is the difference between finishing and an OOM.
    The cap is lowered at runtime by the worker whenever a batch does OOM.
    """

    def __init__(self, lengths, max_batch, max_bins, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths)
        self.max_batch, self.max_bins, self.shuffle = max_batch, max_bins, shuffle
        self.rng = np.random.RandomState(seed)
        self._batches = self._make()

    def _make(self):
        L = self.lengths
        if self.shuffle:
            order = np.argsort(L * self.rng.uniform(0.9, 1.1, size=len(L)))
        else:
            order = np.argsort(L)
        batches, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, L[i])
            if cur and (len(cur) + 1 > self.max_batch or m * (len(cur) + 1) > self.max_bins):
                batches.append(cur)
                cur, m = [], L[i]
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches
        self._batches = self._make()      # reshuffle for the next epoch
        return iter(batches)

    def __len__(self):
        return len(self._batches)


# ---------------------------------------------------------------------------
# GPU augmentation (train only)
# ---------------------------------------------------------------------------
def speed_perturb(x, lengths, lo, hi):
    """x [B,T,C] -> resampled along time by ONE factor per batch.

    One factor for the whole batch (rather than per trial) keeps the two CR-CTC
    views time-aligned, which the consistency loss requires.
    """
    f = float(np.random.uniform(lo, hi))
    if abs(f - 1.0) < 1e-3:
        return x, lengths
    B, T, C = x.shape
    newT = max(int(round(T * f)), 2)
    y = F.interpolate(x.transpose(1, 2).float(), size=newT, mode='linear', align_corners=False)
    new_len = torch.clamp((lengths.float() * f).round().long(), min=1, max=newT)
    return y.transpose(1, 2).to(x.dtype), new_len


def view_augment(x, lengths, cfg):
    """Independent per-view noise: static gain, white noise, constant offset and
    electrode dropout. Electrode dropout drops the tx and sbp channels of the
    SAME electrode together (channels c and c + C/2), which is how the recording
    actually fails — dropping them independently would be an easier problem than
    the one the model faces at test time."""
    B, T, C = x.shape
    dev = x.device
    if cfg.get('static_gain_std', 0) > 0:
        x = x * (1.0 + torch.randn(B, 1, C, device=dev, dtype=x.dtype) * cfg['static_gain_std'])
    if cfg.get('white_noise_std', 0) > 0:
        x = x + torch.randn_like(x) * cfg['white_noise_std']
    if cfg.get('constant_offset_std', 0) > 0:
        x = x + torch.randn(B, 1, C, device=dev, dtype=x.dtype) * cfg['constant_offset_std']
    p = cfg.get('electrode_drop_p', 0.0)
    if p > 0 and C % 2 == 0:
        n_el = C // 2
        apply = (torch.rand(B, 1, device=dev) < cfg.get('electrode_drop_apply', 0.5)).to(x.dtype)
        keep_el = (torch.rand(B, n_el, device=dev) >= p).to(x.dtype)
        keep_el = 1.0 - apply * (1.0 - keep_el)                       # only for 'apply' trials
        keep = torch.cat([keep_el, keep_el], dim=1)                   # tx + sbp of one electrode
        x = x * keep[:, None, :]
    return x


def make_time_mask(lengths, T, frac, min_span, max_span, device):
    """bool [B,T], True = masked. ~frac of each trial's valid bins, in spans of
    [min_span, max_span]. Built on CPU and moved once: masking span-by-span on
    the GPU costs hundreds of kernel launches per step."""
    B = len(lengths)
    mask = np.zeros((B, T), dtype=bool)
    if frac > 0:
        mean_span = 0.5 * (min_span + max_span)
        for b in range(B):
            n = int(lengths[b])
            k = int(round(frac * n / mean_span))
            if k <= 0 or n <= min_span:
                continue
            spans = np.random.randint(min_span, max_span + 1, size=k)
            starts = np.random.randint(0, max(n - min_span, 1), size=k)
            for s0, w in zip(starts, spans):
                mask[b, s0:min(s0 + w, n)] = True
    return torch.from_numpy(mask).to(device, non_blocking=True)
