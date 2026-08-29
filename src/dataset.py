"""
src/dataset.py — HDF5 data loading, neural augmentation, the phoneme-target
Dataset, its collate function, and streaming (Welford) per-channel
normalization statistics.

Targets are phoneme IDs read directly from each trial's ``seq_class_ids``;
normalization stats are computed once on TRAIN and reused unchanged for
val/test so the day-adaptive input layer sees a stable input scale.
"""

import random
from glob import glob
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class NeuralAugmentation:
    """Light stochastic augmentation: time-warp, additive noise, channel dropout."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, neural):
        if random.random() > self.p:
            return neural
        if random.random() < 0.3:
            neural = self.time_warp(neural)
        if random.random() < 0.3:
            noise_level = random.uniform(0.01, 0.05)
            neural = neural + torch.randn_like(neural) * noise_level
        if random.random() < 0.2:
            n_channels = neural.shape[1]
            n_drop = int(n_channels * 0.1)
            drop_indices = random.sample(range(n_channels), n_drop)
            neural[:, drop_indices] = 0
        return neural

    def time_warp(self, neural):
        seq_len = len(neural)
        warp_factor = random.uniform(0.9, 1.1)
        new_len = int(seq_len * warp_factor)
        if new_len < 10:
            return neural
        indices = torch.linspace(0, seq_len - 1, new_len)
        indices_floor = indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=seq_len - 1)
        alpha = (indices - indices_floor.float()).unsqueeze(1)
        warped = (1 - alpha) * neural[indices_floor] + alpha * neural[indices_ceil]
        final_indices = torch.linspace(0, new_len - 1, seq_len).long()
        return warped[final_indices]


def load_split(data_dir, split='train'):
    """Load every trial of a split into memory as parallel lists."""
    files = sorted(glob(f'{data_dir}/**/data_{split}.hdf5', recursive=True))
    print(f"\nLoading {split} split ...")
    all_data = {k: [] for k in ['neural', 'n_steps', 'sentence', 'phonemes',
                                 'phoneme_len', 'session', 'block', 'trial']}
    for filepath in tqdm(files):
        session_name = Path(filepath).parent.name
        with h5py.File(filepath, 'r') as f:
            for trial_key in f.keys():
                trial = f[trial_key]
                all_data['neural'].append(trial['input_features'][:])
                all_data['n_steps'].append(trial.attrs['n_time_steps'])
                all_data['session'].append(session_name)
                all_data['block'].append(trial.attrs['block_num'])
                all_data['trial'].append(trial.attrs['trial_num'])

                sentence = trial.attrs.get('sentence_label')
                all_data['sentence'].append(
                    sentence.decode('utf-8') if isinstance(sentence, bytes) else sentence)

                phon = trial['seq_class_ids'][:] if 'seq_class_ids' in trial \
                    else np.array([], dtype=np.int64)
                phon_len = int(trial.attrs['seq_len']) if 'seq_len' in trial.attrs else len(phon)
                all_data['phonemes'].append(phon)
                all_data['phoneme_len'].append(phon_len)
    print(f"Loaded {len(all_data['neural'])} samples")
    return all_data


def compute_channel_stats(data):
    """Streaming (Welford) per-channel mean/std over every TRAIN timestep."""
    n_channels = data['neural'][0].shape[1]
    count = 0
    mean = np.zeros(n_channels, dtype=np.float64)
    M2 = np.zeros(n_channels, dtype=np.float64)
    for feat, n_steps in zip(data['neural'], data['n_steps']):
        x = feat[:n_steps].astype(np.float64)
        for row in x:
            count += 1
            delta = row - mean
            mean += delta / count
            M2 += delta * (row - mean)
    std = np.sqrt(M2 / max(count - 1, 1))
    std[std < 1e-6] = 1e-6
    return (torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32))


class BrainToTextDataset(Dataset):
    def __init__(self, data, session2idx, feat_mean, feat_std, augment=False, clip=5.0):
        self.neural = data['neural']
        self.n_steps = data['n_steps']
        self.sentences = data['sentence']
        self.sessions = data['session']
        self.phonemes = data['phonemes']
        self.phoneme_len = data['phoneme_len']
        self.session2idx = session2idx
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.clip = clip
        self.augment = augment
        self.augmentation = NeuralAugmentation(p=0.5) if augment else None

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, idx):
        neural = self.neural[idx][:self.n_steps[idx]]
        neural = torch.FloatTensor(neural)

        # Global, split-independent normalization (same stats for train/val/test).
        neural = (neural - self.feat_mean) / self.feat_std
        neural = torch.clamp(neural, -self.clip, self.clip)

        if self.augment and self.augmentation:
            neural = self.augmentation(neural)

        target = torch.LongTensor(self.phonemes[idx])
        target_length = self.phoneme_len[idx]

        return {
            'neural': neural,
            'target': target,
            'length': len(neural),
            'target_length': target_length,
            'sentence': self.sentences[idx] if self.sentences[idx] else "",
            'day_idx': self.session2idx[self.sessions[idx]],
        }


def collate_fn(batch):
    """Sort by length (for pack_padded_sequence) and pad neural + target tensors."""
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    neurals = pad_sequence([item['neural'] for item in batch], batch_first=True)
    targets = pad_sequence([item['target'] for item in batch], batch_first=True)
    return {
        'neural': neurals,
        'target': targets,
        'lengths': torch.LongTensor([item['length'] for item in batch]),
        'target_lengths': torch.LongTensor([item['target_length'] for item in batch]),
        'sentences': [item['sentence'] for item in batch],
        'day_idx': torch.LongTensor([item['day_idx'] for item in batch]),
    }
