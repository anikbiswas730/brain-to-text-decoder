"""
Data augmentation, HDF5 loading, and the PyTorch Dataset for the
Brain-to-Text neural recordings.

Expects the Kaggle "brain-to-text-25" competition layout:
    <data_dir>/<session_name>/data_train.hdf5
    <data_dir>/<session_name>/data_val.hdf5
    <data_dir>/<session_name>/data_test.hdf5

Each HDF5 file contains one group per trial, with:
    trial['input_features']       -> [T, 512] neural feature array
    trial.attrs['n_time_steps']   -> valid length T (features may be padded)
    trial.attrs['sentence_label'] -> ground-truth sentence (train/val only)
    trial.attrs['block_num'], trial.attrs['trial_num']
    trial['seq_class_ids'], trial.attrs['seq_len'] -> optional phoneme labels
"""

import random
from glob import glob
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class NeuralAugmentation:
    """Light augmentation for neural time-series: time warping, additive
    Gaussian noise, and random channel dropout."""

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
    """Loads every trial for a given split ('train' | 'val' | 'test') across
    all session directories into memory as a dict of lists."""
    pattern = f'{data_dir}/**/data_{split}.hdf5'
    files = sorted(glob(pattern, recursive=True))

    print(f"\nLoading {split} split...")
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
                all_data['sentence'].append(sentence.decode('utf-8') if isinstance(sentence, bytes) else sentence)

                all_data['phonemes'].append(trial.get('seq_class_ids')[:] if 'seq_class_ids' in trial else None)
                all_data['phoneme_len'].append(trial.attrs.get('seq_len'))
    print(f"\u2713 Loaded {len(all_data['neural'])} samples")
    return all_data


class BrainToTextDataset(Dataset):
    """Wraps loaded neural/sentence data, handling normalization,
    augmentation, and character-level tokenization."""

    def __init__(self, data, session2idx, char2idx=None, normalize=True, augment=False):
        self.neural = data['neural']
        self.n_steps = data['n_steps']
        self.sentences = data['sentence']
        self.sessions = data['session']
        self.session2idx = session2idx
        self.normalize = normalize
        self.augment = augment
        self.augmentation = NeuralAugmentation(p=0.5) if augment else None

        self.char2idx = char2idx if char2idx is not None else self._build_vocab()
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def _build_vocab(self):
        chars = set()
        for sent in self.sentences:
            if sent:
                chars.update(sent.lower())
        chars = sorted(list(chars))
        char2idx = {'<BLANK>': 0}
        for i, ch in enumerate(chars, start=1):
            char2idx[ch] = i
        return char2idx

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, idx):
        neural = self.neural[idx][:self.n_steps[idx]]
        neural = torch.FloatTensor(neural)

        if self.normalize:
            neural = (neural - neural.mean()) / (neural.std() + 1e-8)
        if self.augment and self.augmentation:
            neural = self.augmentation(neural)

        sentence = self.sentences[idx] if self.sentences[idx] else ""
        target = [self.char2idx.get(ch.lower(), 0) for ch in sentence]

        return {
            'neural': neural,
            'target': torch.LongTensor(target),
            'length': len(neural),
            'target_length': len(target),
            'sentence': sentence,
            'day_idx': self.session2idx[self.sessions[idx]]
        }


def collate_fn(batch):
    """Pads a batch of variable-length neural sequences/targets."""
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    neurals = pad_sequence([item['neural'] for item in batch], batch_first=True)
    targets = pad_sequence([item['target'] for item in batch], batch_first=True)
    return {
        'neural': neurals,
        'target': targets,
        'lengths': torch.LongTensor([item['length'] for item in batch]),
        'target_lengths': torch.LongTensor([item['target_length'] for item in batch]),
        'sentences': [item['sentence'] for item in batch],
        'day_idx': torch.LongTensor([item['day_idx'] for item in batch])
    }


def load_test_samples(data_dir, session2idx):
    """Loads the test split into the flat sample-dict format used by the
    prediction/decoding scripts (as opposed to the batched Dataset above,
    since test-time inference iterates over samples directly)."""
    pattern = f'{data_dir}/**/data_test.hdf5'
    files = sorted(glob(pattern, recursive=True))

    all_samples = []
    sample_id = 0
    for filepath in tqdm(files, desc="Loading test HDF5"):
        session = Path(filepath).parent.name
        with h5py.File(filepath, 'r') as f:
            trial_keys = [k for k in f.keys() if 'trial' in k.lower()]
            for trial_key in trial_keys:
                trial = f[trial_key]
                if 'input_features' not in trial:
                    continue
                features = trial['input_features'][:trial.attrs['n_time_steps']]
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
                import numpy as np
                features = np.clip(features, -5, 5)
                all_samples.append({
                    'id': sample_id,
                    'session': session,
                    'day_idx': session2idx[session],
                    'trial_key': trial_key,
                    'features': torch.FloatTensor(features)
                })
                sample_id += 1
    return all_samples
