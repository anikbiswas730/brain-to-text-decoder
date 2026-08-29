"""
src/inference.py — turning neural trials into CTC emission tensors that the
beam-search decoder consumes.

    extract_emissions_from_loader : validation loader (has references + day idx)
    block_load_test_data          : load + normalize the competition test HDF5
    extract_emissions             : test samples -> padded emission tensor
"""

from glob import glob
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import amp_autocast


def extract_emissions_from_loader(model, loader, device, use_amp=True):
    """Emissions for a DataLoader that also carries reference sentences + days."""
    model.eval()
    all_emissions, all_lengths, all_refs, all_days = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting Val Emissions'):
            neural = batch['neural'].to(device)
            lengths = batch['lengths']
            day_idx = batch['day_idx'].to(device)
            with amp_autocast(enabled=use_amp):
                log_probs, output_lengths = model(neural, lengths, day_idx)
            log_probs = log_probs.transpose(0, 1).float().cpu()  # [B, T, V]
            all_emissions.append(log_probs)
            all_lengths.append(output_lengths.cpu())
            all_refs.extend(batch['sentences'])
            all_days.extend(day_idx.cpu().tolist())

    max_T = max(e.shape[1] for e in all_emissions)
    padded = [F.pad(e, (0, 0, 0, max_T - e.shape[1])) for e in all_emissions]
    return torch.cat(padded, dim=0), torch.cat(all_lengths, dim=0), all_refs, all_days


def block_load_test_data(data_dir, session2idx, feat_mean, feat_std, clip=5.0):
    """Load the competition test split, normalized with the TRAIN stats, in
    chronological (session, block, trial) order for a valid submission."""
    files = sorted(glob(f'{data_dir}/**/data_test.hdf5', recursive=True))
    all_samples, sample_id = [], 0
    for filepath in tqdm(files, desc="Loading Test HDF5"):
        session = Path(filepath).parent.name
        with h5py.File(filepath, 'r') as f:
            trial_keys = [k for k in f.keys() if 'trial' in k.lower()]
            entries = []
            for trial_key in trial_keys:
                trial = f[trial_key]
                if 'input_features' not in trial:
                    continue
                entries.append((int(trial.attrs.get('block_num', 0)),
                                int(trial.attrs.get('trial_num', 0)), trial_key))
            entries.sort(key=lambda e: (e[0], e[1]))
            for block_num, trial_num, trial_key in entries:
                trial = f[trial_key]
                features = trial['input_features'][:trial.attrs['n_time_steps']]
                features = torch.FloatTensor(features)
                features = (features - feat_mean) / feat_std
                features = torch.clamp(features, -clip, clip)
                all_samples.append({
                    'id': sample_id, 'session': session, 'day_idx': session2idx[session],
                    'block_num': block_num, 'trial_num': trial_num,
                    'trial_key': trial_key, 'features': features,
                })
                sample_id += 1
    return all_samples


def extract_emissions(model, samples, device, use_amp=True, batch_size=32):
    """Emissions for a list of test samples (returns emissions, lengths, ids)."""
    model.eval()
    all_emissions, all_lengths, all_ids = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc='Extracting Test Emissions'):
            batch = samples[i:i + batch_size]
            features = [s['features'] for s in batch]
            lengths = torch.LongTensor([len(f) for f in features])
            day_idx = torch.LongTensor([s['day_idx'] for s in batch]).to(device)
            features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).to(device)
            sorted_lengths, sorted_idx = lengths.sort(descending=True)
            with amp_autocast(enabled=use_amp):
                log_probs, output_lengths = model(
                    features_padded[sorted_idx], sorted_lengths, day_idx[sorted_idx])
            log_probs = log_probs.transpose(0, 1).float().cpu()

            unsorted = torch.empty_like(log_probs)
            unsorted_lengths = torch.empty_like(output_lengths.cpu())
            unsorted[sorted_idx] = log_probs
            unsorted_lengths[sorted_idx] = output_lengths.cpu()
            all_emissions.append(unsorted)
            all_lengths.append(unsorted_lengths)
            all_ids.extend([s['id'] for s in batch])

    max_T = max(e.shape[1] for e in all_emissions)
    all_emissions = [F.pad(e, (0, 0, 0, max_T - e.shape[1])) for e in all_emissions]
    return torch.cat(all_emissions, dim=0), torch.cat(all_lengths, dim=0), all_ids
