#!/usr/bin/env python
"""
predict.py — fast baseline submission: greedy CTC phoneme decode -> words via
lexicon exact-match, no beam search, no KenLM, no LLM.

This is the quick sanity path. For the accurate (7.55% WER) submission use
decode_llm.py.

Example
-------
    python predict.py --checkpoint checkpoints/best_model.pt \
        --output submission/submission.csv
"""

import os
import argparse
from glob import glob

import pandas as pd
import torch
from tqdm import tqdm

from config import (CONFIG, resolve_device, PRETRAINED_CKPT_DATASET,
                    LEXICON_DATASET_SLUG)
from src.utils import (set_seed, get_session2idx, resolve_kaggle_dataset, find_file)
from src.model import build_model
from src.metrics import greedy_decode_phonemes, build_lexicon_reverse, phoneme_ids_to_words
from src.inference import block_load_test_data, extract_emissions


def load_norm_and_model(cfg, n_days):
    device = cfg['device']
    search_dir = cfg['checkpoint_dir']
    if PRETRAINED_CKPT_DATASET and not os.path.exists(os.path.join(search_dir, 'norm_stats.pt')):
        search_dir = resolve_kaggle_dataset(PRETRAINED_CKPT_DATASET)
        print(f"Checkpoint source: {search_dir}")

    norm_stats = torch.load(find_file(search_dir, 'norm_stats.pt'))
    feat_mean, feat_std = norm_stats['mean'], norm_stats['std']

    candidates = [cfg.get('checkpoint'),
                  os.path.join(search_dir, 'swa_model.pt'),
                  os.path.join(search_dir, 'best_model.pt')]
    ckpt_path = next((p for p in candidates if p and os.path.exists(p)), None)
    assert ckpt_path is not None, f"No checkpoint found (tried {candidates})."
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(cfg, n_days).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, feat_mean, feat_std


def parse_args():
    p = argparse.ArgumentParser(description="Baseline greedy-CTC submission.")
    p.add_argument('--data_dir', default=CONFIG['data_dir'])
    p.add_argument('--checkpoint_dir', default=CONFIG['checkpoint_dir'])
    p.add_argument('--checkpoint', default=None, help='explicit checkpoint path')
    p.add_argument('--lexicon', default=None)
    p.add_argument('--output', default='submission/submission_baseline.csv')
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    cfg.update({'data_dir': args.data_dir, 'checkpoint_dir': args.checkpoint_dir,
                'checkpoint': args.checkpoint, 'use_amp': not args.no_amp})
    cfg['device'] = resolve_device(cfg['device'])
    set_seed(cfg['seed'])

    session2idx = get_session2idx(cfg['data_dir'])
    n_days = len(session2idx)

    model, feat_mean, feat_std = load_norm_and_model(cfg, n_days)

    lexicon_path = args.lexicon
    if lexicon_path is None:
        lexicon_path = os.path.join(resolve_kaggle_dataset(LEXICON_DATASET_SLUG), 'lexicon.txt')
    lexicon_reverse = build_lexicon_reverse(lexicon_path)

    print("Loading test set ...")
    samples = block_load_test_data(cfg['data_dir'], session2idx, feat_mean, feat_std,
                                   clip=cfg['feature_clip'])
    emissions, lengths, ids = extract_emissions(model, samples, cfg['device'],
                                                use_amp=cfg['use_amp'])

    print("Greedy CTC decoding ...")
    log_probs = emissions.transpose(0, 1)  # [T, B, V] for greedy_decode_phonemes
    predictions = []
    B = 64
    for i in tqdm(range(0, log_probs.shape[1], B), desc="Greedy decode"):
        chunk = log_probs[:, i:i + B, :]
        chunk_lengths = lengths[i:i + B]
        decoded = greedy_decode_phonemes(chunk, chunk_lengths)
        for phon_ids in decoded:
            predictions.append(phoneme_ids_to_words(phon_ids, lexicon_reverse))

    write_submission(cfg['data_dir'], ids, predictions, args.output)


def write_submission(data_dir, ids, predictions, output):
    sample_sub_path = None
    for pattern in [os.path.join(os.path.dirname(data_dir), '**', 'sample_submission.csv'),
                    '/kaggle/input/**/sample_submission.csv']:
        matches = glob(pattern, recursive=True)
        if matches:
            sample_sub_path = matches[0]
            break
    if sample_sub_path:
        cols = pd.read_csv(sample_sub_path).columns
        id_col, text_col = cols[0], cols[1]
    else:
        id_col, text_col = 'id', 'text'

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    df = pd.DataFrame({id_col: ids, text_col: predictions})
    df = df.sort_values(id_col).reset_index(drop=True)
    df[id_col] = range(len(df))
    df.to_csv(output, index=False)
    print(f"\nWrote {output}")
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
