"""
predict.py — the fast baseline: greedy CTC decode mapped through the lexicon,
straight to a submission. No beam search, no KenLM, no LLM.

    python predict.py --checkpoint_dir /path/to/b2t_v6_ckpt \
                      --output submission/submission_baseline.csv

Use it to prove a checkpoint bundle is loadable and wired correctly before
spending hours in `decode_llm.py`, and as the emergency submission if the full
pipeline cannot finish inside a session. Expect roughly 43-44% WER from it: the
collapse-and-look-up decoder has no language model at all, and every phoneme
chunk that is not a lexicon entry becomes `<unk>`. That gap between ~44% here and
single digits after beam + KenLM + LLM is the whole point of the decoding stage —
do not read this number as the model's quality.

With several folds, each test trial is decoded by the first model only; the
exact-CTC ensembling that makes multiple folds worth having lives in
`decode_llm.py`.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from src.dataset import CacheReader, build_cache, make_crossfit_split
from src.inference import (build_day_override, extract_emissions, find_checkpoint_dir,
                           group_checkpoints, load_am)
from src.metrics import greedy_collapse, load_lexicon, official_per, official_wer, phones_to_words
from src.utils import (find_file, get_session2idx, human_time, pick_cache_dir,
                       resolve_competition_path, resolve_kaggle_dataset, set_seed)


def parse_args():
    p = argparse.ArgumentParser(description='Greedy-lexicon baseline submission')
    p.add_argument('--data_dir', default=C.DEFAULT_DATA_DIR)
    p.add_argument('--checkpoint_dir', default=None)
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--lexicon', default=None)
    p.add_argument('--output', default=os.path.join(C.DEFAULT_SUBMISSION_DIR,
                                                    'submission_baseline.csv'))
    p.add_argument('--report_val', action='store_true',
                   help='also report PER / WER on each fold\'s held-out val trials')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(C.SEED)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    device = C.resolve_device()
    t0 = time.time()

    data_dir = resolve_competition_path(args.data_dir, C.COMPETITION_SLUG)
    session2idx = get_session2idx(data_dir)
    idx2session = {v: k for k, v in session2idx.items()}
    cache_dir = pick_cache_dir([args.cache_dir] if args.cache_dir else C.CACHE_CANDIDATES,
                               C.CACHE_NEED_GB)
    build_cache(data_dir, cache_dir)
    readers = {s: CacheReader(cache_dir, s) for s in ('train', 'val', 'test')}
    n_val, n_test = len(readers['val']), len(readers['test'])

    ckpt_dir = find_checkpoint_dir(args.checkpoint_dir, C.PRETRAINED_CKPT_DIR,
                                   C.DEFAULT_CHECKPOINT_DIR, '/kaggle/input')
    if ckpt_dir is None:
        raise FileNotFoundError('No fold*_seed*_*.pt found — run train.py first or pass '
                                '--checkpoint_dir.')
    print('checkpoints:', ckpt_dir)
    norm = torch.load(find_file(ckpt_dir, 'norm_stats.pt'), weights_only=False)
    day_override, _ = build_day_override(ckpt_dir, len(session2idx), idx2session)

    groups = group_checkpoints(ckpt_dir, C.CKPT_PREFERENCE)
    if not groups:
        raise FileNotFoundError(f'no checkpoint matching {C.CKPT_PREFERENCE} under {ckpt_dir}')
    tag = sorted(groups)[0]
    name = next(n for n in C.CKPT_PREFERENCE if n in groups[tag])
    model, ck = load_am(groups[tag][name], device)
    print(f'using {tag}/{name} (epoch {ck.get("epoch")}, step {ck.get("step")}, '
          f'trained on val labels {sorted(ck.get("train_val_labels", []))})')

    if args.lexicon:
        lexicon_path = args.lexicon
    else:
        lexicon_path = find_file(resolve_kaggle_dataset(C.LEXICON_DATASET_SLUG), 'lexicon.txt')
    _, pron2words = load_lexicon(lexicon_path)

    def decode(ems):
        out = []
        for em in ems:
            ids = greedy_collapse(torch.from_numpy(em.astype(np.float32))[:, None, :],
                                  [em.shape[0]])[0]
            out.append((ids, phones_to_words(ids, pron2words)))
        return out

    if args.report_val:
        try:
            split = __import__('json').load(open(find_file(ckpt_dir, 'split.json')))
        except Exception:
            split = {'labels': make_crossfit_split(readers['val'].meta, C.CROSSFIT_PATTERN, C.SEED)}
        labels = [split['labels'].get(m['key'], 'C') for m in readers['val'].meta]
        seen = set(ck.get('train_val_labels', []))
        hold = [j for j in range(n_val) if labels[j] not in seen]
        got = decode(extract_emissions(model, readers, 'val', hold, session2idx, norm,
                                       C.TRAIN_CFG['clip'], device, day_override))
        per = official_per([g[0] for g in got],
                           [[int(p) for p in readers['val'].meta[j]['phonemes']] for j in hold])[0]
        wer = official_wer([readers['val'].meta[j]['sentence'] for j in hold], [g[1] for g in got])[0]
        print(f'held-out ({len(hold)} trials, folds {sorted(set(labels[j] for j in hold))}): '
              f'PER {per * 100:.2f}% | greedy-lexicon WER {wer * 100:.2f}%')

    got = decode(extract_emissions(model, readers, 'test', list(range(n_test)), session2idx, norm,
                                   C.TRAIN_CFG['clip'], device, day_override))
    pd.DataFrame({'id': range(n_test), 'text': [g[1] for g in got]}).to_csv(args.output, index=False)
    print(f'submission -> {args.output} ({n_test} rows) | {human_time(time.time() - t0)}')


if __name__ == '__main__':
    main()
