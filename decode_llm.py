#!/usr/bin/env python
"""
decode_llm.py — the accurate submission pipeline.

    CTC emissions -> KenLM beam search (lm_weight=4.0, beam=250, nbest=30)
    -> calibrate confidence gate on validation
    -> margin-adaptive, verbatim-conditioned Qwen2.5-7B SCORE FUSION (λ=0.5)
    -> submission.csv

Validation result of this exact pipeline: WER 7.62% / CER 5.31% / PER 12.12%.

The gate thresholds are calibrated on the VALIDATION set (which has ground
truth) and reused on the test set, since the decoder's score scale is stable
between the two.

Example
-------
    python decode_llm.py --checkpoint checkpoints/best_model.pt \
        --output submission/submission.csv
"""

import os
import argparse
from glob import glob

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (CONFIG, DECODING, resolve_device, PRETRAINED_CKPT_DATASET,
                    LEXICON_DATASET_SLUG, KENLM_DATASET_SLUG)
from src.utils import (set_seed, get_session2idx, resolve_kaggle_dataset, find_file)
from src.dataset import load_split, BrainToTextDataset, collate_fn
from src.model import build_model
from src.metrics import corpus_wer, corpus_cer, PHONEME_VOCAB
from src.decoding import build_decoder, LLMRescorer
from src.inference import (extract_emissions_from_loader, block_load_test_data,
                           extract_emissions)


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------
def resolve_assets(args):
    if args.lexicon and args.tokens and args.kenlm_binary:
        return args.lexicon, args.tokens, args.kenlm_binary
    lexicon_ds = resolve_kaggle_dataset(LEXICON_DATASET_SLUG)
    kenlm_ds = resolve_kaggle_dataset(KENLM_DATASET_SLUG)
    lexicon_path = args.lexicon or find_file(lexicon_ds, 'lexicon.txt')
    tokens_path = args.tokens or find_file(lexicon_ds, 'tokens.txt')
    kenlm_binary = args.kenlm_binary or find_file(kenlm_ds, '*.bin')
    return lexicon_path, tokens_path, kenlm_binary


def check_tokens_match_vocab(tokens_path):
    with open(tokens_path) as f:
        file_tokens = [line.rstrip('\n') for line in f]
    mismatches = 0
    for i in range(max(len(file_tokens), len(PHONEME_VOCAB))):
        a = file_tokens[i] if i < len(file_tokens) else '<missing>'
        b = PHONEME_VOCAB[i] if i < len(PHONEME_VOCAB) else '<missing>'
        if a.strip() != b.strip():
            mismatches += 1
            print(f"  [{i:2d}] tokens.txt={a!r:12s}  PHONEME_VOCAB={b!r:12s}  <-- MISMATCH")
    print("tokens.txt matches PHONEME_VOCAB." if mismatches == 0
          else f"WARNING: {mismatches} token mismatch(es) - WER will be wrong until fixed.")


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


def beam_search(decoder, emissions, lengths, batch=None, desc="Beam search"):
    batch = DECODING['decode_batch'] if batch is None else batch
    results = []
    for i in tqdm(range(0, len(emissions), batch), desc=desc):
        results.extend(decoder(emissions[i:i + batch], lengths[i:i + batch]))
    return results


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = dict(CONFIG)
    cfg.update({'data_dir': args.data_dir, 'checkpoint_dir': args.checkpoint_dir,
                'checkpoint': args.checkpoint, 'use_amp': not args.no_amp})
    cfg['device'] = resolve_device(cfg['device'])
    set_seed(cfg['seed'])
    device = cfg['device']

    session2idx = get_session2idx(cfg['data_dir'])
    n_days = len(session2idx)

    model, feat_mean, feat_std = load_norm_and_model(cfg, n_days)

    lexicon_path, tokens_path, kenlm_binary = resolve_assets(args)
    print(f"Lexicon: {lexicon_path}\nTokens: {tokens_path}\nKenLM: {kenlm_binary}")
    check_tokens_match_vocab(tokens_path)

    decoder = build_decoder(lexicon_path, tokens_path, kenlm_binary,
                            lm_weight=args.lm_weight, beam_size=args.beam_width,
                            nbest=args.nbest)
    print(f"Decoder ready (beam={args.beam_width}, nbest={args.nbest}, lm_weight={args.lm_weight}).")

    rescorer = LLMRescorer.load(model_name=args.llm_name, fusion_weight=args.llm_fusion_weight)

    # ---- 1. Calibrate the gate on validation (has ground truth) -----------
    print("\nCalibrating confidence gate on the validation set ...")
    val_data = load_split(cfg['data_dir'], 'val')
    val_ds = BrainToTextDataset(val_data, session2idx, feat_mean, feat_std,
                                augment=False, clip=cfg['feature_clip'])
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg['num_workers'])
    val_em, val_len, val_refs, _ = extract_emissions_from_loader(
        model, val_loader, device, use_amp=cfg['use_amp'])
    val_results = beam_search(decoder, val_em, val_len, batch=args.decode_batch,
                              desc="Val beam search")
    rescorer.calibrate(val_results)

    if args.report_val:
        preds, n_low, n_conf = rescorer.rescore_all(val_results, desc="Val LLM fusion")
        pairs = [(r, h) for r, h in zip(val_refs, preds) if r and r.strip()]
        rc = [r for r, _ in pairs]
        hc = [h if h.strip() else "<empty>" for _, h in pairs]
        print(f"\nValidation WER (full pipeline): {corpus_wer(rc, hc) * 100:.2f}%")
        print(f"Validation CER (full pipeline): {corpus_cer(rc, hc) * 100:.2f}%")
        print(f"Gated {n_low + n_conf}/{len(val_results)} "
              f"({n_low} low-score, {n_conf} confident).")

    # ---- 2. Test set: beam search -> gated LLM fusion ---------------------
    print("\nLoading test set ...")
    samples = block_load_test_data(cfg['data_dir'], session2idx, feat_mean, feat_std,
                                   clip=cfg['feature_clip'])
    test_em, test_len, test_ids = extract_emissions(model, samples, device,
                                                    use_amp=cfg['use_amp'])
    test_results = beam_search(decoder, test_em, test_len, batch=args.decode_batch,
                               desc="Test beam search")

    print(f"Score-fusion LLM rescoring with fusion_weight={rescorer.fusion_weight} ...")
    predictions, n_low, n_conf = rescorer.rescore_all(test_results, desc="Gated LLM rescoring")
    n_gated = n_low + n_conf
    print(f"Skipped LLM on {n_gated}/{len(test_results)} "
          f"({100 * n_gated / max(len(test_results), 1):.1f}%): "
          f"{n_low} low-score, {n_conf} high-confidence.")

    write_submission(cfg['data_dir'], test_ids, predictions, args.output)


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
        print(f"Using columns '{id_col}', '{text_col}' from {sample_sub_path}")
    else:
        id_col, text_col = 'id', 'text'
        print("No sample_submission.csv found - defaulting to 'id','text'.")

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    df = pd.DataFrame({id_col: ids, text_col: predictions})
    df = df.sort_values(id_col).reset_index(drop=True)
    df[id_col] = range(len(df))
    df.to_csv(output, index=False)
    print(f"\nWrote {output}")
    print(df.head(10).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="Full beam + KenLM + Qwen fusion submission.")
    p.add_argument('--data_dir', default=CONFIG['data_dir'])
    p.add_argument('--checkpoint_dir', default=CONFIG['checkpoint_dir'])
    p.add_argument('--checkpoint', default=None, help='explicit checkpoint path')
    p.add_argument('--output', default='submission/submission.csv')

    # decoder assets (default: resolve from the Kaggle datasets)
    p.add_argument('--lexicon', default=None)
    p.add_argument('--tokens', default=None)
    p.add_argument('--kenlm_binary', default=None)

    # BEST hyperparameters as defaults
    p.add_argument('--lm_weight', type=float, default=DECODING['lm_weight'])
    p.add_argument('--beam_width', type=int, default=DECODING['beam_width'])
    p.add_argument('--nbest', type=int, default=DECODING['nbest'])
    p.add_argument('--decode_batch', type=int, default=DECODING['decode_batch'])
    p.add_argument('--llm_name', default=DECODING['llm_name'])
    p.add_argument('--llm_fusion_weight', type=float, default=DECODING['llm_fusion_weight'])

    p.add_argument('--report_val', action='store_true',
                   help='also print full-pipeline WER/CER on validation before submitting')
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    main()
