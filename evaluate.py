#!/usr/bin/env python
"""
evaluate.py — reproduce the paper's validation numbers on the val split.

Reports:
    * WER / CER  — beam + KenLM only (no LLM)
    * WER / CER  — beam + KenLM + margin-adaptive Qwen fusion (submission pipeline)
    * PER        — CTC-greedy vs. ground-truth phoneme IDs (LLM-independent)
    * per-recording-day WER/CER/PER
Saves val_wer.json + figures/metrics_summary.png under --checkpoint_dir.

Example
-------
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import os
import json
import argparse

import numpy as np
import editdistance
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (CONFIG, DECODING, resolve_device, PRETRAINED_CKPT_DATASET,
                    LEXICON_DATASET_SLUG, KENLM_DATASET_SLUG)
from src.utils import (set_seed, get_session2idx, resolve_kaggle_dataset, find_file,
                       amp_autocast)
from src.dataset import load_split, BrainToTextDataset, collate_fn
from src.model import build_model
from src.metrics import (greedy_decode_phonemes, corpus_wer, corpus_cer, PHONEME_VOCAB)
from src.decoding import build_decoder, LLMRescorer
from src.inference import extract_emissions_from_loader


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
    idx2session = {v: k for k, v in session2idx.items()}

    model, feat_mean, feat_std = load_norm_and_model(cfg, n_days)

    # Assets + decoder + LLM.
    lexicon_ds = resolve_kaggle_dataset(LEXICON_DATASET_SLUG)
    kenlm_ds = resolve_kaggle_dataset(KENLM_DATASET_SLUG)
    lexicon_path = args.lexicon or find_file(lexicon_ds, 'lexicon.txt')
    tokens_path = args.tokens or find_file(lexicon_ds, 'tokens.txt')
    kenlm_binary = args.kenlm_binary or find_file(kenlm_ds, '*.bin')
    decoder = build_decoder(lexicon_path, tokens_path, kenlm_binary,
                            lm_weight=args.lm_weight, beam_size=args.beam_width,
                            nbest=args.nbest)
    rescorer = LLMRescorer.load(model_name=args.llm_name, fusion_weight=args.llm_fusion_weight)

    val_data = load_split(cfg['data_dir'], 'val')
    val_ds = BrainToTextDataset(val_data, session2idx, feat_mean, feat_std,
                                augment=False, clip=cfg['feature_clip'])
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg['num_workers'])

    # ---- Emissions + beam search (top-1, no LLM) --------------------------
    val_em, val_len, val_refs, val_days = extract_emissions_from_loader(
        model, val_loader, device, use_amp=cfg['use_amp'])

    print("Beam search (n-best) ...")
    val_results = []
    B = args.decode_batch
    for i in tqdm(range(0, len(val_em), B), desc="Val beam"):
        val_results.extend(decoder(val_em[i:i + B], val_len[i:i + B]))

    top1_hyps = [" ".join(r[0].words) if r and r[0].words else "" for r in val_results]
    pairs = [(r, h, d) for r, h, d in zip(val_refs, top1_hyps, val_days) if r and r.strip()]
    refs_clean = [r for r, _, _ in pairs]
    hyps_clean = [h if h.strip() else "<empty>" for _, h, _ in pairs]
    clean_days = [d for _, _, d in pairs]
    overall_wer = corpus_wer(refs_clean, hyps_clean)
    overall_cer = corpus_cer(refs_clean, hyps_clean)
    print(f"\nBeam+KenLM (no LLM):  WER={overall_wer * 100:.2f}%  CER={overall_cer * 100:.2f}%")

    # ---- Full pipeline (+LLM fusion) --------------------------------------
    rescorer.calibrate(val_results)
    preds, n_low, n_conf = rescorer.rescore_all(val_results, desc="LLM fusion")
    fpairs = [(r, h) for r, h in zip(val_refs, preds) if r and r.strip()]
    fr = [r for r, _ in fpairs]
    fh = [h if h.strip() else "<empty>" for _, h in fpairs]
    full_wer = corpus_wer(fr, fh)
    full_cer = corpus_cer(fr, fh)
    print(f"Beam+KenLM+LLM:       WER={full_wer * 100:.2f}%  CER={full_cer * 100:.2f}%")
    print(f"Gated {n_low + n_conf}/{len(val_results)} ({n_low} low-score, {n_conf} confident).")

    # ---- PER (CTC-greedy vs. ground-truth phoneme IDs) --------------------
    per_day_edits, per_day_counts = {}, {}
    total_edits, total_count = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Greedy PER"):
            neural = batch['neural'].to(device)
            lengths = batch['lengths']
            day_idx = batch['day_idx']
            targets = batch['target']
            target_lengths = batch['target_lengths']
            with amp_autocast(enabled=cfg['use_amp']):
                log_probs, out_len = model(neural, lengths, day_idx.to(device))
            decoded = greedy_decode_phonemes(log_probs, out_len)
            for b, phon_ids in enumerate(decoded):
                ref = targets[b, :target_lengths[b]].tolist()
                if len(ref) == 0:
                    continue
                e = editdistance.eval(phon_ids, ref)
                d = int(day_idx[b])
                per_day_edits[d] = per_day_edits.get(d, 0) + e
                per_day_counts[d] = per_day_counts.get(d, 0) + len(ref)
                total_edits += e
                total_count += len(ref)
    overall_per = total_edits / max(total_count, 1)
    print(f"PER (CTC-greedy):     {overall_per * 100:.2f}%")

    # ---- Save JSON + summary figure ---------------------------------------
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    figures_dir = os.path.join(cfg['checkpoint_dir'], 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    with open(os.path.join(cfg['checkpoint_dir'], 'val_wer.json'), 'w') as f:
        json.dump({
            'wer_beam_ngram_only': overall_wer, 'cer_beam_ngram_only': overall_cer,
            'wer_full_pipeline': full_wer, 'cer_full_pipeline': full_cer,
            'per_greedy': overall_per, 'lm_weight': args.lm_weight,
            'llm_fusion_weight_base': rescorer.fusion_weight,
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ['WER\n(Beam+KenLM)', 'WER\n(+LLM)', 'CER\n(Beam+KenLM)', 'CER\n(+LLM)', 'PER\n(CTC)']
    values = [overall_wer * 100, full_wer * 100, overall_cer * 100, full_cer * 100, overall_per * 100]
    bars = ax.bar(labels, values, color=['tab:purple', 'tab:pink', 'tab:blue', 'tab:cyan', 'tab:green'])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{v:.1f}%',
                ha='center', va='bottom')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Validation Error Rates: WER vs CER vs PER')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved val_wer.json and figures/metrics_summary.png under {cfg['checkpoint_dir']}.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WER/CER/PER on validation.")
    p.add_argument('--data_dir', default=CONFIG['data_dir'])
    p.add_argument('--checkpoint_dir', default=CONFIG['checkpoint_dir'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--lexicon', default=None)
    p.add_argument('--tokens', default=None)
    p.add_argument('--kenlm_binary', default=None)
    p.add_argument('--lm_weight', type=float, default=DECODING['lm_weight'])
    p.add_argument('--beam_width', type=int, default=DECODING['beam_width'])
    p.add_argument('--nbest', type=int, default=DECODING['nbest'])
    p.add_argument('--decode_batch', type=int, default=DECODING['decode_batch'])
    p.add_argument('--llm_name', default=DECODING['llm_name'])
    p.add_argument('--llm_fusion_weight', type=float, default=DECODING['llm_fusion_weight'])
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    main()
