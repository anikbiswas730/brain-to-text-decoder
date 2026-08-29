#!/usr/bin/env python
"""
train.py — train HybridLSTMTransformerCTC with phoneme CTC + day-drift loss.

Saves (under --checkpoint_dir):
    norm_stats.pt        per-channel TRAIN normalization (reused at inference)
    latest_model.pt      every epoch
    best_model.pt        whenever validation WER improves
    swa_model.pt         SWA average of the top-k checkpoints (usually best)
    training_history.json + figures/training_{loss,wer,per}.png

Example
-------
    python train.py --data_dir /path/to/hdf5_data_final \
        --checkpoint_dir checkpoints --num_epochs 82
    python train.py --resume checkpoints/latest_model.pt
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG, resolve_device, LEXICON_DATASET_SLUG
from src.utils import (set_seed, get_session2idx, resolve_kaggle_dataset,
                       amp_autocast, make_grad_scaler)
from src.dataset import (load_split, compute_channel_stats, BrainToTextDataset, collate_fn)
from src.model import build_model
from src.metrics import build_lexicon_reverse, validate_model


def average_state_dicts(state_dicts):
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state_dicts[0].items()}
    for sd in state_dicts:
        for k, v in sd.items():
            avg[k] += v.float()
    for k in avg:
        avg[k] /= len(state_dicts)
    return avg


def train(cfg, lexicon_reverse):
    device = cfg['device']
    session2idx = get_session2idx(cfg['data_dir'])
    n_days = len(session2idx)
    print(f"Found {n_days} recording sessions/days.")

    train_data = load_split(cfg['data_dir'], 'train')
    val_data = load_split(cfg['data_dir'], 'val')

    print("Computing global per-channel normalization stats from TRAIN only ...")
    feat_mean, feat_std = compute_channel_stats(train_data)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    torch.save({'mean': feat_mean, 'std': feat_std},
               os.path.join(cfg['checkpoint_dir'], 'norm_stats.pt'))

    train_ds = BrainToTextDataset(train_data, session2idx, feat_mean, feat_std,
                                  augment=cfg['use_augmentation'], clip=cfg['feature_clip'])
    val_ds = BrainToTextDataset(val_data, session2idx, feat_mean, feat_std,
                                augment=False, clip=cfg['feature_clip'])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg['num_workers'], pin_memory=True)

    model = build_model(cfg, n_days).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                            weight_decay=cfg['weight_decay'])
    scaler = make_grad_scaler(enabled=cfg['use_amp'])

    start_epoch = 0
    if cfg.get('resume_from_checkpoint'):
        print(f"Resuming from {cfg['resume_from_checkpoint']} ...")
        ckpt = torch.load(cfg['resume_from_checkpoint'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', -1) + 1
        print(f"Resumed at epoch {start_epoch}.")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['learning_rate'], epochs=cfg['num_epochs'],
        steps_per_epoch=len(train_loader), pct_start=0.1,
        last_epoch=(start_epoch * len(train_loader) - 1) if start_epoch else -1,
    )

    epoch_losses, epoch_wers, epoch_pers = [], [], []
    best_wer, no_improve = float('inf'), 0
    top_checkpoints = []

    for epoch in range(start_epoch, cfg['num_epochs']):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg["num_epochs"]}'):
            neural = batch['neural'].to(device)
            target = batch['target'].to(device)
            lengths = batch['lengths']
            target_lengths = batch['target_lengths']
            day_idx = batch['day_idx'].to(device)

            optimizer.zero_grad()
            with amp_autocast(enabled=cfg['use_amp']):
                log_probs, output_lengths = model(neural, lengths, day_idx)
                ctc_loss = criterion(log_probs.float(), target, output_lengths, target_lengths)

                drift_loss = 0.0
                if cfg['drift_lambda'] > 0 and n_days > 1:
                    for d in range(1, n_days):
                        w_diff = model.day_weights[d] - model.day_weights[d - 1]
                        b_diff = model.day_biases[d] - model.day_biases[d - 1]
                        drift_loss += (torch.sum(w_diff ** 2) + torch.sum(b_diff ** 2))
                    drift_loss = drift_loss / (n_days - 1)

                loss = ctc_loss + cfg['drift_lambda'] * drift_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        avg_loss = running / len(train_loader)
        val_wer, val_per = validate_model(model, val_loader, device, lexicon_reverse,
                                          use_amp=cfg['use_amp'])
        epoch_losses.append(avg_loss)
        epoch_wers.append(val_wer)
        epoch_pers.append(val_per)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, WER={val_wer:.2f}%, PER={val_per:.2f}%")

        ckpt = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch, 'config': cfg}
        torch.save(ckpt, os.path.join(cfg['checkpoint_dir'], 'latest_model.pt'))

        if val_wer < best_wer:
            best_wer, no_improve = val_wer, 0
            torch.save({**ckpt, 'wer': best_wer, 'per': val_per},
                       os.path.join(cfg['checkpoint_dir'], 'best_model.pt'))
        else:
            no_improve += 1

        top_checkpoints.append((val_wer, {k: v.cpu().clone() for k, v in model.state_dict().items()}))
        top_checkpoints = sorted(top_checkpoints, key=lambda x: x[0])[:cfg['swa_topk']]

        if no_improve >= cfg['patience']:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    if len(top_checkpoints) > 1:
        swa_state = average_state_dicts([sd for _, sd in top_checkpoints])
        torch.save({'model_state_dict': swa_state, 'config': cfg},
                   os.path.join(cfg['checkpoint_dir'], 'swa_model.pt'))
        print(f"Saved SWA average of top {len(top_checkpoints)} checkpoints -> swa_model.pt")

    # History + curves.
    with open(os.path.join(cfg['checkpoint_dir'], 'training_history.json'), 'w') as f:
        json.dump({'loss': epoch_losses, 'wer': epoch_wers, 'per': epoch_pers}, f, indent=2)

    figures_dir = os.path.join(cfg['checkpoint_dir'], 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    epochs_range = range(1, len(epoch_losses) + 1)
    for series, ylabel, title, color, fname in [
        (epoch_losses, 'CTC Loss', 'Training Loss over Epochs', 'tab:blue', 'training_loss.png'),
        (epoch_wers, 'Word Error Rate (%)', 'Validation WER over Epochs', 'tab:red', 'training_wer.png'),
        (epoch_pers, 'Phoneme Error Rate (%)', 'Validation PER over Epochs', 'tab:green', 'training_per.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, series, color=color, marker='o', markersize=3)
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Training finished. Best validation WER: {best_wer:.2f}%")
    return best_wer


def parse_args():
    p = argparse.ArgumentParser(description="Train the brain-to-text CTC model.")
    p.add_argument('--data_dir', default=CONFIG['data_dir'])
    p.add_argument('--checkpoint_dir', default=CONFIG['checkpoint_dir'])
    p.add_argument('--num_epochs', type=int, default=CONFIG['num_epochs'])
    p.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    p.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'])
    p.add_argument('--resume', default=None, help='checkpoint to resume from')
    p.add_argument('--lexicon', default=None,
                   help='path to lexicon.txt (defaults to the Kaggle lexicon dataset)')
    p.add_argument('--no_amp', action='store_true', help='disable mixed precision')
    p.add_argument('--seed', type=int, default=CONFIG['seed'])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    cfg.update({
        'data_dir': args.data_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'resume_from_checkpoint': args.resume,
        'use_amp': not args.no_amp,
        'seed': args.seed,
    })
    cfg['device'] = resolve_device(cfg['device'])
    set_seed(cfg['seed'])
    print(f"Device: {cfg['device']} | epochs: {cfg['num_epochs']} | batch: {cfg['batch_size']}")

    lexicon_path = args.lexicon
    if lexicon_path is None:
        lexicon_ds = resolve_kaggle_dataset(LEXICON_DATASET_SLUG)
        lexicon_path = os.path.join(lexicon_ds, 'lexicon.txt')
    print(f"Loading phoneme lexicon for end-to-end WER: {lexicon_path}")
    lexicon_reverse = build_lexicon_reverse(lexicon_path)
    print(f"Lexicon reverse-map: {len(lexicon_reverse)} unique phoneme sequences.")

    train(cfg, lexicon_reverse)


if __name__ == '__main__':
    main()
