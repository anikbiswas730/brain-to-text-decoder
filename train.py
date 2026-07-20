"""
Train the Hybrid LSTM-Transformer CTC model on the Brain-to-Text neural
recordings.

This is the full training pipeline (CTC loss + a "drift loss" that keeps
each day's adaptation layer close to its neighbours). In the original
Kaggle notebook this block was disabled for the final submission run
because a checkpoint had already been trained in an earlier kernel version
-- this script is that training code, cleaned up and made runnable
end-to-end and standalone.

Usage:
    python train.py
    python train.py --data_dir /path/to/hdf5_data_final --num_epochs 50
    python train.py --resume checkpoints/latest_model.pt   # continue training

Outputs (written to --checkpoint_dir, default "checkpoints/"):
    latest_model.pt  - overwritten every epoch
    best_model.pt    - overwritten whenever validation WER improves
    loss_curve.png   - training loss curve
"""

import argparse
import copy
import os

import matplotlib
matplotlib.use('Agg')  # headless-safe backend for saving plots to file
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CONFIG
from src.dataset import BrainToTextDataset, collate_fn, load_split
from src.metrics import validate_model
from src.model import HybridLSTMTransformerCTC
from src.utils import get_session2idx, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train the Brain-to-Text CTC model")
    p.add_argument('--data_dir', default=CONFIG['data_dir'],
                    help='Root dir containing <session>/data_{train,val}.hdf5 files')
    p.add_argument('--checkpoint_dir', default=CONFIG['checkpoint_dir'])
    p.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    p.add_argument('--num_epochs', type=int, default=CONFIG['num_epochs'])
    p.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'])
    p.add_argument('--patience', type=int, default=CONFIG['early_stop_patience'])
    p.add_argument('--resume', default=None,
                    help='Optional path to a checkpoint (model + optimizer state) to resume from')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def train_model(train_loader, val_loader, char2idx, config, n_days, checkpoint_dir,
                 num_epochs, learning_rate, patience, resume_path=None):
    device = config['device']
    model = HybridLSTMTransformerCTC.from_config(config, n_days=n_days, vocab_size=len(char2idx)).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Resumed successfully. Starting training...")

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=num_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_losses = []
    idx2char = {v: k for k, v in char2idx.items()}
    best_wer = float('inf')
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            neural = batch['neural'].to(device)
            target = batch['target'].to(device)
            lengths = batch['lengths']
            target_lengths = batch['target_lengths']
            day_idx = batch['day_idx'].to(device)

            log_probs, output_lengths = model(neural, lengths, day_idx)
            ctc_loss = criterion(log_probs, target, output_lengths, target_lengths)

            drift_loss = 0.0
            if config['drift_lambda'] > 0 and n_days > 1:
                for d in range(1, n_days):
                    w_diff = model.day_weights[d] - model.day_weights[d - 1]
                    b_diff = model.day_biases[d] - model.day_biases[d - 1]
                    drift_loss += (torch.sum(w_diff ** 2) + torch.sum(b_diff ** 2))
                drift_loss = drift_loss / (n_days - 1)

            loss = ctc_loss + (config['drift_lambda'] * drift_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        val_wer = validate_model(model, val_loader, idx2char, device)

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, WER={val_wer:.2f}%")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, os.path.join(checkpoint_dir, 'latest_model.pt'))

        if val_wer < best_wer:
            best_wer = val_wer
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'char2idx': char2idx,
                'config': config,
                'wer': best_wer,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs (best WER: {best_wer:.2f}%)")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'))
    plt.close()

    return model, best_wer


def main():
    args = parse_args()
    set_seed(args.seed)

    config = copy.deepcopy(CONFIG)
    config['data_dir'] = args.data_dir
    config['batch_size'] = args.batch_size

    print("=" * 80)
    print("BRAIN-TO-TEXT: TRAINING")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"PyTorch version: {torch.__version__}")

    session2idx = get_session2idx(args.data_dir)
    n_days = len(session2idx)
    print(f"Discovered {n_days} unique sessions.")

    train_data = load_split(args.data_dir, 'train')
    val_data = load_split(args.data_dir, 'val')

    train_dataset = BrainToTextDataset(train_data, session2idx, augment=config['use_augmentation'])
    val_dataset = BrainToTextDataset(val_data, session2idx, char2idx=train_dataset.char2idx, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    model, best_wer = train_model(
        train_loader, val_loader, train_dataset.char2idx, config, n_days,
        checkpoint_dir=args.checkpoint_dir, num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, patience=args.patience, resume_path=args.resume,
    )

    print(f"\n\u2713 Training complete. Best validation WER: {best_wer:.2f}%")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
