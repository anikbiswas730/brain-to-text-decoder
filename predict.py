"""
Generate a submission.csv from a trained checkpoint using plain greedy CTC
decoding (no n-gram LM, no LLM rescoring). This is the fast baseline path;
see decode_llm.py for the higher-accuracy pipeline that was used for the
final (~15% WER) submission.

Usage:
    python predict.py --checkpoint checkpoints/best_model.pt
    python predict.py --checkpoint checkpoints/best_model.pt --output submission.csv
"""

import argparse

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import CONFIG
from src.dataset import load_test_samples
from src.metrics import ctc_greedy_decode
from src.model import HybridLSTMTransformerCTC
from src.utils import get_session2idx


def parse_args():
    p = argparse.ArgumentParser(description="Generate a baseline (greedy-decode) submission")
    p.add_argument('--data_dir', default=CONFIG['data_dir'])
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--output', default='submission.csv')
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


def generate_predictions(model, test_samples, idx2char, device, batch_size=32):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), batch_size), desc="Generating predictions"):
            batch = test_samples[i:i + batch_size]
            features = [s['features'] for s in batch]
            lengths = torch.LongTensor([len(f) for f in features])
            day_idx = torch.LongTensor([s['day_idx'] for s in batch]).to(device)

            features_padded = pad_sequence(features, batch_first=True).to(device)
            sorted_lengths, sorted_idx = lengths.sort(descending=True)

            features_sorted = features_padded[sorted_idx]
            day_idx_sorted = day_idx[sorted_idx]

            log_probs, output_lengths = model(features_sorted, sorted_lengths, day_idx_sorted)
            _, max_indices = log_probs.max(dim=-1)

            batch_preds = []
            for b in range(max_indices.size(1)):
                seq = max_indices[:output_lengths[b], b].cpu().numpy()
                batch_preds.append(ctc_greedy_decode(seq, idx2char))

            unsorted_preds = [''] * len(batch_preds)
            for j, pred in zip(sorted_idx.tolist(), batch_preds):
                unsorted_preds[j] = pred
            predictions.extend(unsorted_preds)
    return predictions


def main():
    args = parse_args()
    device = CONFIG['device']

    print("=" * 80)
    print("BRAIN-TO-TEXT: BASELINE PREDICTION (greedy CTC decode)")
    print("=" * 80)

    session2idx = get_session2idx(args.data_dir)
    n_days = len(session2idx)

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    char2idx = checkpoint['char2idx']
    idx2char = {v: k for k, v in char2idx.items()}
    config = checkpoint.get('config', CONFIG)

    model = HybridLSTMTransformerCTC.from_config(config, n_days=n_days, vocab_size=len(char2idx)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_samples = load_test_samples(args.data_dir, session2idx)
    print(f"Loaded {len(test_samples)} test samples")

    predictions = generate_predictions(model, test_samples, idx2char, device, batch_size=args.batch_size)

    df = pd.DataFrame({'id': [s['id'] for s in test_samples], 'text': predictions})
    df = df.sort_values('id').reset_index(drop=True)
    df['id'] = range(len(df))
    df.to_csv(args.output, index=False)

    print(f"\n\u2713 Wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
