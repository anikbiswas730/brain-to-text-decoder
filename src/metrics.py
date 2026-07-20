"""
Evaluation helpers: greedy CTC decoding and Word Error Rate (WER).
"""

import jiwer
import torch


def ctc_greedy_decode(indices, idx2char):
    """Collapses a raw CTC path (repeated tokens + blanks) into a string.
    `indices` is a 1D iterable of predicted class ids for one sequence."""
    decoded, prev = [], None
    for token in indices:
        if token != 0 and token != prev:
            decoded.append(idx2char.get(token, ''))
        prev = token
    return ''.join(decoded)


def validate_model(model, val_loader, idx2char, device):
    """Runs the model over a validation DataLoader and returns the corpus
    Word Error Rate (as a percentage)."""
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in val_loader:
            neural = batch['neural'].to(device)
            lengths = batch['lengths']
            day_idx = batch['day_idx'].to(device)

            log_probs, output_lengths = model(neural, lengths, day_idx)
            _, max_indices = log_probs.max(dim=-1)  # [T, B]

            for b in range(max_indices.size(1)):
                seq = max_indices[:output_lengths[b], b].cpu().numpy()
                predictions.append(ctc_greedy_decode(seq, idx2char))
                ground_truths.append(batch['sentences'][b])

    # jiwer chokes on empty ground-truth/prediction strings, so filter/guard them.
    valid_preds, valid_truths = [], []
    for p, t in zip(predictions, ground_truths):
        if t and t.strip():
            valid_preds.append(p if p.strip() else "<empty>")
            valid_truths.append(t.strip())

    if not valid_truths:
        return 100.0

    return jiwer.wer(valid_truths, valid_preds) * 100.0
