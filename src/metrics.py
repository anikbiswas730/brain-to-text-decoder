"""
src/metrics.py — decoding helpers and error-rate metrics.

Includes:
    * CTC greedy phoneme decoding (blank/duplicate collapse)
    * phoneme <-> word helpers driven by lexicon.txt
    * official brain-to-text '25 text normalization + corpus WER/CER (jiwer)
    * phoneme error rate (PER) vs. ground-truth phoneme IDs
    * validate_model(): per-epoch WER (lexicon) + PER used during training
"""

import re

import editdistance
import jiwer
import torch

from config import PHONEME_VOCAB
from src.utils import amp_autocast


# ---------------------------------------------------------------------------
# CTC greedy decode + phoneme/word helpers
# ---------------------------------------------------------------------------
def greedy_decode_phonemes(log_probs, output_lengths):
    """CTC greedy decode -> list of phoneme-id sequences (blanks/dupes collapsed)."""
    _, max_indices = log_probs.max(dim=-1)  # [T, B]
    decoded = []
    for b in range(max_indices.size(1)):
        seq = max_indices[:output_lengths[b], b].cpu().numpy()
        out, prev = [], None
        for token in seq:
            if token != 0 and token != prev:
                out.append(int(token))
            prev = token
        decoded.append(out)
    return decoded


def phoneme_ids_to_str(phoneme_ids):
    """[7, 11, 32] -> 'B EH T' (space-joined phoneme symbols)."""
    return ' '.join(PHONEME_VOCAB[p] for p in phoneme_ids)


def build_lexicon_reverse(lexicon_path):
    """Parse lexicon.txt into a phoneme-tuple -> word map (first occurrence wins)."""
    reverse = {}
    with open(lexicon_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                word, phon_str = parts[0], parts[1]
                phonemes = [p for p in phon_str.rstrip(' |').split(' ') if p]
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                word, phonemes = parts[0], parts[1:]
            key = tuple(phonemes)
            if key not in reverse:
                reverse[key] = word
    return reverse


def phoneme_ids_to_words(phoneme_ids, lexicon_reverse):
    """Fast, LM-free phoneme->word decode (used only for train-time WER)."""
    sil_id = len(PHONEME_VOCAB) - 1
    words, current = [], []
    for pid in phoneme_ids:
        if pid == sil_id:
            if current:
                key = tuple(PHONEME_VOCAB[p] for p in current)
                words.append(lexicon_reverse.get(key, '<unk>'))
                current = []
        else:
            current.append(pid)
    if current:
        key = tuple(PHONEME_VOCAB[p] for p in current)
        words.append(lexicon_reverse.get(key, '<unk>'))
    return ' '.join(words)


# ---------------------------------------------------------------------------
# Official text normalization + corpus WER/CER
# ---------------------------------------------------------------------------
def normalize_text(sentence):
    """Official brain-to-text '25 normalization: keep letters/spaces/hyphens/
    apostrophes, lowercase, collapse stray hyphens & whitespace."""
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([w for w in sentence.split() if w != ''])
    return sentence


def corpus_wer(refs, hyps):
    return jiwer.wer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps])


def corpus_cer(refs, hyps):
    return jiwer.cer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps])


def utterance_wer(ref, hyp):
    return jiwer.wer(normalize_text(ref), normalize_text(hyp))


def utterance_cer(ref, hyp):
    return jiwer.cer(normalize_text(ref), normalize_text(hyp))


# ---------------------------------------------------------------------------
# Per-epoch validation (WER via lexicon + PER via raw phoneme IDs)
# ---------------------------------------------------------------------------
def validate_model(model, val_loader, device, lexicon_reverse, use_amp=True):
    """Compute (WER%, PER%) for one pass over the validation loader."""
    model.eval()
    total_word_edits, total_words = 0, 0
    total_phon_edits, total_phons = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            neural = batch['neural'].to(device)
            lengths = batch['lengths']
            day_idx = batch['day_idx'].to(device)
            sentences = batch['sentences']
            targets = batch['target']
            target_lengths = batch['target_lengths']

            with amp_autocast(enabled=use_amp):
                log_probs, output_lengths = model(neural, lengths, day_idx)
            decoded_phonemes = greedy_decode_phonemes(log_probs, output_lengths)

            for b, phon_ids in enumerate(decoded_phonemes):
                ref_phon_ids = targets[b, :target_lengths[b]].tolist()
                if len(ref_phon_ids) > 0:
                    total_phon_edits += editdistance.eval(phon_ids, ref_phon_ids)
                    total_phons += len(ref_phon_ids)

                ref = sentences[b]
                if not ref or not ref.strip():
                    continue
                hyp = phoneme_ids_to_words(phon_ids, lexicon_reverse)
                ref_words, hyp_words = ref.split(), hyp.split()
                total_word_edits += editdistance.eval(hyp_words, ref_words)
                total_words += max(len(ref_words), 1)

    wer = 100.0 * total_word_edits / max(total_words, 1)
    per = 100.0 * total_phon_edits / max(total_phons, 1)
    return wer, per
