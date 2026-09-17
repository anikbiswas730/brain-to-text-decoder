"""
src/metrics.py — the OFFICIAL text normalisation and scoring, plus the lexicon
and greedy-decode helpers that turn phoneme IDs into words.

Everything here mirrors the competition's own evaluation code
(`nejm-brain-to-text/model_training/evaluate_model_helpers.remove_punctuation`
and the aggregate WER in `evaluate_model.py`) so that an internal number is
directly comparable with the leaderboard:

    WER = (S + D + I) / N
        = sum_i editdistance(ref_i.split(), hyp_i.split()) / sum_i len(ref_i.split())

i.e. a CORPUS-level ratio of summed edits to summed reference words — not the
mean of per-utterance WERs, which is a different (and usually more optimistic)
number.
"""

import re

import editdistance
import numpy as np

from config import PHONEME_VOCAB, N_CLASSES, BLANK_ID, SIL_ID  # noqa: F401

PHONE2ID = {p.strip(): i for i, p in enumerate(PHONEME_VOCAB)}   # '|' -> 40


# ---------------------------------------------------------------------------
# Official normalisation + metrics
# ---------------------------------------------------------------------------
def remove_punctuation(sentence):
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])
    return sentence


def official_wer(refs, hyps):
    """Corpus-level WER exactly as evaluate_model.py. -> (wer, edits, n_ref_words)."""
    tot_ed, tot_n = 0, 0
    for r, h in zip(refs, hyps):
        rw = remove_punctuation(r or '').split()
        hw = remove_punctuation(h or '').split()
        tot_ed += editdistance.eval(rw, hw)
        tot_n += len(rw)
    return (tot_ed / max(tot_n, 1)), tot_ed, tot_n


def official_cer(refs, hyps):
    """Corpus-level character error rate on the same normalised text."""
    tot_ed, tot_n = 0, 0
    for r, h in zip(refs, hyps):
        rc = remove_punctuation(r or '')
        hc = remove_punctuation(h or '')
        tot_ed += editdistance.eval(list(rc), list(hc))
        tot_n += len(rc)
    return (tot_ed / max(tot_n, 1)), tot_ed, tot_n


def official_per(pred_seqs, true_seqs):
    """Aggregate phoneme error rate like rnn_trainer.validation:
    sum edit distance / sum true length."""
    ed = sum(editdistance.eval(list(p), list(t)) for p, t in zip(pred_seqs, true_seqs))
    n = sum(len(t) for t in true_seqs)
    return ed / max(n, 1), ed, n


def utt_word_errors(ref, hyp):
    """(edits, n_ref_words) for one utterance — the building block of the
    utterance bootstrap used to put a noise floor under sweep margins."""
    rw = remove_punctuation(ref or '').split()
    return editdistance.eval(rw, remove_punctuation(hyp or '').split()), len(rw)


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
def load_lexicon(path):
    """-> (word2prons {word: [tuple(phones), ...]}, pron2words {tuple: [words]}).

    Accepts both the tab-separated form (`word\\tP1 P2 |`) and the whitespace
    form used by flashlight lexicons. Entries containing a phone outside
    PHONEME_VOCAB are dropped rather than silently mapped.
    """
    word2prons, pron2words = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                w, rest = line.split('\t', 1)
                ph = [p for p in rest.replace('|', ' ').split() if p]
            else:
                parts = line.split()
                w, ph = parts[0], [p for p in parts[1:] if p != '|']
            if not ph:
                continue
            ph = tuple(ph)
            if any(p not in PHONE2ID for p in ph):
                continue
            lst = word2prons.setdefault(w, [])
            if ph not in lst:
                lst.append(ph)
            pw = pron2words.setdefault(ph, [])
            if w not in pw:
                pw.append(w)
    return word2prons, pron2words


# ---------------------------------------------------------------------------
# Greedy CTC decode
# ---------------------------------------------------------------------------
def greedy_collapse(lp_tb, out_len):
    """lp [T,B,C] -> list of collapsed phoneme-id lists (blank removed, repeats merged)."""
    am = lp_tb.argmax(-1).transpose(0, 1).cpu().numpy()
    res = []
    for b in range(am.shape[0]):
        seq = am[b, :int(out_len[b])]
        out, prev = [], -1
        for t in seq:
            t = int(t)
            if t != BLANK_ID and t != prev:
                out.append(t)
            prev = t
        res.append(out)
    return res


def phones_to_words(ids, pron2words, word_rank=None):
    """Map a collapsed phoneme-id sequence to words by splitting on the word
    boundary token and looking each chunk up in the lexicon. Unknown chunks
    become '<unk>'. This is a diagnostic decoder (no LM, no beam) — it is what
    the training worker reports as `greedy_wer`, which is why that number is
    ~44% while the full beam+LM+LLM pipeline is single digits."""
    words, cur = [], []

    def flush():
        if cur:
            ws = pron2words.get(tuple(PHONEME_VOCAB[p].strip() for p in cur))
            if not ws:
                words.append('<unk>')
            else:
                words.append(min(ws, key=lambda w: word_rank.get(w, 1e9)) if word_rank else ws[0])

    for p in ids:
        if p == SIL_ID:
            flush()
            cur = []
        else:
            cur.append(p)
    flush()
    return ' '.join(words)


def detect_sil_convention(train_meta, lex_path, n_max=2000):
    """Does the ground-truth phoneme sequence end with ' | ' after the last word?

    Getting this wrong silently costs ~1 phoneme per utterance in every CTC
    likelihood computed during rescoring, so it is measured from the data rather
    than assumed, and the answer is stored next to the checkpoints.
    """
    w2p, _ = load_lexicon(lex_path)
    n = end_sil = sil_eq_words = sil_eq_words_m1 = exact_t = exact_nt = 0
    for m in train_meta[:n_max]:
        words = remove_punctuation(m['sentence']).split()
        ph = [int(p) for p in m['phonemes']]
        if not words or not ph or any(w not in w2p for w in words):
            continue
        n += 1
        end_sil += int(ph[-1] == SIL_ID)
        ns = sum(1 for p in ph if p == SIL_ID)
        sil_eq_words += int(ns == len(words))
        sil_eq_words_m1 += int(ns == len(words) - 1)
        seq = []
        for w in words:
            seq += [PHONE2ID[p] for p in w2p[w][0]]
            seq.append(SIL_ID)
        exact_t += int(seq == ph)
        exact_nt += int(seq[:-1] == ph)
    trailing = end_sil >= 0.5 * max(n, 1)
    return {'n_checked': n, 'frac_end_with_sil': end_sil / max(n, 1),
            'frac_nsil_eq_nwords': sil_eq_words / max(n, 1),
            'frac_nsil_eq_nwords_minus1': sil_eq_words_m1 / max(n, 1),
            'exact_match_trailing': exact_t / max(n, 1),
            'exact_match_no_trailing': exact_nt / max(n, 1),
            'sil_trailing': bool(trailing)}


def bootstrap_wer_sd(refs, hyps, n_boot=400, seed=1337):
    """Utterance-bootstrap standard deviation of the corpus WER.

    A sweep that evaluates hundreds of configurations WILL find a winner; this
    is the noise floor its margin has to clear before the win means anything.
    """
    if not n_boot or len(refs) < 10:
        return float('nan')
    en = [utt_word_errors(r, h) for r, h in zip(refs, hyps)]
    e = np.array([x[0] for x in en], dtype=np.float64)
    n = np.array([x[1] for x in en], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(refs), (int(n_boot), len(refs)))
    return float(np.std(e[idx].sum(1) / np.maximum(n[idx].sum(1), 1.0)))
