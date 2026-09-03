"""
src/decoding.py — the KenLM beam-search decoder plus the Qwen2.5-7B
score-fusion rescorer.

Two public pieces:
    build_decoder(...)        -> a torchaudio/flashlight ctc_decoder
    class LLMRescorer         -> verbatim-conditioned LLM scoring + margin-
                                 adaptive SCORE FUSION (never a pure override)
                                 + confidence gating.

The rescorer keeps its calibrated gate thresholds as instance state, so no
module-level globals leak between the val and test passes.
"""

import sys
import subprocess
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DECODING, VERBATIM_PREFIX


# ---------------------------------------------------------------------------
# flashlight-text availability (torchaudio's ctc_decoder wraps it)
# ---------------------------------------------------------------------------
def ensure_flashlight():
    try:
        import flashlight.lib.text.decoder  # noqa: F401
        return True
    except Exception:
        pass
    print("flashlight not importable - installing flashlight-text ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "flashlight-text"], check=False)
    importlib.invalidate_caches()
    try:
        import flashlight.lib.text.decoder  # noqa: F401
        return True
    except Exception:
        return False


def build_decoder(lexicon_path, tokens_path, kenlm_binary_path,
                  lm_weight=None, beam_size=None, nbest=None, word_score=None):
    """Build a lexicon-constrained CTC beam-search decoder over KenLM."""
    if not ensure_flashlight():
        raise RuntimeError(
            "flashlight-text could not be imported after install. On Kaggle this "
            "usually means Internet is OFF, or a stale import cache (Restart & Run "
            "All after the install succeeds). Offline: attach the flashlight_text "
            "wheel and `pip install --no-index --find-links <dir> flashlight-text`."
        )
    from torchaudio.models.decoder import ctc_decoder

    lm_weight = DECODING['lm_weight'] if lm_weight is None else lm_weight
    beam_size = DECODING['beam_width'] if beam_size is None else beam_size
    nbest = DECODING['nbest'] if nbest is None else nbest
    word_score = DECODING['word_score'] if word_score is None else word_score

    return ctc_decoder(
        lexicon=lexicon_path,
        tokens=tokens_path,
        lm=kenlm_binary_path,
        nbest=nbest,
        beam_size=beam_size,
        lm_weight=lm_weight,
        word_score=word_score,
        blank_token='BLANK',
        sil_token='|',
    )


# ---------------------------------------------------------------------------
# LLM score-fusion rescorer
# ---------------------------------------------------------------------------
def _standardize(vals):
    """Zero-mean/unit-std within one n-best list (scale-free fusion weight)."""
    v = np.asarray(vals, dtype=np.float64)
    sd = v.std()
    if sd < 1e-8:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


class LLMRescorer:
    """Verbatim-conditioned LLM scoring + margin-adaptive score fusion + gating.

    Parameters
    ----------
    fusion_weight : base fusion λ (best = 0.5). Applied per-utterance, scaled
        down as the decoder's own top1-vs-runnerup margin grows.
    gate_percentile / gate_margin_percentile : calibrate() sets the low-score
        (incoherent) and high-confidence gate thresholds from these.
    """

    def __init__(self, llm_model, tokenizer, fusion_weight=None,
                 gate_percentile=None, gate_margin_percentile=None,
                 prefix=VERBATIM_PREFIX):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.fusion_weight = DECODING['llm_fusion_weight'] if fusion_weight is None else fusion_weight
        self.gate_percentile = (DECODING['llm_gate_percentile']
                                if gate_percentile is None else gate_percentile)
        self.gate_margin_percentile = (DECODING['llm_gate_margin_percentile']
                                       if gate_margin_percentile is None else gate_margin_percentile)
        self.prefix = prefix
        self.gate_threshold = None
        self.gate_margin_threshold = None

    # ---- factory ----------------------------------------------------------
    @classmethod
    def load(cls, model_name=None, fusion_weight=None, gate_percentile=None,
             gate_margin_percentile=None, prefix=VERBATIM_PREFIX):
        """Load a 4-bit quantized causal LM and wrap it in a rescorer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = DECODING['llm_name'] if model_name is None else model_name
        print(f"Loading rescoring LLM ({model_name}) in 4-bit ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto")
        mdl.eval()
        return cls(mdl, tok, fusion_weight=fusion_weight,
                   gate_percentile=gate_percentile,
                   gate_margin_percentile=gate_margin_percentile, prefix=prefix)

    # ---- scoring ----------------------------------------------------------
    def compute_scores(self, sentences, batch_size=16, prefix=None):
        """Length-normalized conditional log-likelihood P(candidate | prefix).
        The prefix is fed but excluded from the score (its labels are masked)."""
        prefix = self.prefix if prefix is None else prefix
        tok = self.tokenizer
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        n_prefix = len(prefix_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        scores = []
        for i in range(0, len(sentences), batch_size):
            chunk = [s if s.strip() else "<empty>" for s in sentences[i:i + batch_size]]
            seqs = []
            for s in chunk:
                cand_ids = tok(s, add_special_tokens=False)["input_ids"]
                if len(cand_ids) == 0:
                    cand_ids = [tok.eos_token_id]
                seqs.append(prefix_ids + cand_ids)

            maxlen = max(len(x) for x in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            score_mask = torch.zeros((len(seqs), maxlen), dtype=torch.float)
            for r, ids in enumerate(seqs):
                L = len(ids)
                input_ids[r, :L] = torch.tensor(ids, dtype=torch.long)
                attn[r, :L] = 1
                score_mask[r, n_prefix:L] = 1.0

            input_ids = input_ids.to(self.llm_model.device)
            attn = attn.to(self.llm_model.device)
            score_mask = score_mask.to(self.llm_model.device)

            with torch.no_grad():
                logits = self.llm_model(input_ids=input_ids, attention_mask=attn).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_smask = score_mask[:, 1:]
            nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(shift_labels.shape)
            token_counts = shift_smask.sum(dim=1).clamp(min=1)
            per_seq_nll = (nll * shift_smask).sum(dim=1) / token_counts
            scores.extend((-per_seq_nll).tolist())
        return scores

    # ---- decoder-score helpers -------------------------------------------
    @staticmethod
    def normalized_ngram_score(hyp):
        return hyp.score / max(len(hyp.words), 1)

    def top1_margin(self, nbest_result):
        if len(nbest_result) < 2:
            return float('inf')
        return (self.normalized_ngram_score(nbest_result[0])
                - self.normalized_ngram_score(nbest_result[1]))

    # ---- gating -----------------------------------------------------------
    def calibrate(self, decoder_results, percentile=None, margin_percentile=None):
        """Set both gate thresholds from this set's own score distribution."""
        percentile = self.gate_percentile if percentile is None else percentile
        margin_percentile = self.gate_margin_percentile if margin_percentile is None else margin_percentile

        top1_scores = [self.normalized_ngram_score(r[0]) for r in decoder_results if r]
        self.gate_threshold = float(np.percentile(top1_scores, percentile))

        margins = [self.top1_margin(r) for r in decoder_results
                   if r and np.isfinite(self.top1_margin(r))]
        self.gate_margin_threshold = float(np.percentile(margins, margin_percentile)) \
            if margins else float('inf')

        print(f"LLM gate calibrated on {len(top1_scores)} utterances:\n"
              f"  low-score cutoff ({percentile}th pct): {self.gate_threshold:.4f}\n"
              f"  margin cutoff ({margin_percentile}th pct): {self.gate_margin_threshold:.4f}")
        return self.gate_threshold, self.gate_margin_threshold

    def is_gated(self, nbest_result):
        if not nbest_result:
            return True
        low_score = (self.gate_threshold is not None and
                     self.normalized_ngram_score(nbest_result[0]) < self.gate_threshold)
        confident = (self.gate_margin_threshold is not None and
                     self.top1_margin(nbest_result) > self.gate_margin_threshold)
        return bool(low_score or confident)

    def gate_reason(self, nbest_result):
        """Return 'low_score', 'confident' or None (for reporting skip counts)."""
        if not nbest_result:
            return None
        if self.normalized_ngram_score(nbest_result[0]) < self.gate_threshold:
            return 'low_score'
        if self.top1_margin(nbest_result) > self.gate_margin_threshold:
            return 'confident'
        return None

    # ---- fusion -----------------------------------------------------------
    def fuse_pick(self, nbest_result, llm_scores, lam):
        """Index maximizing standardize(decoder) + lam * standardize(llm).
        lam == 0 (or mismatched scores) reproduces the decoder top-1 exactly."""
        dec = [self.normalized_ngram_score(h) for h in nbest_result]
        if lam == 0 or not llm_scores or len(llm_scores) != len(dec):
            return int(np.argmax(dec))
        combined = _standardize(dec) + lam * _standardize(llm_scores)
        return int(np.argmax(combined))

    def adaptive_lambda(self, nbest_result, base_lambda=None):
        """Per-utterance fusion weight, scaled down as decoder confidence grows."""
        base_lambda = self.fusion_weight if base_lambda is None else base_lambda
        if base_lambda == 0:
            return 0.0
        if self.gate_margin_threshold in (None, 0) or not np.isfinite(self.gate_margin_threshold):
            return base_lambda
        margin = self.top1_margin(nbest_result)
        if not np.isfinite(margin):
            return 0.0
        scale = 1.0 - min(margin / self.gate_margin_threshold, 1.0)
        return base_lambda * scale

    def gated_rescore(self, nbest_result, lam=None, prefix=None):
        """Final transcript for one utterance: decoder top-1 if gated, else fused."""
        if not nbest_result:
            return ""
        hyps = [" ".join(h.words) if h.words else "" for h in nbest_result]
        if not any(hyps):
            return ""
        if self.is_gated(nbest_result):
            return hyps[0]
        lam = self.adaptive_lambda(nbest_result) if lam is None else lam
        if lam == 0:
            return hyps[0]
        llm_scores = self.compute_scores(hyps, batch_size=len(hyps), prefix=prefix)
        return hyps[self.fuse_pick(nbest_result, llm_scores, lam)]

    # ---- batch driver -----------------------------------------------------
    def rescore_all(self, decoder_results, desc="Gated LLM rescoring"):
        """Run gated_rescore over a list of n-best results; also return skip counts."""
        preds = []
        n_low, n_conf = 0, 0
        for res in tqdm(decoder_results, desc=desc):
            reason = self.gate_reason(res) if res else None
            if reason == 'low_score':
                n_low += 1
            elif reason == 'confident':
                n_conf += 1
            preds.append(self.gated_rescore(res))
        return preds, n_low, n_conf
