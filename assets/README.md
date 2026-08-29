# assets/

## `B-T-S pipeline.jpg` — the pipeline architecture figure

Following the original repository's convention, the paper's **Brain-to-Speech
(B-T-S) pipeline** architecture diagram lives at the repository root as
`B-T-S pipeline.jpg` and is embedded at the top of `README.md`.

**Upload your own `B-T-S pipeline.jpg` to the repository root manually** (same
as the model weights and the submission CSV). It is intentionally *not*
regenerated or fabricated here — for academic integrity the figure used in the
paper should be the authors' own, not a machine-generated substitute.

Recommended figure contents (matches the code in `src/`):

    Neural features (512-ch)
        └─ Day-specific linear adaptation  (per-session W, b + drift loss)
        └─ Gaussian smoothing (depthwise Conv1D)
        └─ CNN  →  BiLSTM  →  Patch embedding  →  RoPE Transformer (drop-path)
        └─ Phoneme CTC head (41-class: 39 phonemes + blank + word boundary)
                 │  emissions
                 ▼
        KenLM 4-gram beam search (lexicon-constrained, lm_weight = 4.0)
                 │  n-best
                 ▼
        Qwen2.5-7B verbatim-conditioned score fusion (λ = 0.5, gated)
                 │
                 ▼
             Decoded text  →  submission.csv
