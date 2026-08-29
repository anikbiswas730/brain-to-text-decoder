# notebooks/

`iccit-brain-to-text-hybrid-qwen2-5-7b.ipynb` — the original single-file Kaggle
notebook, kept **for reference** (exploratory sweeps, per-day/error-breakdown
plots, and the LLM/lm_weight sweep history that identified the best combo:
`lm_weight = 4.0`, fusion `λ = 0.5`).

The repository root (`config.py`, `train.py`, `predict.py`, `decode_llm.py`,
`evaluate.py`, `src/`) is the modularized, runnable version of this notebook and
is what you should use for training and submission. The notebook is not
import-clean and is not meant to be run as-is outside Kaggle.
