# notebooks/

The two Kaggle notebooks this repository was modularized from. They are kept verbatim,
outputs included, because the numbers in the top-level README come from their run logs.

| Notebook | `MODE` | What it did |
| -------- | ------ | ----------- |
| `b2t-v6-crossfit-ctc-verified.ipynb` | `train` | v6.2 continue-training run: warm start from the v6.1 checkpoints, ~10.7 h per fold on 2x T4, final PER 6.88% (A) / 6.89% (B) |
| `b2t-25-crossfit-llama3-1-8b-sweep_best.ipynb` | `llm` | decoding + sweep run: Llama-3.1-8B QLoRA rescorer, pool and stage-2 sweeps, verify-fold WER 3.87% |

Both carry the same `b2t_core.py` and `train_worker.py` module cells; they differ in the
decoding module (the LLM notebook adds QLoRA 4-bit fine-tuning and the `tol_rel` tie-break)
and in the sweep configuration.

Mapping to this repo:

| Notebook cell | Repo file |
| ------------- | --------- |
| BLOCK 1 config | `config.py` |
| `b2t_core.py` | `src/{dataset,model,metrics,utils}.py` |
| `train_worker.py` | `src/train_worker.py` |
| BLOCK 2 data + BLOCK 3.1/3.2 | `train.py` |
| `b2t_decode.py` | `src/decoding.py` |
| BLOCK 4.1-4.6 | `decode_llm.py` + `src/inference.py` |
| BLOCK 4.5b/4.5c/4.5d analysis | `evaluate.py` |

The notebooks run top-to-bottom on Kaggle with a single `MODE` switch; the repo replaces
that switch with separate entry points and CLI flags. The algorithms are unchanged.
