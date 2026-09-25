# notebooks/

The Kaggle notebooks this repository was modularized from, kept verbatim with their outputs — every number
in the top-level README and in `results/` comes from their run logs.

| Notebook | `MODE` | What it did |
| -------- | ------ | ----------- |
| `b2t-v6-crossfit-ctc-verified.ipynb` | `train` | v6.2 continue-training run: warm start from the v6.1 checkpoints, ~10.7 h per fold on 2 × T4, final PER 6.88 % (A) / 6.89 % (B). **Produces the checkpoints every decode uses.** |
| `b2t-25-crossfit-llama3-1-8b-hyperparamter-tuning.ipynb` | `llm` | **v6.4 reference decode** (current): beam 1200, 7 KenLM weights, expansion (8, 20, 600), joint n-best × λ-subset CV sweep, Llama-3.1-8B QLoRA, fluency gate, optional GEC. Fold-C WER **3.38 %**, 8 h 53 m. |
| `archive/b2t-25-crossfit-llama3-1-8b-sweep_best.ipynb` | `llm` | v6.2 decode (superseded): beam 400, staged pool sweep, stage-2 range / `tol_rel` / `llm_topk` sweep. Fold-C WER 3.87 %. |

The three notebooks carry byte-identical `b2t_core.py`, `train_worker.py` and `b2t_decode.py` module cells;
they differ only in the config cell and in the BLOCK 4.x decoding cells.

Mapping to this repo:

| Notebook cell | Repo file |
| ------------- | --------- |
| BLOCK 1 config | `config.py` |
| `b2t_core.py` | `src/{dataset,model,metrics,utils}.py` |
| `train_worker.py` | `src/train_worker.py` |
| `b2t_decode.py` | `src/decoding.py` (unchanged since v6.2) |
| BLOCK 2 data + BLOCK 3.1 / 3.2 | `train.py` |
| BLOCK 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 | `decode_llm.py` + `src/inference.py` |
| BLOCK 4.2b joint sweep | `src/sweep.py` |
| BLOCK 4.5b / 4.5c / 4.5d analysis | `src/analysis.py` (called by `decode_llm.py`); `evaluate.py` adds CER, S/D/I and bootstrap sd |
| BLOCK 4.5e selective GEC | `src/gec.py` |

The notebooks run top-to-bottom on Kaggle with a single `MODE` switch; the repo replaces that switch with
separate entry points and CLI flags. The algorithms are unchanged.
