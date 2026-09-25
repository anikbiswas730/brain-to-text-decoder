# Changelog

## v6.4 — 2026-09-25 · wide pinned search, CV pool sweep, selective GEC

Reference run: `notebooks/b2t-25-crossfit-llama3-1-8b-hyperparamter-tuning.ipynb`.
**Fold-C WER 3.87 % → 3.38 %**, all out-of-fold 4.35 % → 3.59 %. Same checkpoints as v6.2.

Training: **no change** — `src/{model,dataset,metrics,utils,train_worker,decoding}.py` are identical to v6.2
(the notebook module cells are byte-identical), and `train.py` is untouched.

Decoding hyperparameters (`config.LLM_CFG`):
- `gen_beam` 400 → **1200**; `gen_nbest` 75 → **100**
- `gen_lm_weights` [0.5, 2.0, 3.5, 5.0] → **[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]** (+ 3.0 decoded for the 1-best reference)
- expansion (`expand_top`, `expand_max_nb`, `expand_max_new`) (3, 12, 600) → **(8, 20, 600)**
- new: `nbest_sweep=[25, 50, 75, 100]`, `sweep_cv_folds=5`, `sweep_se_mult=1.0`, `sweep_budget_min=40`
- removed: `tol_rel`; `SWEEP_CFG`, `STAGE1_GRIDS`, `STAGE2_GRIDS` → single `STAGE1_GRID`, `STAGE2_GRID`
- new: `GEC_CFG`
- `PRETRAINED_CKPT_DIR` → the `b2t-v6-crossfit-2` dataset the reference run decoded

Pipeline (`decode_llm.py`, rewritten to follow notebook BLOCKs 4.1 – 4.6):
- beam search runs once and is cached to `gen_cache.pkl.gz` with a fingerprint (beam, n-best, λ,
  checkpoints, emission signature, task list); `--gen_cache` reuses it (6 h 49 m → seconds)
- BLOCK 4.2b **joint sweep** (`src/sweep.py`) replaces the staged coordinate-descent pool sweep: every
  (n-best × non-empty λ subset) is an index view of one maximal pool, scored by 5-fold session-grouped CV;
  the pinned pool is kept unless the CV winner beats it by ≥ 1 paired SE. Reference run: 508 combos, plateau,
  pinned pool kept.
- stage 2: one fixed grid + smoothed argmin per gate percentile (the range / `tol_rel` / `llm_topk` sweeps
  are gone); gate veto on C and refit on A∪B∪C unchanged; warns when a final weight sits on a grid edge
- BLOCK 4.5e **selective generative error correction** (`src/gec.py`, optional, `--no_gec`): LoRA corrector on
  low-fluency / near-tie trials, acoustically re-scored, adopted only if it wins on C by ≥ 1 paired-bootstrap
  SE. Reference run: not adopted (3.11 → 3.28 %). New `reject_oov` switch closes an OOV loophole.
- reporting moved into `src/analysis.py` and now runs inside `decode_llm.py` exactly as in the notebook
  (ablation, oracle-vs-k, fluency figure, per-day, search-vs-model error analysis) plus two new figures
  (`joint_sweep.png`, `error_analysis.png`); new tables `gate_search.csv`, `joint_sweep_choice.json`,
  `gec_summary.json`, `test_predictions_detail.csv`
- stages follow the notebook's `%%run_if --soft / --needs-ok / --optional` semantics

Repository:
- architecture figure `assets/bts_pipeline.{svg,png}` + generator `scripts/draw_pipeline.py`
  (the old README linked a `B-T-S pipeline.jpg` that was never in the repo)
- `results/` with the reference run's aggregate tables; `assets/figures/` with its figures
- `.gitignore` restored (it had been committed as a file named `download`)
- `evaluate.py` no longer overwrites the notebook-format outlier / clean day tables
- v6.2 decode notebook moved to `notebooks/archive/`

## v6.2 — 2026-09-18

Continue-training run (warm start from v6.1) with memory guard and resume; decode with Llama-3.1-8B QLoRA,
staged pool sweep and stage-2 sweep. Fold-C WER 3.87 %, all OOF 4.35 %.
