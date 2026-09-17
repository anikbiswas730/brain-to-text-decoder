# Brain-to-Text v6 — Cross-Fit Conformer CTC + KenLM + LLM Rescoring

![B-T-S Pipeline Architecture](B-T-S%20pipeline.jpg)

Neural-speech decoding pipeline for the Kaggle
**[brain-to-text-25](https://www.kaggle.com/competitions/brain-to-text-25)** competition.
Two **Patch-Conv → BiGRU → Conformer** acoustic models are trained with **phoneme CTC +
inter-CTC + CR-CTC** on a **trial-level cross-fit** of the validation set, decoded with a
lexicon-constrained **KenLM 4-gram** beam search, expanded with **phonetic-neighbour**
candidates, rescored by **exact CTC sequence likelihood** plus a **QLoRA-adapted
Llama-3.1-8B**, and combined by **fluency-gated log-linear fusion** whose weights are tuned
on out-of-fold trials and verified on a fold nothing was selected on.

This is the modularized, repo-ready version of the two Kaggle notebooks in `notebooks/`.

> The architecture figure is the **B-T-S pipeline** diagram. Upload the paper's own
> `B-T-S pipeline.jpg` to the repository root (see `assets/README.md`) — it is not
> machine-generated here, to keep the figure authentic for the paper.

## Results

Acoustic models (EMA weights, each fold scored on the trials it never trained on):

| Fold | Init | Epochs | Hours | PER | greedy-lexicon WER |
| ---- | ---- | -----: | ----: | --: | -----------------: |
| A (seed 0) | warm start from v6.1 | 83 | 10.7 | **6.88%** | 43.87% |
| B (seed 1) | warm start from v6.1 | 82 | 10.7 | **6.89%** | 42.30% |

Greedy-lexicon WER is a no-LM diagnostic, not a pipeline number — it is what the training
worker can compute without loading KenLM. The decoding stage is where the WER actually is:

| System | tune (A∪B) | **verify (C)** | all OOF |
| ------ | ---------: | -------------: | ------: |
| greedy lexicon, 1 model | 42.62% | 44.63% | 43.02% |
| flashlight 1-best, KenLM lm_weight 3.0, 1 model | 6.07% | 6.66% | 6.18% |
| pool rescoring: exact-CTC ensemble + KenLM + #words | 5.68% | 5.42% | 5.63% |
| + phonetic-neighbour expansion | 4.80% | 4.67% | 4.78% |
| + task-adapted Llama-3.1-8B (no gate) | 4.64% | 4.14% | 4.54% |
| **+ fluency gate** | **4.47%** | **3.87%** | **4.35%** |
| FINAL (refit on A∪B∪C — in-sample on C) | 4.56% | 3.76% | 4.40% |
| *oracle ceiling: pool + expansion* | *3.26%* | *2.20%* | — |

**Read the verify (C) column.** Fold C was seen by neither acoustic model and selected on by
no sweep, so it is the only honest estimate. The tune column is what the sweep optimised and
is optimistic by construction: the pool sweep's winning margin had a bootstrap sd of 0.35 pp
with 21 rival configurations inside it. The FINAL row is refit on all three folds and is
in-sample on C — it is the system that produced the submission, not an estimate of anything.

Winning decode configuration (pinned as the defaults in `config.py`):

```
beam 400 · n-best 25 · lm_weights [0.5, 1.5, 2.5, 3.0, 4.0, 5.0] · expansion (top 3, ≤12 neighbours, ≤600 new)
stage-1 grid 'wide'  ->  ng 1.6, nw -8.0
stage-2 grid 'wide', smooth, tol_rel 0.005, llm_topk 12, fluency gate at the 10th percentile
   hi-fluency weights: ng 4.0, llm 1.4, nw  1.5
   lo-fluency weights: ng 1.0, llm 0.8, nw -6.5
rescorer: meta-llama/Llama-3.1-8B, QLoRA nf4, dev NLL 8.20 -> 3.09 in 37 min on one T4
```

## Why this pipeline looks the way it does

Each design choice below is a fix for a measured failure, not a preference.

**Trial-level cross-fit, not a held-out block.** Kaggle's test trials are interleaved with
val trials *inside the same recording blocks*. Holding out whole blocks would measure a
different, easier problem. So val is labelled `A/B/A/B/C` by trial inside every
(session, block); fold A's model trains on train + val-A, fold B's on train + val-B, and C
is left for verification.

**Exact CTC likelihood, not posterior averaging.** Averaging two models' frame posteriors
destroyed WER in v2 (66%) because independently trained CTC models spike at different
frames. Candidates are instead scored by `log P(phones | x)` from a CTC forward pass per
model and averaged — alignment-free, so ensembling is well-defined.

**Expansion before a bigger reranker.** ~95% of remaining errors are substitutions, and a
reranker can only pick from what is in the list. Single-word phonetic-neighbour substitution
of the top hypotheses bought 0.88 WER points (5.68 → 4.80 tune); the whole LLM stage on top
of it bought 0.33.

**A fluency gate, because a fluency prior is wrong for some trials.** Part of the stimulus
set is non-grammatical word sequences, where an LM that prefers fluent English actively
hurts. The gate splits utterances at a percentile of the LLM's per-token score and tunes a
separate weight vector for each side — the low-fluency side lands on a much smaller LLM
weight (0.8 vs 1.4) and a strongly negative word bonus.

**Sweeping grid *ranges*, not just points.** In an earlier run the stage-2 optimum landed on
the boundary of the search grid in all three weights at once. That is not an optimum, it is
the edge of the grid. Every sweep row now carries an `edge` flag, and the grid range itself
is a swept axis.

**A memory guard instead of hoping.** v6.0's workers were OOM-killed by the kernel after 27
minutes; its successor lost 11 workers the same way. The worker now watches its own RSS,
saves state and exits with code 75 before the kernel can act, and the launcher relaunches it
with `--resume` for free. Evaluation and resume timers live inside the resume state, because
restarts used to reset them and one fold got a single evaluation in five hours.

## Project structure

```
.
├── config.py                 # single source of truth: paths + every hyperparameter
├── train.py                  # cross-fit training launcher (supervises one worker per GPU)
├── predict.py                # fast baseline: greedy CTC -> submission
├── decode_llm.py             # full pipeline: beam + expansion + LLM fusion -> submission
├── evaluate.py               # ablation, per-day analysis, error breakdown, figures
├── B-T-S pipeline.jpg        # architecture figure (upload manually)
├── src/
│   ├── utils.py              # seeding, path resolvers, scratch-disk picker, atomic save, memory probes
│   ├── dataset.py            # HDF5 -> float16 cache, pread reader, cross-fit split, bucketing, GPU augs
│   ├── model.py              # B2TNetV4 (day adapter, patch Conv1d, BiGRU, Conformer), losses, EMA
│   ├── metrics.py            # official normalisation, WER/CER/PER, lexicon, greedy decode, bootstrap
│   ├── decoding.py           # flashlight beam pool, KenLM, expansion, exact-CTC features, LLM, tuner
│   ├── inference.py          # checkpoint screening, emissions, pools, feature cache, weight grids
│   └── train_worker.py       # one process = one GPU = one fold (crash-proof)
├── scripts/
│   └── build_kenlm.sh        # OPTIONAL one-time KenLM CLI build
├── checkpoints/              # training output bundle lands here (gitignored)
├── submission/               # submission.csv lands here (gitignored)
├── notebooks/                # the two original Kaggle notebooks, kept for reference
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```

## The acoustic model

`B2TNetV4` (see `src/model.py`), 30.15M parameters, ~7.1 GB peak on a T4:

1. **Low-rank day adapter** — rank-32 residual per session, plus a generic slot for sessions
   no fold trained on. Exact identity at init (`V` zero-init, scale as `1 + gain`), so weight
   decay pulls a session toward the shared solution rather than toward zero. Day dropout
   (0.15) keeps the generic slot usable.
2. **Gaussian smoothing** — fixed depthwise kernel, std 2 bins.
3. **In-model time masking** — applied after smoothing, before patching.
4. **Patch embedding as a strided Conv1d** (patch 14, stride 4) — identical to a linear layer
   on unfolded patches, but never materialises the `[B, T', 512·14]` tensor, which with its
   LayerNorm was ~40% of GPU memory.
5. **4 × residual BiGRU** (hidden 384, packed, `enforce_sorted=False`).
6. **3 × Conformer block** — RoPE attention, conv module with per-frame LayerNorm (the old
   `GroupNorm(1, C)` over `[C, T]` leaked padded-frame statistics into valid frames), macaron
   FFNs, stochastic depth.
7. **Phoneme CTC head** (41 classes) + two inter-CTC auxiliary heads.

Training: AdamW (lr 7e-4, day params at 2× and their own decay), **time-based** cosine so the
schedule lands on `lr_min` exactly at the session deadline whatever throughput turns out to
be, CR-CTC consistency between two augmented views, EMA weights for every checkpoint and
every evaluation.

## Data

Kaggle **brain-to-text-25** HDF5 files (512-channel neural features + phoneme IDs + sentence
transcripts, split `train` / `val` / `test`, one folder per recording day). On Kaggle it is
mounted at `DEFAULT_DATA_DIR` in `config.py`. Locally:

```bash
kaggle competitions download -c brain-to-text-25
python train.py --data_dir /path/to/hdf5_data_final
```

The first run writes a ~13 GB float16 cache to scratch disk (`/tmp/b2t_cache` by default).
Both workers read it with `os.pread` and share it through the OS page cache — **do not** put
it on a tmpfs mount; that is what OOM-killed v6.0.

## Setup

```bash
pip install -r requirements.txt
# OPTIONAL — only to rebuild a KenLM binary from scratch (the pipeline resolves a
# prebuilt 4-gram binary from a Kaggle dataset):
bash scripts/build_kenlm.sh
```

`meta-llama/Llama-3.1-8B` is gated: export `HF_TOKEN` with an accepted licence, or pass
`--llm_name Qwen/Qwen2.5-1.5B` for the ungated (and much cheaper) alternative.

## Usage

### 1. Train (≈11.5 h on 2× T4)

```bash
python train.py --data_dir /path/to/hdf5_data_final --checkpoint_dir checkpoints
# continue a previous run (same folds, same split, same normalisation):
python train.py --init_ckpt_dir /path/to/previous_bundle
# one GPU only:
python train.py --folds A
```

Writes the bundle `decode_llm.py` needs: `fold{A,B}_seed{0,1}_{best_wer,best_per,ema_final}.pt`,
`norm_stats.pt`, `split.json`, `sil_convention.json`, `trained_days.json`, `manifest.json`,
plus `tables/training_summary.csv` and `figures/training_curves.png`.

On Kaggle: run this as one session, *Save Version*, then create a dataset from the output
folder and pass it as `--checkpoint_dir` to the decoding run.

### 2. Baseline predictions (minutes, no LM)

```bash
python predict.py --checkpoint_dir checkpoints \
                  --output submission/submission_baseline.csv --report_val
```

### 3. Full pipeline — beam + expansion + LLM fusion (≈7 h with the sweep)

```bash
python decode_llm.py --checkpoint_dir /path/to/b2t_v6_ckpt \
                     --output submission/submission.csv
```

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `--no_sweep` | one static decode setting instead of the search | sweep on |
| `--no_llm` | acoustic + n-gram rescoring only | LLM on |
| `--no_finetune` | skip QLoRA, score with the base LLM | fine-tune on |
| `--beam` / `--nbest` | override generation width / depth | 400 / 75 |
| `--llm_name` / `--llm_batch` | rescoring LLM and its scoring batch | Llama-3.1-8B / 16 |
| `--n_proc` | beam-search processes (auto-capped by free RAM) | 3 |

A valid `submission.csv` exists from the first stage onward and is overwritten as better
artefacts appear (greedy → flashlight 1-best → full pipeline), so a stage that fails late
still leaves you something to submit. Tables land in `tables/`, figures in `figures/`.

### 4. Reproduce the report

```bash
python evaluate.py
# -> tables/{ablation,per_day_analysis,outlier_days_analysis,clean_days_analysis,error_breakdown}.csv
#    figures/{ablation_wer,wer_by_day}.png
```

## Cost notes for the sweep

Measured on the 2h44m decode run — worth knowing before you widen an axis:

| Axis | Cost |
| ---- | ---- |
| `beam` | **~1h23m per value** — full flashlight regeneration |
| `lm_weights` | free — the union is generated once, subsets are sliced |
| `nbest` | free — n-best lists are nested |
| `expand_top` / `max_nb` / `max_new` | cheap — only new strings hit the CTC scorer |
| stage-1 / stage-2 grid range | ~1 s per grid |
| `tol_rel`, `smooth_grid` | free — re-picks an already computed grid |
| `llm_topk` | free — the largest k is scored once, smaller k is its prefix |
| gate percentile | already a sweep |

Only `beam` is expensive. Everything else is close to free, which is why the sweep is on by
default and why the noise floor (`bootstrap`) is printed next to every margin.

## Manual uploads

Three artifacts are intentionally kept out of git:

- **Model weights** → `checkpoints/` (the whole bundle, not just the `.pt` files), or attach
  as a Kaggle dataset and point `--checkpoint_dir` at it.
- **Submission CSV** → `submission/`.
- **Architecture figure** → `B-T-S pipeline.jpg` at the repository root.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff).
