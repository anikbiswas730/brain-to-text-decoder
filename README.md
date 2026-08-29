# Brain-to-Text: Hybrid LSTM–Transformer CTC Decoder

![B-T-S Pipeline Architecture](B-T-S%20pipeline.jpg)

Neural-speech decoding pipeline for the Kaggle **[brain-to-text-25](https://www.kaggle.com/competitions/brain-to-text-25)**
competition: a hybrid **CNN + BiLSTM + patch-Transformer** acoustic model trained
with **phoneme CTC**, decoded with a lexicon-constrained **KenLM 4-gram** beam
search, and rescored with a verbatim-conditioned **Qwen2.5-7B** language model via
**margin-adaptive score fusion**. This is the modularized, repo-ready version of the
original single Kaggle notebook (`notebooks/iccit-brain-to-text-hybrid-qwen2-5-7b.ipynb`).

> The architecture figure above is the **B-T-S pipeline** diagram. Upload the paper's
> own `B-T-S pipeline.jpg` to the repository root (see `assets/README.md`) — it is not
> machine-generated here, to keep the figure authentic for the paper.

## Results (validation set)

| Stage | WER | CER | PER |
| ----- | --: | --: | --: |
| CTC-greedy (acoustic only) | — | — | **12.12%** |
| Beam + KenLM (no LLM) | 7.71% | 5.43% | — |
| **Beam + KenLM + Qwen2.5-7B fusion** | **7.62%** | **5.31%** | — |

Best decoding configuration (pinned as defaults in `config.py`):
`lm_weight = 4.0`, fusion `λ = 0.5`, beam width `250`, n-best `30`, gate
percentiles `1 / 75`, rescoring LLM `Qwen/Qwen2.5-7B` (4-bit nf4).
*(The KenLM `lm_weight` sweep was still improving at the grid edge, so `4.0`
is the best **confirmed** point, not a proven optimum.)*

## Project structure

```
.
├── config.py                 # Single source of truth: paths + BEST hyperparameters
├── train.py                  # Training pipeline (phoneme CTC + drift loss + SWA)
├── predict.py                # Fast baseline: greedy CTC decode -> submission
├── decode_llm.py             # Full pipeline: KenLM beam + Qwen fusion -> submission
├── evaluate.py               # Validation WER / CER / PER (paper numbers)
├── B-T-S pipeline.jpg        # Architecture figure (upload manually)
├── src/
│   ├── utils.py              # seeding, drop_path, session map, path resolvers, AMP shim
│   ├── dataset.py            # HDF5 loading, augmentation, Dataset, collate, channel stats
│   ├── model.py              # RoPE attention, Transformer blocks, HybridLSTMTransformerCTC
│   ├── metrics.py            # greedy decode, phoneme<->word, WER/CER/PER, normalization
│   ├── decoding.py           # KenLM beam decoder + Qwen LLMRescorer (fusion + gating)
│   └── inference.py          # emission extraction (val loader / test HDF5)
├── scripts/
│   └── build_kenlm.sh        # OPTIONAL one-time KenLM CLI build
├── checkpoints/              # model weights land here (upload manually; gitignored)
├── submission/               # submission.csv lands here (upload manually; gitignored)
├── notebooks/
│   └── iccit-brain-to-text-hybrid-qwen2-5-7b.ipynb   # original reference notebook
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```

## The model

`HybridLSTMTransformerCTC` (see `src/model.py`) processes each trial as:

1. **Day-specific linear adaptation** — a learned per-session linear layer
   (`day_weights` / `day_biases`) absorbs electrode drift between recording days,
   regularized during training by a **drift loss** keeping adjacent days close.
2. **Gaussian smoothing** — a fixed depthwise Conv1D smooths the 512-channel features.
3. **CNN → BiLSTM** — local temporal features, then a 2-layer bidirectional LSTM.
4. **Patch embedding + Transformer** — LSTM outputs are grouped into patches and
   passed through a small Transformer encoder using **RoPE** and **stochastic depth**.
5. **Phoneme CTC head** — 41-class per-timestep phoneme log-probabilities (39 CMU phonemes + CTC blank + word boundary).

Best model/training hyperparameters (`config.py → CONFIG`): `d_model=384, n_heads=6,
n_layers=4, d_ff=1536, patch_size=3, lstm_hidden=256, lstm_layers=2; dropout=0.4,
head_dim=256, attn_dropout=0.5, drop_path_rate=0.2, smooth_std=2.0, drift_lambda=0.01;
AdamW lr=5e-4, wd=1e-4, batch=16, epochs=82, OneCycleLR, grad_clip=5.0, AMP, top-3 SWA,
early-stop patience=10`.

## The decoding pipeline

`emissions → KenLM beam search → confidence gate → verbatim-conditioned Qwen2.5-7B
score fusion → text`. The LLM never overrides the acoustic + n-gram evidence; it is
**fused** as `standardize(decoder_score) + λ · standardize(llm_score)` and only re-ranks
*within* the n-best. A per-utterance **margin-adaptive** λ applies the most LLM weight
where the decoder is least confident. Gating skips clearly incoherent trials (1st
percentile) and already-confident trials (75th percentile). See `src/decoding.py`.

## Data

Kaggle **brain-to-text-25** HDF5 files (512-channel neural features + phoneme IDs +
sentence transcripts, split `train` / `val` / `test`, one folder per recording day).
On Kaggle it is mounted at the `DEFAULT_DATA_DIR` in `config.py`. Locally:

```bash
kaggle competitions download -c brain-to-text-25
python train.py --data_dir /path/to/hdf5_data_final
```

## Setup

```bash
pip install -r requirements.txt
# OPTIONAL — only to rebuild a KenLM binary from scratch (the pipeline ships with
# a prebuilt 4-gram binary resolved from a Kaggle dataset):
bash scripts/build_kenlm.sh
```

## Usage

### 1. Train

```bash
python train.py \
    --data_dir /path/to/hdf5_data_final \
    --checkpoint_dir checkpoints \
    --num_epochs 82
# resume:
python train.py --resume checkpoints/latest_model.pt
```

Writes `checkpoints/{norm_stats,latest_model,best_model,swa_model}.pt`,
`training_history.json`, and `figures/training_{loss,wer,per}.png`.

### 2. Baseline predictions (fast, no LM)

```bash
python predict.py --checkpoint checkpoints/best_model.pt \
    --output submission/submission_baseline.csv
```

### 3. Full pipeline — KenLM + LLM rescoring (final submission)

```bash
python decode_llm.py \
    --checkpoint checkpoints/best_model.pt \
    --output submission/submission.csv \
    --report_val          # optional: also print full-pipeline val WER/CER
```

Key flags (all default to the best values in `config.py`):

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `--lm_weight` | KenLM weight in beam search | `4.0` |
| `--beam_width` / `--nbest` | beam size / n-best depth | `250` / `30` |
| `--llm_fusion_weight` | base fusion λ (margin-adaptive at inference) | `0.5` |
| `--llm_name` | HuggingFace causal LM for rescoring | `Qwen/Qwen2.5-7B` |
| `--lexicon` / `--tokens` / `--kenlm_binary` | override auto-resolved assets | Kaggle datasets |

### 4. Reproduce the paper numbers

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
# -> prints WER/CER (beam-only and +LLM) + PER, saves val_wer.json + metrics_summary.png
```

## Manual uploads

Three artifacts are intentionally kept out of git and uploaded by hand:

- **Model weights** → `checkpoints/` (`norm_stats.pt` + `best_model.pt`/`swa_model.pt`), or attach as a Kaggle dataset and set `PRETRAINED_CKPT_DATASET` in `config.py`.
- **Submission CSV** → `submission/`.
- **Architecture figure** → `B-T-S pipeline.jpg` at the repository root.

## License

MIT — see [`LICENSE`](LICENSE). Update the copyright holder/year as needed.

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff).
