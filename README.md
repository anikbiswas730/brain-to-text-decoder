# Brain-to-Text: Hybrid LSTM-Transformer CTC Decoder

![B-T-S Pipeline Architecture](B-T-S%20pipeline.jpg)

Neural-speech decoding pipeline for the Kaggle **[brain-to-text-25](https://www.kaggle.com/competitions/brain-to-text-25)**
competition: a hybrid CNN + BiLSTM + Transformer acoustic model trained with
CTC loss, decoded with a KenLM n-gram language model, and rescored with a
causal LLM for fluency. This is the modularized, repo-ready version of the
original single Kaggle notebook (`notebooks/kaggle_submission.ipynb`) that
produced the final submission (~15% WER).

## Project structure

```
.
├── config.py              # Single source of truth for paths & hyperparameters
├── train.py                # Full training pipeline (model + drift loss + WER validation)
├── predict.py               # Baseline submission: greedy CTC decode, no LM
├── decode_llm.py            # Full pipeline: KenLM beam search + causal-LLM rescoring
├── scripts/
│   └── build_kenlm.sh        # One-time build of the KenLM CLI binaries
├── src/
│   ├── dataset.py            # HDF5 loading, augmentation, Dataset, collate_fn
│   ├── model.py               # RoPE attention, Transformer blocks, HybridLSTMTransformerCTC
│   ├── metrics.py             # CTC greedy decode + WER
│   └── utils.py                # drop_path, session->day-index mapping, seeding
├── checkpoints/               # Trained model checkpoints land here (gitignored)
├── notebooks/
│   └── kaggle_submission.ipynb  # Original exploratory Kaggle notebook, kept for reference
├── requirements.txt
└── .gitignore
```

## The model

`HybridLSTMTransformerCTC` (see `src/model.py`) processes each trial as:

1. **Day-specific linear adaptation** — a learned per-session linear layer
   (`day_weights` / `day_biases`) absorbs electrode drift between recording
   sessions, regularized during training by a "drift loss" that keeps
   adjacent days' adaptation layers close to each other.
2. **Gaussian smoothing** — a fixed depthwise Conv1D smooths the 512-channel
   neural features.
3. **CNN → BiLSTM** — local temporal feature extraction, then a 2-layer
   bidirectional LSTM.
4. **Patch embedding + Transformer** — LSTM outputs are grouped into patches
   and passed through a small Transformer encoder using RoPE and stochastic
   depth (drop path).
5. **CTC head** — outputs per-timestep character log-probabilities.

## Data

The dataset is the Kaggle **brain-to-text-25** competition data (HDF5 files
with 512-channel neural features + sentence transcripts, split into
`train` / `val` / `test`, one folder per recording session/"day"). On Kaggle
this is already mounted at:

```
/kaggle/input/competitions/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final
```

which is the default in `config.py`. Running locally: download it with the
Kaggle CLI and point `--data_dir` at the local copy:

```bash
kaggle competitions download -c brain-to-text-25
```

## Setup

```bash
pip install -r requirements.txt

# One-time: build the KenLM CLI tools (lmplz, build_binary), needed only
# for decode_llm.py
bash scripts/build_kenlm.sh
```

## Usage

### 1. Train

```bash
python train.py \
    --data_dir /path/to/hdf5_data_final \
    --checkpoint_dir checkpoints \
    --num_epochs 82
```

Saves `checkpoints/latest_model.pt` (every epoch) and `checkpoints/best_model.pt`
(whenever validation WER improves), plus a `loss_curve.png`. To continue
training from an existing checkpoint:

```bash
python train.py --resume checkpoints/latest_model.pt
```

### 2. Baseline predictions (fast, no LM)

```bash
python predict.py --checkpoint checkpoints/best_model.pt --output submission.csv
```

### 3. Full pipeline: KenLM + LLM rescoring (final submission)

This is the higher-accuracy path — beam search against a 4-gram KenLM
model trained on the training transcripts, with each beam candidate
re-ranked by a causal LLM's fluency score:

```bash
python decode_llm.py \
    --checkpoint checkpoints/best_model.pt \
    --llm_name Qwen/Qwen2.5-7B \
    --beam_width 100 \
    --llm_weight 1.0 \
    --output submission_llm.csv
```

Key knobs (all have sensible defaults in `config.py`):

| Flag | Meaning |
|---|---|
| `--kenlm_alpha` | Acoustic/LM balance in beam search |
| `--kenlm_beta` | Word insertion bonus in beam search |
| `--llm_weight` | How much the LLM fluency score influences beam re-ranking |
| `--llm_name` | Any HuggingFace causal LM (bigger = slower, usually more fluent) |
| `--corpus_file` | Skip re-loading the full train split by passing a pre-built one-sentence-per-line corpus |

## Notes on this being a "modularized" repo

The original notebook trained a checkpoint in an earlier run, then in its
final submission run **skipped training entirely** (that block was disabled)
and just reloaded the trained checkpoint straight into the KenLM+LLM
decoding pipeline. `train.py` in this repo is that training code, restored
and made runnable/standalone (with `--resume` support for continuing from
a previous run, and CLI args instead of hardcoded Kaggle-kernel-specific
paths) — so the repo works as a complete pipeline: train → predict →
decode with rescoring, or you can drop straight into `decode_llm.py` if you
already have a `best_model.pt`.

## License

No license file is included yet — add one (e.g. MIT, Apache-2.0) before
making the repo public if you want to specify usage terms.
