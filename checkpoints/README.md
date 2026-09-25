# checkpoints/

`train.py` writes the whole decoding bundle here. It is gitignored — upload the folder by
hand, or (on Kaggle) create a dataset from the training run's output and pass it to
`decode_llm.py --checkpoint_dir`.

```
fold{A,B}_seed{0,1}_best_wer.pt     EMA weights, best greedy-lexicon WER on this fold's holdout
fold{A,B}_seed{0,1}_best_per.pt     EMA weights, best phoneme error rate
fold{A,B}_seed{0,1}_ema_final.pt    EMA weights at the deadline
fold{A,B}_seed{0,1}_metrics.jsonl   one line per evaluation (curves come from this)
norm_stats.pt                       per-channel mean/std of the train split
split.json                          the trial-level A/B/C cross-fit labels
sil_convention.json                 whether targets carry a trailing word-boundary token
trained_days.json                   per fold, which day indices it actually trained on
run_fold*.json                      the exact config each worker ran
manifest.json                       file list + model/train/aug config
```

**All four of the small JSON/PT files matter**, not just the weights:

* `split.json` decides which model is out-of-fold for which trial. Decode with the wrong
  split and every reported number is contaminated.
* `norm_stats.pt` must be the statistics the model was trained under, or the day adapters
  no longer mean the same thing.
* `sil_convention.json` decides whether a trailing `' | '` is appended to CTC targets —
  getting it wrong costs about one phoneme per utterance in every likelihood.
* `trained_days.json` tells the decoder which sessions to route through the generic adapter
  slot instead of an untouched, randomly-initialised one.

Each `.pt` holds `model_state_dict` (EMA weights), `config` (the `MODEL_CFG` it was built
with), `n_days`, `arch`, `fold`, `seed`, `train_val_labels`, `epoch`, `step`, `metrics`.
`decode_llm.py` rebuilds the model from the checkpoint's own `config`, so a bundle stays
loadable even if `config.py` moves on.

Checkpoint *selection* is not done by these filenames: every candidate is screened by a real
KenLM beam search on held-out trials, and `CKPT_PREFERENCE` in `config.py` is only the
tie-break. In the reference runs (v6.2 and v6.4, same bundle) fold A's `best_per.pt` won (screen WER 7.61% vs
7.86% for `best_wer.pt`) and fold B's `best_wer.pt` won (7.46%, tied with `best_per.pt` and taken by the
`CKPT_PREFERENCE` tie-break). See `results/checkpoint_screening.csv`.
