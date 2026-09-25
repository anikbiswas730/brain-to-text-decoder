# results/

Aggregate tables of the v6.4 reference run
(`notebooks/b2t-25-crossfit-llama3-1-8b-hyperparamter-tuning.ipynb`, 2 × T4, 8 h 53 m), copied verbatim
from its `tables/` output. `decode_llm.py` writes the same files to `tables/` on every run.

| File | Contents |
| ---- | -------- |
| `ablation_decoding.csv` | WER per decoding stage on tune (A∪B), verify (C) and all out-of-fold trials, plus oracle rows |
| `best_weights.json` | the shipped fusion weights: stage-1, gate percentile / τ, hi- and lo-regime weights, models used |
| `checkpoint_screening.csv` | real beam-search WER (beam 50, λ 3.0, 300 held-out trials) of every checkpoint per fold |
| `joint_sweep_all_combos.csv` | all 508 (n-best × KenLM-weight subset) pools: session-CV WER, in-sample tune WER, fitted weights, pool size; fold-C WER and paired gap / SE for the top rows and the pinned pool |
| `oracle_vs_k.csv` | oracle WER of the top-k candidates under the stage-1 ranking |
| `outlier_days_analysis.csv` / `clean_days_analysis.csv` | per-session greedy PER, flashlight and final WER, share of all errors, share in the low-fluency regime |
| `error_analysis_search_vs_model.csv` | final-system word errors by cause (search error / not in pool / outscored / OOV) |
| `llm_finetune_history.json` | dev NLL/token of the Llama-3.1-8B QLoRA fine-tune per epoch |
| `dataset_summary.csv` / `dataset_trials_per_day.csv` | trials, sessions and lengths per split and per session |

Not committed: `oof_predictions.csv`, `error_analysis_utterances.csv` and `test_predictions_detail.csv`
(per-utterance, they contain competition transcripts), and `submission.csv`.
