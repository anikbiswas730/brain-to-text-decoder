# submission/

**Upload / write the competition submission CSV here.**

- `python predict.py    --output submission/submission_baseline.csv` — fast greedy-CTC baseline
- `python decode_llm.py --output submission/submission.csv`          — full pipeline (WER ≈ 7.62%)

`*.csv` files are gitignored so large prediction files are not committed. The CSV
columns are auto-matched to the competition's `sample_submission.csv` (falling
back to `id,text`).
