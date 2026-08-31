# checkpoints/

**Download the trained model weights and place them here manually.**

The required model weights can be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/158r--SA2euXlFlWWAIrZ7fQjg0t-rgc1?usp=drive_link). 

> **Access Note:** This Drive link is restricted. When you click the link, you will be prompted to request access. Please request **Viewer** permission so you can download the files. You do not need to ask for Editor access.

The inference scripts (`predict.py`, `decode_llm.py`, `evaluate.py`) look for
these files in this folder (or in the Kaggle dataset named by
`PRETRAINED_CKPT_DATASET` in `config.py`):

| File               | Produced by            | Required for inference |
| ------------------ | ---------------------- | ---------------------- |
| `norm_stats.pt`    | `train.py`             | **Yes** — per-channel TRAIN normalization (mean/std) |
| `best_model.pt`    | `train.py`             | Yes (unless `swa_model.pt` is present) |
| `swa_model.pt`     | `train.py` (SWA avg)   | Optional — preferred over `best_model.pt` when present |
| `latest_model.pt`  | `train.py` (per epoch) | Only for `--resume` |

`*.pt` files are gitignored (they are large binaries), so commit them via Git LFS
or attach them as a Kaggle dataset rather than pushing them directly.

Loading priority used by the scripts: `swa_model.pt` → `best_model.pt`
(override with `--checkpoint /path/to/file.pt`).
