"""
config.py — single source of truth for paths, the phoneme inventory, and every
hyperparameter used across training, decoding and LLM rescoring (pipeline v6).

v6 replaces the single-model v5 pipeline with a **trial-level cross-fit** of the
validation set: two acoustic models are trained in parallel (one per T4), each
seeing a disjoint half of `val`, and every val trial is then decoded by the
model(s) that never saw it. Decoding ensembles the folds by *exact CTC sequence
log-likelihood* (not posterior averaging, which destroyed WER in v2), expands the
candidate pool with phonetic neighbours, and rescores with a task-adapted LLM
under a fluency gate.

The training / decoding defaults below are the ones actually used by the runs in
`notebooks/`:

    acoustic      B2TNetV4, 30.15M params, 2 folds x ~11h on a Kaggle T4
                  fold A: PER 6.88%  |  fold B: PER 6.95%  (EMA, held-out trials)
    decoding      flashlight lexicon beam search + KenLM 4-gram
                  beam 300-400, n-best 50-75, lm_weight union {0.5, 2.0, 3.5, 5.0}
    rescoring     Llama-3.1-8B (QLoRA nf4) next-word-prediction fine-tune,
                  log-linear fusion `am + ng*kenlm + llm*LLM + nw*#words`,
                  weights tuned on folds A u B, verified on the untouched fold C

Every entry point (train.py / predict.py / decode_llm.py / evaluate.py) imports
from here and lets the CLI override a subset of these values.
"""

import os

# ---------------------------------------------------------------------------
# Phoneme inventory (index 0 = CTC blank, last entry = word boundary/silence).
# These IDs match `seq_class_ids` in every trial's HDF5 record, so the model
# predicts phonemes directly rather than characters.
# PROTECTED: must stay byte-identical to tokens.txt of the lexicon dataset.
# ---------------------------------------------------------------------------
PHONEME_VOCAB = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W',
    'Y', 'Z', 'ZH', ' | ',
]
N_CLASSES = len(PHONEME_VOCAB)      # 41 = 39 CMU phonemes + CTC blank + word boundary
BLANK_ID = 0
SIL_ID = N_CLASSES - 1              # 40  (' | ')

# ---------------------------------------------------------------------------
# Paths. On Kaggle the competition data is already mounted at DEFAULT_DATA_DIR;
# running locally, pass --data_dir on the command line.
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = (
    '/kaggle/input/competitions/brain-to-text-25/'
    't15_copyTask_neuralData/hdf5_data_final'
)
COMPETITION_SLUG = 'brain-to-text-25'

DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_SUBMISSION_DIR = 'submission'
DEFAULT_FIGURES_DIR = 'figures'
DEFAULT_TABLES_DIR = 'tables'
DEFAULT_LOG_DIR = 'logs'

# Disk-backed scratch for the float16 trial cache (~13 GB). The first candidate
# that is writable, NOT tmpfs, and has enough free space wins — a RAM-backed
# cache is what OOM-killed v6.0.
CACHE_CANDIDATES = [
    '/tmp/b2t_cache', '/kaggle/temp/b2t_cache', '/var/tmp/b2t_cache', './work/_b2t_cache',
]
CACHE_NEED_GB = 14.0

# Kaggle dataset slugs for the prebuilt lexicon / tokens / KenLM binary.
# (scripts/build_kenlm.sh builds a KenLM binary from scratch instead.)
LEXICON_DATASET_SLUG = 'heyyousum/quality-english-dataset-for-ngram-model-v2'
KENLM_DATASET_SLUG = 'heyyousum/custom-4-gram-wiki-news-switchboard-updated-v3'

# Where `decode_llm.py` looks for trained fold checkpoints when --checkpoint_dir
# is not given: a Kaggle dataset built from a previous train run's output.
PRETRAINED_CKPT_DATASET = None
PRETRAINED_CKPT_DIR = '/kaggle/input/b2t-v6-crossfit-ckpt'

# Per fold, which checkpoint file to decode with. All candidates present are
# screened by REAL beam-search WER on held-out trials first; this list is only
# the tie-break order. PROTECTED: changing it changes which weights ship.
CKPT_PREFERENCE = ['best_wer.pt', 'ema_final.pt', 'best_per.pt']

# Warm start: continue training from a previous run's checkpoints (per fold,
# first name found wins). None -> train from scratch.
INIT_CKPT_DIR = None
INIT_PREFERENCE = ['best_per.pt', 'best_wer.pt', 'ema_final.pt']

SEED = 1337

# ---------------------------------------------------------------------------
# Cross-fit plan: one worker per GPU, one fold each.
#   Inside every (session, block) of `val`, trials are labelled A/B/A/B/C in
#   trial order with a per-block random offset. Fold A's model trains on train +
#   val-A and is held out on B/C; fold B's model trains on train + val-B.
#   Fold C is seen by neither and is never tuned on — it is the honest check.
#   Trial-level (not block-level) because Kaggle's test trials are interleaved
#   with val trials inside the SAME blocks.
# ---------------------------------------------------------------------------
CROSSFIT_PATTERN = ['A', 'B', 'A', 'B', 'C']
TRAIN_RUNS = [
    {'fold': 'A', 'seed': 0, 'gpu': 0, 'train_val_labels': ['A']},
    {'fold': 'B', 'seed': 1, 'gpu': 1, 'train_val_labels': ['B']},
]
# FINAL-FIT variant (only after v6 is validated): 'train_val_labels': ['A','B','C']
# for both runs. No clean val is left, so decoding must reuse weights tuned on a
# previous cross-fit run.

# ---------------------------------------------------------------------------
# Session time budget (training). The LR schedule is TIME-based, not epoch-based:
# the cosine is parameterised by wall-clock progress toward `train_end`, so it
# always lands on lr_min at the deadline no matter how many epochs fit.
# ---------------------------------------------------------------------------
SESSION = {
    'session_hours': 11.5,            # Kaggle session limit to respect
    'notebook_reserve_min': 10,       # after workers stop: curves/tables/output flush
    'worker_final_reserve_min': 8,    # inside each worker: final EMA eval + checkpoint write
    'monitor_print_min': 10,          # how often the launcher echoes worker progress
    'max_worker_restarts': 10,        # a CRASHED worker is relaunched from its resume state
    'worker_rss_limit_gb': 11.0,      # memory guard: save state + exit(75) above this RSS
    'worker_min_avail_gb': 3.0,       # ...or when the machine drops below this much free RAM
    'restart_min_remaining_min': 10,  # ...unless less than this much training time is left
}

# ---------------------------------------------------------------------------
# Acoustic model: B2TNetV4  (PROTECTED — do not edit without retraining)
#   low-rank day adapter -> Gaussian smoothing -> strided-Conv1d patching
#   -> 4x residual BiGRU -> 3x Conformer (RoPE) -> phoneme CTC head
#   + two inter-CTC auxiliary heads. 30.15M parameters.
# ---------------------------------------------------------------------------
MODEL_CFG = dict(
    input_size=512, patch_size=14, patch_stride=4, d_model=512, gru_hidden=384, gru_layers=4,
    n_conformer=3, n_heads=8, d_ff=1536, conv_kernel=15, day_rank=32, day_dropout_p=0.15,
    dropout=0.35, attn_dropout=0.1, drop_path_rate=0.15, smooth_kernel_std=2.0, smooth_kernel_size=100,
)

# ---------------------------------------------------------------------------
# Training: AdamW + time-based cosine, EMA weights, CTC + inter-CTC + CR-CTC.
# ---------------------------------------------------------------------------
TRAIN_CFG = dict(
    batch_size=32, max_bins_per_batch=32 * 1400,
    lr_max=7.0e-4, lr_min=1.0e-5, warmup_steps=2000, lr_day_scale=2.0,
    weight_decay=0.01, weight_decay_day=0.01, grad_clip=2.0, clip=5.0, ema_decay=0.999,
    interctc_weight=0.3, cr_weight=0.2,
    log_every=200, eval_every_min=30, eval_every_min_late=15, late_frac=0.6, resume_every_min=20,
    init_from=None,
)

# ---------------------------------------------------------------------------
# GPU augmentation (train only). Applied per CR-CTC view except speed perturbation
# and random_cut, which are shared so the two views stay time-aligned.
# ---------------------------------------------------------------------------
AUG_CFG = dict(
    random_cut=3, speed_p=0.5, speed_lo=0.9, speed_hi=1.1,
    white_noise_std=0.8, constant_offset_std=0.2, static_gain_std=0.05,
    electrode_drop_p=0.10, electrode_drop_apply=0.5,
    time_mask_frac=0.20, time_mask_min=4, time_mask_max=24,
)

# ---------------------------------------------------------------------------
# Decoding + LLM rescoring. These are the values from the latest sweep run
# (notebooks/b2t-25-crossfit-llama3-1-8b-sweep_best.ipynb); SWEEP_CFG below can
# re-search most of them at negligible cost.
#
#   For a cheap run, set llm_name='Qwen/Qwen2.5-1.5B', llm_ft_batch=32,
#   llm_ft_grad_accum=1, llm_batch=32 (the config used by the v6.2 train notebook).
# ---------------------------------------------------------------------------
LLM_CFG = dict(
    # checkpoint screening: real beam WER on a subsample decides which file per fold
    screen_n=300, screen_beam=50, screen_lm_weight=3.0,
    # candidate generation (flashlight lexicon beam search, CPU process pool)
    gen_lm_weights=[0.5, 2.0, 3.5, 5.0], gen_beam=400, gen_nbest=75, n_proc=3,
    # phonetic-neighbour expansion of the top hypotheses
    expansion=True, expand_top=3, expand_max_nb=12, expand_max_new=600,
    # task-adapted LLM rescorer
    use_llm=True, llm_name='meta-llama/Llama-3.1-8B', llm_finetune=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05, llm_ft_epochs=2, llm_ft_lr=2e-4,
    llm_ft_batch=6, llm_ft_grad_accum=6,      # QLoRA on a single T4: micro-batch + accumulation
    llm_topk=24, llm_batch=16,
    # stage-2 weight tuning
    gate_percentiles=[0, 5, 10, 15, 20, 30, 40], smooth_grid=True, tol_rel=0.0,
    refit_on_all_oof=True,
    tune_labels=['A', 'B'], verify_labels=['C'],
)

# ---------------------------------------------------------------------------
# Decoding sweep: search RANGES instead of one static decode setting.
#
# Measured cost per axis on a 2h44m run:
#   gen_beam    -> FULL flashlight regeneration                ~1h23m per value
#   lm_weights  -> FREE: generate the union once, score any subset
#   nbest       -> FREE: n-best lists are nested, slice [:n] off the generated max
#   expand_*    -> cheap: only genuinely new strings hit the CTC scorer
#   stage-1/2 weight grid ranges -> ~1s per grid
#   tol_rel / smooth_grid        -> FREE: re-pick from an already computed grid
#   llm_topk    -> FREE: score the largest k once, smaller k is a prefix of it
#
# Why the grid RANGES are swept: in the 4.67% run the stage-2 optimum landed on
# the edge of the grid in all three weights at once (llm=max, ng=max, nw=min).
# A tuner that stops at a boundary has not found an optimum, it has run out of
# grid. Every sweep row is flagged when its weights still land on an edge.
#
# Selection is on the TUNE folds (A u B) ONLY. Fold C rides along in every table
# so you can see whether the choice transferred, and is never selected on.
# ---------------------------------------------------------------------------
SWEEP_CFG = dict(
    enable=True,
    mode='staged',               # 'staged' = coordinate descent (cheap) | 'grid' = full product
    passes=2,
    # ---- pool side ----
    beam=[300],                  # add 400 only if you can afford another ~1h30m of generation
    nbest=[25, 50, 75],
    lm_weight_sets=[
        [3.0],
        [2.0, 3.0, 4.0],
        [0.5, 2.0, 3.0, 3.5, 5.0],
        [0.5, 1.5, 2.5, 3.0, 4.0, 5.0],
    ],
    expand_top=[0, 3, 5, 8],
    expand_max_nb=[12, 20],
    expand_max_new=[600, 1200],
    stage1_grid=['base', 'wide'],
    # ---- stage-2 side ----
    stage2_grid=['base', 'wide', 'wider'],
    tol_rel=[0.0, 0.005, 0.02],  # 1-SE-style tie-break toward a smaller |llm| weight
    smooth_grid=[True, False],
    llm_topk=[12, 24, 40],
    # ---- budget / reporting ----
    time_budget_min=75,          # pool sweep only (generation excluded)
    stage2_budget_min=25,
    min_minutes_left_for_extra_beam=240,
    cache_features=True,
    bootstrap=400,               # utterance bootstrap sd = the noise floor a margin must beat
)

STAGE1_GRIDS = {                 # (kenlm start, stop, step), (word-bonus start, stop, step)
    'base': ((0.0, 3.0, 0.1), (-4.0, 6.0, 0.25)),
    'wide': ((0.0, 4.0, 0.1), (-8.0, 8.0, 0.25)),
}
STAGE2_GRIDS = {                 # (kenlm...), (llm...), (word-bonus...)
    'base':  ((0.0, 2.5, 0.25), (0.0, 2.0, 0.1), (-4.0, 6.0, 0.5)),
    'wide':  ((0.0, 4.0, 0.25), (0.0, 4.0, 0.1), (-8.0, 8.0, 0.5)),
    'wider': ((0.0, 6.0, 0.25), (0.0, 6.0, 0.2), (-10.0, 10.0, 0.5)),
}


def resolve_device(requested='cuda'):
    """Return 'cuda' only if a GPU is actually available, else 'cpu'."""
    import torch
    return 'cuda' if (requested == 'cuda' and torch.cuda.is_available()) else 'cpu'


def work_dir(base=None):
    """Kaggle writes to /kaggle/working; locally everything lands in ./work."""
    if base:
        return os.path.abspath(base)
    return '/kaggle/working' if os.path.isdir('/kaggle') else os.path.abspath('./work')
