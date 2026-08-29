"""
config.py — single source of truth for paths, the phoneme inventory, and every
hyperparameter used across training, decoding and LLM rescoring.

The decoding / LLM defaults below are the BEST combination found by the
validation sweep in the original notebook:

    lm_weight (KenLM)     = 4.5
    llm_fusion_weight (λ) = 0.75
    rescoring LLM         = Qwen/Qwen2.5-7B (4-bit nf4)
    beam_width / n-best   = 250 / 30
    gate percentiles      = 1 (incoherent) / 75 (already-confident)

  ->  validation  WER 7.62%  |  CER 5.31%  |  PER 12.12%

Every entry-point script (train.py / predict.py / decode_llm.py / evaluate.py)
imports from here and lets the CLI override a subset of these values.
"""

import os

# ---------------------------------------------------------------------------
# Phoneme inventory (index 0 = CTC blank, last entry = word-boundary/silence).
# These IDs match `seq_class_ids` stored in every trial's HDF5 record, so the
# model predicts phonemes directly rather than characters.
# ---------------------------------------------------------------------------
PHONEME_VOCAB = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W',
    'Y', 'Z', 'ZH', ' | ',
]
N_CLASSES = len(PHONEME_VOCAB)  # 41 = 39 CMU phonemes + CTC blank + word boundary

# ---------------------------------------------------------------------------
# Default paths. On Kaggle the competition data is already mounted at
# DEFAULT_DATA_DIR; running locally, pass --data_dir on the command line.
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = (
    '/kaggle/input/competitions/brain-to-text-25/'
    't15_copyTask_neuralData/hdf5_data_final'
)
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_CHECKPOINT_DIR, 'figures')

# Kaggle dataset slugs for the prebuilt lexicon / tokens / KenLM binary.
# (See scripts/build_kenlm.sh to build a KenLM binary from scratch instead.)
LEXICON_DATASET_SLUG = 'heyyousum/quality-english-dataset-for-ngram-model-v2'
KENLM_DATASET_SLUG = 'heyyousum/custom-4-gram-wiki-news-switchboard-updated-v3'

# Kaggle dataset holding a previously-trained checkpoint (norm_stats.pt +
# best_model.pt / swa_model.pt). Set to None to always read from --checkpoint_dir.
PRETRAINED_CKPT_DATASET = 'shohan3125/brain-to-text-checkpoint'

# ---------------------------------------------------------------------------
# Model architecture + training hyperparameters (BEST).
# ---------------------------------------------------------------------------
CONFIG = {
    # runtime
    'data_dir': DEFAULT_DATA_DIR,
    'checkpoint_dir': DEFAULT_CHECKPOINT_DIR,
    'device': 'cuda',                 # resolved to 'cpu' at runtime if no GPU
    'seed': 42,

    # dataloader / schedule
    'batch_size': 16,
    'num_epochs': 82,
    'num_workers': 2,

    # architecture
    'input_size': 512,
    'd_model': 384,
    'n_heads': 6,
    'n_layers': 4,
    'd_ff': 1536,
    'patch_size': 3,
    'lstm_hidden': 256,
    'lstm_layers': 2,
    'n_classes': N_CLASSES,

    # regularization & day adaptation
    'dropout': 0.4,
    'head_dim': 256,
    'attn_dropout': 0.5,
    'drop_path_rate': 0.2,
    'smooth_kernel_std': 2.0,
    'smooth_kernel_size': 100,
    'drift_lambda': 0.01,

    # optimizer
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'grad_clip': 5.0,
    'use_augmentation': True,

    # mixed precision & feature clipping
    'use_amp': True,
    'feature_clip': 5.0,

    # early stopping / SWA
    'patience': 10,
    'swa_topk': 3,

    # resume
    'resume_from_checkpoint': None,
}

# ---------------------------------------------------------------------------
# Beam-search + LLM-rescoring hyperparameters (BEST).
# ---------------------------------------------------------------------------
DECODING = {
    'beam_width': 250,
    'nbest': 30,
    'lm_weight': 4.5,           # KenLM weight (best confirmed point of the sweep)
    'word_score': 0.0,
    'decode_batch': 16,

    'llm_name': 'Qwen/Qwen2.5-7B',
    'llm_fusion_weight': 0.75,   # base fusion λ (best); margin-adaptive at inference

    # confidence gate percentiles
    'llm_gate_percentile': 1,          # skip only clearly incoherent trials
    'llm_gate_margin_percentile': 75,  # skip already-confident trials
}

# Verbatim-aware conditioning prefix used when scoring candidates with the LLM.
VERBATIM_PREFIX = (
    "The following is a raw, word-for-word transcript of someone speaking out "
    "loud. It is unedited and may contain repetitions, false starts, filler "
    "words, or unusual phrasing exactly as spoken:\n"
)


def resolve_device(requested='cuda'):
    """Return 'cuda' only if a GPU is actually available, else 'cpu'."""
    import torch
    return 'cuda' if (requested == 'cuda' and torch.cuda.is_available()) else 'cpu'
