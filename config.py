"""
Central configuration for the Brain-to-Text decoding pipeline.

Nothing in here is secret or environment-specific in a sensitive way -- it's
just the single source of truth for paths and hyperparameters so that
train.py / predict.py / decode_llm.py all agree with each other.

Override any of these on the command line (see each script's --help) instead
of editing this file, unless you're changing a genuine default.
"""

import torch

CONFIG = {
    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------
    # Default path assumes this is run inside a Kaggle notebook/kernel with
    # the "brain-to-text-25" competition dataset attached. If you're running
    # locally, download the dataset via the Kaggle API/CLI and point this at
    # the local copy, e.g.:
    #   kaggle competitions download -c brain-to-text-25
    'data_dir': '/kaggle/input/competitions/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final',

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 16,
    'num_epochs': 82,

    # --------------------------------------------------------------
    # Model architecture
    # --------------------------------------------------------------
    'd_model': 384,
    'n_heads': 6,
    'n_layers': 4,
    'd_ff': 1536,
    'patch_size': 3,
    'lstm_hidden': 256,
    'lstm_layers': 2,

    # --------------------------------------------------------------
    # Regularization & Adaptation
    # --------------------------------------------------------------
    'dropout': 0.4,
    'head_dim': 256,
    'attn_dropout': 0.5,
    'drop_path_rate': 0.2,        # Stochastic depth for Transformer blocks
    'smooth_kernel_std': 2.0,     # Gaussian smoothing std-dev
    'smooth_kernel_size': 100,    # Gaussian smoothing kernel size
    'drift_lambda': 0.01,         # Regularization strength for day-specific layers

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'use_augmentation': True,

    # --------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------
    'checkpoint_dir': 'checkpoints',
    'early_stop_patience': 10,

    # --------------------------------------------------------------
    # LM rescoring / decoding
    # --------------------------------------------------------------
    'beam_width': 100,
    'kenlm_alpha': 0.4,           # Acoustic/LM base weight (pyctcdecode alpha)
    'kenlm_beta': 1.0,            # Word insertion weight (pyctcdecode beta)
    'llm_weight': 1.0,            # Weight of the causal-LM fluency score
    'llm_name': 'Qwen/Qwen2.5-7B',
    'kenlm_ngram_order': 4,
}
