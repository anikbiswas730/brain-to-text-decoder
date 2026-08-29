"""Brain-to-Text decoder — modular source package.

Submodules:
    utils      : seeding, drop_path, session mapping, Kaggle/data path resolvers, AMP shim
    dataset    : HDF5 loading, augmentation, Dataset, collate_fn, channel stats
    model      : RoPE attention, Transformer blocks, HybridLSTMTransformerCTC
    metrics    : CTC greedy decode, phoneme<->word helpers, WER/CER/PER, text normalization
    decoding   : KenLM beam-search decoder + Qwen LLM score-fusion rescorer
    inference  : emission extraction (val loader / test HDF5 samples)
"""

__all__ = ["utils", "dataset", "model", "metrics", "decoding", "inference"]
