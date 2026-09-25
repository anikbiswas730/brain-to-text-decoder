"""Brain-to-Text v6.4 — modular source package.

Submodules:
    utils         : seeding, path resolvers, scratch-disk selection, atomic saves,
                    memory probes, figure/table writers
    dataset       : HDF5 -> float16 cache, pread reader, cross-fit split,
                    length-bucketed batching, GPU augmentation
    model         : B2TNetV4 (day adapter -> patch Conv1d -> BiGRU -> Conformer),
                    CTC / inter-CTC / CR-CTC losses, EMA
    metrics       : official normalisation, WER/CER/PER, lexicon, greedy decode
    decoding      : flashlight beam pool, KenLM scorer, phonetic expansion,
                    exact-CTC rescoring features, LLM fine-tune + scorer, tuner
    inference     : checkpoint discovery/screening, emission extraction, pools,
                    beam-search cache, weight grids
    sweep         : joint n-best x KenLM-weight-subset sweep, session-grouped CV
    gec           : selective generative error correction (optional)
    analysis      : ablation / oracle / fluency / per-day / error-analysis tables + figures
    train_worker  : one process = one GPU = one cross-fit fold
"""

__all__ = ['utils', 'dataset', 'model', 'metrics', 'decoding', 'inference', 'sweep', 'gec', 'analysis',
           'train_worker']
