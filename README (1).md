# assets/

## `B-T-S pipeline.jpg`

The README links an architecture figure at the repository root:

```
![B-T-S Pipeline Architecture](B-T-S%20pipeline.jpg)
```

Upload the paper's own `B-T-S pipeline.jpg` there. It is deliberately not generated from
code, so the figure in the repo is the same one the paper uses.

If you need to redraw it, the pipeline it depicts is:

```
neural features [T, 512]
  -> low-rank day adapter (+ generic slot)
  -> Gaussian smoothing -> time masking
  -> patch embedding (Conv1d, patch 14 / stride 4)
  -> 4x residual BiGRU -> 3x Conformer (RoPE)
  -> phoneme CTC head (41)            [+ 2 inter-CTC aux heads, training only]

  x2 folds (trial-level cross-fit of val: A/B/A/B/C)
  -> flashlight lexicon beam search x KenLM 4-gram x several lm_weights -> n-best pool
  -> phonetic-neighbour expansion
  -> features: exact CTC log-likelihood (averaged over folds), KenLM ln P, LLM ln P, #words
  -> fluency-gated log-linear fusion, weights tuned on A u B, verified on C
  -> text
```

Figures produced by the code itself go to `figures/` (gitignored), not here.
