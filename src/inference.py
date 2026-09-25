"""
src/inference.py — the glue between trained checkpoints and the decoding stage:
locate a checkpoint bundle, decide WHICH file per fold to decode with, pull the
emissions out, assemble candidate pools, cache the (expensive) beam-search
output, and hold the weight-grid helpers the tuners share.

Two things here are easy to get wrong and expensive to get wrong:

* **Which checkpoint.** Each fold writes `best_wer`, `best_per` and `ema_final`.
  Those are selected by a greedy, lexicon-only decode, which is not the metric
  that ships. So every candidate is screened by a REAL KenLM beam search on a
  subsample of trials the fold never trained on, and the winner by that measure
  is used; `CKPT_PREFERENCE` is only the tie-break.

* **Which model may see which trial.** A val trial is decoded only by models
  whose training labels exclude that trial's fold (`oof_models`). Feeding a
  trial to the model that trained on it would report a number no test set can
  reproduce.
"""

import gzip
import hashlib
import os
import pickle
import random
import re
import time
from glob import glob

import numpy as np
import torch

from config import STAGE1_GRID, STAGE2_GRID
from src.dataset import BucketBatchSampler, TrialSet, collate_trials
from src.decoding import Pool, pick_from_grid, Tuner
from src.metrics import official_wer
from src.model import B2TNetV4, forward_batch


# ---------------------------------------------------------------------------
# checkpoint bundle
# ---------------------------------------------------------------------------
def find_checkpoint_dir(*candidates):
    """First directory (searched recursively) that actually contains fold
    checkpoints. Pass the CLI value, then config fallbacks, then /kaggle/input."""
    for c in candidates:
        if not c:
            continue
        if os.path.isdir(str(c)):
            hits = sorted(glob(os.path.join(c, '**', 'fold*_seed*_*.pt'), recursive=True))
            if hits:
                return os.path.dirname(hits[0])
    return None


def group_checkpoints(ckpt_dir, preference):
    """-> {'foldA_seed0': {'best_wer.pt': path, ...}, ...}"""
    groups = {}
    for f in sorted(glob(os.path.join(ckpt_dir, '**', 'fold*_seed*_*.pt'), recursive=True)):
        m = re.match(r'(fold[^_]+_seed\d+)_(.+\.pt)$', os.path.basename(f))
        if m and m.group(2) in preference:
            groups.setdefault(m.group(1), {})[m.group(2)] = f
    return groups


def build_day_override(ckpt_dir, n_days, idx2session=None, log=print):
    """Route sessions that NO fold ever trained on through the generic adapter
    slot (index n_days) instead of their own untouched, randomly-initialised one.

    A day is only counted as untrained if every fold agrees it is untrained —
    hence the intersection over all `trained_days.json` files.
    """
    override = torch.arange(n_days, dtype=torch.long)
    untrained = None
    for f in glob(os.path.join(ckpt_dir, '**', 'trained_days.json'), recursive=True):
        import json
        s = set(int(d) for d in json.load(open(f)).get('untrained_days', []))
        untrained = s if untrained is None else (untrained & s)
    untrained = untrained or set()
    for d in untrained:
        override[d] = n_days
    if idx2session is not None:
        log('days routed through the generic slot:',
            [idx2session[d] for d in sorted(untrained)] or 'none')
    return override, untrained


def load_am(path, device):
    """Load one acoustic checkpoint (EMA weights) -> (model.eval(), raw dict)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    mdl = B2TNetV4(ck['n_days'], ck['config'])
    mdl.load_state_dict(ck['model_state_dict'])
    return mdl.to(device).eval(), ck


# ---------------------------------------------------------------------------
# emissions
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_emissions(mdl, readers, split, idxs, session2idx, norm, clip, device, day_override=None):
    """-> list of np.float16 [T', 41] log-prob matrices, in the order of `idxs`.

    Kept in float16 on the host: the pool needs every model's emissions for
    every trial simultaneously, and float32 would roughly double a multi-GB
    working set for no measurable WER difference.
    """
    ds = TrialSet(readers, [(split, i) for i in idxs], session2idx)
    out = [None] * len(idxs)
    for bidx in BucketBatchSampler(ds.lengths, 24, 24 * 2500, False, 0):
        b = collate_trials([ds[i] for i in bidx])
        lp, ol = forward_batch(mdl, b, device, norm, clip, day_override=day_override)
        lp = lp.cpu()
        for k, j in enumerate(b['j'].tolist()):
            out[j] = lp[:int(ol[k]), k].numpy().astype(np.float16)
    return out


def screen_checkpoints(groups, readers, val_labels, session2idx, norm, clip, device,
                       lexicon_path, tokens_path, kenlm_path, cfg, seed=1337,
                       day_override=None, log=print):
    """Pick one checkpoint per fold by REAL beam-search WER on held-out trials.

    Returns (models, rows) where each model is
        {'tag', 'ckpt', 'labels_seen', 'model', 'screen_wer', 'arch_metrics'}
    and `rows` is the screening table (empty if the beam decode was unavailable).
    """
    from src.decoding import run_beam_pool

    n_val = len(readers['val'])
    val_refs = [m['sentence'] for m in readers['val'].meta]
    spec = dict(lexicon=lexicon_path, tokens=tokens_path, lm=kenlm_path,
                beam=cfg['screen_beam'], nbest=1, lm_weights=[cfg['screen_lm_weight']])
    tasks, meta, t0 = [], {}, time.time()
    for tag, cands in groups.items():
        ck_any = torch.load(next(iter(cands.values())), map_location='cpu', weights_only=False)
        seen = set(ck_any.get('train_val_labels', []))
        hold = [j for j in range(n_val) if val_labels[j] not in seen]
        sub = (sorted(random.Random(seed).sample(hold, min(cfg['screen_n'], len(hold))))
               if hold else [])
        meta[tag] = {'seen': seen, 'sub': sub}
        if len(cands) == 1 or not sub:
            continue                      # nothing to choose between
        for name, path in cands.items():
            try:
                mdl, _ = load_am(path, device)
                ems = extract_emissions(mdl, readers, 'val', sub, session2idx, norm, clip,
                                        device, day_override)
                tasks += [((tag, name, k), ems[k]) for k in range(len(sub))]
                del mdl
            except Exception as e:
                log(f'  {tag}/{name}: unreadable checkpoint skipped ({e!r})')
            if device == 'cuda':
                torch.cuda.empty_cache()
    try:
        out = run_beam_pool(tasks, spec, cfg['n_proc'], desc='screen') if tasks else {}
    except Exception as e:
        log('screening decode failed -> CKPT_PREFERENCE order used:', repr(e))
        out = {}

    models, rows = [], []
    lw0 = cfg['screen_lm_weight']
    pref = list(cfg.get('preference', []))
    for tag, cands in groups.items():
        sub = meta[tag]['sub']
        scores = {}
        for name in cands:
            if out and (tag, name, 0) in out:
                hyps = [(out[(tag, name, k)][lw0] or [''])[0] for k in range(len(sub))]
                scores[name] = official_wer([val_refs[j] for j in sub], hyps)[0]
                rows.append({'group': tag, 'checkpoint': name,
                             'screen_wer_%': round(scores[name] * 100, 2), 'n_trials': len(sub)})
        order = (sorted(scores, key=lambda n: (scores[n], pref.index(n))) if scores else [])
        order += [n for n in sorted(cands, key=pref.index) if n not in order]
        mdl = None
        for best_name in order:
            try:
                mdl, ck = load_am(cands[best_name], device)
                break
            except Exception as e:
                log(f'  {tag}/{best_name}: load failed ({e!r})')
        if mdl is None:
            continue
        models.append({'tag': tag, 'ckpt': cands[best_name],
                       'labels_seen': set(ck.get('train_val_labels', [])), 'model': mdl,
                       'screen_wer': scores.get(best_name), 'arch_metrics': ck.get('metrics', {})})
        log(f"{tag}: using {best_name} (screen WER "
            f"{(scores.get(best_name) or float('nan')) * 100:.2f}%), trained on val labels "
            f"{sorted(models[-1]['labels_seen'])}")
    log(f'screening took {time.time() - t0:.0f}s')
    return models, rows


def oof_models(models, val_labels):
    """For each val trial, the indices of models that never trained on its fold."""
    return [[mi for mi, M in enumerate(models) if val_labels[j] not in M['labels_seen']]
            for j in range(len(val_labels))]


# ---------------------------------------------------------------------------
# candidate pools
# ---------------------------------------------------------------------------
def build_pool(gen, models, split, n, model_lists, lm_weights, nbest):
    """Merge the n-best lists of (model x lm_weight) into one deduplicated pool
    per utterance. `src` keeps provenance as 'tag@lm_weight#rank' — the '#0'
    suffix is what protects each generator's 1-best from the top-k LLM filter."""
    pool = Pool()
    for j in range(n):
        texts, src, seen = [], [], set()
        for mi in model_lists[j]:
            for lw in lm_weights:
                for r, t in enumerate(gen[(split, j, mi)][lw][:nbest]):
                    if t not in seen:
                        seen.add(t)
                        texts.append(t)
                        src.append(f"{models[mi]['tag']}@{lw}#{r}")
        if not texts:
            texts, src = [''], ['empty#0']
        pool.add_utt(texts, src)
    return pool


# ---------------------------------------------------------------------------
# beam-search cache
# ---------------------------------------------------------------------------
# Generation is by far the most expensive stage (6h49m for 4610 decodes at beam 1200
# in the reference run, on ONE process because a 19 GB KenLM leaves room for no
# more). Everything after it is minutes. So the raw n-best output is written once as
# a small gzip pickle and reused whenever the fingerprint below still matches.
def gen_fingerprint(spec, models, val_ems, tasks):
    """What the cached decodes depend on: decode settings, which checkpoints, an
    emission signature (catches a different bundle with identical file names) and
    the exact task list."""
    return {'beam': int(spec['beam']), 'nbest': int(spec['nbest']),
            'lm_weights': [float(w) for w in spec['lm_weights']],
            'ckpts': [os.path.basename(str(M['ckpt'])) for M in models],
            'em_sig': [round(float(np.asarray(val_ems[M['tag']][0], dtype=np.float64).sum()), 1)
                       for M in models],
            'tasks_md5': hashlib.md5(repr([t[0] for t in tasks]).encode()).hexdigest()}


def load_gen_cache(paths, fp, n_tasks, log=print):
    """First cache file among `paths` whose fingerprint matches -> the GEN dict, else None."""
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with gzip.open(p, 'rb') as f:
                blob = pickle.load(f)
            if blob.get('fp') == fp and len(blob.get('gen', {})) == n_tasks:
                log(f'loaded cached beam-search results from {p} -> skipping the decode')
                return blob['gen']
            log(f'[gen cache] {p} does not match this run (beam/nbest/lm_weights/checkpoints/tasks '
                f'differ) -> ignoring')
        except Exception as e:
            log(f'[gen cache] could not read {p}: {e!r}')
    return None


def save_gen_cache(path, fp, gen, log=print):
    """Atomic write; attach the file as a Kaggle dataset (or pass --gen_cache) to reuse it."""
    try:
        with gzip.open(path + '.tmp', 'wb', compresslevel=3) as f:
            pickle.dump({'fp': fp, 'gen': gen}, f, protocol=4)
        os.replace(path + '.tmp', path)
        log(f'saved decode cache -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)')
    except Exception as e:
        log(f'[gen cache] save failed (continuing): {e!r}')


# ---------------------------------------------------------------------------
# weight grids
# ---------------------------------------------------------------------------
def mk_axis(spec):
    lo, hi, st = spec
    return np.round(np.arange(lo, hi + 1e-9, st), 3)


def stage1_axes():
    """(KenLM weight axis, word-bonus axis) of config.STAGE1_GRID."""
    return mk_axis(STAGE1_GRID['ng']), mk_axis(STAGE1_GRID['nw'])


def stage2_axes():
    """(KenLM, LLM, word-bonus) axes of config.STAGE2_GRID."""
    return mk_axis(STAGE2_GRID['ng']), mk_axis(STAGE2_GRID['llm']), mk_axis(STAGE2_GRID['nw'])


def on_edge(W, A, B=None, G=None):
    """True if a tuned weight sits on the boundary of its own search grid.

    A tuner that stops at a boundary has not found an optimum, it has run out of
    grid. decode_llm.py prints a warning whenever a final weight lands on one.
    """
    f = []
    if A is not None:
        f.append(W['ng'] <= A[0] + 1e-9 or W['ng'] >= A[-1] - 1e-9)
    if B is not None and len(B) > 1:
        f.append(W['llm'] <= B[0] + 1e-9 or W['llm'] >= B[-1] - 1e-9)
    if G is not None:
        f.append(W['nw'] <= G[0] + 1e-9 or W['nw'] >= G[-1] - 1e-9)
    return any(f)


def stage1_on(pool, errs, nref, idx, A, G, smooth=True):
    """Tune (kenlm weight, word bonus) with the LLM term switched off."""
    tu = Tuner(pool, errs, nref, idx)
    err = tu.grid(A, [0.0], G)
    W, Wraw = pick_from_grid(err, A, [0.0], G, smooth=smooth)
    return W, Wraw, err


def oracle_wer(errs, nref, idx):
    """Best achievable WER if an oracle picked the best candidate in the pool —
    the ceiling any reranker is working against."""
    return sum(errs[j].min() for j in idx) / max(nref[idx].sum(), 1)
