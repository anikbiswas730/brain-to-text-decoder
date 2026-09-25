"""
src/sweep.py — BLOCK 4.2b: the joint (n-best x KenLM-weight subset) pool sweep.

The beam search runs ONCE, at n-best = max(nbest_sweep) and at every KenLM weight
that may be needed. Because flashlight returns n-best lists best-first, "n-best 25"
is just the first 25 entries of the n-best-100 list, and a pool built from a subset
of KenLM weights is just a subset of the (model, weight) lists. So every
(subset, n-best) combination is a *view* of one maximal pool — nothing here decodes.

How a combination is scored
---------------------------
1. One maximal pool is built (all weights x n-best max) and the exact features
   (CTC log-likelihood averaged over out-of-fold models, KenLM ln P, #words) and
   per-candidate word errors are computed ONCE per unique (utterance, text).
2. A combination = per utterance, the union of the first `nb` entries of each
   selected (model, weight) list -> integer indices into the maximal pool.
3. Its score is SESSION-GROUPED K-FOLD CROSS-VALIDATION over the tune folds:
   stage-1 weights (KenLM, word bonus) are fit on K-1 groups of recording
   sessions and the held-out group is decoded with them. Picking the minimum of
   ~500 same-data "fit and score" numbers would be badly optimistic; CV is much
   less so. The grid error is additive over utterances, so each fold's grid is
   computed once and "fit on the other K-1 folds" is simply (total - fold).
4. Guard: the pinned pool (config gen_lm_weights x gen_nbest) is replaced only
   if the CV winner beats it by >= sweep_se_mult paired standard errors.

Fold C is evaluated for the top rows and the pinned pool so you can SEE whether a
choice transfers — it never picks anything.

Reference run: all 508 combinations were evaluated in 4 min; 24 of them tied at
the CV minimum 4.985% (paired gap 0.00 pp), the pinned pool was one of them, so it
was kept. Pools built from >= 5 of the 7 weights sit on that plateau; a single
weight costs 0.2 pp (lm 4.5) up to 3.4 pp (lm 0.5).
"""

import itertools
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.decoding import Pool, Tuner, errors_lists, fill_features, pick_from_grid


def _all_subsets(ws):
    return [c for k in range(1, len(ws) + 1) for c in itertools.combinations(ws, k)]


def _structured_subsets(ws, base):
    """Budget fallback: singletons, contiguous windows, leave-one-out, all, and the pinned set."""
    n = len(ws)
    S = {tuple(ws)} | {(w,) for w in ws} | {tuple(ws[i:j]) for i in range(n) for j in range(i + 1, n + 1)}
    S |= {tuple(w for w in ws if w != x) for x in ws if n > 1}
    S.add(tuple(base))
    return sorted(S, key=lambda s: (len(s), s))


def session_folds(tune_idx, sessions, k):
    """Greedy balanced assignment of whole sessions to k folds (biggest sessions first)."""
    groups = {}
    for j, s in zip(tune_idx, sessions):
        groups.setdefault(s, []).append(j)
    k = max(1, min(int(k), len(groups)))
    folds = [[] for _ in range(k)]
    for _, js in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        folds[int(np.argmin([len(x) for x in folds]))] += js
    return [sorted(x) for x in folds], len(groups)


def joint_sweep(gen, models, oof, val_refs, val_sessions, tune_idx, verify_idx, sweep_lw, base_lw,
                nbest_list, A, G, lex, ngram, utt_ems_val, device, cv_folds=5, se_mult=1.0,
                budget_min=40.0, smooth=True, human_time=None, log=print):
    """Run the sweep. Returns (table: DataFrame, best_lw: list, best_nbest: int, info: dict).

    gen          {(split, j, mi): {lm_weight: [texts best-first]}} from run_beam_pool
    oof          per val trial, the indices of the models that never trained on it
    sweep_lw     every decoded KenLM weight (incl. the screening weight)
    base_lw      the pinned pool's weights (config gen_lm_weights)
    nbest_list   n-best values to try; max must be <= what was decoded
    A, G         stage-1 grid axes (KenLM weight, word bonus)
    """
    ht = human_time or (lambda s: f'{s:.0f}s')
    n_val = len(val_refs)
    sweep_lw = sorted(float(w) for w in sweep_lw)
    nb_list = sorted(int(n) for n in nbest_list)
    nb_max = max(nb_list)
    base_lw = tuple(sorted(float(w) for w in base_lw))
    base_combo = (base_lw, nb_max)
    t_all = time.time()

    # ---- 1) maximal pool: unique texts per utterance + rank -> index maps -------------------
    MP, MP_IDX = Pool(), []
    for j in range(n_val):
        texts, src, pos, idx_j = [], [], {}, {}
        for mi in oof[j]:
            for lw in sweep_lw:
                ids = []
                for r, t in enumerate(gen[('val', j, mi)][lw][:nb_max]):
                    if t not in pos:
                        pos[t] = len(texts)
                        texts.append(t)
                        src.append(f"{models[mi]['tag']}@{lw}#{r}")
                    ids.append(pos[t])
                idx_j[(mi, lw)] = np.asarray(ids, dtype=np.int64)       # rank order, best first
        if not texts:
            texts, src = [''], ['empty#0']
        MP.add_utt(texts, src)
        MP_IDX.append(idx_j)
    log(f'maximal pool: {len(sweep_lw)} lm_weights x nbest {nb_max}: val mean '
        f'{np.mean([len(t) for t in MP.texts]):.1f} unique candidates/utt (features computed once)')
    fill_features(MP, utt_ems_val, lex, ngram, device, log=log)
    MP_ERRS, MP_NREF = errors_lists(MP, val_refs)
    log(f'features + edit distances for the maximal pool: {ht(time.time() - t_all)}')

    # ---- 2) a combination = an index view of the maximal pool -------------------------------
    def combo_view(subset, nb, sel):
        v = SimpleNamespace(texts=[None] * n_val, am=[None] * n_val, ng=[None] * n_val,
                            nw=[None] * n_val, llm=[None] * n_val)
        errs = [None] * n_val
        for u in sel:
            parts = [MP_IDX[u][(mi, lw)][:nb] for mi in oof[u] for lw in subset]
            c = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
            if c.size == 0:
                c = np.array([0])
            v.texts[u] = c
            v.am[u], v.ng[u], v.nw[u] = MP.am[u][c], MP.ng[u][c], MP.nw[u][c]
            v.llm[u] = np.full(c.size, np.nan)
            errs[u] = MP_ERRS[u][c]
        return v, errs

    # ---- session-grouped CV folds over TUNE ----------------------------------------------------
    fold_idx, n_sess = session_folds(tune_idx, val_sessions, cv_folds)
    K = len(fold_idx)
    n_tune = float(MP_NREF[tune_idx].sum())
    log(f'CV: {K} session-grouped folds over {n_sess} tune sessions, utterances/fold '
        f'{[len(x) for x in fold_idx]}')
    VEC = {}          # (subset, nbest) -> held-out word errors per utterance (fold order), for paired SEs

    def eval_combo(subset, nb, with_verify=False):
        tus, grids = [], []
        for idx in fold_idx:
            v, errs = combo_view(subset, nb, idx)
            tu = Tuner(v, errs, MP_NREF, idx)
            tus.append((tu, v, idx))
            grids.append(tu.grid(A, [0.0], G))
        total = np.sum(grids, axis=0)                                   # error grid over ALL tune utterances
        W, _ = pick_from_grid(total, A, [0.0], G, smooth=smooth)        # final weights: fit on all tune
        vecs, cv_err = [], 0.0
        for f, (tu, v, idx) in enumerate(tus):
            Wf = W if K == 1 else pick_from_grid(total - grids[f], A, [0.0], G, smooth=smooth)[0]
            S = tu.AM + Wf['ng'] * tu.NG + Wf['nw'] * tu.NW              # held-out fold, weights fit on the rest
            pick = S.argmax(1)
            e = tu.E[np.arange(len(pick)), pick]
            vecs.append(e)
            cv_err += float(e.sum())
        VEC[(tuple(subset), nb)] = np.concatenate(vecs)
        row = {'nbest': nb, 'n_lw': len(subset), 'lm_weights': ' '.join(f'{w:g}' for w in subset),
               'cv_wer_%': cv_err / n_tune * 100, 'tune_wer_%': W['err'] / n_tune * 100,
               'ng': W['ng'], 'nw': W['nw'],
               'mean_pool': float(np.mean([len(v.texts[u]) for _, v, idx in tus for u in idx]))}
        if with_verify and verify_idx:
            vv, ve = combo_view(subset, nb, verify_idx)
            tv = Tuner(vv, ve, MP_NREF, verify_idx)
            row['verify_wer_%'] = tv.errors(W['ng'], 0.0, W['nw'])[0] / tv.N * 100
        return row

    # ---- 3) sweep ------------------------------------------------------------------------------
    subsets = _all_subsets(sweep_lw)
    n_full = len(subsets)
    combos = [(s, nb) for s in subsets for nb in nb_list]
    t = time.time()
    first = eval_combo(*base_combo, with_verify=True)
    dt = time.time() - t
    budget = float(budget_min) * 60
    if dt * len(combos) > budget:
        subsets = _structured_subsets(sweep_lw, base_lw)
        combos = [(s, nb) for s in subsets for nb in nb_list]
        log(f'full power-set sweep would take ~{ht(dt * n_full * len(nb_list))} (> {budget / 60:.0f} min '
            f'budget) -> structured subsets only (singletons, contiguous windows, leave-one-out, all)')
    log(f'joint sweep: {len(subsets)} lm_weight subsets x {len(nb_list)} nbest = {len(combos)} combos '
        f'(~{ht(dt * len(combos))} est.)')
    rows = []
    for i, (s_, nb_) in enumerate(combos):
        rows.append(first if (s_, nb_) == base_combo else eval_combo(s_, nb_))
        if (i + 1) % 100 == 0:
            log(f'  {i + 1}/{len(combos)} combos | best CV WER so far {min(r["cv_wer_%"] for r in rows):.2f}%')
    SW = pd.DataFrame(rows).sort_values(['cv_wer_%', 'mean_pool']).reset_index(drop=True)
    lwkey = lambda sub: ' '.join(f'{w:g}' for w in sub)                                   # noqa: E731
    rowkey = lambda r: (tuple(float(x) for x in r['lm_weights'].split()), int(r['nbest']))  # noqa: E731
    is_base = (SW['lm_weights'] == lwkey(base_lw)) & (SW['nbest'] == nb_max)

    # fold C (report only) for the top rows and for the pinned pool
    need = pd.concat([SW.head(10), SW[is_base]]).drop_duplicates(subset=['lm_weights', 'nbest'])
    if 'verify_wer_%' not in SW:
        SW['verify_wer_%'] = np.nan
    for _, r in need.iterrows():
        m = (SW['lm_weights'] == r['lm_weights']) & (SW['nbest'] == r['nbest'])
        SW.loc[m, 'verify_wer_%'] = eval_combo(rowkey(r)[0], int(r['nbest']),
                                               with_verify=True).get('verify_wer_%', np.nan)
    best_key = rowkey(SW.iloc[0])
    vb = VEC[best_key]

    def paired(key):
        """gap (pp) of `key` vs the CV-best combo on held-out errors, and its paired SE (pp)."""
        d = VEC[key] - vb
        return d.sum() / n_tune * 100, (np.sqrt(len(d)) * d.std(ddof=1)) / n_tune * 100

    SW['gap_vs_best_pp'] = np.nan
    SW['gap_se_pp'] = np.nan
    for i in SW.index[:10].tolist() + SW.index[is_base].tolist():
        SW.loc[i, ['gap_vs_best_pp', 'gap_se_pp']] = paired(rowkey(SW.loc[i]))
    SW = SW.round(3)
    base_row = SW[is_base]
    show = pd.concat([SW.head(10), base_row]).drop_duplicates(subset=['lm_weights', 'nbest'])
    log('\nTop combos by session-CV WER on TUNE (verify = fold C, report only) + pinned pool:')
    log(show[['nbest', 'lm_weights', 'cv_wer_%', 'tune_wer_%', 'verify_wer_%', 'gap_vs_best_pp',
              'gap_se_pp', 'mean_pool']].to_string(index=False))
    log('\nbest CV WER per nbest:\n' + SW.groupby('nbest')['cv_wer_%'].min().to_string())

    # ---- 4) adopt the winner, unless the pinned pool is within se_mult SE of it ------------------
    best = SW.iloc[0]
    kept_pinned = False
    if len(base_row) and rowkey(base_row.iloc[0]) != best_key and se_mult > 0:
        gap, se = paired(base_combo)
        if gap <= se_mult * se:
            best, kept_pinned = base_row.iloc[0], True
            log(f'pinned pool is within {se_mult:g} SE of the CV-best (gap {gap:.2f} pp, SE {se:.2f} pp) '
                f'-> keeping the pinned pool')
        else:
            log(f'CV-best beats the pinned pool by {gap:.2f} pp (SE {se:.2f} pp, >{se_mult:g} SE) -> switching')
    best_lw = sorted(float(x) for x in best['lm_weights'].split())
    best_nb = int(best['nbest'])
    info = {'cv_wer_%': float(best['cv_wer_%']), 'tune_wer_%': float(best['tune_wer_%']),
            'verify_wer_%': float(best.get('verify_wer_%', np.nan)), 'n_combos': len(combos),
            'kept_pinned': bool(kept_pinned or rowkey(best) == base_combo),
            'pinned_cv_wer_%': float(base_row['cv_wer_%'].iloc[0]) if len(base_row) else float('nan'),
            'pinned_verify_wer_%': float(base_row['verify_wer_%'].iloc[0]) if len(base_row) else float('nan'),
            'seconds': round(time.time() - t_all, 1)}
    log(f'\nCHOSEN: lm_weights {best_lw} | nbest {best_nb} | CV {info["cv_wer_%"]:.2f}% | in-sample tune '
        f'{info["tune_wer_%"]:.2f}% | verify(C) {info["verify_wer_%"]:.2f}% | pinned pool: CV '
        f'{info["pinned_cv_wer_%"]:.2f}%, verify {info["pinned_verify_wer_%"]:.2f}%')
    log(f'note: CV is a max over {len(combos)} combos so it is still slightly optimistic; '
        f'fold C is the honest check.')
    return SW, best_lw, best_nb, info
