"""
decode_llm.py — the full v6 decoding pipeline: trained cross-fit checkpoints in,
`submission.csv` out.

    python decode_llm.py --checkpoint_dir /path/to/b2t_v6_ckpt \
                         --output submission/submission.csv
    python decode_llm.py --no_sweep            # single static decode setting
    python decode_llm.py --no_llm              # acoustic + n-gram rescoring only
    python decode_llm.py --llm_name Qwen/Qwen2.5-1.5B --llm_batch 32   # cheap run

Stages (the BLOCK 4.x sequence of the notebook):

  4.1  locate the checkpoint bundle, route never-trained days through the generic
       adapter slot, screen each fold's checkpoints by REAL beam WER, extract
       emissions for val and test, write a greedy fallback submission
  4.2  generate the candidate pool ONCE, at the union of every KenLM weight the
       sweep may ask for and at max(nbest) — subsets are then free, because
       n-best lists are nested
  4.2b sweep the pool axes (lm-weight set x nbest x beam x expansion x stage-1
       grid range) by coordinate descent, selecting on the TUNE folds only
  4.3  exact features (CTC / KenLM / #words), stage-1 weights, phonetic expansion
  4.4  task-adapted LLM: QLoRA next-word-prediction fine-tune on the training
       transcripts, then score the top-K candidates per utterance
  4.5  stage-2 sweep (grid range x tol_rel x smoothing x llm_topk x fluency gate),
       verify on the untouched fold C, refit on all out-of-fold trials
  4.6  emit the test predictions under exactly the same pool / features / weights

Two rules the whole file is built around:

  * **Selection happens on A u B only.** Fold C rides along in every table so you
    can see whether a choice transferred, and is never selected on. With this
    many swept axes on ~570 tune trials, some tune noise WILL get fitted — read
    the C column, not the tune column, as the estimate of what generalises.
  * **Nothing is allowed to lose the submission.** Every stage after the first is
    soft: it logs its traceback and the run continues with the best artefact
    produced so far, and a valid `submission.csv` exists from stage 4.1 onward.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from glob import glob

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from src.dataset import CacheReader, build_cache, make_crossfit_split
from src.decoding import (LLMScorer, NgramScorer, PronLexicon, Tuner, errors_lists, expand_pool,
                          finetune_llm_nwp, fluency, pick_from_grid, predict_texts, run_beam_pool,
                          subpool, topk_masks)
from src.metrics import (bootstrap_wer_sd, greedy_collapse, load_lexicon, official_wer,
                         phones_to_words, remove_punctuation)
from src.inference import (FeatCache, build_day_override, build_pool, extract_emissions,
                           find_checkpoint_dir, group_checkpoints, oof_models, on_edge,
                           oracle_wer, screen_checkpoints, stage1_axes, stage1_on, stage2_axes)
from src.utils import (figure_guard, find_file, get_session2idx, human_time, init_matplotlib,
                       pick_cache_dir, resolve_competition_path, resolve_kaggle_dataset, save_fig,
                       save_table, set_seed)

T0 = time.time()
STAGE_FAILED = []


def soft(name):
    """Decorator-free stage guard: `with soft('4.4 LLM'):` — a failure is logged
    and recorded, the pipeline continues with whatever it already has."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        try:
            yield
        except Exception as e:
            STAGE_FAILED.append(name)
            print('\n' + '!' * 90)
            print(f'SOFT STAGE FAILED: {name} ({e!r}) - the pipeline continues.\n'
                  + traceback.format_exc() + '!' * 90)
    return _ctx()


def parse_args():
    p = argparse.ArgumentParser(description='Brain-to-Text v6 decoding + LLM rescoring')
    p.add_argument('--data_dir', default=C.DEFAULT_DATA_DIR)
    p.add_argument('--checkpoint_dir', default=None,
                   help='bundle written by train.py (defaults to config.PRETRAINED_CKPT_DIR)')
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--output', default=os.path.join(C.DEFAULT_SUBMISSION_DIR, 'submission.csv'))
    p.add_argument('--figures_dir', default=C.DEFAULT_FIGURES_DIR)
    p.add_argument('--tables_dir', default=C.DEFAULT_TABLES_DIR)
    p.add_argument('--lexicon', default=None)
    p.add_argument('--tokens', default=None)
    p.add_argument('--kenlm_binary', default=None)
    p.add_argument('--beam', type=int, default=None, help='override LLM_CFG.gen_beam')
    p.add_argument('--nbest', type=int, default=None)
    p.add_argument('--n_proc', type=int, default=None, help='beam-search processes (RAM-capped)')
    p.add_argument('--llm_name', default=None)
    p.add_argument('--llm_batch', type=int, default=None)
    p.add_argument('--no_llm', action='store_true')
    p.add_argument('--no_sweep', action='store_true')
    p.add_argument('--no_finetune', action='store_true')
    p.add_argument('--session_hours', type=float, default=C.SESSION['session_hours'])
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(C.SEED)
    plt = init_matplotlib()
    for d in (args.figures_dir, args.tables_dir, os.path.dirname(args.output) or '.'):
        os.makedirs(d, exist_ok=True)

    cfg = dict(C.LLM_CFG)
    for k, v in [('gen_beam', args.beam), ('gen_nbest', args.nbest), ('n_proc', args.n_proc),
                 ('llm_name', args.llm_name), ('llm_batch', args.llm_batch)]:
        if v is not None:
            cfg[k] = v
    if args.no_llm:
        cfg['use_llm'] = False
    if args.no_finetune:
        cfg['llm_finetune'] = False
    cfg['preference'] = C.CKPT_PREFERENCE
    sweep = dict(C.SWEEP_CFG)
    if args.no_sweep:
        sweep['enable'] = False

    device = C.resolve_device()
    n_gpus = torch.cuda.device_count()

    # ---- data -------------------------------------------------------------
    data_dir = resolve_competition_path(args.data_dir, C.COMPETITION_SLUG)
    session2idx = get_session2idx(data_dir)
    idx2session = {v: k for k, v in session2idx.items()}
    n_days = len(session2idx)
    cache_dir = pick_cache_dir([args.cache_dir] if args.cache_dir else C.CACHE_CANDIDATES,
                               C.CACHE_NEED_GB)
    build_cache(data_dir, cache_dir)
    readers = {s: CacheReader(cache_dir, s) for s in ('train', 'val', 'test')}
    n_val, n_test = len(readers['val']), len(readers['test'])
    val_refs = [m['sentence'] for m in readers['val'].meta]
    print(f'data: {data_dir} | {n_days} sessions | val {n_val} | test {n_test}')

    # ---- assets -----------------------------------------------------------
    if args.lexicon:
        lexicon_path = args.lexicon
        tokens_path = args.tokens or os.path.join(os.path.dirname(args.lexicon), 'tokens.txt')
    else:
        lex_ds = resolve_kaggle_dataset(C.LEXICON_DATASET_SLUG)
        lexicon_path, tokens_path = find_file(lex_ds, 'lexicon.txt'), find_file(lex_ds, 'tokens.txt')
    kenlm_path = args.kenlm_binary or find_file(resolve_kaggle_dataset(C.KENLM_DATASET_SLUG), '*.bin')
    print('lexicon:', lexicon_path, '\nkenlm  :', kenlm_path)

    # =======================================================================
    # 4.1 checkpoints, day override, screening, emissions
    # =======================================================================
    ckpt_dir = find_checkpoint_dir(args.checkpoint_dir, C.PRETRAINED_CKPT_DIR,
                                   C.DEFAULT_CHECKPOINT_DIR, '/kaggle/input')
    if ckpt_dir is None:
        raise FileNotFoundError('No fold*_seed*_*.pt found. Run train.py first, then pass '
                                '--checkpoint_dir (on Kaggle: attach the training output as a dataset).')
    print('checkpoints:', ckpt_dir)
    norm = torch.load(find_file(ckpt_dir, 'norm_stats.pt'), weights_only=False)
    try:
        sil_info = json.load(open(find_file(ckpt_dir, 'sil_convention.json')))
    except Exception:
        sil_info = {'sil_trailing': True}
    clip = C.TRAIN_CFG['clip']

    # the split must be the one training used, or "out of fold" means nothing
    try:
        split = json.load(open(find_file(ckpt_dir, 'split.json')))
        print('split: loaded from the checkpoint bundle')
    except Exception as e:
        print('split.json not next to the checkpoints -> regenerating deterministically:', repr(e))
        split = {'labels': make_crossfit_split(readers['val'].meta, C.CROSSFIT_PATTERN, C.SEED)}
    val_labels = [split['labels'].get(m['key'], 'C') for m in readers['val'].meta]
    print('val split counts:', pd.Series(val_labels).value_counts().to_dict())

    day_override, _ = build_day_override(ckpt_dir, n_days, idx2session)
    groups = group_checkpoints(ckpt_dir, C.CKPT_PREFERENCE)
    print('checkpoint groups:', {k: sorted(v) for k, v in groups.items()})
    models, screen_rows = screen_checkpoints(groups, readers, val_labels, session2idx, norm, clip,
                                             device, lexicon_path, tokens_path, kenlm_path, cfg,
                                             C.SEED, day_override)
    if not models:
        raise RuntimeError('no loadable acoustic checkpoint')
    if screen_rows:
        save_table(pd.DataFrame(screen_rows), 'checkpoint_screening.csv', args.tables_dir)

    val_ems, test_ems = {}, {}
    for M in models:
        val_ems[M['tag']] = extract_emissions(M['model'], readers, 'val', list(range(n_val)),
                                              session2idx, norm, clip, device, day_override)
        test_ems[M['tag']] = extract_emissions(M['model'], readers, 'test', list(range(n_test)),
                                               session2idx, norm, clip, device, day_override)
    oof = oof_models(models, val_labels)
    n_no_oof = sum(1 for o in oof if not o)
    if n_no_oof:
        print(f'WARNING: {n_no_oof} val trials were trained on by every model (final-fit run?) '
              f'-> excluded from tuning.')

    tune_idx = [j for j in range(n_val) if oof[j] and val_labels[j] in cfg['tune_labels']]
    verify_idx = [j for j in range(n_val) if oof[j] and val_labels[j] in cfg['verify_labels']]
    all_idx = [j for j in range(n_val) if oof[j]]
    if not tune_idx:
        print('WARNING: no out-of-fold A/B trials -> tuning on all OOF trials')
        tune_idx = all_idx

    def wer_on(idx, hyps):
        hyps = hyps if isinstance(hyps, dict) else dict(enumerate(hyps))
        return official_wer([val_refs[j] for j in idx], [hyps[j] for j in idx])[0]

    # greedy reference rows + fallback submission #1
    _, pron2words = load_lexicon(lexicon_path)

    def greedy_text(em):
        ids = greedy_collapse(torch.from_numpy(em.astype(np.float32))[:, None, :], [em.shape[0]])[0]
        return ids, phones_to_words(ids, pron2words)

    greedy_val = [''] * n_val
    for j in range(n_val):
        if oof[j]:
            greedy_val[j] = greedy_text(val_ems[models[oof[j][0]]['tag']][j])[1]
    pd.DataFrame({'id': range(n_test),
                  'text': [greedy_text(test_ems[models[0]['tag']][j])[1] for j in range(n_test)]}
                 ).to_csv(args.output, index=False)
    print(f'OOF greedy-lexicon WER {wer_on(all_idx, greedy_val) * 100:.2f}% '
          f'| fallback submission #1 written (greedy lexicon)')

    # =======================================================================
    # 4.2 candidate generation (ONE pass at the union of everything swept)
    # =======================================================================
    lw_union = set(cfg['gen_lm_weights']) | {cfg['screen_lm_weight']}
    for s in (sweep.get('lm_weight_sets', []) if sweep.get('enable') else []):
        lw_union |= set(float(w) for w in s)
    gen_lw = sorted(lw_union)
    nbest_max = int(max([cfg['gen_nbest']] + [int(n) for n in (sweep.get('nbest', [])
                                                               if sweep.get('enable') else [])]))
    beam_grid = sorted(set([int(cfg['gen_beam'])] + [int(b) for b in (sweep.get('beam', [])
                                                                      if sweep.get('enable') else [])]))

    tasks = [(('val', j, mi), val_ems[models[mi]['tag']][j]) for j in range(n_val) for mi in oof[j]]
    tasks += [(('test', j, mi), test_ems[models[mi]['tag']][j])
              for j in range(n_test) for mi in range(len(models))]

    gen_by_beam = {}
    for bi, beam in enumerate(beam_grid):
        left = args.session_hours * 60 - (time.time() - T0) / 60
        if bi and left < sweep.get('min_minutes_left_for_extra_beam', 0):
            print(f'skipping beam={beam}: only {left:.0f} min of session left')
            continue
        spec = dict(lexicon=lexicon_path, tokens=tokens_path, lm=kenlm_path, beam=beam,
                    nbest=nbest_max, lm_weights=gen_lw)
        print(f'beam search: {len(tasks)} decodes x {len(gen_lw)} lm_weights {gen_lw}, '
              f'beam={beam}, nbest={nbest_max}, {cfg["n_proc"]} processes')
        t0 = time.time()
        gen_by_beam[beam] = run_beam_pool(tasks, spec, cfg['n_proc'], desc=f'generate b{beam}')
        print(f'  beam={beam} generation took {human_time(time.time() - t0)}')
    if not gen_by_beam:
        raise RuntimeError('no beam width could be generated')

    beam_sel, lw_sel, nbest_sel = max(gen_by_beam), list(gen_lw), nbest_max
    gen = gen_by_beam[beam_sel]
    pool_val = build_pool(gen, models, 'val', n_val, oof, lw_sel, nbest_sel)
    pool_test = build_pool(gen, models, 'test', n_test, [list(range(len(models)))] * n_test,
                           lw_sel, nbest_sel)
    lw0 = cfg['screen_lm_weight']
    base_val = [(gen[('val', j, oof[j][0])][lw0] or [''])[0] if oof[j] else '' for j in range(n_val)]
    base_test = [(gen[('test', j, 0)][lw0] or [''])[0] for j in range(n_test)]
    pd.DataFrame({'id': range(n_test), 'text': base_test}).to_csv(args.output, index=False)
    print(f'flashlight 1-best (lm_weight={lw0}): tune {wer_on(tune_idx, base_val) * 100:.2f}% | '
          f'verify(C) {wer_on(verify_idx, base_val) * 100:.2f}% | '
          f'all OOF {wer_on(all_idx, base_val) * 100:.2f}%  | fallback submission #2 written')

    # ---- shared objects for every rescoring stage --------------------------
    lex = PronLexicon(lexicon_path, sil_trailing=sil_info['sil_trailing'])
    try:
        ngram = NgramScorer(kenlm_path)
    except Exception as e:
        ngram = None
        print('WARNING: kenlm python module unavailable -> n-gram feature disabled:', repr(e))
    utt_ems_val = [[val_ems[models[mi]['tag']][j] for mi in oof[j]] for j in range(n_val)]
    utt_ems_test = [[test_ems[M['tag']][j] for M in models] for j in range(n_test)]
    s1_grid = (sweep['stage1_grid'][0] if sweep.get('enable') else 'base')
    A1, G1 = stage1_axes(s1_grid)
    fcv = FeatCache(n_val, enabled=bool(sweep.get('cache_features', True)))
    fct = FeatCache(n_test, enabled=False)       # the test pool is built once
    sweep_target = None

    # =======================================================================
    # 4.2b pool-side sweep
    # =======================================================================
    if sweep.get('enable'):
        with soft('4.2b pool sweep'):
            rows, seen_variants, t_sw = [], {}, time.time()

            def eval_variant(beam, lw_set, nbest, exp, tag='', s1g=None):
                s1g = s1g or s1_grid
                key = (int(beam), tuple(float(w) for w in lw_set), int(nbest), int(exp['top']),
                       int(exp['max_nb']), int(exp['max_new']), s1g)
                if key in seen_variants:
                    return seen_variants[key]
                _A, _G = stage1_axes(s1g)
                t0 = time.time()
                pv = build_pool(gen_by_beam[beam], models, 'val', n_val, oof, lw_set, nbest)
                fcv.fill(pv, utt_ems_val, lex, ngram, device, log=lambda *a: None)
                errs, nref = errors_lists(pv, val_refs)
                W, _, _ = stage1_on(pv, errs, nref, tune_idx, _A, _G, cfg['smooth_grid'])
                if exp['top'] > 0 and cfg['expansion']:
                    st = expand_pool(pv, lex, ngram, W, exp['top'], exp['max_nb'], exp['max_new'])
                    fcv.fill(pv, utt_ems_val, lex, ngram, device, only_new_from=st,
                             log=lambda *a: None)
                    errs, nref = errors_lists(pv, val_refs)
                    W, _, _ = stage1_on(pv, errs, nref, tune_idx, _A, _G, cfg['smooth_grid'])
                pred = predict_texts(pv, W)[0]
                row = {'beam': int(beam), 'nbest': int(nbest), 'lm_weights': repr(list(lw_set)),
                       'expand_top': int(exp['top']), 'expand_max_nb': int(exp['max_nb']),
                       'expand_max_new': int(exp['max_new']), 'stage1_grid': s1g,
                       'edge': bool(on_edge(W, _A, None, _G)),
                       'pool_mean': round(float(np.mean([len(t) for t in pv.texts])), 1),
                       'tune_WER_%': round(100 * wer_on(tune_idx, pred), 3),
                       'verify_C_WER_%': (round(100 * wer_on(verify_idx, pred), 3)
                                          if verify_idx else float('nan')),
                       'oracle_tune_%': round(100 * oracle_wer(errs, nref, tune_idx), 3),
                       'secs': round(time.time() - t0, 1), 'stage': tag, '_pred': pred}
                rows.append(row)
                seen_variants[key] = row
                print(f"  [{tag}] beam={beam} nbest={nbest:>3} lw={row['lm_weights']:<28} "
                      f"exp=({exp['top']},{exp['max_nb']},{exp['max_new']}) s1={s1g:<4} "
                      f"pool={row['pool_mean']:>6.1f}{' EDGE' if row['edge'] else '     '} | "
                      f"tune {row['tune_WER_%']:.2f}% | C {row['verify_C_WER_%']:.2f}% | "
                      f"oracle {row['oracle_tune_%']:.2f}% | {row['secs']:.0f}s")
                del pv
                gc.collect()
                return row

            def budget_left():
                return sweep['time_budget_min'] - (time.time() - t_sw) / 60

            BEAMS = sorted(gen_by_beam)
            NBESTS = sorted(set(int(n) for n in sweep['nbest'] if int(n) <= nbest_max))
            LWSETS = [sorted(set(float(w) for w in s) & set(gen_lw)) for s in sweep['lm_weight_sets']]
            LWSETS = [s for s in LWSETS if s] or [list(gen_lw)]
            LWSETS = list(dict.fromkeys(tuple(s) for s in LWSETS))
            ETOP = sorted(set(int(v) for v in sweep['expand_top']))
            ENB = sorted(set(int(v) for v in sweep['expand_max_nb']))
            ENEW = sorted(set(int(v) for v in sweep['expand_max_new']))
            S1G = [g for g in sweep['stage1_grid'] if g in C.STAGE1_GRIDS] or ['base']
            exp0 = {'top': cfg['expand_top'], 'max_nb': cfg['expand_max_nb'],
                    'max_new': cfg['expand_max_new']}
            print(f'sweep: beams {BEAMS} x nbest {NBESTS} x {len(LWSETS)} weight sets x stage-1 '
                  f'grid {S1G}, then expansion {ETOP} x {ENB} x {ENEW} | mode={sweep["mode"]} | '
                  f'budget {sweep["time_budget_min"]} min')

            if sweep['mode'] == 'grid':
                for beam in BEAMS:
                    for lw in LWSETS:
                        for nb in NBESTS:
                            for et in ETOP:
                                for en in ENB:
                                    for ew in (ENEW if et > 0 else ENEW[:1]):
                                        for sg in S1G:
                                            if budget_left() <= 0:
                                                break
                                            eval_variant(beam, list(lw), nb,
                                                         {'top': et, 'max_nb': en, 'max_new': ew},
                                                         'grid', sg)
            else:
                cur = {'beam': BEAMS[-1], 'lw': list(LWSETS[-1]), 'nbest': max(NBESTS),
                       's1grid': S1G[0], **exp0}
                for p in range(int(sweep.get('passes', 2))):
                    for axis in ('lw', 'nbest', 'beam', 'top', 'max_nb', 'max_new', 's1grid'):
                        if budget_left() <= 0:
                            print(f'  budget exhausted after pass {p + 1}, axis {axis}')
                            break
                        grid = {'lw': [list(s) for s in LWSETS], 'nbest': NBESTS, 'beam': BEAMS,
                                'top': ETOP, 'max_nb': ENB, 'max_new': ENEW, 's1grid': S1G}[axis]
                        if len(grid) < 2 and p > 0:
                            continue
                        if axis in ('max_nb', 'max_new') and cur['top'] == 0:
                            continue
                        cand = []
                        for v in grid:
                            c = dict(cur)
                            c[axis] = v
                            cand.append((v, eval_variant(c['beam'], c['lw'], c['nbest'],
                                                         {'top': c['top'], 'max_nb': c['max_nb'],
                                                          'max_new': c['max_new']},
                                                         f'p{p + 1}:{axis}', c['s1grid'])))
                        v, r = min(cand, key=lambda vr: vr[1]['tune_WER_%'])
                        if cur[axis] != v:
                            print(f'   -> {axis}: {cur[axis]} -> {v}  (tune {r["tune_WER_%"]:.2f}%)')
                        cur[axis] = v
                    if budget_left() <= 0:
                        break

            best = min(rows, key=lambda r: r['tune_WER_%'])
            df = pd.DataFrame([{k: v for k, v in r.items() if k != '_pred'} for r in rows])
            save_table(df.sort_values('tune_WER_%').reset_index(drop=True),
                       'decoding_sweep.csv', args.tables_dir)
            sd = bootstrap_wer_sd([val_refs[j] for j in tune_idx],
                                  [best['_pred'][j] for j in tune_idx],
                                  sweep.get('bootstrap', 0), C.SEED)
            print(f'\n{len(rows)} variants evaluated in {(time.time() - t_sw) / 60:.1f} min')
            print(f'best on TUNE: beam={best["beam"]} nbest={best["nbest"]} lw={best["lm_weights"]} '
                  f'expand=({best["expand_top"]},{best["expand_max_nb"]},{best["expand_max_new"]}) '
                  f'-> tune {best["tune_WER_%"]:.2f}% (bootstrap sd {sd * 100:.2f} pp), '
                  f'C {best["verify_C_WER_%"]:.2f}%')
            rival = [r for r in rows if r is not best and r['tune_WER_%'] - best['tune_WER_%'] < 100 * sd]
            if rival:
                print(f'  {len(rival)} other configs are within one bootstrap sd on tune - the '
                      f'margin is inside the noise, so do not read the winner as strictly better.')
            if best['edge']:
                print("NOTE: the winning stage-1 weights sit on the edge of their grid - widen "
                      "STAGE1_GRIDS['wide'] and re-run if you want that axis genuinely optimised.")

            beam_sel, lw_sel, nbest_sel = best['beam'], eval(best['lm_weights']), best['nbest']
            s1_grid = best['stage1_grid']
            A1, G1 = stage1_axes(s1_grid)
            cfg.update({'gen_beam': beam_sel, 'gen_nbest': nbest_sel, 'gen_lm_weights': lw_sel,
                        'expand_top': best['expand_top'], 'expand_max_nb': best['expand_max_nb'],
                        'expand_max_new': best['expand_max_new'],
                        'expansion': cfg['expansion'] and best['expand_top'] > 0})
            bestj = {k: best[k] for k in ('beam', 'nbest', 'lm_weights', 'expand_top',
                                          'expand_max_nb', 'expand_max_new', 'stage1_grid', 'edge',
                                          'tune_WER_%', 'verify_C_WER_%', 'pool_mean')}
            bestj['tune_bootstrap_sd_pp'] = round(100 * sd, 3) if sd == sd else None
            json.dump(bestj, open(os.path.join(args.tables_dir, 'decoding_sweep_best.json'), 'w'),
                      indent=1, default=float)
            for b in list(gen_by_beam):
                if b != beam_sel:
                    del gen_by_beam[b]
            gc.collect()
            gen = gen_by_beam[beam_sel]
            pool_val = build_pool(gen, models, 'val', n_val, oof, lw_sel, nbest_sel)
            pool_test = build_pool(gen, models, 'test', n_test, [list(range(len(models)))] * n_test,
                                   lw_sel, nbest_sel)
            base_val = [(gen[('val', j, oof[j][0])][lw0] or [''])[0] if oof[j] else ''
                        for j in range(n_val)]
            sweep_target = best['tune_WER_%']
            for r in rows:
                r.pop('_pred', None)
            gc.collect()

    # =======================================================================
    # 4.3 exact features, stage-1 weights, phonetic expansion
    # =======================================================================
    abl = {'greedy lexicon (1 model)': dict(enumerate(greedy_val)),
           f'flashlight 1-best lm={lw0} (1 model)': dict(enumerate(base_val))}
    t0 = time.time()
    fcv.fill(pool_val, utt_ems_val, lex, ngram, device)
    fct.fill(pool_test, utt_ems_test, lex, ngram, device)
    print(f'features: {human_time(time.time() - t0)} '
          f'(val feature cache: {fcv.hits} hits / {fcv.miss} computed)')

    errs, nref = errors_lists(pool_val, val_refs)
    W1, W1raw, err1 = stage1_on(pool_val, errs, nref, tune_idx, A1, G1, cfg['smooth_grid'])
    abl['pool rescoring: CTC-ens + KenLM + #words'] = predict_texts(pool_val, W1)[0]
    oracle = {'pool (no expansion)': (oracle_wer(errs, nref, tune_idx),
                                      oracle_wer(errs, nref, verify_idx) if verify_idx else float('nan'))}
    print(f'stage-1 weights (smoothed) {W1} | raw-best {W1raw}')
    W_stage1 = W1

    if cfg['expansion']:
        with soft('4.3 expansion'):
            t0 = time.time()
            stv = expand_pool(pool_val, lex, ngram, W1, cfg['expand_top'], cfg['expand_max_nb'],
                              cfg['expand_max_new'])
            fcv.fill(pool_val, utt_ems_val, lex, ngram, device, only_new_from=stv)
            stt = expand_pool(pool_test, lex, ngram, W1, cfg['expand_top'], cfg['expand_max_nb'],
                              cfg['expand_max_new'])
            fct.fill(pool_test, utt_ems_test, lex, ngram, device, only_new_from=stt)
            errs, nref = errors_lists(pool_val, val_refs)
            W1x, _, err1 = stage1_on(pool_val, errs, nref, tune_idx, A1, G1, cfg['smooth_grid'])
            abl['+ phonetic-neighbour expansion'] = predict_texts(pool_val, W1x)[0]
            oracle['pool + expansion'] = (oracle_wer(errs, nref, tune_idx),
                                          oracle_wer(errs, nref, verify_idx) if verify_idx else float('nan'))
            print(f'expansion: {human_time(time.time() - t0)} | pool mean '
                  f'{np.mean([len(t) for t in pool_val.texts]):.0f} | weights {W1x} | '
                  f'tune {wer_on(tune_idx, abl["+ phonetic-neighbour expansion"]) * 100:.2f}% | '
                  f'oracle tune {oracle["pool + expansion"][0] * 100:.2f}%')
            W_stage1 = W1x

    if sweep_target is not None:
        got = 100 * wer_on(tune_idx, abl.get('+ phonetic-neighbour expansion',
                                             abl['pool rescoring: CTC-ens + KenLM + #words']))
        print(f'sweep reproduction check: the sweep reported {sweep_target:.2f}% on tune, '
              f'this stage reproduces {got:.2f}%'
              + ('  OK' if abs(got - sweep_target) < 0.05 else '  <-- MISMATCH, investigate'))
        fcv.release()          # the sweep is over; give the RAM back before the LLM

    with figure_guard('stage1_surface'):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(err1[:, 0, :] / nref[tune_idx].sum() * 100, aspect='auto', origin='lower',
                       cmap='viridis', extent=[G1[0], G1[-1], A1[0], A1[-1]])
        ax.scatter([W_stage1['nw']], [W_stage1['ng']], c='red', marker='*', s=150)
        ax.set_xlabel('word bonus (per word)')
        ax.set_ylabel('KenLM weight (ln units)')
        ax.set_title('Stage-1 tune-set WER (%)')
        fig.colorbar(im, ax=ax)
        save_fig(fig, 'stage1_weight_surface', args.figures_dir)

    # =======================================================================
    # 4.4 task-adapted LLM
    # =======================================================================
    llm_ok, llm_hist = False, {}
    topk_max = (max([cfg['llm_topk']] + [int(k) for k in sweep['llm_topk']])
                if sweep.get('enable') else cfg['llm_topk'])
    mask_val = mask_test = None
    if cfg['use_llm'] and (n_gpus > 0):
        with soft('4.4 LLM'):
            for M in models:                  # emissions are cached; free the GPU for the LLM
                M['model'] = None
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.time()
            train_sents = [remove_punctuation(m['sentence']) for m in readers['train'].meta]
            if ngram is not None:
                ngram.m = None                # free the KenLM RAM (the score cache is kept)
                gc.collect()
            try:
                llm_model, llm_tok, llm_hist = finetune_llm_nwp(cfg['llm_name'], train_sents, cfg)
            except Exception as e:
                traceback.print_exc()
                print('LoRA fine-tuning failed -> using the base LLM without fine-tuning:', repr(e))
                gc.collect()
                torch.cuda.empty_cache()
                llm_model, llm_tok, llm_hist = finetune_llm_nwp(
                    cfg['llm_name'], train_sents, dict(cfg, llm_finetune=False))
            # single GPU only: copy.deepcopy() to a 2nd GPU clones on the CURRENT
            # device first, which briefly needs 2x the footprint on GPU0
            scorer = LLMScorer(llm_model, llm_tok, 1)
            print(f'LLM ready in {human_time(time.time() - t0)}: {llm_hist}')
            print(f'  scoring top-{topk_max} per utterance')
            mask_val = topk_masks(pool_val, W_stage1, topk_max)
            mask_test = topk_masks(pool_test, W_stage1, topk_max)
            for pool, masks in ((pool_val, mask_val), (pool_test, mask_test)):
                texts = sorted(set(pool.texts[u][i] for u in range(len(pool))
                                   for i in np.where(masks[u] > 0)[0]))
                t1 = time.time()
                lp, nt = scorer.score(texts, bs=cfg['llm_batch'])
                lut = dict(zip(texts, zip(lp, nt)))
                for u in range(len(pool)):
                    for i in np.where(masks[u] > 0)[0]:
                        pool.llm[u][i], pool.ntok[u][i] = lut[pool.texts[u][i]]
                print(f'  scored {len(texts)} unique hypotheses in {human_time(time.time() - t1)}')
            llm_ok = True
            json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in llm_hist.items()},
                      open(os.path.join(args.tables_dir, 'llm_finetune_history.json'), 'w'), indent=1)
    else:
        print('LLM disabled (use_llm=False or no GPU).')

    # =======================================================================
    # 4.5 stage-2 sweep, fluency gate, verify on C, refit
    # =======================================================================
    A2, B2, G2 = stage2_axes((sweep['stage2_grid'][0] if sweep.get('enable') else 'base'))

    def gate_grids(pool, e, idx, flu, percentiles, A, B, G):
        """Error grids per fluency percentile. Picking is separate, so tol_rel and
        smoothing can be swept without recomputing a single grid."""
        tu = Tuner(pool, e, nref, idx)
        f = flu[idx]
        out = []
        for p in percentiles:
            tau = -np.inf if p == 0 else float(np.percentile(f, p))
            lo, hi = np.where(f < tau)[0], np.where(f >= tau)[0]
            out.append({'percentile': p, 'tau': tau, 'n_lo': len(lo),
                        'hi_grid': tu.grid(A, B, G, rows=hi),
                        'lo_grid': tu.grid(A, B, G, rows=lo) if len(lo) else None})
        return out

    def pick_gate(g, A, B, G, smooth, tol, nref_sum):
        W_hi, _ = pick_from_grid(g['hi_grid'], A, B, G, smooth=smooth, tol_rel=tol)
        W_lo, e_lo = None, 0.0
        if g['lo_grid'] is not None:
            W_lo, _ = pick_from_grid(g['lo_grid'], A, B, G, smooth=smooth, tol_rel=tol)
            e_lo = W_lo['err']
        return {'percentile': g['percentile'], 'tau': g['tau'], 'hi': W_hi, 'lo': W_lo,
                'err': W_hi['err'] + e_lo, 'wer': (W_hi['err'] + e_lo) / max(nref_sum, 1),
                'n_lo': g['n_lo']}

    final_pool_val, final_errs, final_pool_test = pool_val, errs, pool_test
    final_flu_val, final_flu_test = np.zeros(n_val), np.zeros(n_test)
    final, bestv = None, None

    if llm_ok:
        with soft('4.5 stage-2 sweep'):
            K_GRID = (sorted(set(int(k) for k in sweep['llm_topk'] if int(k) <= topk_max))
                      if sweep.get('enable') else [cfg['llm_topk']]) or [topk_max]
            S2 = ([g for g in sweep['stage2_grid'] if g in C.STAGE2_GRIDS] or ['base']
                  if sweep.get('enable') else ['base'])
            TOL = sorted(set(float(t) for t in sweep['tol_rel'])) if sweep.get('enable') else [cfg['tol_rel']]
            SMO = (list(dict.fromkeys(sweep['smooth_grid'])) if sweep.get('enable')
                   else [cfg['smooth_grid']])
            t0, s2_rows = time.time(), []
            ns = nref[tune_idx].sum()
            budget = sweep.get('stage2_budget_min', 25) if sweep.get('enable') else 1e9
            for k in K_GRID:
                Mv = mask_val if k == topk_max else topk_masks(pool_val, W_stage1, k)
                sv, se = subpool(pool_val, Mv, errs)
                flu = fluency(sv, W_stage1)
                for pre in S2:
                    if (time.time() - t0) / 60 > budget:
                        print('  stage-2 budget exhausted - keeping the best so far')
                        break
                    _A, _B, _G = stage2_axes(pre)
                    grids = gate_grids(sv, se, tune_idx, flu, cfg['gate_percentiles'], _A, _B, _G)
                    for sm in SMO:
                        for tol in TOL:
                            cand = [pick_gate(g, _A, _B, _G, sm, tol, ns) for g in grids]
                            bg = min(cand, key=lambda r: (round(r['err']), r['percentile']))
                            edge = on_edge(bg['hi'], _A, _B, _G) or (
                                bg['lo'] is not None and on_edge(bg['lo'], _A, _B, _G))
                            s2_rows.append({'llm_topk': k, 'grid': pre, 'smooth': bool(sm),
                                            'tol_rel': tol, 'gate_percentile': bg['percentile'],
                                            'tune_WER_%': round(100 * bg['wer'], 3),
                                            'llm_w': bg['hi']['llm'], 'ng_w': bg['hi']['ng'],
                                            'nw_w': bg['hi']['nw'], 'edge': bool(edge)})
                            if bestv is None or bg['err'] < bestv['bg']['err']:
                                bestv = {'k': k, 'grid': pre, 'smooth': bool(sm), 'tol': tol,
                                         'mask': Mv, 'sv': sv, 'se': se, 'flu': flu,
                                         'gs': cand, 'bg': bg, 'axes': (_A, _B, _G)}
            s2_df = pd.DataFrame(s2_rows).sort_values('tune_WER_%').reset_index(drop=True)
            save_table(s2_df, 'stage2_sweep.csv', args.tables_dir)
            print(f'stage-2 sweep: {len(s2_rows)} settings in {time.time() - t0:.0f}s')
            print(s2_df.head(12).to_string(index=False))

            cfg.update({'llm_topk': bestv['k'], 'smooth_grid': bestv['smooth'], 'tol_rel': bestv['tol']})
            A2, B2, G2 = bestv['axes']
            mask_test = topk_masks(pool_test, W_stage1, bestv['k'])
            sub_val, sub_errs = bestv['sv'], bestv['se']
            sub_test, _ = subpool(pool_test, mask_test)
            flu_val, flu_test = bestv['flu'], fluency(sub_test, W_stage1)
            wh = bestv['bg']['hi']
            print(f"stage-2 selected on tune: llm_topk={bestv['k']} grid={bestv['grid']} "
                  f"smooth={bestv['smooth']} tol_rel={bestv['tol']} "
                  f"gate p={bestv['bg']['percentile']} -> tune {bestv['bg']['wer'] * 100:.2f}%")
            print(f"   hi weights ng={wh['ng']} llm={wh['llm']} nw={wh['nw']}"
                  + ('   <-- STILL ON A GRID EDGE: widen STAGE2_GRIDS and re-run'
                     if on_edge(wh, *bestv['axes']) else '   (inside the grid)'))
            oracle[f"top-{cfg['llm_topk']} (LLM-scored)"] = (
                oracle_wer(sub_errs, nref, tune_idx),
                oracle_wer(sub_errs, nref, verify_idx) if verify_idx else float('nan'))

            nogate, best_g = bestv['gs'][0], bestv['bg']
            abl['+ adapted LLM (no gate)'] = predict_texts(sub_val, W=nogate['hi'])[0]
            abl['+ fluency gate'] = predict_texts(sub_val, gate=best_g, flu=flu_val)[0]
            final_pool_val, final_errs, final_pool_test = sub_val, sub_errs, sub_test
            final = {'gate': best_g, 'percentile': best_g['percentile']}
            # the gate is kept only if it ALSO holds on C; otherwise the simpler system wins
            if best_g['percentile'] != 0 and verify_idx:
                v_gate = wer_on(verify_idx, abl['+ fluency gate'])
                v_nog = wer_on(verify_idx, abl['+ adapted LLM (no gate)'])
                if v_gate > v_nog:
                    print(f'gate did not transfer to C ({v_gate * 100:.2f}% vs {v_nog * 100:.2f}%) '
                          f'-> using no-gate weights')
                    final = {'gate': nogate, 'percentile': 0}
            if cfg['refit_on_all_oof'] and verify_idx:
                gs = gate_grids(sub_val, sub_errs, all_idx, flu_val, [final['percentile']], A2, B2, G2)
                final['gate'] = pick_gate(gs[0], A2, B2, G2, bestv['smooth'], bestv['tol'],
                                          nref[all_idx].sum())
            final_flu_val, final_flu_test = flu_val, flu_test

    if final is None:
        Wf = W_stage1
        if cfg['refit_on_all_oof'] and verify_idx:
            Wf, _, _ = stage1_on(pool_val, errs, nref, all_idx, A1, G1, cfg['smooth_grid'])
        final = {'gate': {'tau': -np.inf, 'hi': Wf, 'lo': None}, 'percentile': 0}

    final_pred_val, _ = predict_texts(final_pool_val, gate=final['gate'], flu=final_flu_val,
                                      sel=all_idx)
    abl['FINAL (refit on A u B u C, in-sample on C)'] = final_pred_val

    fg = final['gate']
    best_w = {'percentile': final['percentile'], 'tau': fg['tau'],
              'hi': {k: fg['hi'][k] for k in ('ng', 'llm', 'nw')},
              'lo': None if fg['lo'] is None else {k: fg['lo'][k] for k in ('ng', 'llm', 'nw')},
              'stage1': W_stage1, 'llm_used': llm_ok, 'llm': cfg['llm_name'],
              'gen_beam': cfg['gen_beam'], 'gen_nbest': cfg['gen_nbest'],
              'gen_lm_weights': cfg['gen_lm_weights'], 'llm_topk': cfg['llm_topk'],
              'expand': [cfg['expand_top'], cfg['expand_max_nb'], cfg['expand_max_new']],
              'stage1_grid': s1_grid, 'stage2_grid': bestv['grid'] if bestv else None,
              'smooth_grid': cfg['smooth_grid'], 'tol_rel': cfg.get('tol_rel', 0.0),
              'swept': bool(sweep.get('enable')),
              'models': [M['tag'] + ':' + os.path.basename(M['ckpt']) for M in models]}
    json.dump(best_w, open(os.path.join(args.tables_dir, 'best_weights.json'), 'w'),
              indent=1, default=float)
    print('final weights:', best_w)

    # ---- validation predictions for evaluate.py ---------------------------
    rows = []
    for j in all_idx:
        m = readers['val'].meta[j]
        r = {'idx': j, 'session': m['session'], 'block': m['block'], 'fold': val_labels[j],
             'reference': m['sentence']}
        for name, preds in abl.items():
            r[name] = preds.get(j, '')
        rows.append(r)
    save_table(pd.DataFrame(rows), 'val_predictions.csv', args.tables_dir)
    save_table(pd.DataFrame([{'pool': k, 'oracle_tune_%': round(v[0] * 100, 3),
                              'oracle_verify_%': round(v[1] * 100, 3)} for k, v in oracle.items()]),
               'oracle_ceiling.csv', args.tables_dir)

    # =======================================================================
    # 4.6 submission
    # =======================================================================
    test_pred, _ = predict_texts(final_pool_test, gate=final['gate'], flu=final_flu_test)
    texts = [test_pred[j] for j in range(n_test)]
    pd.DataFrame({'id': range(n_test), 'text': texts}).to_csv(args.output, index=False)
    print(f'\nsubmission -> {args.output}  ({n_test} rows, '
          f'{np.mean([len(t.split()) for t in texts]):.1f} words/utterance)')
    if STAGE_FAILED:
        print(f'NOTE: stages that failed softly: {STAGE_FAILED} — the submission above comes from '
              f'the last stage that succeeded.')
    print(f'session clock: {human_time(time.time() - T0)}')


if __name__ == '__main__':
    main()
