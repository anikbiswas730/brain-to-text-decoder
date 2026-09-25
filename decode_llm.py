"""
decode_llm.py — the full v6.4 decoding pipeline: trained cross-fit checkpoints in,
`submission.csv` out. The stages are the BLOCK 4.x sequence of
notebooks/b2t-25-crossfit-llama3-1-8b-hyperparamter-tuning.ipynb.

    python decode_llm.py --checkpoint_dir /path/to/b2t_v6_ckpt --output submission/submission.csv
    python decode_llm.py --gen_cache work/gen_cache.pkl.gz     # reuse a previous ~7 h beam search
    python decode_llm.py --no_gec                              # skip the optional GEC block (~45 min)
    python decode_llm.py --no_llm                              # acoustic + n-gram rescoring only
    python decode_llm.py --llm_name Qwen/Qwen2.5-1.5B --llm_batch 32 --no_gec   # cheap run

  4.1  locate the checkpoint bundle, route never-trained days through the generic
       adapter slot, screen each fold's checkpoints by REAL beam WER, extract
       emissions for val and test, write fallback submission #1 (greedy lexicon)
  4.2  flashlight lexicon beam search ONCE: beam 1200, n-best 100, 7 KenLM weights,
       every out-of-fold model per val trial and every model per test trial;
       cached to gen_cache.pkl.gz; fallback submission #2 (flashlight 1-best)
  4.2b joint sweep nbest x KenLM-weight subset by session-grouped CV on the tune
       folds; the pinned pool is kept unless the CV winner beats it by >= 1 paired SE
  4.3  exact features (CTC ensemble / KenLM / #words), stage-1 weights,
       phonetic-neighbour expansion, stage-1 weights again
  4.4  task-adapted LLM: QLoRA next-word-prediction fine-tune on the training
       transcripts, then score the top-24 candidates per utterance
  4.5  stage-2 weights per fluency-gate percentile, gate kept only if it holds on
       fold C, refit on all out-of-fold trials -> tables/best_weights.json
  4.5b-d  ablation, oracle-vs-k, fluency figure, per-day and error analysis
  4.5e optional selective generative error correction, adopted only if it wins on C
  4.6  test predictions under exactly the same pool / features / weights / gate

Two rules the whole file is built around:

  * **Selection happens on A u B only.** Fold C is scored in every table so you can
    see whether a choice transferred; it never picks anything (the one exception is
    a VETO: the gate and GEC are dropped if they do not also hold on C).
  * **Nothing is allowed to lose the submission.** Every stage after 4.1 is soft: it
    logs its traceback and the run continues; stages that need a failed stage's
    output are skipped, and `submission.csv` keeps the last good artefact.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from glob import glob
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from src.dataset import CacheReader, build_cache, make_crossfit_split
from src.decoding import (LLMScorer, NgramScorer, PronLexicon, Tuner, acoustic_scores, errors_lists,
                          expand_pool, fill_features, finetune_llm_nwp, fluency, pick_from_grid,
                          predict_texts, run_beam_pool, subpool, topk_masks, utt_scores)
from src.metrics import (greedy_collapse, load_lexicon, official_per, official_wer, phones_to_words,
                         remove_punctuation)
from src.inference import (build_day_override, build_pool, extract_emissions, find_checkpoint_dir,
                           gen_fingerprint, group_checkpoints, load_gen_cache, on_edge, oof_models,
                           oracle_wer, save_gen_cache, screen_checkpoints, stage1_axes, stage1_on,
                           stage2_axes)
from src import analysis
from src.utils import (figure_guard, find_file, get_session2idx, human_time, init_matplotlib,
                       pick_cache_dir, resolve_competition_path, resolve_kaggle_dataset, save_fig,
                       save_table, set_seed)

T0 = time.time()
STAGE_FAILED = []


def run_stage(name, fn, needs_ok=True, optional=False):
    """%%run_if semantics of the notebook.
    needs_ok : skipped when an earlier (non-optional) soft stage failed
    optional : a failure is logged but does not block later stages (analysis / figures)"""
    if needs_ok and STAGE_FAILED:
        print(f'[skip] {name}: an earlier stage failed ({STAGE_FAILED}); the fallback submission stays')
        return False
    try:
        fn()
        return True
    except Exception as e:
        if not optional:
            STAGE_FAILED.append(name)
        print('\n' + '!' * 90 + f'\n{"OPTIONAL" if optional else "SOFT"} STAGE FAILED: {name} ({e!r}) '
              f'- the pipeline continues. Traceback:\n' + traceback.format_exc() + '!' * 90)
        return False


def parse_args():
    p = argparse.ArgumentParser(description='Brain-to-Text v6.4 decoding + LLM rescoring')
    p.add_argument('--data_dir', default=C.DEFAULT_DATA_DIR)
    p.add_argument('--checkpoint_dir', default=None,
                   help='bundle written by train.py (defaults to config.PRETRAINED_CKPT_DIR, then a search)')
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--output', default=os.path.join(C.DEFAULT_SUBMISSION_DIR, 'submission.csv'))
    p.add_argument('--figures_dir', default=C.DEFAULT_FIGURES_DIR)
    p.add_argument('--tables_dir', default=C.DEFAULT_TABLES_DIR)
    p.add_argument('--lexicon', default=None)
    p.add_argument('--tokens', default=None)
    p.add_argument('--kenlm_binary', default=None)
    p.add_argument('--gen_cache', default=None,
                   help='beam-search cache to reuse/write (default: <work>/gen_cache.pkl.gz)')
    p.add_argument('--beam', type=int, default=None, help='override LLM_CFG.gen_beam')
    p.add_argument('--nbest', type=int, default=None, help='override LLM_CFG.gen_nbest (caps the sweep)')
    p.add_argument('--n_proc', type=int, default=None, help='beam-search processes (RAM-capped)')
    p.add_argument('--llm_name', default=None)
    p.add_argument('--llm_batch', type=int, default=None)
    p.add_argument('--no_llm', action='store_true')
    p.add_argument('--no_finetune', action='store_true')
    p.add_argument('--no_sweep', action='store_true', help='skip BLOCK 4.2b, keep the pinned pool')
    p.add_argument('--no_gec', action='store_true', help='skip BLOCK 4.5e')
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(C.SEED)
    plt = init_matplotlib()
    for d in (args.figures_dir, args.tables_dir, os.path.dirname(args.output) or '.'):
        os.makedirs(d, exist_ok=True)
    out_root = os.path.dirname(os.path.abspath(args.output))

    cfg = dict(C.LLM_CFG)
    for k, v in [('gen_beam', args.beam), ('gen_nbest', args.nbest), ('n_proc', args.n_proc),
                 ('llm_name', args.llm_name), ('llm_batch', args.llm_batch)]:
        if v is not None:
            cfg[k] = v
    cfg['nbest_sweep'] = [n for n in cfg['nbest_sweep'] if n <= cfg['gen_nbest']] or [cfg['gen_nbest']]
    if args.no_llm:
        cfg['use_llm'] = False
    if args.no_finetune:
        cfg['llm_finetune'] = False
    cfg['preference'] = C.CKPT_PREFERENCE
    gec_cfg = dict(C.GEC_CFG, enable=C.GEC_CFG['enable'] and not args.no_gec)

    device = C.resolve_device()
    n_gpus = torch.cuda.device_count()
    S = SimpleNamespace()                          # everything the stages share

    # ---- data -------------------------------------------------------------
    data_dir = resolve_competition_path(args.data_dir, C.COMPETITION_SLUG)
    session2idx = get_session2idx(data_dir)
    idx2session = {v: k for k, v in session2idx.items()}
    n_days = len(session2idx)
    cache_dir = pick_cache_dir([args.cache_dir] if args.cache_dir else C.CACHE_CANDIDATES, C.CACHE_NEED_GB)
    build_cache(data_dir, cache_dir)
    readers = {s: CacheReader(cache_dir, s) for s in ('train', 'val', 'test')}
    n_val, n_test = len(readers['val']), len(readers['test'])
    val_meta = readers['val'].meta
    val_refs = [m['sentence'] for m in val_meta]
    print(f'data: {data_dir} | {n_days} sessions | val {n_val} | test {n_test}')
    try:
        analysis.dataset_statistics(readers, args.tables_dir)
    except Exception as e:
        print('dataset statistics skipped:', repr(e))

    # ---- assets -----------------------------------------------------------
    if args.lexicon:
        lexicon_path = args.lexicon
        tokens_path = args.tokens or os.path.join(os.path.dirname(args.lexicon), 'tokens.txt')
    else:
        lex_ds = resolve_kaggle_dataset(C.LEXICON_DATASET_SLUG)
        lexicon_path, tokens_path = find_file(lex_ds, 'lexicon.txt'), find_file(lex_ds, 'tokens.txt')
    kenlm_path = args.kenlm_binary or find_file(resolve_kaggle_dataset(C.KENLM_DATASET_SLUG), '*.bin')
    print('lexicon:', lexicon_path, '\nkenlm  :', kenlm_path)
    with open(tokens_path) as f:
        toks = [l.rstrip('\n').strip() for l in f]
    assert [t.strip() for t in toks[:C.N_CLASSES]] == [p.strip() for p in C.PHONEME_VOCAB], \
        'tokens.txt != config.PHONEME_VOCAB'

    # =======================================================================
    # 4.1 checkpoints, day override, screening, emissions   (hard stage)
    # =======================================================================
    ckpt_dir = find_checkpoint_dir(args.checkpoint_dir, C.PRETRAINED_CKPT_DIR, C.DEFAULT_CHECKPOINT_DIR,
                                   '/kaggle/input')
    if ckpt_dir is None:
        raise FileNotFoundError('No fold*_seed*_*.pt found. Run train.py first, then pass --checkpoint_dir '
                                '(on Kaggle: attach the training output as a dataset).')
    print('checkpoints:', ckpt_dir)
    norm = torch.load(find_file(ckpt_dir, 'norm_stats.pt'), weights_only=False)
    try:
        sil_info = json.load(open(find_file(ckpt_dir, 'sil_convention.json')))
    except Exception:
        sil_info = {'sil_trailing': True}
    clip = C.TRAIN_CFG['clip']
    try:                                           # the split must be the one training used
        split = json.load(open(find_file(ckpt_dir, 'split.json')))
        print('split: loaded from the checkpoint bundle')
    except Exception as e:
        print('split.json not next to the checkpoints -> regenerating deterministically:', repr(e))
        split = {'labels': make_crossfit_split(val_meta, C.CROSSFIT_PATTERN, C.SEED)}
    val_labels = [split['labels'].get(m['key'], 'C') for m in val_meta]
    print('val split counts:', pd.Series(val_labels).value_counts().to_dict())

    day_override, _ = build_day_override(ckpt_dir, n_days, idx2session)
    groups = group_checkpoints(ckpt_dir, C.CKPT_PREFERENCE)
    print('checkpoint groups:', {k: sorted(v) for k, v in groups.items()})
    models, screen_rows = screen_checkpoints(groups, readers, val_labels, session2idx, norm, clip, device,
                                             lexicon_path, tokens_path, kenlm_path, cfg, C.SEED, day_override)
    if not models:
        raise RuntimeError('no loadable acoustic checkpoint')
    if screen_rows:
        save_table(pd.DataFrame(screen_rows), 'checkpoint_screening.csv', args.tables_dir)
    val_ems, test_ems = {}, {}
    for M in models:
        val_ems[M['tag']] = extract_emissions(M['model'], readers, 'val', list(range(n_val)), session2idx,
                                              norm, clip, device, day_override)
        test_ems[M['tag']] = extract_emissions(M['model'], readers, 'test', list(range(n_test)), session2idx,
                                               norm, clip, device, day_override)
    oof = oof_models(models, val_labels)
    if sum(1 for o in oof if not o):
        print(f'WARNING: {sum(1 for o in oof if not o)} val trials were trained on by every model '
              f'(final-fit run?) -> excluded from tuning.')
    tune_idx = [j for j in range(n_val) if oof[j] and val_labels[j] in cfg['tune_labels']]
    verify_idx = [j for j in range(n_val) if oof[j] and val_labels[j] in cfg['verify_labels']]
    all_idx = [j for j in range(n_val) if oof[j]]
    if not tune_idx:
        print('WARNING: no out-of-fold A/B trials -> tuning on all OOF trials')
        tune_idx = all_idx

    def wer_on(idx, hyps):
        hyps = hyps if isinstance(hyps, dict) else dict(enumerate(hyps))
        return official_wer([val_refs[j] for j in idx], [hyps[j] for j in idx])[0]

    _, pron2words = load_lexicon(lexicon_path)

    def greedy(em):
        ids = greedy_collapse(torch.from_numpy(em.astype(np.float32))[:, None, :], [em.shape[0]])[0]
        return ids, phones_to_words(ids, pron2words)

    greedy_ids, greedy_val = [None] * n_val, [''] * n_val
    for j in all_idx:
        greedy_ids[j], greedy_val[j] = greedy(val_ems[models[oof[j][0]]['tag']][j])
    print(f"OOF greedy PER {official_per([greedy_ids[j] for j in all_idx], [list(val_meta[j]['phonemes']) for j in all_idx])[0] * 100:.2f}% | "
          f"OOF greedy-lexicon WER {wer_on(all_idx, greedy_val) * 100:.2f}%")
    pd.DataFrame({'id': range(n_test), 'text': [greedy(test_ems[models[0]['tag']][j])[1] for j in range(n_test)]}
                 ).to_csv(args.output, index=False)
    print(f'fallback submission #1 written (greedy lexicon) | session clock {human_time(time.time() - T0)}')

    utt_ems_val = [[val_ems[models[mi]['tag']][j] for mi in oof[j]] for j in range(n_val)]
    utt_ems_test = [[test_ems[M['tag']][j] for M in models] for j in range(n_test)]
    lw0 = cfg['screen_lm_weight']
    S.abl = {'greedy lexicon (1 model)': dict(enumerate(greedy_val))}
    S.oracle, S.llm_ok, S.scorer = {}, False, None
    S.lex = PronLexicon(lexicon_path, sil_trailing=sil_info['sil_trailing'])
    try:
        S.ngram = NgramScorer(kenlm_path)
    except Exception as e:
        S.ngram = None
        print('WARNING: kenlm python module unavailable -> n-gram feature disabled:', repr(e))
    A1, G1 = stage1_axes()
    A2, B2, G2 = stage2_axes()

    # =======================================================================
    # 4.2 candidate generation (ONE pass) + fallback submission #2
    # =======================================================================
    def stage_generate():
        S.gen_lw = sorted(set(float(w) for w in cfg['gen_lm_weights']) | {float(lw0)})
        S.pool_lw = sorted(set(float(w) for w in cfg['gen_lm_weights']))
        S.pool_nbest = int(cfg['gen_nbest'])
        spec = dict(lexicon=lexicon_path, tokens=tokens_path, lm=kenlm_path, beam=cfg['gen_beam'],
                    nbest=cfg['gen_nbest'], lm_weights=S.gen_lw)
        tasks = [(('val', j, mi), val_ems[models[mi]['tag']][j]) for j in range(n_val) for mi in oof[j]]
        tasks += [(('test', j, mi), test_ems[models[mi]['tag']][j]) for j in range(n_test)
                  for mi in range(len(models))]
        print(f'beam search: {len(tasks)} decodes x {len(S.gen_lw)} lm_weights {S.gen_lw}, beam={spec["beam"]}, '
              f'nbest={spec["nbest"]} (max of sweep {cfg["nbest_sweep"]}), {cfg["n_proc"]} processes')
        cache_path = args.gen_cache or os.path.join(C.work_dir(), 'gen_cache.pkl.gz')
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        fp = gen_fingerprint(spec, models, val_ems, tasks)
        S.gen = load_gen_cache([cache_path] + sorted(glob('/kaggle/input/*/gen_cache.pkl.gz')
                                                     + glob('/kaggle/input/*/*/gen_cache.pkl.gz')),
                               fp, len(tasks))
        if S.gen is None:
            t0 = time.time()
            S.gen = run_beam_pool(tasks, spec, cfg['n_proc'], desc='generate')
            print(f'generation took {human_time(time.time() - t0)}')
            save_gen_cache(cache_path, fp, S.gen)
        S.base_val = [(S.gen[('val', j, oof[j][0])][lw0] or [''])[0] if oof[j] else '' for j in range(n_val)]
        S.base_test = [(S.gen[('test', j, 0)][lw0] or [''])[0] for j in range(n_test)]
        S.abl[f'flashlight 1-best lm={lw0} (1 model)'] = dict(enumerate(S.base_val))
        print(f'flashlight 1-best (lm_weight={lw0}, first OOF model): tune {wer_on(tune_idx, S.base_val) * 100:.2f}% | '
              f'verify(C) {wer_on(verify_idx, S.base_val) * 100:.2f}% | all OOF {wer_on(all_idx, S.base_val) * 100:.2f}%')
        pd.DataFrame({'id': range(n_test), 'text': S.base_test}).to_csv(args.output, index=False)
        print('fallback submission #2 written (flashlight 1-best)')

    def rebuild_pools():
        S.pool_val = build_pool(S.gen, models, 'val', n_val, oof, S.pool_lw, S.pool_nbest)
        S.pool_test = build_pool(S.gen, models, 'test', n_test, [list(range(len(models)))] * n_test,
                                 S.pool_lw, S.pool_nbest)
        print(f'pool: lm_weights {S.pool_lw} | nbest {S.pool_nbest} | val mean '
              f'{np.mean([len(t) for t in S.pool_val.texts]):.1f} | test mean '
              f'{np.mean([len(t) for t in S.pool_test.texts]):.1f}')

    # =======================================================================
    # 4.2b joint sweep: nbest x lm-weight subset, session-grouped CV on TUNE
    # =======================================================================
    def stage_sweep():
        if args.no_sweep:
            print('joint sweep disabled (--no_sweep) -> pinned pool')
            return
        from src.sweep import joint_sweep
        table, best_lw, best_nb, info = joint_sweep(
            S.gen, models, oof, val_refs, [val_meta[j]['session'] for j in tune_idx], tune_idx, verify_idx,
            S.gen_lw, cfg['gen_lm_weights'], cfg['nbest_sweep'], A1, G1, S.lex, S.ngram, utt_ems_val, device,
            cv_folds=cfg['sweep_cv_folds'], se_mult=cfg['sweep_se_mult'], budget_min=cfg['sweep_budget_min'],
            smooth=cfg['smooth_grid'], human_time=human_time)
        save_table(table, 'joint_sweep_all_combos.csv', args.tables_dir)
        json.dump(dict(info, lm_weights=best_lw, nbest=best_nb),
                  open(os.path.join(args.tables_dir, 'joint_sweep_choice.json'), 'w'), indent=1)
        analysis.joint_sweep_figure(table, args.figures_dir)
        S.pool_lw, S.pool_nbest = best_lw, best_nb

    # =======================================================================
    # 4.3 exact features, stage-1 weights, phonetic expansion
    # =======================================================================
    def stage_features():
        rebuild_pools()
        t0 = time.time()
        fill_features(S.pool_val, utt_ems_val, S.lex, S.ngram, device)
        fill_features(S.pool_test, utt_ems_test, S.lex, S.ngram, device)
        print(f'features: {human_time(time.time() - t0)}')
        S.errs, S.nref = errors_lists(S.pool_val, val_refs)
        W1, W1raw, err1 = stage1_on(S.pool_val, S.errs, S.nref, tune_idx, A1, G1, cfg['smooth_grid'])
        S.abl['pool rescoring: CTC-ens + KenLM + #words'] = predict_texts(S.pool_val, W1)[0]
        S.oracle['pool (no expansion)'] = (oracle_wer(S.errs, S.nref, tune_idx), oracle_wer(S.errs, S.nref, verify_idx))
        print(f'stage-1 weights (smoothed) {W1} | raw-best {W1raw}')
        print(f"  tune {wer_on(tune_idx, S.abl['pool rescoring: CTC-ens + KenLM + #words']) * 100:.2f}% | "
              f"verify {wer_on(verify_idx, S.abl['pool rescoring: CTC-ens + KenLM + #words']) * 100:.2f}% | "
              f"oracle tune {S.oracle['pool (no expansion)'][0] * 100:.2f}%")
        S.W_stage1 = W1
        if cfg['expansion']:
            t0 = time.time()
            ex = (cfg['expand_top'], cfg['expand_max_nb'], cfg['expand_max_new'])
            st_v = expand_pool(S.pool_val, S.lex, S.ngram, W1, *ex)
            fill_features(S.pool_val, utt_ems_val, S.lex, S.ngram, device, only_new_from=st_v)
            st_t = expand_pool(S.pool_test, S.lex, S.ngram, W1, *ex)
            fill_features(S.pool_test, utt_ems_test, S.lex, S.ngram, device, only_new_from=st_t)
            S.errs, S.nref = errors_lists(S.pool_val, val_refs)
            W1x, _, err1 = stage1_on(S.pool_val, S.errs, S.nref, tune_idx, A1, G1, cfg['smooth_grid'])
            S.abl['+ phonetic-neighbour expansion'] = predict_texts(S.pool_val, W1x)[0]
            S.oracle['pool + expansion'] = (oracle_wer(S.errs, S.nref, tune_idx), oracle_wer(S.errs, S.nref, verify_idx))
            print(f'expansion {ex}: {human_time(time.time() - t0)} | pool mean '
                  f'{np.mean([len(t) for t in S.pool_val.texts]):.0f} | weights {W1x} | '
                  f'tune {wer_on(tune_idx, S.abl["+ phonetic-neighbour expansion"]) * 100:.2f}% | '
                  f'verify {wer_on(verify_idx, S.abl["+ phonetic-neighbour expansion"]) * 100:.2f}% | '
                  f"oracle tune {S.oracle['pool + expansion'][0] * 100:.2f}%")
            S.W_stage1 = W1x
        if on_edge(S.W_stage1, A1, None, G1):
            print('NOTE: the stage-1 weights sit on the edge of config.STAGE1_GRID - widen it to optimise that axis.')
        with figure_guard('stage1_surface'):
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(err1[:, 0, :] / S.nref[tune_idx].sum() * 100, aspect='auto', origin='lower',
                           cmap='viridis', extent=[G1[0], G1[-1], A1[0], A1[-1]])
            ax.scatter([S.W_stage1['nw']], [S.W_stage1['ng']], c='red', marker='*', s=150)
            ax.set_xlabel('word bonus (per word)')
            ax.set_ylabel('KenLM weight (ln units)')
            ax.set_title('Stage-1 tune-set WER (%)')
            fig.colorbar(im, ax=ax)
            save_fig(fig, 'stage1_weight_surface', args.figures_dir)

    # =======================================================================
    # 4.4 task-adapted LLM (a failure here degrades to "no LLM", not to a stop)
    # =======================================================================
    def stage_llm():
        if not (cfg['use_llm'] and n_gpus > 0):
            print('LLM disabled (use_llm=False or no GPU).')
            return
        try:
            for M in models:                       # emissions are cached; free the GPU for the LLM
                M['model'] = None
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.time()
            train_sents = [remove_punctuation(m['sentence']) for m in readers['train'].meta]
            if S.ngram is not None:
                S.ngram.m = None                   # free the KenLM RAM (the score cache is kept)
                gc.collect()
            ft_ok = True
            try:
                model, tok, hist = finetune_llm_nwp(cfg['llm_name'], train_sents, cfg)
            except Exception as e:
                traceback.print_exc()
                print('LoRA fine-tuning failed -> using the base LLM without fine-tuning:', repr(e))
                ft_ok = False
            if not ft_ok:                          # outside the except: the failed model is released first
                gc.collect()
                torch.cuda.empty_cache()
                model, tok, hist = finetune_llm_nwp(cfg['llm_name'], train_sents, dict(cfg, llm_finetune=False))
            # single GPU only: copy.deepcopy() to a 2nd GPU clones on the CURRENT device
            # first, which briefly needs 2x the footprint on GPU0
            S.scorer = LLMScorer(model, tok, 1)
            print(f'LLM ready in {human_time(time.time() - t0)}: {hist}')
            S.mask_val = topk_masks(S.pool_val, S.W_stage1, cfg['llm_topk'])
            S.mask_test = topk_masks(S.pool_test, S.W_stage1, cfg['llm_topk'])
            for pool, masks in ((S.pool_val, S.mask_val), (S.pool_test, S.mask_test)):
                texts = sorted(set(pool.texts[u][i] for u in range(len(pool)) for i in np.where(masks[u] > 0)[0]))
                t1 = time.time()
                lp, nt = S.scorer.score(texts, bs=cfg['llm_batch'])
                lut = dict(zip(texts, zip(lp, nt)))
                for u in range(len(pool)):
                    for i in np.where(masks[u] > 0)[0]:
                        pool.llm[u][i], pool.ntok[u][i] = lut[pool.texts[u][i]]
                print(f'  scored {len(texts)} unique hypotheses in {human_time(time.time() - t1)}')
            S.llm_ok = True
            json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in hist.items()},
                      open(os.path.join(args.tables_dir, 'llm_finetune_history.json'), 'w'), indent=1)
        except Exception as e:
            traceback.print_exc()
            print('LLM stage failed -> continuing with acoustic + n-gram rescoring only:', repr(e))

    # =======================================================================
    # 4.5 stage-2 weights + fluency gate, verify on C, refit
    # =======================================================================
    def gate_search(pool, errs, idx, flu, percentiles):
        tu = Tuner(pool, errs, S.nref, idx)
        f = flu[idx]
        res = []
        for p in percentiles:
            tau = -np.inf if p == 0 else float(np.percentile(f, p))
            lo, hi = np.where(f < tau)[0], np.where(f >= tau)[0]
            err_hi = tu.grid(A2, B2, G2, rows=hi)
            W_hi, _ = pick_from_grid(err_hi, A2, B2, G2, smooth=cfg['smooth_grid'])
            W_lo, e_lo = None, 0.0
            if len(lo):
                W_lo, _ = pick_from_grid(tu.grid(A2, B2, G2, rows=lo), A2, B2, G2, smooth=cfg['smooth_grid'])
                e_lo = W_lo['err']
            res.append({'percentile': p, 'tau': tau, 'hi': W_hi, 'lo': W_lo, 'err': W_hi['err'] + e_lo,
                        'wer': (W_hi['err'] + e_lo) / S.nref[idx].sum(), 'n_lo': len(lo), 'err_hi_grid': err_hi})
        return res

    def stage_fusion():
        if S.llm_ok:
            sub_val, sub_errs = subpool(S.pool_val, S.mask_val, S.errs)
            sub_test, _ = subpool(S.pool_test, S.mask_test)
            flu_val, flu_test = fluency(sub_val, S.W_stage1), fluency(sub_test, S.W_stage1)
            S.oracle[f"top-{cfg['llm_topk']} (LLM-scored)"] = (oracle_wer(sub_errs, S.nref, tune_idx),
                                                               oracle_wer(sub_errs, S.nref, verify_idx))
            t0 = time.time()
            GS = gate_search(sub_val, sub_errs, tune_idx, flu_val, cfg['gate_percentiles'])
            print(f'gate search {time.time() - t0:.0f}s')
            for r in GS:
                print(f"  gate p={r['percentile']:>2} tau={r['tau']:8.3f} n_lo={r['n_lo']:4d} tune WER {r['wer'] * 100:.2f}% | "
                      f"hi {dict((k, r['hi'][k]) for k in ('ng', 'llm', 'nw'))} "
                      f"lo {None if r['lo'] is None else dict((k, r['lo'][k]) for k in ('ng', 'llm', 'nw'))}")
            save_table(pd.DataFrame([{'gate_percentile': r['percentile'], 'tau': r['tau'], 'n_lo': r['n_lo'],
                                      'tune_WER_%': round(r['wer'] * 100, 3),
                                      **{f'hi_{k}': r['hi'][k] for k in ('ng', 'llm', 'nw')},
                                      **{f'lo_{k}': (r['lo'][k] if r['lo'] else np.nan) for k in ('ng', 'llm', 'nw')}}
                                     for r in GS]), 'gate_search.csv', args.tables_dir)
            nogate = GS[0]
            best_g = min(GS, key=lambda r: (round(r['err']), r['percentile']))
            S.abl['+ adapted LLM (no gate)'] = predict_texts(sub_val, W=nogate['hi'])[0]
            S.abl['+ fluency gate'] = predict_texts(sub_val, gate=best_g, flu=flu_val)[0]
            S.final_pool_val, S.final_errs, S.final_pool_test = sub_val, sub_errs, sub_test
            S.final = {'gate': best_g, 'percentile': best_g['percentile']}
            # VETO: keep the gate only if it also holds on C; otherwise the simpler no-gate system
            v_gate = wer_on(verify_idx, S.abl['+ fluency gate']) if verify_idx else 0
            v_nog = wer_on(verify_idx, S.abl['+ adapted LLM (no gate)']) if verify_idx else 0
            if best_g['percentile'] != 0 and verify_idx and v_gate > v_nog:
                print(f'gate did not transfer to C ({v_gate * 100:.2f}% vs {v_nog * 100:.2f}%) -> no-gate weights')
                S.final = {'gate': nogate, 'percentile': 0}
            if cfg['refit_on_all_oof'] and verify_idx:
                S.final['gate'] = gate_search(sub_val, sub_errs, all_idx, flu_val, [S.final['percentile']])[0]
            S.final_flu_val, S.final_flu_test = flu_val, flu_test
        else:
            S.final_pool_val, S.final_errs, S.final_pool_test = S.pool_val, S.errs, S.pool_test
            Wf = S.W_stage1
            if cfg['refit_on_all_oof'] and verify_idx:
                Wf, _, _ = stage1_on(S.pool_val, S.errs, S.nref, all_idx, A1, G1, cfg['smooth_grid'])
            S.final = {'gate': {'tau': -np.inf, 'hi': Wf, 'lo': None}, 'percentile': 0}
            S.final_flu_val, S.final_flu_test = np.zeros(n_val), np.zeros(n_test)

        S.final_pred_val, S.final_reg_val = predict_texts(S.final_pool_val, gate=S.final['gate'],
                                                          flu=S.final_flu_val, sel=all_idx)
        S.abl['FINAL (refit on A∪B∪C, in-sample on C)'] = S.final_pred_val
        fg = S.final['gate']
        for side in ('hi', 'lo'):
            if fg.get(side) is not None and S.llm_ok and on_edge(fg[side], A2, B2, G2):
                print(f'NOTE: final {side}-regime weights {dict((k, fg[side][k]) for k in ("ng", "llm", "nw"))} '
                      f'sit on the edge of config.STAGE2_GRID - the optimum may lie outside it.')
        best_w = {'percentile': S.final['percentile'], 'tau': fg['tau'],
                  'hi': {k: fg['hi'][k] for k in ('ng', 'llm', 'nw')},
                  'lo': None if fg['lo'] is None else {k: fg['lo'][k] for k in ('ng', 'llm', 'nw')},
                  'stage1': S.W_stage1, 'llm_used': S.llm_ok, 'llm': cfg['llm_name'], 'gen_lm_weights': S.gen_lw,
                  'pool_lm_weights': S.pool_lw, 'pool_nbest': S.pool_nbest, 'gen_beam': cfg['gen_beam'],
                  'expand': [cfg['expand_top'], cfg['expand_max_nb'], cfg['expand_max_new']],
                  'llm_topk': cfg['llm_topk'],
                  'models': [M['tag'] + ':' + os.path.basename(M['ckpt']) for M in models]}
        json.dump(best_w, open(os.path.join(args.tables_dir, 'best_weights.json'), 'w'), indent=1, default=float)
        print('final weights:', best_w)

    # =======================================================================
    # 4.5b-d reporting (optional: never blocks the submission)
    # =======================================================================
    def stage_ablation():
        analysis.ablation_table(S.abl, S.oracle, wer_on, tune_idx, verify_idx, all_idx, args.tables_dir,
                                args.figures_dir)
        analysis.oracle_vs_k(S.pool_val, S.errs, S.nref, all_idx, S.W_stage1, utt_scores,
                             wer_on(all_idx, S.final_pred_val), args.tables_dir, args.figures_dir)
        if S.llm_ok:
            analysis.fluency_gate_figure(S.final_pool_val, S.final_errs, S.final_flu_val, all_idx, S.W_stage1,
                                         S.final['gate'], A2, B2, G2, S.nref, utt_scores, args.figures_dir)

    def stage_per_day():
        analysis.per_day_analysis(val_meta, val_refs, all_idx, greedy_ids, S.base_val, S.final_pred_val,
                                  S.final_reg_val, args.tables_dir, args.figures_dir, out_root=out_root)

    def stage_error_analysis():
        if S.ngram is not None and S.ngram.m is None:
            try:
                S.ngram._load()                    # BLOCK 4.4 freed the KenLM RAM
            except Exception as e:
                print('kenlm reload failed:', repr(e))
                S.ngram = None
        summ, _ = analysis.error_analysis(
            all_idx, val_meta, val_refs, S.final_pred_val, S.final_pool_val, S.final_flu_val, S.final['gate'],
            S.lex, S.ngram, utt_ems_val, (S.scorer.score if (S.llm_ok and S.scorer) else None), device,
            acoustic_scores, utt_scores, args.tables_dir)
        if summ is not None:
            analysis.error_analysis_figure(summ, args.figures_dir)
        pd.DataFrame({'session': [val_meta[j]['session'] for j in all_idx], 'label': [val_labels[j] for j in all_idx],
                      'ref': [remove_punctuation(val_refs[j]) for j in all_idx],
                      'flashlight': [S.base_val[j] for j in all_idx],
                      'final': [S.final_pred_val[j] for j in all_idx]}
                     ).to_csv(os.path.join(args.tables_dir, 'oof_predictions.csv'), index=False)

    # =======================================================================
    # 4.5e selective GEC (optional; adopted only if it wins on fold C)
    # =======================================================================
    S.gec_adopted, S.gec_test = False, {}

    def stage_gec():
        if not (gec_cfg['enable'] and S.llm_ok):
            print('GEC disabled or LLM unavailable -> skipping BLOCK 4.5e')
            return
        from src.gec import run_selective_gec
        if S.ngram is not None and S.ngram.m is None:
            S.ngram._load()
        S.scorer = None                            # free the scoring LLM before loading a 2nd 8B model
        gc.collect()
        res = run_selective_gec(gec_cfg, cfg['llm_name'], S.final_pool_val, S.final_pool_test, S.final_flu_val,
                                S.final_flu_test, S.final['gate'], S.final_pred_val, all_idx, tune_idx,
                                verify_idx, oof, val_refs, utt_ems_val, utt_ems_test, S.lex, S.ngram, device,
                                wer_on, seed=C.SEED, human_time=human_time)
        json.dump(res['summary'], open(os.path.join(args.tables_dir, 'gec_summary.json'), 'w'), indent=1, default=float)
        if res['adopted']:
            S.gec_adopted, S.gec_test = True, res['test_pred']
            S.final_pred_val = res['val_pred']
            S.abl['+ selective GEC (pool top-k, acoustic-grounded)'] = S.final_pred_val

    # =======================================================================
    # 4.6 submission
    # =======================================================================
    def stage_submission():
        test_pred, test_reg = predict_texts(S.final_pool_test, gate=S.final['gate'], flu=S.final_flu_test)
        if S.gec_adopted:
            test_pred.update(S.gec_test)
            print(f'merged {len(S.gec_test)} selective-GEC test predictions')
        texts = [test_pred[j] for j in range(n_test)]
        sub = pd.DataFrame({'id': range(n_test), 'text': texts})
        sub.to_csv(args.output, index=False)
        tm = readers['test'].meta
        save_table(pd.DataFrame({'id': range(n_test), 'session': [m['session'] for m in tm],
                                 'block': [m['block'] for m in tm], 'trial': [m['trial'] for m in tm],
                                 'text': texts, 'flashlight_1best': S.base_test,
                                 'regime': [test_reg.get(j, 'hi') for j in range(n_test)],
                                 'pool_size': [len(S.pool_test.texts[j]) for j in range(n_test)]}),
                   'test_predictions_detail.csv', args.tables_dir)
        print(f"submission -> {args.output}: {len(sub)} rows | empty: {int((sub['text'].str.strip() == '').sum())} | "
              f"changed vs flashlight 1-best: {np.mean([a != b for a, b in zip(texts, S.base_test)]) * 100:.1f}% | "
              f"lo-regime: {np.mean([test_reg.get(j) == 'lo' for j in range(n_test)]) * 100:.1f}%")

    def stage_val_predictions():
        """Per-system OOF predictions for evaluate.py (CER, S/D/I breakdown, bootstrap sd)."""
        rows = []
        for j in all_idx:
            m = val_meta[j]
            r = {'idx': j, 'session': m['session'], 'block': m['block'], 'fold': val_labels[j],
                 'reference': m['sentence']}
            for name, preds in S.abl.items():
                r[name] = preds.get(j, '')
            rows.append(r)
        save_table(pd.DataFrame(rows), 'val_predictions.csv', args.tables_dir)

    run_stage('4.2 candidate generation', stage_generate, needs_ok=False)
    run_stage('4.2b joint sweep', stage_sweep)
    run_stage('4.3 features + expansion', stage_features)
    run_stage('4.4 LLM', stage_llm)
    run_stage('4.5 stage-2 + fluency gate', stage_fusion)
    run_stage('4.5b ablation', stage_ablation, optional=True)
    run_stage('4.5c per-day', stage_per_day, optional=True)
    run_stage('4.5d error analysis', stage_error_analysis, optional=True)
    run_stage('4.5e selective GEC', stage_gec, optional=True)
    run_stage('val predictions', stage_val_predictions, optional=True)
    run_stage('4.6 submission', stage_submission)
    if STAGE_FAILED:
        print(f'NOTE: stages that failed softly: {STAGE_FAILED} — {args.output} comes from the last stage '
              f'that succeeded.')
    print(f'\nDONE. session clock: {human_time(time.time() - T0)}')


if __name__ == '__main__':
    main()
