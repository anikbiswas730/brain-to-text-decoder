"""
train.py — the v6 training entry point.

    python train.py --data_dir /path/to/hdf5_data_final --checkpoint_dir checkpoints
    python train.py --init_ckpt_dir /path/to/previous_run    # warm start (continue)
    python train.py --session_hours 11.5 --folds A,B

What it does, in order:

  1. builds (or reuses) the flat float16 trial cache on scratch disk
  2. computes per-channel normalisation statistics, or copies them from the run
     being continued
  3. draws the trial-level cross-fit split of `val` (A/B/A/B/C inside every
     session-block) and saves it next to the checkpoints
  4. writes one JSON config per fold and launches ONE worker process per GPU
  5. monitors them: a worker that hits the memory guard (exit 75) is relaunched
     from its saved state immediately and for free; a worker that CRASHES is
     relaunched up to --max_restarts times; both keep their resume state
  6. writes curves, a summary table and a manifest for the checkpoint bundle

This script holds no model and no data in memory while training runs — it is a
supervisor. All the real work happens in src/train_worker.py.

Output bundle (`--checkpoint_dir`, one Kaggle dataset when you are done):

    fold{A,B}_seed{0,1}_{best_wer,best_per,ema_final}.pt   EMA weights
    fold{A,B}_seed{0,1}_metrics.jsonl                      eval history
    norm_stats.pt  split.json  sil_convention.json  trained_days.json
    run_fold*.json  manifest.json

`decode_llm.py` needs that whole bundle, not just the .pt files: the split
decides which model is out-of-fold for which trial, and the normalisation
statistics have to be the ones the model was trained under.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from src.dataset import CacheReader, build_cache, compute_norm_stats, make_crossfit_split
from src.metrics import PHONE2ID, detect_sil_convention, remove_punctuation  # noqa: F401
from src.utils import (figure_guard, find_file, human_time, init_matplotlib, pick_cache_dir,
                       resolve_competition_path, resolve_kaggle_dataset, save_fig, save_table,
                       set_seed)

T0 = time.time()


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Brain-to-Text v6 cross-fit training')
    p.add_argument('--data_dir', default=C.DEFAULT_DATA_DIR)
    p.add_argument('--checkpoint_dir', default=C.DEFAULT_CHECKPOINT_DIR)
    p.add_argument('--cache_dir', default=None, help='scratch dir for the float16 trial cache')
    p.add_argument('--log_dir', default=C.DEFAULT_LOG_DIR)
    p.add_argument('--figures_dir', default=C.DEFAULT_FIGURES_DIR)
    p.add_argument('--tables_dir', default=C.DEFAULT_TABLES_DIR)
    p.add_argument('--lexicon', default=None, help='lexicon.txt (else resolved from the Kaggle dataset)')
    p.add_argument('--session_hours', type=float, default=C.SESSION['session_hours'])
    p.add_argument('--folds', default=None, help="comma-separated subset, e.g. 'A' or 'A,B'")
    p.add_argument('--init_ckpt_dir', default=C.INIT_CKPT_DIR,
                   help='continue training from a previous run (per fold, same fold only)')
    p.add_argument('--max_restarts', type=int, default=C.SESSION['max_worker_restarts'])
    p.add_argument('--rss_limit_gb', type=float, default=C.SESSION['worker_rss_limit_gb'])
    p.add_argument('--min_avail_gb', type=float, default=C.SESSION['worker_min_avail_gb'])
    p.add_argument('--keep_cache', action='store_true', help='do not delete the cache when done')
    return p.parse_args()


# ---------------------------------------------------------------------------
def prepare_data(args):
    """cache -> readers -> normalisation -> split -> lexicon. Returns a dict."""
    data_dir = resolve_competition_path(args.data_dir, C.COMPETITION_SLUG)
    from src.utils import get_session2idx
    session2idx = get_session2idx(data_dir)
    print(f'data: {data_dir} | {len(session2idx)} sessions')

    cands = [args.cache_dir] if args.cache_dir else C.CACHE_CANDIDATES
    cache_dir = pick_cache_dir(cands, C.CACHE_NEED_GB)
    resume_dir = os.path.join(os.path.dirname(cache_dir), 'b2t_resume')
    os.makedirs(resume_dir, exist_ok=True)

    t0 = time.time()
    build_cache(data_dir, cache_dir)
    readers = {s: CacheReader(cache_dir, s) for s in ('train', 'val', 'test')}
    print(f'cache ready in {time.time() - t0:.0f}s:', {s: len(r) for s, r in readers.items()})

    # ---- normalisation: a warm start MUST inherit its source run's statistics --
    norm_path = os.path.join(cache_dir, 'norm_stats.pt')
    init_dir = None
    if args.init_ckpt_dir:
        if not os.path.isdir(args.init_ckpt_dir):
            raise FileNotFoundError(f'--init_ckpt_dir not found: {args.init_ckpt_dir}')
        hits = sorted(glob(os.path.join(args.init_ckpt_dir, '**', 'fold*_seed*_*.pt'), recursive=True))
        if not hits:
            raise FileNotFoundError(f'no fold*_seed*_*.pt under {args.init_ckpt_dir}')
        init_dir = os.path.dirname(hits[0])
        print('continue-training source:', init_dir)
        src_norm = os.path.join(init_dir, 'norm_stats.pt')
        if os.path.exists(src_norm):
            shutil.copy(src_norm, norm_path)
            print('normalisation: norm_stats.pt copied from the init run (identical preprocessing)')
    if not os.path.exists(norm_path):
        torch.save(compute_norm_stats(readers['train']), norm_path)

    # ---- lexicon / tokens (the worker needs pron2words for greedy WER) --------
    if args.lexicon:
        lexicon_path = args.lexicon
        tokens_path = os.path.join(os.path.dirname(args.lexicon), 'tokens.txt')
    else:
        lex_ds = resolve_kaggle_dataset(C.LEXICON_DATASET_SLUG)
        lexicon_path, tokens_path = find_file(lex_ds, 'lexicon.txt'), find_file(lex_ds, 'tokens.txt')
    print('lexicon:', lexicon_path)
    if os.path.exists(tokens_path):
        with open(tokens_path) as f:
            toks = [l.rstrip('\n').strip() for l in f]
        assert [t.strip() for t in toks[:C.N_CLASSES]] == [p.strip() for p in C.PHONEME_VOCAB], \
            'tokens.txt does not match PHONEME_VOCAB — the phoneme IDs would be silently permuted'

    # ---- cross-fit split -----------------------------------------------------
    split = {'labels': make_crossfit_split(readers['val'].meta, C.CROSSFIT_PATTERN, C.SEED),
             'pattern': C.CROSSFIT_PATTERN, 'seed': C.SEED}
    if init_dir and os.path.exists(os.path.join(init_dir, 'split.json')):
        init_split = json.load(open(os.path.join(init_dir, 'split.json')))
        same = init_split.get('labels') == split['labels']
        print(f'split: using split.json of the init run (identical to regenerated split: {same})')
        split = init_split
    split_path = os.path.join(args.checkpoint_dir, 'split.json')
    json.dump(split, open(split_path, 'w'))
    labels = [split['labels'].get(m['key'], 'C') for m in readers['val'].meta]
    print('val split counts:', pd.Series(labels).value_counts().to_dict())

    # ---- phoneme target convention (trailing ' | ' or not) -------------------
    try:
        sil_info = detect_sil_convention(readers['train'].meta, lexicon_path)
    except Exception as e:
        print('SIL convention check failed, assuming trailing silence:', repr(e))
        sil_info = {'sil_trailing': True}
    print('phoneme target convention:', sil_info)
    json.dump(sil_info, open(os.path.join(args.checkpoint_dir, 'sil_convention.json'), 'w'))

    return {'data_dir': data_dir, 'session2idx': session2idx, 'cache_dir': cache_dir,
            'resume_dir': resume_dir, 'readers': readers, 'norm_path': norm_path,
            'lexicon_path': lexicon_path, 'split_path': split_path, 'init_dir': init_dir}


def dataset_tables(readers, tables_dir):
    """Trials per day / per split, and how much of val is verbatim in train."""
    try:
        rows = []
        for s, r in readers.items():
            for m in r.meta:
                rows.append({'split': s, 'session': m['session'], 'block': m['block'], 'T': m['n'],
                             'n_words': len(remove_punctuation(m['sentence']).split()),
                             'n_phon': len(m['phonemes'])})
        df = pd.DataFrame(rows)
        per_day = df.pivot_table(index='session', columns='split', values='T',
                                 aggfunc='count', fill_value=0).reset_index()
        save_table(per_day, 'dataset_trials_per_day.csv', tables_dir)
        summ = df.groupby('split').agg(trials=('T', 'size'), sessions=('session', 'nunique'),
                                       mean_T=('T', 'mean'), median_T=('T', 'median'),
                                       max_T=('T', 'max'), mean_words=('n_words', 'mean')).reset_index()
        print(summ.to_string(index=False))
        save_table(summ, 'dataset_summary.csv', tables_dir)
        train_sent = set(remove_punctuation(m['sentence']) for m in readers['train'].meta)
        val_sent = [remove_punctuation(m['sentence']) for m in readers['val'].meta]
        print(f'val sentences also present verbatim in train: '
              f'{np.mean([s in train_sent for s in val_sent]) * 100:.1f}%')
    except Exception as e:
        print('dataset statistics skipped:', repr(e))


# ---------------------------------------------------------------------------
def build_worker_config(run, args, data, deadline_ts):
    cfg = dict(C.TRAIN_CFG)
    cfg.update({'fold': run['fold'], 'seed': run['seed'], 'train_val_labels': run['train_val_labels'],
                'out_dir': os.path.abspath(args.checkpoint_dir), 'resume_dir': data['resume_dir'],
                'cache_dir': data['cache_dir'], 'split_path': os.path.abspath(data['split_path']),
                'session2idx': data['session2idx'], 'deadline_ts': deadline_ts,
                'final_reserve_s': C.SESSION['worker_final_reserve_min'] * 60,
                'model': C.MODEL_CFG, 'aug': C.AUG_CFG, 'lexicon_path': data['lexicon_path'],
                'rss_limit_gb': args.rss_limit_gb, 'min_avail_gb': args.min_avail_gb})
    tag = f"fold{run['fold']}_seed{run['seed']}"

    # warm start: same fold only. Initialising fold A from fold B's weights would
    # leak A's holdout trials into A's model through B's training set.
    cfg['init_from'] = C.TRAIN_CFG.get('init_from')
    if data['init_dir']:
        init = None
        for name in C.INIT_PREFERENCE:
            c = sorted(glob(os.path.join(args.init_ckpt_dir, '**', f'{tag}_{name}'), recursive=True))
            if c:
                init = c[0]
                break
        if init is None:
            raise FileNotFoundError(f'no {tag}_{{{"|".join(C.INIT_PREFERENCE)}}} under {args.init_ckpt_dir}')
        ck = torch.load(init, map_location='cpu', weights_only=False)
        if sorted(ck.get('train_val_labels', [])) != sorted(run['train_val_labels']):
            raise ValueError(f"{init} was trained on val labels {ck.get('train_val_labels')}, "
                             f"this run expects {run['train_val_labels']}")
        m = ck.get('metrics', {}) or {}
        print(f"{tag}: init <- {init} (step {ck.get('step')}, epoch {ck.get('epoch')}, "
              f"PER {m.get('per', float('nan')) * 100:.2f}%, "
              f"greedyWER {m.get('greedy_wer', float('nan')) * 100:.2f}%)")
        cfg['init_from'] = init
        del ck
    return tag, cfg


def ram_line():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f'host RAM used {vm.percent:.0f}% ({vm.available / 1e9:.1f} GB available)'
    except Exception:
        return ''


def tail(path, n=400):
    try:
        with open(path, errors='replace') as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def launch(entry, resume, repo_root):
    run = entry['run']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run['gpu']), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='64')
    if resume:
        env.pop('B2T_FAKE_CRASH_STEP', None)
    cmd = [sys.executable, '-u', '-m', 'src.train_worker', '--config', entry['cfg_path']]
    if resume:
        cmd.append('--resume')
    logf = open(entry['log_path'], 'a')
    entry['proc'] = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=repo_root)
    entry['logf'] = logf
    print(f"{'RE' if resume else ''}launched fold {run['fold']} seed {run['seed']} on GPU{run['gpu']} "
          f"(pid {entry['proc'].pid}{', --resume' if resume else ''}) -> {entry['log_path']}")


def monitor(entries, args, deadline_ts, train_end_ts, repo_root):
    last_print = 0.0
    while True:
        alive = False
        for e in entries:
            if e['done']:
                continue
            rc = e['proc'].poll()
            if rc is None:
                alive = True
                continue
            e['logf'].close()
            if rc == 0:
                e['done'] = True
                print(f"fold {e['run']['fold']} finished normally")
                continue
            if rc == 75 and deadline_ts - time.time() > 60:
                # planned memory recycle: state is already on disk, not a crash
                e['recycles'] += 1
                print(f"[memory guard] fold {e['run']['fold']} saved its state and exited "
                      f"(recycle #{e['recycles']}, {ram_line()}) -> relaunching with --resume")
                time.sleep(3)
                launch(e, True, repo_root)
                alive = True
                continue
            why = 'killed by the OS, usually host-RAM OOM' if rc in (-9, 137) else 'exception, see log'
            print(f"\n!!! fold {e['run']['fold']} exited with code {rc} ({why}) | {ram_line()}")
            for l in tail(e['log_path'], 25):
                print('   ' + l.rstrip()[:300])
            if (e['restarts'] < args.max_restarts
                    and train_end_ts - time.time() > C.SESSION['restart_min_remaining_min'] * 60):
                e['restarts'] += 1
                print(f"-> relaunching from the last resume state "
                      f"(restart {e['restarts']}/{args.max_restarts})")
                time.sleep(3)
                launch(e, True, repo_root)
                alive = True
            else:
                e['done'] = True
                print(f"-> not relaunching fold {e['run']['fold']} (restarts used or too close to "
                      f"the deadline); its saved checkpoints are kept")
        if not alive:
            break
        if time.time() - last_print >= C.SESSION['monitor_print_min'] * 60:
            last_print = time.time()
            print(f"\n===== monitor @ session {human_time(time.time() - T0)} | train end in "
                  f"{human_time(train_end_ts - time.time())} | {ram_line()} =====")
            for e in entries:
                lines = tail(e['log_path'])
                key = [l for l in lines if any(k in l for k in
                                               ('EVAL', 'BUDGET', 'OOM', 'RESUMED', 'MEMORY GUARD'))][-3:]
                status = 'done' if e['done'] else 'running'
                print(f"-- fold {e['run']['fold']} [{status}, crash restarts {e['restarts']}, "
                      f"memory recycles {e['recycles']}]")
                for l in key + lines[-1:]:
                    print('   ' + l.rstrip()[:300])
            try:
                print(subprocess.run(['nvidia-smi',
                                      '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader'],
                                     capture_output=True, text=True, timeout=30).stdout.strip())
            except Exception:
                pass
        if time.time() > deadline_ts + 20 * 60:
            for e in entries:
                if not e['done'] and e['proc'].poll() is None:
                    e['proc'].kill()
            print('killed workers still running 20 min past the deadline')
        time.sleep(15)


# ---------------------------------------------------------------------------
def training_report(args):
    """Curves + summary from the workers' metrics.jsonl, and a bundle manifest."""
    plt = init_matplotlib()
    curves = {}
    for f in sorted(glob(os.path.join(args.checkpoint_dir, '*_metrics.jsonl'))):
        tag = os.path.basename(f).replace('_metrics.jsonl', '')
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        curves[tag] = pd.DataFrame(rows)
        curves[tag]['run'] = tag
    if curves:
        save_table(pd.concat(curves.values(), ignore_index=True),
                   'training_eval_history.csv', args.tables_dir)
        with figure_guard('training_curves'):
            panels = [('per', 'PER (greedy, %)'), ('greedy_wer', 'WER greedy lexicon (%)'),
                      ('train_ctc', 'train CTC loss')]
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
            for ax, (col, title) in zip(axes, panels):
                for tag, df in curves.items():
                    if col in df:
                        ax.plot(df['hours'], df[col] * (100 if col != 'train_ctc' else 1),
                                marker='o', ms=3, lw=1.5, label=tag)
                ax.set_title(title)
                ax.set_xlabel('hours')
                ax.grid(alpha=0.3)
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle('B2TNetV4 cross-fit training (EMA weights, fold holdout)', y=1.03)
            save_fig(fig, 'training_curves', args.figures_dir)
        rows = []
        for tag, df in curves.items():
            i = int(df['greedy_wer'].idxmin())
            rows.append({'run': tag, 'evals': len(df), 'last_epoch': int(df['epoch'].iloc[-1]),
                         'last_step': int(df['step'].iloc[-1]),
                         'hours': round(float(df['hours'].iloc[-1]), 2),
                         'best_greedy_wer_%': round(float(df['greedy_wer'].min()) * 100, 2),
                         'at_hours': round(float(df['hours'][i]), 2),
                         'best_per_%': round(float(df['per'].min()) * 100, 2),
                         'final_per_%': round(float(df['per'].iloc[-1]) * 100, 2),
                         'final_greedy_wer_%': round(float(df['greedy_wer'].iloc[-1]) * 100, 2)})
        summ = pd.DataFrame(rows)
        print(summ.to_string(index=False))
        save_table(summ, 'training_summary.csv', args.tables_dir)

    manifest = sorted(os.path.relpath(p, args.checkpoint_dir)
                      for p in glob(os.path.join(args.checkpoint_dir, '*')))
    json.dump({'files': manifest, 'model_cfg': C.MODEL_CFG, 'train_cfg': C.TRAIN_CFG,
               'aug_cfg': C.AUG_CFG, 'runs': C.TRAIN_RUNS, 'session_hours': args.session_hours},
              open(os.path.join(args.checkpoint_dir, 'manifest.json'), 'w'), indent=1)
    print('\ncheckpoint bundle:', args.checkpoint_dir)
    for m in manifest:
        print('  ', m)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for d in (args.checkpoint_dir, args.log_dir, args.figures_dir, args.tables_dir):
        os.makedirs(d, exist_ok=True)
    set_seed(C.SEED)
    n_gpus = torch.cuda.device_count()
    print(f'device={C.resolve_device()} x{n_gpus} | torch {torch.__version__}')

    data = prepare_data(args)
    dataset_tables(data['readers'], args.tables_dir)
    del data['readers']          # the supervisor holds no data while training runs

    runs = C.TRAIN_RUNS
    if args.folds:
        want = [f.strip() for f in args.folds.split(',')]
        runs = [r for r in runs if r['fold'] in want]
    if n_gpus and n_gpus < len(runs):
        print(f'WARNING: {n_gpus} GPU(s) visible -> running only fold(s) '
              f'{[r["fold"] for r in runs[:n_gpus]]}.')
        runs = runs[:n_gpus]
    if not runs:
        raise SystemExit('no fold selected')

    deadline_ts = T0 + args.session_hours * 3600 - C.SESSION['notebook_reserve_min'] * 60
    train_end_ts = deadline_ts - C.SESSION['worker_final_reserve_min'] * 60

    # the decoding stage needs these three next to the weights
    shutil.copy(data['norm_path'], os.path.join(args.checkpoint_dir, 'norm_stats.pt'))

    entries = []
    for run in runs:
        tag, cfg = build_worker_config(run, args, data, deadline_ts)
        cfg_path = os.path.join(args.checkpoint_dir, f'run_{tag}.json')
        json.dump(cfg, open(cfg_path, 'w'), indent=1)
        stale = os.path.join(data['resume_dir'], f'{tag}_resume.pt')
        if os.path.exists(stale):
            os.remove(stale)                 # never resume from an older session's state
        log_path = os.path.join(args.log_dir, f'train_{tag}.log')
        open(log_path, 'w').close()
        e = {'run': run, 'tag': tag, 'cfg_path': cfg_path, 'log_path': log_path,
             'restarts': 0, 'recycles': 0, 'done': False}
        launch(e, False, repo_root)
        entries.append(e)
    print(f'train end in {human_time(train_end_ts - time.time())} | '
          f'{(time.time() - T0) / 60:.1f} min used for setup | {ram_line()}')

    monitor(entries, args, deadline_ts, train_end_ts, repo_root)

    for e in entries:
        print(f"\nfold {e['run']['fold']} (restarts {e['restarts']}, recycles {e['recycles']}) last lines:")
        for l in tail(e['log_path'], 5):
            print('   ' + l.rstrip()[:300])
    written = sorted(glob(os.path.join(args.checkpoint_dir, 'fold*_seed*_*.pt')))
    print(f'\ncheckpoints written: {len(written)}')
    if not written:
        raise RuntimeError('no checkpoint was produced by any worker - see the log excerpts above')

    training_report(args)
    if not args.keep_cache and data['cache_dir'].startswith(C.work_dir()):
        shutil.rmtree(data['cache_dir'], ignore_errors=True)
        shutil.rmtree(data['resume_dir'], ignore_errors=True)
        print('removed the data cache from the working directory so it is not saved as output')
    print(f'\nNEXT: package {args.checkpoint_dir} (on Kaggle: Save Version -> New Dataset from the '
          f'output folder), then run decode_llm.py --checkpoint_dir <that dataset>.')
    print(f'session clock: {human_time(time.time() - T0)}')


if __name__ == '__main__':
    main()
