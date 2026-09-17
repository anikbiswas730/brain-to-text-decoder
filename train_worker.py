"""
src/train_worker.py — one process = one GPU = one cross-fit fold.

    python -m src.train_worker --config checkpoints/run_foldA_seed0.json [--resume]

`train.py` writes the JSON config, launches one of these per GPU with
CUDA_VISIBLE_DEVICES pinned, and relaunches it with --resume whenever it exits.
You would normally not run it by hand.

Robustness design (v6.0 was OOM-killed by the OS after 27 minutes, and its
successor lost 11 workers to the same kernel OOM killer):
  * no KenLM / flashlight in the worker — the beam evaluator was the single
    biggest host-RAM user, so checkpoint selection here uses PER + greedy WER
    and real beam WER is measured once, later, in the decoding stage
  * no DataLoader subprocesses, no pinned memory, no memmap (pread per trial)
  * MEMORY GUARD: above `rss_limit_gb` RSS, or below `min_avail_gb` free, the
    worker saves its resume state and exits with code 75. The launcher relaunches
    it immediately with --resume; nothing is lost and the OS never has to kill it
  * CUDA OOM on a batch -> batch skipped, the bins-per-batch cap is lowered 15%,
    training continues with a fresh GradScaler at the same scale
  * a non-finite loss skips the batch instead of poisoning the weights
  * evaluation failures never stop training
  * resume state every `resume_every_min` minutes, and the eval/resume TIMERS are
    part of that state — otherwise every restart postpones the next evaluation
    (which is why one fold got a single eval in 5 hours)
  * every checkpoint is written to a temp file and atomically renamed
  * --init_from must load with ZERO missing/unexpected keys, so a warm start can
    never silently degrade into training from random weights
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback

os.environ.setdefault('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT', '64')  # bound the cuDNN plan cache

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (BucketBatchSampler, CacheReader, TrialSet, collate_trials,
                         make_time_mask, speed_perturb, view_augment)
from src.metrics import load_lexicon, official_per, official_wer, greedy_collapse, phones_to_words
from src.model import B2TNetV4, EMA, cr_ctc_consistency, ctc_mean, forward_batch
from src.utils import atomic_save, human_time, is_oom, mem_status, rss_gb

RECYCLE_EXIT_CODE = 75     # memory guard: 'state saved, please relaunch me with --resume'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


@torch.no_grad()
def evaluate(model, hold_ds, readers, norm, cfg, device, pron2words):
    """PER and greedy-lexicon WER on this fold's held-out trials (no LM, no beam)."""
    model.eval()
    preds, trues, refs, hyps = [], [], [], []
    for idx in BucketBatchSampler(hold_ds.lengths, 32, 32 * 2000, False, 0):
        batch = collate_trials([hold_ds[i] for i in idx])
        lp, out_len = forward_batch(model, batch, device, norm, cfg['clip'])
        ids = greedy_collapse(lp, out_len)
        for k, j in enumerate(batch['j'].tolist()):
            r, i = hold_ds.items[j]
            m = readers[r].meta[i]
            preds.append(ids[k])
            trues.append([int(p) for p in m['phonemes']])
            refs.append(m['sentence'])
            hyps.append(phones_to_words(ids[k], pron2words))
    return {'per': float(official_per(preds, trues)[0]),
            'greedy_wer': float(official_wer(refs, hyps)[0]), 'n': len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    out_dir = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    resume_dir = cfg['resume_dir']
    os.makedirs(resume_dir, exist_ok=True)
    tag = f"fold{cfg['fold']}_seed{cfg['seed']}"
    resume_path = os.path.join(resume_dir, f'{tag}_resume.pt')
    seed = int(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    deadline = float(cfg['deadline_ts'])
    train_end = deadline - float(cfg['final_reserve_s'])

    # ---- data -------------------------------------------------------------
    readers = {'train': CacheReader(cfg['cache_dir'], 'train'), 'val': CacheReader(cfg['cache_dir'], 'val')}
    session2idx = cfg['session2idx']
    n_days = len(session2idx)
    norm = torch.load(os.path.join(cfg['cache_dir'], 'norm_stats.pt'), weights_only=False)
    labels = json.load(open(cfg['split_path']))['labels']
    fold_labels = set(cfg['train_val_labels'])

    train_items = [('train', i) for i, m in enumerate(readers['train'].meta) if len(m['phonemes']) > 0]
    hold_items = []
    for i, m in enumerate(readers['val'].meta):
        if len(m['phonemes']) == 0:
            continue
        (train_items if labels.get(m['key']) in fold_labels else hold_items).append(('val', i))
    train_ds = TrialSet(readers, train_items, session2idx)
    hold_ds = TrialSet(readers, hold_items, session2idx)

    # sessions this fold never trained on -> decoding routes them through the
    # generic adapter slot instead of an untouched random one
    trained_days = sorted(set(session2idx[readers[r].meta[i]['session']] for r, i in train_items))
    json.dump({'untrained_days': [d for d in range(n_days) if d not in set(trained_days)],
               'trained_days': trained_days},
              open(os.path.join(out_dir, 'trained_days.json'), 'w'))

    mcfg = cfg['model']
    use_amp = device == 'cuda'

    def build_all():
        mdl = B2TNetV4(n_days, mcfg).to(device)
        day_ids = set(id(p) for p in mdl.day_parameters())
        decay = [p for p in mdl.parameters() if id(p) not in day_ids and p.ndim >= 2]
        no_decay = [p for p in mdl.parameters() if id(p) not in day_ids and p.ndim < 2]
        o = torch.optim.AdamW([
            {'params': decay, 'weight_decay': cfg['weight_decay'], 'lr_scale': 1.0},
            {'params': no_decay, 'weight_decay': 0.0, 'lr_scale': 1.0},
            {'params': mdl.day_parameters(), 'weight_decay': cfg['weight_decay_day'],
             'lr_scale': cfg['lr_day_scale']},
        ], lr=cfg['lr_max'], betas=(0.9, 0.98), eps=1e-6)
        return mdl, o, torch.amp.GradScaler('cuda', enabled=use_amp)

    model, opt, scaler = build_all()

    st = {'step': 0, 'epoch': 0, 't_start': time.time(), 't_warm_end': None, 'best_sel': float('inf'),
          'best_per': float('inf'), 'max_bins': int(cfg['max_bins_per_batch']), 'restarts': 0,
          'oom_skips': 0, 'nan_skips': 0}
    resumed = False
    ema = None

    # ---- resume / warm start ---------------------------------------------
    if args.resume and os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            scaler.load_state_dict(ck['scaler'])
            ema = EMA(model, cfg['ema_decay'])
            ema.model.load_state_dict(ck['ema'])
            ema.n = ck['ema_n']
            st.update(ck['state'])
            st['restarts'] += 1
            resumed = True
            del ck
            log(f'[{tag}] RESUMED from {resume_path} at step {st["step"]} epoch {st["epoch"]} '
                f'(restart #{st["restarts"]})')
        except Exception as e:
            log(f'[{tag}] resume state unreadable ({e!r}) -> starting fresh')
            model, opt, scaler = build_all()
            ema = None
    elif args.resume:
        log(f'[{tag}] --resume given but no resume state yet -> starting fresh')

    if not resumed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        if cfg.get('init_from'):
            ck = torch.load(cfg['init_from'], map_location=device, weights_only=False)
            res = model.load_state_dict(ck['model_state_dict'], strict=False)
            if res.missing_keys or res.unexpected_keys:
                raise RuntimeError(f'init_from {cfg["init_from"]} does not match the model: '
                                   f'missing={res.missing_keys[:5]} unexpected={res.unexpected_keys[:5]}')
            log(f'[{tag}] initialised from {cfg["init_from"]} (0 missing / 0 unexpected keys; '
                f'source step {ck.get("step")}, epoch {ck.get("epoch")}, '
                f'metrics {ck.get("metrics", {}).get("per")})')
            del ck
        ema = EMA(model, cfg['ema_decay'])
    else:
        rs = seed + 1000 * st['restarts']       # a restart must not replay the same batch order
        random.seed(rs)
        np.random.seed(rs)
        torch.manual_seed(rs)

    log(f'[{tag}] device={device} {torch.cuda.get_device_name(0) if device == "cuda" else ""} | '
        f'params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | train trials {len(train_items)} '
        f'(val labels {sorted(fold_labels)}) | holdout {len(hold_items)} | '
        f'train-end in {human_time(train_end - time.time())} | {mem_status()}')

    pron2words = load_lexicon(cfg['lexicon_path'])[1] if cfg.get('lexicon_path') else {}
    aug = cfg['aug']
    metrics_path = os.path.join(out_dir, f'{tag}_metrics.jsonl')
    warmup = int(cfg['warmup_steps'])
    run_loss = {'ctc': 0.0, 'aux': 0.0, 'cr': 0.0, 'n': 0}
    next_eval = st.get('next_eval_ts') or (time.time() + 60 * cfg['eval_every_min'])
    next_resume = st.get('next_resume_ts') or (time.time() + 60 * cfg['resume_every_min'])
    st['next_eval_ts'], st['next_resume_ts'] = next_eval, next_resume
    rss0, _ = rss_gb()
    rss_limit = max(float(cfg.get('rss_limit_gb', 1e9)),
                    rss0 + float(os.environ.get('B2T_RSS_MARGIN_GB', '2.0')))
    min_avail = float(cfg.get('min_avail_gb', 0.0))
    steps_this_life = 0
    log(f'[{tag}] memory guard: recycle above RSS {rss_limit:.1f} GB or below {min_avail:.1f} GB '
        f'available (RSS now {rss0:.1f} GB)')
    epoch_times = []

    def lr_now():
        """Warmup by STEP, then cosine by WALL CLOCK: the schedule always lands
        on lr_min exactly at train_end, whatever throughput turns out to be."""
        if st['step'] < warmup:
            return cfg['lr_max'] * (st['step'] + 1) / warmup
        if st['t_warm_end'] is None:
            st['t_warm_end'] = time.time()
        prog = (time.time() - st['t_warm_end']) / max(train_end - st['t_warm_end'], 1.0)
        prog = min(max(prog, 0.0), 1.0)
        return cfg['lr_min'] + 0.5 * (cfg['lr_max'] - cfg['lr_min']) * (1 + math.cos(math.pi * prog))

    def save_ckpt(name, metrics):
        atomic_save({'model_state_dict': ema.model.state_dict(), 'config': mcfg, 'n_days': n_days,
                     'arch': B2TNetV4.ARCH, 'fold': cfg['fold'], 'seed': seed,
                     'train_val_labels': sorted(fold_labels), 'epoch': st['epoch'], 'step': st['step'],
                     'metrics': metrics}, os.path.join(out_dir, f'{tag}_{name}.pt'))

    def save_resume():
        try:
            atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(),
                         'scaler': scaler.state_dict(), 'ema': ema.model.state_dict(),
                         'ema_n': ema.n, 'state': dict(st)}, resume_path)
        except Exception as e:
            log(f'[{tag}] resume save failed (training continues): {e!r}')

    def do_eval(final=False):
        try:
            t0 = time.time()
            m = evaluate(ema.model, hold_ds, readers, norm, cfg, device, pron2words)
            m.update({'epoch': st['epoch'], 'step': st['step'],
                      'hours': (time.time() - st['t_start']) / 3600, 'lr': float(lr_now()),
                      'train_ctc': run_loss['ctc'] / max(run_loss['n'], 1),
                      'train_aux': run_loss['aux'] / max(run_loss['n'], 1),
                      'train_cr': run_loss['cr'] / max(run_loss['n'], 1),
                      'eval_s': time.time() - t0, 'max_bins': st['max_bins'],
                      'oom_skips': st['oom_skips'], 'nan_skips': st['nan_skips']})
            flag = ''
            if m['greedy_wer'] < st['best_sel'] or (m['greedy_wer'] == st['best_sel'] and m['per'] < st['best_per']):
                st['best_sel'] = m['greedy_wer']
                save_ckpt('best_wer', m)
                flag += ' *best_wer*'
            if m['per'] < st['best_per']:
                st['best_per'] = m['per']
                save_ckpt('best_per', m)
                flag += ' *best_per*'
            if final:
                save_ckpt('ema_final', m)
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(m) + '\n')
            log(f"[{tag}] EVAL ep{st['epoch']} step{st['step']} {m['hours']:.2f}h | "
                f"PER {m['per'] * 100:.2f}% | greedyWER {m['greedy_wer'] * 100:.2f}% | "
                f"ctc {m['train_ctc']:.3f} aux {m['train_aux']:.3f} cr {m['train_cr']:.3f} | "
                f"lr {m['lr']:.2e} | {m['eval_s']:.0f}s{flag} | {mem_status()}")
        except Exception as e:
            log(f'[{tag}] evaluation failed (training continues): {e!r}\n{traceback.format_exc()}')
            if final:
                try:
                    save_ckpt('ema_final', {})
                except Exception:
                    pass
        finally:
            for k in run_loss:
                run_loss[k] = 0.0
            model.train()
            if device == 'cuda':
                torch.cuda.empty_cache()

    fake_crash = int(os.environ.get('B2T_FAKE_CRASH_STEP', '-1'))     # smoke test only
    fake_oom = int(os.environ.get('B2T_FAKE_OOM_STEP', '-1'))         # smoke test only

    # ---- training loop ----------------------------------------------------
    model.train()
    stop = False
    while not stop:
        st['epoch'] += 1
        te = time.time()
        sampler = BucketBatchSampler(train_ds.lengths, cfg['batch_size'], st['max_bins'], True,
                                     seed * 7919 + st['epoch'] + 100 * st['restarts'])
        for idx in sampler:
            if time.time() >= train_end:
                stop = True
                break
            if st['step'] == fake_crash and not resumed:
                log(f'[{tag}] (smoke test) simulated hard crash')
                sys.stdout.flush()
                os._exit(137)
            batch = collate_trials([train_ds[i] for i in idx])
            lr = lr_now()
            for g in opt.param_groups:
                g['lr'] = lr * g['lr_scale']
            xv = lp = auxs = loss = ctc = aux = cr = tmask = None
            try:
                if st['step'] == fake_oom:
                    fake_oom = -1
                    raise torch.cuda.OutOfMemoryError('CUDA out of memory (smoke test)')
                x = batch['neural'].to(device).float()
                x = torch.clamp((x - norm['mean'].to(device)) / norm['std'].to(device),
                                -cfg['clip'], cfg['clip'])
                lengths = batch['lengths'].to(device)
                tgt, tlen = batch['target'].to(device), batch['target_lengths'].to(device)
                day = batch['day_idx'].to(device)
                B = x.shape[0]
                if aug['random_cut'] > 0:
                    cut = int(np.random.randint(0, aug['random_cut']))
                    if cut and x.shape[1] - cut > 20:
                        x = x[:, cut:]
                        lengths = (lengths - cut).clamp(min=1)
                if np.random.rand() < aug['speed_p']:
                    x, lengths = speed_perturb(x, lengths, aug['speed_lo'], aug['speed_hi'])
                n_views = 2 if cfg['cr_weight'] > 0 else 1
                xv = torch.cat([view_augment(x, lengths, aug) for _ in range(n_views)], 0)
                del x
                lv, dv = lengths.repeat(n_views), day.repeat(n_views)
                tv, tlv = tgt.repeat(n_views, 1), tlen.repeat(n_views)
                tmask = make_time_mask(lv.tolist(), xv.shape[1], aug['time_mask_frac'],
                                       aug['time_mask_min'], aug['time_mask_max'], device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    lp, out_len, auxs = model(xv, lv, dv, time_mask=tmask, return_aux=True)
                ctc = ctc_mean(lp, tv, out_len, tlv)
                aux = torch.stack([ctc_mean(a, tv, out_len, tlv) for a in auxs]).mean() if auxs else ctc * 0
                loss = (1 - cfg['interctc_weight']) * ctc + cfg['interctc_weight'] * aux
                cr = cr_ctc_consistency(lp, out_len, tlv, B) if n_views == 2 else ctc * 0
                loss = loss + cfg['cr_weight'] * cr
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    st['nan_skips'] += 1
                    if st['nan_skips'] % 20 == 1:
                        log(f'[{tag}] non-finite loss, batch skipped (total {st["nan_skips"]})')
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                st['step'] += 1
                run_loss['ctc'] += float(ctc.detach())
                run_loss['aux'] += float(aux.detach())
                run_loss['cr'] += float(cr.detach())
                run_loss['n'] += 1
            except Exception as e:
                if not is_oom(e):
                    raise
                st['oom_skips'] += 1
                xv = lp = auxs = loss = ctc = aux = cr = tmask = None
                opt.zero_grad(set_to_none=True)
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
                # a fresh scaler at the same scale avoids "unscale_() already called"
                # if the OOM landed in the middle of an optimizer step
                scaler = torch.amp.GradScaler('cuda',
                                              init_scale=float(scaler.get_scale()) if use_amp else 2.0 ** 16,
                                              enabled=use_amp)
                st['max_bins'] = max(int(st['max_bins'] * 0.85), 4000)
                log(f'[{tag}] CUDA OOM on batch (T={int(batch["lengths"].max())}, B={len(idx)}) -> '
                    f'skipped, max_bins_per_batch now {st["max_bins"]} (from next epoch) | {mem_status()}')
                continue

            if st['step'] % cfg['log_every'] == 0:
                log(f"[{tag}] ep{st['epoch']} step{st['step']} loss {float(loss):.3f} "
                    f"ctc {float(ctc):.3f} cr {float(cr):.4f} lr {lr:.2e} | "
                    f"train-end in {human_time(train_end - time.time())} | {mem_status()}")
            now = time.time()
            steps_this_life += 1
            if now >= next_eval:
                do_eval()
                late = (now - st['t_start']) / max(train_end - st['t_start'], 1) > cfg['late_frac']
                next_eval = time.time() + 60 * (cfg['eval_every_min_late'] if late else cfg['eval_every_min'])
                st['next_eval_ts'] = next_eval
            if now >= next_resume:
                next_resume = time.time() + 60 * cfg['resume_every_min']
                st['next_resume_ts'] = next_resume
                save_resume()
            if steps_this_life % 10 == 0 and steps_this_life >= 50:
                r, av = rss_gb()
                if (r > rss_limit or av < min_avail) and (deadline - now > 180 or av < 1.5):
                    st['next_eval_ts'], st['next_resume_ts'] = next_eval, next_resume
                    save_resume()
                    log(f'[{tag}] MEMORY GUARD: RSS {r:.1f} GB / available {av:.1f} GB at '
                        f'step {st["step"]} -> state saved, exiting for a clean relaunch '
                        f'(code {RECYCLE_EXIT_CODE})')
                    sys.stdout.flush()
                    os._exit(RECYCLE_EXIT_CODE)
        if not stop:
            epoch_times.append(time.time() - te)
            if len(epoch_times) in (2, 5) or st['epoch'] % 25 == 0:
                ept = float(np.median(epoch_times[-5:]))
                rem = max(train_end - time.time(), 0)
                log(f"[{tag}] BUDGET: epoch time ~{ept / 60:.2f} min -> ~{int(rem / max(ept, 1e-6))} "
                    f"more epochs, projected TOTAL ~{st['epoch'] + int(rem / max(ept, 1e-6))} epochs "
                    f"(time-based cosine lands on lr_min at the deadline)")

    log(f"[{tag}] training loop finished at epoch {st['epoch']}, step {st['step']}; final EMA eval...")
    do_eval(final=True)
    log(f"[{tag}] DONE. best greedyWER {st['best_sel'] * 100:.2f}% | best PER {st['best_per'] * 100:.2f}% | "
        f"OOM skips {st['oom_skips']} | NaN skips {st['nan_skips']} | restarts {st['restarts']}")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        print(time.strftime('%H:%M:%S'), 'FATAL worker exception:\n' + traceback.format_exc(), flush=True)
        sys.exit(3)
