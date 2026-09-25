"""
evaluate.py — the reporting stage: turn `tables/val_predictions.csv` (written by
decode_llm.py) into the numbers and figures a paper needs.

    python evaluate.py                       # reads tables/val_predictions.csv
    python evaluate.py --tables_dir tables --figures_dir figures

Produces:

    tables/ablation.csv              WER/CER per system, per fold
    tables/per_day_analysis.csv      WER per recording session, sorted, with an `outlier`
                                     flag (WER > max(2 x median, median + 5 pp))
    tables/error_breakdown.csv       substitutions / deletions / insertions
    tables/val_summary.json          headline numbers of the final system

(`decode_llm.py` already writes the notebook-format `outlier_days_analysis.csv` /
`clean_days_analysis.csv`; this script does not overwrite them.)
    figures/ablation_wer.png         the stack, tune vs verify
    figures/wer_by_day.png           per-day WER with the outlier threshold

Read the columns in this order:

  * **verify_C** is the honest number. Nothing in the pipeline was selected on
    fold C, so it is the only column that estimates unseen-data performance.
  * **tune_AuB** is what the sweep optimised, so it is optimistic by construction
    — the gap between the two columns is how much tune noise got fitted.
  * the **FINAL** row is refit on A u B u C and is therefore IN-SAMPLE on C. It is
    reported because it is the system that produced the submission, not as an
    estimate of anything.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from src.metrics import bootstrap_wer_sd, official_cer, official_wer, remove_punctuation
from src.utils import figure_guard, init_matplotlib, save_fig, save_table

META_COLS = ['idx', 'session', 'block', 'fold', 'reference']


def parse_args():
    p = argparse.ArgumentParser(description='Brain-to-Text v6.4 validation report')
    p.add_argument('--predictions', default=None,
                   help='val_predictions.csv (default: <tables_dir>/val_predictions.csv)')
    p.add_argument('--tables_dir', default=C.DEFAULT_TABLES_DIR)
    p.add_argument('--figures_dir', default=C.DEFAULT_FIGURES_DIR)
    p.add_argument('--bootstrap', type=int, default=400)
    return p.parse_args()


def error_counts(ref, hyp):
    """(substitutions, deletions, insertions) from the edit-distance alignment."""
    r, h = remove_punctuation(ref or '').split(), remove_punctuation(hyp or '').split()
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            s += int(r[i - 1] != h[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dele, ins


def main():
    args = parse_args()
    plt = init_matplotlib()
    path = args.predictions or os.path.join(args.tables_dir, 'val_predictions.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — run decode_llm.py first (it writes this file).')
    df = pd.read_csv(path).fillna('')
    systems = [c for c in df.columns if c not in META_COLS]
    refs = df['reference'].tolist()
    tune = df['fold'].isin(C.LLM_CFG['tune_labels']).values
    verify = df['fold'].isin(C.LLM_CFG['verify_labels']).values
    print(f'{len(df)} out-of-fold val trials | tune(A u B) {tune.sum()} | verify(C) {verify.sum()} '
          f'| {len(systems)} systems')

    # ---- ablation ---------------------------------------------------------
    rows = []
    for s in systems:
        hyps = df[s].tolist()
        row = {'system': s,
               'tune_AuB_WER_%': round(100 * official_wer(list(np.array(refs)[tune]),
                                                          list(np.array(hyps)[tune]))[0], 2),
               'verify_C_WER_%': (round(100 * official_wer(list(np.array(refs)[verify]),
                                                           list(np.array(hyps)[verify]))[0], 2)
                                  if verify.any() else float('nan')),
               'all_OOF_WER_%': round(100 * official_wer(refs, hyps)[0], 2),
               'all_OOF_CER_%': round(100 * official_cer(refs, hyps)[0], 2)}
        if verify.any():
            row['verify_C_sd_pp'] = round(100 * bootstrap_wer_sd(list(np.array(refs)[verify]),
                                                                 list(np.array(hyps)[verify]),
                                                                 args.bootstrap, C.SEED), 2)
        rows.append(row)
    abl = pd.DataFrame(rows)
    print('\n' + abl.to_string(index=False))
    save_table(abl, 'ablation.csv', args.tables_dir)
    print('\nverify_C is the only column nothing was selected on; the FINAL row is refit on '
          'A u B u C and is in-sample on C.')

    final = systems[-1]

    # ---- error breakdown --------------------------------------------------
    brk = []
    for s in systems:
        S = D = I = N = 0
        for r, h in zip(refs, df[s].tolist()):
            a, b, c = error_counts(r, h)
            S += a
            D += b
            I += c
            N += len(remove_punctuation(r or '').split())
        brk.append({'system': s, 'sub_%': round(100 * S / max(N, 1), 2),
                    'del_%': round(100 * D / max(N, 1), 2), 'ins_%': round(100 * I / max(N, 1), 2),
                    'sub_share_of_errors_%': round(100 * S / max(S + D + I, 1), 1)})
    brk = pd.DataFrame(brk)
    print('\n' + brk.to_string(index=False))
    save_table(brk, 'error_breakdown.csv', args.tables_dir)

    # ---- per-day ----------------------------------------------------------
    # Columns for the first system (greedy), the beam baseline, and the final one,
    # so a bad day can be traced to the acoustic model or to the decoder.
    track = [s for s in (systems[0], systems[1] if len(systems) > 1 else None, final) if s]
    day_rows = []
    for sess, g in df.groupby('session'):
        row = {'session': sess, 'n_trials': len(g)}
        for s in track:
            w, ed, n = official_wer(g['reference'].tolist(), g[s].tolist())
            row[f'WER_% [{s}]'] = round(100 * w, 2)
            if s == final:
                row.update({'ref_words': int(n), 'edits': int(ed), 'WER_%': round(100 * w, 2)})
        day_rows.append(row)
    day = pd.DataFrame(day_rows).sort_values('WER_%', ascending=False).reset_index(drop=True)
    # the notebook's rule: a day is an outlier at twice the median, or 5 points above it,
    # whichever is larger — robust to the handful of very short sessions
    med = float(day['WER_%'].median())
    thr = max(2 * med, med + 5.0)
    day['outlier'] = day['WER_%'] > thr
    save_table(day, 'per_day_analysis.csv', args.tables_dir)
    print(f'\nper-day WER on "{final}": median {med:.2f}%, '
          f'outlier threshold {thr:.2f}%, {int(day["outlier"].sum())} outlier day(s): '
          f'{day[day["outlier"]]["session"].tolist() or "none"}')

    # ---- figures ----------------------------------------------------------
    with figure_guard('ablation_wer'):
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(abl) + 2))
        y = np.arange(len(abl))
        ax.barh(y - 0.2, abl['tune_AuB_WER_%'], height=0.4, label='tune (A u B)')
        ax.barh(y + 0.2, abl['verify_C_WER_%'], height=0.4, label='verify (C)')
        ax.set_yticks(y)
        ax.set_yticklabels(abl['system'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('WER (%)')
        ax.legend(frameon=False)
        ax.grid(axis='x', alpha=0.3)
        ax.set_title('Ablation: what each decoding stage is worth')
        save_fig(fig, 'ablation_wer', args.figures_dir)

    with figure_guard('wer_by_day'):
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(day)), 3.6))
        colors = ['tab:red' if o else 'tab:blue' for o in day['outlier']]
        ax.bar(range(len(day)), day['WER_%'], color=colors)
        ax.axhline(thr, ls='--', lw=1, c='k', label=f'outlier threshold {thr:.1f}%')
        ax.set_xticks(range(len(day)))
        ax.set_xticklabels(day['session'], rotation=90, fontsize=7)
        ax.set_ylabel('WER (%)')
        ax.legend(frameon=False)
        ax.set_title(f'Per-session WER — {final}')
        save_fig(fig, 'wer_by_day', args.figures_dir)

    summary = {'n_trials': int(len(df)), 'final_system': final,
               'tune_AuB_WER_%': float(abl.iloc[-1]['tune_AuB_WER_%']),
               'verify_C_WER_%': float(abl.iloc[-1]['verify_C_WER_%']),
               'all_OOF_WER_%': float(abl.iloc[-1]['all_OOF_WER_%']),
               'all_OOF_CER_%': float(abl.iloc[-1]['all_OOF_CER_%'])}
    import json
    json.dump(summary, open(os.path.join(args.tables_dir, 'val_summary.json'), 'w'), indent=1)
    print('\n' + json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
