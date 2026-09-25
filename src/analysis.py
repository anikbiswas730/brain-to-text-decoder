"""
src/analysis.py — the reporting half of the decoding run (notebook BLOCKs 4.5b-4.5d),
plus two summary figures derived from tables the run already writes.

Every function takes plain arrays / dicts, writes to `tables_dir` / `figures_dir`
and returns what it wrote, so it can be re-run offline from the saved tables.
Plotting is always wrapped in `figure_guard`: a matplotlib bug must never cost
the submission of an ~9 h run.

    dataset_statistics        tables/dataset_summary.csv, dataset_trials_per_day.csv
    ablation_table            tables/ablation_decoding.csv, figures/ablation_decoding.png
    oracle_vs_k               tables/oracle_vs_k.csv,        figures/oracle_vs_k.png
    fluency_gate_figure       figures/fluency_gate_and_weight_surface.png
    per_day_analysis          tables/{outlier,clean}_days_analysis.csv, figures/per_day_wer.png
    error_analysis            tables/error_analysis_{search_vs_model,utterances}.csv
    joint_sweep_figure        figures/joint_sweep.png        (from joint_sweep_all_combos.csv)
    error_analysis_figure     figures/error_analysis.png     (from error_analysis_search_vs_model.csv)
"""

import os

import numpy as np
import pandas as pd

from src.metrics import official_per, official_wer, remove_punctuation, utt_word_errors
from src.utils import figure_guard, save_fig, save_table

TUNE_COLOR, VERIFY_COLOR = '#4C72B0', '#DD8452'


def _plt():
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
def dataset_statistics(readers, tables_dir, log=print):
    rows = []
    for s, r in readers.items():
        for m in r.meta:
            rows.append({'split': s, 'session': m['session'], 'block': m['block'], 'T': m['n'],
                         'n_words': len(remove_punctuation(m['sentence']).split()),
                         'n_phon': len(m['phonemes'])})
    df = pd.DataFrame(rows)
    per_day = df.pivot_table(index='session', columns='split', values='T', aggfunc='count',
                             fill_value=0).reset_index()
    save_table(per_day, 'dataset_trials_per_day.csv', tables_dir)
    summ = df.groupby('split').agg(trials=('T', 'size'), sessions=('session', 'nunique'),
                                   mean_T=('T', 'mean'), median_T=('T', 'median'), max_T=('T', 'max'),
                                   mean_words=('n_words', 'mean')).reset_index()
    log(summ.to_string(index=False))
    save_table(summ, 'dataset_summary.csv', tables_dir)
    train_sent = set(remove_punctuation(m['sentence']) for m in readers['train'].meta)
    val_sent = [remove_punctuation(m['sentence']) for m in readers['val'].meta]
    log(f'val sentences also present verbatim in train: '
        f'{np.mean([s in train_sent for s in val_sent]) * 100:.1f}%')
    return summ


# ---------------------------------------------------------------------------
def ablation_table(abl, oracle, wer_on, tune_idx, verify_idx, all_idx, tables_dir, figures_dir,
                   log=print):
    """abl: {system name: {utt: text}} in stack order; oracle: {pool name: (tune, verify)}."""
    rows = []
    for name, preds in abl.items():
        rows.append({'system': name, 'tune_A∪B_WER_%': round(wer_on(tune_idx, preds) * 100, 2),
                     'verify_C_WER_%': round(wer_on(verify_idx, preds) * 100, 2) if verify_idx else np.nan,
                     'all_OOF_WER_%': round(wer_on(all_idx, preds) * 100, 2)})
    for name, (ot, ov) in oracle.items():
        rows.append({'system': f'oracle: {name}', 'tune_A∪B_WER_%': round(ot * 100, 2),
                     'verify_C_WER_%': round(ov * 100, 2), 'all_OOF_WER_%': np.nan})
    df = pd.DataFrame(rows)
    log('\n' + df.to_string(index=False))
    save_table(df, 'ablation_decoding.csv', tables_dir)
    plot_ablation(df, figures_dir)
    return df


def plot_ablation(df, figures_dir):
    plt = _plt()
    with figure_guard('ablation'):
        d = df[~df['system'].str.startswith('oracle')]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        x = np.arange(len(d))
        w = 0.38
        ax.bar(x - w / 2, d['tune_A∪B_WER_%'], w, label='tune (A∪B, OOF)', color=TUNE_COLOR)
        ax.bar(x + w / 2, d['verify_C_WER_%'], w, label='verify (C, unseen by both)', color=VERIFY_COLOR)
        for xi, (a, b) in enumerate(zip(d['tune_A∪B_WER_%'], d['verify_C_WER_%'])):
            ax.text(xi - w / 2, a, f'{a:.2f}', ha='center', va='bottom', fontsize=7)
            ax.text(xi + w / 2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(d['system'], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('WER (%)')
        ax.legend(frameon=False)
        ax.set_title('Decoding ablation (official WER)')
        save_fig(fig, 'ablation_decoding', figures_dir)


# ---------------------------------------------------------------------------
def oracle_vs_k(pool, errs, nref, idx, W, utt_scores, final_wer, tables_dir, figures_dir,
                ks=(1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512)):
    """Oracle WER of the top-k candidates under the stage-1 ranking: how much a
    perfect reranker could still recover from the pool at each shortlist size."""
    plt = _plt()
    ov = []
    for k in ks:
        e = 0.0
        for j in idx:
            S = utt_scores(pool, j, W)
            e += errs[j][np.argsort(-S)[:k]].min()
        ov.append(e / nref[idx].sum() * 100)
    df = pd.DataFrame({'k': list(ks), 'oracle_wer_%': ov})
    save_table(df, 'oracle_vs_k.csv', tables_dir)
    with figure_guard('oracle_vs_k'):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(ks, ov, marker='o')
        ax.set_xscale('log', base=2)
        ax.axhline(final_wer * 100, ls='--', c='gray', label='final system')
        ax.set_xlabel('top-k candidates (stage-1 ranking)')
        ax.set_ylabel('oracle WER (%)')
        ax.set_title('How much the pool can still give')
        ax.legend(frameon=False)
        save_fig(fig, 'oracle_vs_k', figures_dir)
    return df


# ---------------------------------------------------------------------------
def fluency_gate_figure(pool, errs, flu, idx, W_stage1, gate, A2, B2, G2, nref, utt_scores, figures_dir):
    """Left: fluency of the acoustic-best hypothesis, split by whether stage 1 got the
    utterance right, with the gate threshold. Right: the hi-regime error surface over
    (KenLM, LLM) at the chosen word bonus, with the chosen weights."""
    plt = _plt()
    with figure_guard('fluency_gate'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        ok = np.array([errs[j][int(np.argmax(utt_scores(pool, j, dict(W_stage1, llm=0.0))))] == 0 for j in idx])
        f = flu[idx]
        axes[0].hist(f[ok], bins=40, alpha=0.6, label='stage-1 correct')
        axes[0].hist(f[~ok], bins=40, alpha=0.6, label='stage-1 wrong')
        if np.isfinite(gate['tau']):
            axes[0].axvline(gate['tau'], c='k', ls='--', label='gate tau')
        axes[0].set_xlabel('adapted-LLM log-prob / token of acoustic-best hypothesis')
        axes[0].legend(frameon=False)
        eh = gate['err_hi_grid']
        gi = int(np.argmin(np.abs(G2 - gate['hi']['nw'])))
        im = axes[1].imshow(eh[:, :, gi] / nref[idx].sum() * 100, origin='lower', aspect='auto',
                            cmap='viridis', extent=[B2[0], B2[-1], A2[0], A2[-1]])
        axes[1].scatter([gate['hi']['llm']], [gate['hi']['ng']], c='red', marker='*', s=150)
        axes[1].set_xlabel('LLM weight')
        axes[1].set_ylabel('KenLM weight')
        axes[1].set_title('hi-regime errors (share of all words, %)')
        fig.colorbar(im, ax=axes[1])
        save_fig(fig, 'fluency_gate_and_weight_surface', figures_dir)


# ---------------------------------------------------------------------------
def per_day_analysis(val_meta, val_refs, idx, greedy_ids, base_pred, final_pred, final_regime,
                     tables_dir, figures_dir, out_root=None, log=print):
    """Per-session OOF WER; a session is an OUTLIER when its final WER exceeds
    max(2 x median, median + 5 pp) — robust to the handful of very short sessions."""
    plt = _plt()
    rows = []
    for s in sorted(set(val_meta[j]['session'] for j in idx)):
        sj = [j for j in idx if val_meta[j]['session'] == s]
        per_d = official_per([greedy_ids[j] for j in sj], [list(val_meta[j]['phonemes']) for j in sj])[0]
        wf, ef, nf = official_wer([val_refs[j] for j in sj], [final_pred[j] for j in sj])
        wb = official_wer([val_refs[j] for j in sj], [base_pred[j] for j in sj])[0]
        rows.append({'session': s, 'n_trials': len(sj), 'n_words': nf,
                     'mean_T': float(np.mean([val_meta[j]['n'] for j in sj])),
                     'per_greedy_%': round(per_d * 100, 2), 'wer_flashlight_%': round(wb * 100, 2),
                     'wer_final_%': round(wf * 100, 2), 'delta_pp': round((wf - wb) * 100, 2),
                     'word_errors': ef, 'frac_lo_regime': float(np.mean([final_regime.get(j) == 'lo' for j in sj]))})
    df = pd.DataFrame(rows)
    df['share_of_all_errors_%'] = (df['word_errors'] / max(df['word_errors'].sum(), 1) * 100).round(2)
    med = df['wer_final_%'].median()
    df['outlier'] = df['wer_final_%'] > max(2 * med, med + 5.0)
    save_table(df[df['outlier']].sort_values('wer_final_%', ascending=False), 'outlier_days_analysis.csv', tables_dir)
    save_table(df[~df['outlier']], 'clean_days_analysis.csv', tables_dir)
    if out_root:                    # the Kaggle convention: both CSVs also in the output root
        import shutil
        for n in ('outlier_days_analysis.csv', 'clean_days_analysis.csv'):
            shutil.copy(os.path.join(tables_dir, n), os.path.join(out_root, n))
    log(f'outlier days (> max(2x median, median+5pp), median={med:.2f}%):')
    log(df[df['outlier']][['session', 'n_trials', 'per_greedy_%', 'wer_flashlight_%', 'wer_final_%']]
        .to_string(index=False))
    with figure_guard('per_day'):
        fig, ax = plt.subplots(figsize=(12, 3.6))
        x = np.arange(len(df))
        ax.bar(x - 0.2, df['wer_flashlight_%'], 0.4, label='flashlight 1-best', color='#BBBBBB')
        ax.bar(x + 0.2, df['wer_final_%'], 0.4, label='final', color=TUNE_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(df['session'].str.replace('t15.', ''), rotation=70, fontsize=7)
        ax.set_ylabel('OOF WER (%)')
        ax.legend(frameon=False)
        ax.set_title('Per-session out-of-fold WER')
        save_fig(fig, 'per_day_wer', figures_dir)
    return df


# ---------------------------------------------------------------------------
def error_analysis(idx, val_meta, val_refs, final_pred, final_pool, final_flu, gate, lex, ngram,
                   utt_ems_val, llm_score_fn, device, acoustic_scores, utt_scores, tables_dir, log=print):
    """Classify every wrong utterance of the final system:

      reference contains OOV word          no lexicon path can ever produce it
      model error: in pool but outscored   the reranker had the answer and preferred another
      search error                         the reference was never generated, but WOULD have
                                           won under the final weights -> more search helps
      model error: not in pool, would lose even if generated it would lose -> the scores
                                           (acoustic model / priors) are the bottleneck
    """
    items = []
    for j in idx:
        ref = remove_punctuation(val_refs[j])
        hyp = final_pred[j]
        if ref == hyp:
            continue
        e, n = utt_word_errors(ref, hyp)
        items.append((j, ref, hyp, e, n, ref in set(final_pool.texts[j]), lex.targets(ref) is None))
    ref_texts = [r for (_, r, _, _, _, ip, oov) in items if not ip and not oov]
    ref_llm = dict(zip(ref_texts, zip(*llm_score_fn(ref_texts)))) if (llm_score_fn and ref_texts) else {}
    rows = []
    for (j, ref, hyp, e, n, in_pool, oov) in items:
        if oov:
            kind = 'reference contains OOV word'
        elif in_pool:
            kind = 'model error: reference in pool but outscored'
        else:
            W = gate['lo'] if (gate['lo'] is not None and final_flu[j] < gate['tau']) else gate['hi']
            am = acoustic_scores([ref], utt_ems_val[j], lex, device).mean(1)[0]
            ng = ngram.ln(ref) if ngram else 0.0
            ll = ref_llm[ref][0] if ref in ref_llm else 0.0
            s_ref = am + W['ng'] * ng + W.get('llm', 0) * ll + W['nw'] * len(ref.split())
            s_hyp = utt_scores(final_pool, j, W)[final_pool.texts[j].index(hyp)]
            kind = ('search error: reference would win but never entered the pool' if s_ref > s_hyp
                    else 'model error: reference not in pool and would lose anyway')
        rows.append({'session': val_meta[j]['session'], 'ref': ref, 'hyp': hyp, 'word_errors': e, 'kind': kind})
    df = pd.DataFrame(rows)
    summ = None
    if len(df):
        summ = df.groupby('kind').agg(utterances=('ref', 'size'), word_errors=('word_errors', 'sum')).reset_index()
        summ['share_of_word_errors_%'] = (summ['word_errors'] / summ['word_errors'].sum() * 100).round(1)
        log('\n' + summ.to_string(index=False))
        save_table(summ, 'error_analysis_search_vs_model.csv', tables_dir)
        save_table(df, 'error_analysis_utterances.csv', tables_dir)
    return summ, df


# ---------------------------------------------------------------------------
# summary figures from saved tables (also used to build the README figures)
# ---------------------------------------------------------------------------
def joint_sweep_figure(sweep_df, figures_dir, pinned_lw=None):
    """Left: best session-CV WER vs how many KenLM weights feed the pool, per n-best.
    Right: CV WER of a pool built from ONE KenLM weight. Shows where the pool
    composition matters (few / extreme weights) and where it is a plateau."""
    plt = _plt()
    d = sweep_df.copy()
    with figure_guard('joint_sweep'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'width_ratios': [1.15, 1]})
        for nb, g in d.groupby('nbest'):
            b = g.groupby('n_lw')['cv_wer_%'].min()
            axes[0].plot(b.index, b.values, marker='o', label=f'n-best {nb}')
        best = d['cv_wer_%'].min()
        axes[0].axhline(best, ls=':', c='gray', lw=1)
        axes[0].set_xlabel('number of KenLM weights in the pool (best subset)')
        axes[0].set_ylabel('session-CV WER on tune (%)')
        axes[0].set_title('Pool composition: plateau from 5 weights on')
        axes[0].text(0.98, 0.55, 'all four n-best curves coincide', transform=axes[0].transAxes,
                     ha='right', fontsize=8, color='gray')
        axes[0].legend(frameon=False, fontsize=8)
        s1 = d[d['n_lw'] == 1].copy()
        s1['lw'] = s1['lm_weights'].astype(float)
        for nb, g in s1.groupby('nbest'):
            g = g.sort_values('lw')
            axes[1].plot(g['lw'], g['cv_wer_%'], marker='o', label=f'n-best {nb}')
        axes[1].axhline(best, ls=':', c='gray', lw=1, label=f'best any subset ({best:.2f}%)')
        axes[1].set_xlabel('single KenLM weight used to generate the pool')
        axes[1].set_ylabel('session-CV WER on tune (%)')
        axes[1].set_title('One weight alone')
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        save_fig(fig, 'joint_sweep', figures_dir)


def error_analysis_figure(summ, figures_dir):
    """Share of the final system's word errors by cause (from error_analysis_search_vs_model.csv)."""
    plt = _plt()
    order = ['model error: reference not in pool and would lose anyway',
             'search error: reference would win but never entered the pool',
             'model error: reference in pool but outscored',
             'reference contains OOV word']
    short = {order[0]: 'not in pool, would lose anyway\n(acoustic model / priors)',
             order[1]: 'search error: would win,\nnever generated',
             order[2]: 'in pool but outscored\n(reranker)',
             order[3]: 'reference has an\nout-of-lexicon word'}
    colors = ['#8172B3', '#55A868', '#C44E52', '#937860']
    s = summ.set_index('kind').reindex([k for k in order if k in set(summ['kind'])])
    with figure_guard('error_analysis'):
        fig, ax = plt.subplots(figsize=(8, 2.9))
        y = np.arange(len(s))
        ax.barh(y, s['share_of_word_errors_%'], color=colors[:len(s)])
        for yi, (sh, we, ut) in enumerate(zip(s['share_of_word_errors_%'], s['word_errors'], s['utterances'])):
            ax.text(sh + 0.6, yi, f'{sh:.1f}%  ({int(we)} words, {int(ut)} utts)', va='center', fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([short[k] for k in s.index], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(s['share_of_word_errors_%']) * 1.45)
        ax.set_xlabel('share of the final system\'s word errors (%)')
        ax.set_title('Where the remaining errors come from (all out-of-fold val trials)')
        save_fig(fig, 'error_analysis', figures_dir)
