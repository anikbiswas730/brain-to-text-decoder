"""
scripts/draw_pipeline.py — draws the B-T-S pipeline figure used in the README.

    python scripts/draw_pipeline.py            # -> assets/bts_pipeline.svg (+ .png if cairosvg is installed)

Every number in the figure is taken from config.py or from the reference run's
tables in results/; change them here if you re-run with different settings.
"""

import os
from xml.sax.saxutils import escape

W, H = 1800, 1140
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
PAL = {  # fill, stroke
    'data': ('#F2F4F7', '#6B7280'), 'am': ('#E3EBF7', '#4C72B0'), 'search': ('#E4F2E7', '#3E8E57'),
    'llm': ('#EDE8F6', '#7462A8'), 'tune': ('#FCEEDF', '#D17A3A'), 'out': ('#2F3640', '#2F3640'),
    'opt': ('#FFFFFF', '#9CA3AF'),
}
out = []
BOLD = ' font-weight="bold"'
ITAL = ' font-style="italic"'


def box(x, y, w, h, kind, title=None, lines=(), num=None, dashed=False, fs=13, title_fs=15, rx=10,
        line_gap=19, center=False, top_pad=0):
    fill, stroke = PAL[kind]
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.8"{dash}/>')
    txt_col = '#FFFFFF' if kind == 'out' else '#1F2937'
    cy = y + 26 + top_pad
    anchor, tx = ('middle', x + w / 2) if center else ('start', x + 14)
    if title:
        t = escape(title)
        if num is not None:
            out.append(f'<circle cx="{x + 24}" cy="{cy - 5}" r="12" fill="{stroke}"/>'
                       f'<text x="{x + 24}" y="{cy}" font-size="13" font-weight="bold" fill="#fff" '
                       f'text-anchor="middle" font-family="{FONT}">{num}</text>')
            tx = x + 44
        out.append(f'<text x="{tx}" y="{cy}" font-size="{title_fs}" font-weight="bold" fill="{txt_col}" '
                   f'text-anchor="{anchor}" font-family="{FONT}">{t}</text>')
        cy += line_gap + 6
    for ln in lines:
        bold = ln.startswith('**')
        ln = ln.strip('*')
        col = '#6B7280' if ln.startswith('(') and kind != 'out' else txt_col
        lx = x + 14 if not center else x + w / 2
        out.append(f'<text x="{lx}" y="{cy}" font-size="{fs}" fill="{col}" text-anchor="{anchor}" '
                   f'font-family="{FONT}"{BOLD if bold else ''}>{escape(ln)}</text>')
        cy += line_gap


def arrow(pts, color='#374151', dashed=False, width=2.2):
    d = 'M ' + ' L '.join(f'{x} {y}' for x, y in pts)
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    mk = 'arrowD' if dashed else 'arrow'
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{mk})"/>')


def text(x, y, s, fs=13, col='#374151', bold=False, anchor='start', italic=False):
    out.append(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{col}" text-anchor="{anchor}" font-family="{FONT}"'
               f'{BOLD if bold else ''}{ITAL if italic else ''}>{escape(s)}</text>')


def lane(x, y, w, h, label):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#FAFBFC" stroke="#D1D5DB" '
               f'stroke-width="1.5"/>')
    text(x + 18, y + 28, label, fs=16, col='#111827', bold=True)


# ---------------------------------------------------------------------------
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
out.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker>'
           '<marker id="arrowD" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9CA3AF"/></marker></defs>')
out.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
text(24, 36, "B-T-S pipeline — Brain-to-Text '25 (v6.4): intracortical neural activity → phonemes → words",
     fs=21, col='#111827', bold=True)
text(W - 24, 36, 'fold-C WER 3.38 %', fs=19, col='#D17A3A', bold=True, anchor='end')

# ======================= lane 1: acoustic model ===========================
lane(20, 56, 1760, 450, '① ACOUSTIC MODEL — two cross-fit folds, one per Kaggle T4 (~10.7 h each)')
box(40, 100, 250, 385, 'data', 'Neural features', [
    'x ∈ ℝ^(T × 512), 20 ms bins', '', '256 electrodes (4 arrays)', '× 2 features:', '  threshold crossings',
    '  spike-band power', '', '45 sessions, 2023 – 2025', 'train 8 072 · val 1 426', 'test 1 450 trials', '',
    'per-channel z-score,', 'clip ±5'])

# cross-fit split
box(310, 100, 310, 385, 'tune', 'Trial-level cross-fit of val', [])
labs = 'ABABCABABC'
cols = {'A': '#4C72B0', 'B': '#3E8E57', 'C': '#D17A3A'}
for i, l in enumerate(labs):
    cx = 326 + i * 28
    out.append(f'<rect x="{cx}" y="140" width="24" height="24" rx="4" fill="{cols[l]}"/>')
    text(cx + 12, 157, l, fs=12, col='#fff', bold=True, anchor='middle')
text(326, 184, 'one (session, block) of val, in trial order', fs=11.5, col='#6B7280', italic=True)
yy = 214
for ln, c in [('Model A: train + val-A (GPU 0)', cols['A']), ('Model B: train + val-B (GPU 1)', cols['B']),
              ('Fold C: seen by neither model', cols['C'])]:
    out.append(f'<rect x="326" y="{yy - 11}" width="12" height="12" rx="2" fill="{c}"/>')
    text(346, yy, ln, fs=13)
    yy += 24
for ln in ['', 'val split: A 572 · B 570 · C 284', '', 'Every val trial is decoded only by', 'the model(s) that never saw it —',
           'C by both, exactly like test.', '', '(trial-level, because Kaggle test trials', '(are interleaved with val trials',
           '(inside the same recording blocks)']:
    text(326, yy, ln.lstrip('('), fs=12.5, col='#6B7280' if ln.startswith('(') else '#1F2937')
    yy += 18

# model
box(640, 100, 850, 385, 'am', 'B2TNetV4 — Patch-Conv · BiGRU · Conformer, CTC (30.15 M params)', [])
blocks = [('Day adapter', ['rank-32 resid.', 'per session', '+ generic slot', 'id. at init']),
          ('Gaussian', ['smoothing', 'σ = 2 bins']),
          ('Time mask', ['(training)', 'in-model']),
          ('Patch', ['Conv1d', 'k 14 · s 4', '→ d 512']),
          ('4 × Res-', ['BiGRU', '384 / dir.', 'packed']),
          ('3 × Con-', ['former', 'RoPE MHSA', '8 heads', 'conv k 15']),
          ('CTC head', ['Linear →', '41 classes', 'log-softmax'])]
bx, bw, gap = 656, 104, 15
for i, (t, ls) in enumerate(blocks):
    x = bx + i * (bw + gap)
    out.append(f'<rect x="{x}" y="150" width="{bw}" height="140" rx="8" fill="#FFFFFF" stroke="#4C72B0" stroke-width="1.5"/>')
    text(x + bw / 2, 174, t, fs=12.5, bold=True, anchor='middle', col='#1F2937')
    for k, l in enumerate(ls):
        text(x + bw / 2, 195 + 18 * k, l, fs=11.5, anchor='middle', col='#6B7280' if l.startswith('(') else '#1F2937')
    if i:
        arrow([(x - gap, 220), (x - 1, 220)], width=1.8)
# aux heads
for i in (4, 5):
    x = bx + i * (bw + gap) + bw / 2
    arrow([(x, 290), (x, 322)], color='#9CA3AF', dashed=True, width=1.6)
out.append(f'<rect x="{bx + 4 * (bw + gap) - 6}" y="324" width="{2 * bw + gap + 12}" height="30" rx="6" fill="#FFFFFF" '
           f'stroke="#9CA3AF" stroke-dasharray="6 4"/>')
text(bx + 4 * (bw + gap) + bw + gap / 2, 344, 'inter-CTC heads (training only)', fs=11.5, anchor='middle', col='#6B7280')
for k, ln in enumerate([
        'Loss = CTC + 0.3 · inter-CTC + 0.2 · CR-CTC (consistency between two augmented views)',
        'Augment: speed ×0.9–1.1, white noise, offsets, gain, electrode dropout, time masking',
        'AdamW 7e-4, time-based cosine to the session deadline · EMA 0.999 · day dropout 0.15',
        'Continue-trained from the previous run; memory guard + resume keeps 11 h runs alive']):
    text(656, 390 + 22 * k, ln, fs=12.5)

box(1510, 100, 250, 385, 'am', 'Phoneme posteriors', [
    'log P(c_t | x)', 'shape [T′ × 41]', 'T′ = ⌊(T − 14)/4⌋ + 1', '', '39 CMU phonemes', '+ CTC blank',
    '+ word boundary “|”', '', 'held-out greedy PER', 'A 6.88 % · B 6.89 %', '',
    'checkpoint per fold picked', 'by real beam WER on its', 'own held-out trials'])
arrow([(290, 290), (309, 290)])
arrow([(620, 290), (639, 290)])
arrow([(1490, 290), (1509, 290)])

# ======================= lane 2: decoding =================================
lane(20, 540, 1760, 525, '② DECODING — per trial (val: out-of-fold models only · test: both models)')
arrow([(1635, 485), (1635, 520), (240, 520), (240, 596)])
text(900, 514, 'emissions of every model that never trained on the trial', fs=12.5, col='#6B7280',
     italic=True, anchor='middle')

bw2, xs = 405, [40, 480, 920, 1360]
yA, hA, yB, hB = 598, 180, 840, 180
box(xs[0], yA, bw2, hA, 'search', 'Lexicon beam search', [
    'flashlight LexiconDecoder + KenLM 4-gram', 'beam 1200 · n-best 100 per LM weight',
    'λ_LM ∈ {0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5}', '4 610 decodes in 6 h 49 m (1 process:', '(19 GB KenLM) → cached to gen_cache.pkl.gz'],
    num=1)
box(xs[1], yA, bw2, hA, 'search', 'Candidate pool + joint sweep', [
    'dedup over models × λ × n-best', '≈ 241 hyps / val trial · ≈ 348 / test trial',
    'sweep n-best × λ-subset: 508 combos,', '5-fold session-grouped CV on A∪B', '→ plateau; pinned pool kept (CV 4.99 %)'], num=2)
box(xs[2], yA, bw2, hA, 'am', 'Exact features + stage-1 fusion', [
    'am = mean over models of log P_CTC(W | x)', '(exact sequence likelihood, not posterior avg.)',
    'ng = KenLM ln P(W)   ·   nw = #words', '**s₁ = am + 2.9 · ng + 0.5 · nw', 'fold C: 5.96 → 4.62 %'], num=3)
box(xs[3], yA, bw2, hA, 'search', 'Phonetic-neighbour expansion', [
    'top 8 hyps: swap one word for a lexicon', 'neighbour at phoneme edit distance ≤ 1',
    '≤ 20 neighbours / word · ≤ 600 new / trial', 'pool ≈ 740 hyps · stage-1 re-tuned', 'fold C: 4.62 → 3.49 %'], num=4)
for i in range(3):
    arrow([(xs[i] + bw2, yA + 90), (xs[i + 1] - 1, yA + 90)])
arrow([(xs[3] + bw2 / 2, yA + hA), (xs[3] + bw2 / 2, yB - 1)])

box(xs[3], yB, bw2, hB, 'llm', 'Task-adapted LLM scoring', [
    'shortlist: top 24 by s₁ (+ every 1-best)', 'Llama-3.1-8B · QLoRA nf4 · LoRA r 16',
    'next-word fine-tune on train transcripts', 'dev NLL/token 8.20 → 3.09 (best epoch 1)',
    'llm = ln P_LLM(W) · fold C: 3.49 → 3.38 %'], num=5)
box(xs[2], yB, bw2, hB, 'tune', 'Fluency-gated log-linear fusion', [
    'f = LLM log-prob/token of the s₁-best hyp', 'τ = 5th percentile of f (−5.96)',
    '**f ≥ τ: s = am + 2.5 ng + 1.6 llm − 0.5 nw', '**f < τ: s = am + 0.5 ng + 0.9 llm − 4.0 nw',
    '(weights on A∪B, gate checked on C, refit)'], num=6)
box(xs[1], yB, bw2, 128, 'opt', 'Selective GEC (optional)', [
    'LoRA corrector rewrites low-fluency /', 'near-tie trials; re-scored with am + ng',
    '(fold C 3.11 → 3.28 % → NOT adopted)'], num=7, dashed=True)
box(xs[0], yB, bw2, hB, 'out', 'Transcript', [
    'argmax_W s → text', 'submission.csv — 1 450 test trials', '4.4 % of test trials in the f < τ regime',
    '27.2 % changed vs flashlight 1-best'], title_fs=16, num=8)
arrow([(xs[3] - 1, yB + 90), (xs[2] + bw2 + 1, yB + 90)])
arrow([(xs[2] - 1, yB + 160), (xs[0] + bw2 + 1, yB + 160)])
arrow([(xs[2] - 1, yB + 64), (xs[1] + bw2 + 1, yB + 64)], color='#9CA3AF', dashed=True)
arrow([(xs[1] - 1, yB + 64), (xs[0] + bw2 + 1, yB + 64)], color='#9CA3AF', dashed=True)
text(xs[1] + bw2 / 2, yB + 153, 'main path', fs=12, col='#6B7280', italic=True, anchor='middle')

# ======================= protocol strip ===================================
text(24, 1094, 'Tuning protocol:', fs=14, bold=True, col='#111827')
text(160, 1094, 'every weight is chosen on out-of-fold folds A∪B (1 142 trials) · verified once on fold C (284 trials, '
     'seen by no model and no tuner) · refit on A∪B∪C for the test submission.', fs=14)
text(24, 1122, 'Fold-C WER:', fs=14, bold=True, col='#111827')
text(125, 1122, 'greedy 44.63 %  →  flashlight 1-best 5.96 %  →  pool rescoring 4.62 %  →  + expansion 3.49 %  →  '
     '+ adapted LLM 3.38 %  →  + fluency gate 3.38 %', fs=14)
out.append('</svg>')

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svg = os.path.join(root, 'assets', 'bts_pipeline.svg')
    os.makedirs(os.path.dirname(svg), exist_ok=True)
    open(svg, 'w', encoding='utf-8').write('\n'.join(out))
    print('->', svg)
    try:
        import cairosvg
        cairosvg.svg2png(url=svg, write_to=svg[:-4] + '.png', output_width=W * 2)
        print('->', svg[:-4] + '.png')
    except ImportError:
        print('pip install cairosvg to also render the PNG')
