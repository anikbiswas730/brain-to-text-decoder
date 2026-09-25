"""
src/gec.py — BLOCK 4.5e: selective generative error correction (optional).

A discriminative reranker can only choose among candidates it was given. GEC
instead lets an LLM *write* a transcript, conditioned on the top-k candidates of
the existing (expansion-augmented, LLM-scored) pool:

    Below are several candidate transcriptions of the same neural-decoded
    utterance ... Candidates:
    - <top-1>
    - <top-2>
    ...
    Corrected:

It is deliberately fenced in, because free generation amplifies exactly the
fluency bias that already dominates the error analysis:

  * SELECTIVE: only trials the fluency gate calls low-confidence (f < tau) or whose
    pool top-1/top-2 score margin is in the bottom `margin_percentile` are touched.
  * GROUNDED: best-of-n (1 greedy + n-1 sampled) generations are re-scored with
    the SAME formula the pool is ranked by — exact CTC log-likelihood of the brain
    signal + tuned KenLM + word-count terms — and a generation must not score below
    the base prediction (`accept_margin_nats`).
  * VERIFIED: a second LoRA adapter is trained on fold A u B only; the block is
    adopted only if its word-error gain on fold C clears `anchor_z` x the paired
    bootstrap SE. Otherwise the submission is untouched.

Reference run: 116/1426 val and 156/1450 test trials triggered; 34 val rewrites
accepted (33 of them contained an out-of-lexicon word); fold C 3.11% -> 3.28%
(-3.0 words, SE 1.6) -> NOT adopted. `reject_oov=True` closes the OOV loophole
(an OOV candidate has no acoustic score, so only LM + length were compared).
"""

import gc
import os
import random
import time

import numpy as np
import torch

from src.decoding import acoustic_scores, predict_texts, utt_scores
from src.metrics import remove_punctuation, utt_word_errors

GEC_PROMPT = (
    "Below are several candidate transcriptions of the same neural-decoded "
    "utterance. They may contain errors. Using the candidates, write the "
    "single most accurate transcription. Respond with only the corrected "
    "sentence, no explanation.\n\nCandidates:\n{cands}\n\nCorrected:"
)


# ---------------------------------------------------------------------------
# candidate selection, same ranking as the final system
# ---------------------------------------------------------------------------
def gate_weight_for(j, flu_arr, gate):
    """The per-utterance hi/lo weights predict_texts() uses."""
    if gate.get('lo') is not None and flu_arr[j] < gate['tau']:
        return gate['lo']
    return gate['hi']


def pool_top2_margin(pool, u, W):
    """Score gap between the pool's top-1 and top-2 candidates (inf if < 2 candidates)."""
    s = utt_scores(pool, u, W)
    if len(s) < 2:
        return float('inf')
    top2 = np.partition(s, -2)[-2:]
    return float(top2[-1] - top2[-2])


def topk_candidates_from_pool(pool, u, W, k):
    scores = utt_scores(pool, u, W)
    seen, out = set(), []
    for i in np.argsort(-scores)[:max(k, 1)]:
        t = pool.texts[u][i]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out or ['']


def gec_prompt_for_pool(pool, j, flu_arr, gate, k):
    cands = topk_candidates_from_pool(pool, j, gate_weight_for(j, flu_arr, gate), k)
    return GEC_PROMPT.format(cands='\n'.join(f'- {c}' for c in cands))


# ---------------------------------------------------------------------------
# the corrector: a separate QLoRA adapter, loss on the target span only
# ---------------------------------------------------------------------------
def _gec_ids(tok, prompt, target=None):
    p_ids = tok(prompt, add_special_tokens=False)['input_ids']
    if target is None:
        return p_ids, len(p_ids)
    t_ids = tok(' ' + target + tok.eos_token, add_special_tokens=False)['input_ids']
    return p_ids + t_ids, len(p_ids)


def finetune_llm_gec(model_name, examples, cfg, log=print):
    """examples: [{'prompt', 'target'}]. Separate adapter from the scoring (NWP) one,
    because the objective differs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    dev0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    use_amp = dev0 != 'cpu'
    hf_token = os.environ.get('HF_TOKEN')
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16 if use_amp else torch.float32,
                                 bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_cfg if use_amp else None,
        torch_dtype=torch.float32 if not use_amp else None,
        device_map={'': 0} if use_amp else None, token=hf_token)
    model.config.use_cache = False
    if use_amp:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
                                      'down_proj'],
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)

    packed = [_gec_ids(tok, ex['prompt'], ex['target']) for ex in examples]
    bs, epochs = cfg['ft_batch'], cfg['ft_epochs']
    accum = max(int(cfg.get('ft_grad_accum', 1)), 1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['ft_lr'])
    amp_dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def batchify(items):
        L = max(len(ids) for ids, _ in items)
        input_ids = torch.full((len(items), L), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), L), -100, dtype=torch.long)
        attn = torch.zeros((len(items), L), dtype=torch.long)
        for i, (ids, plen) in enumerate(items):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            labels[i, plen:len(ids)] = torch.tensor(ids[plen:])        # mask the prompt span
            attn[i, :len(ids)] = 1
        return input_ids.to(dev0), labels.to(dev0), attn.to(dev0)

    model.train()
    rng = random.Random(0)
    out = None
    for ep in range(epochs):
        order = list(range(len(packed)))
        rng.shuffle(order)
        opt.zero_grad()
        for step, s in enumerate(range(0, len(order), bs)):
            input_ids, labels, attn = batchify([packed[i] for i in order[s:s + bs]])
            with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        log(f'  [GEC] epoch {ep + 1}/{epochs} last-batch loss {out.loss.item():.3f}')
    model.eval()
    return model, tok


@torch.no_grad()
def gec_generate(model, tok, prompt, max_new_tokens=40, do_sample=False, temperature=1.0, top_p=1.0):
    dev0 = next(model.parameters()).device
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).to(dev0)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False)
    out = model.generate(**ids, **kw)
    text = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
    return text.strip().split('\n')[0].strip()


def gec_generate_candidates(model, tok, prompt, cfg):
    """1 greedy decode + (n_samples - 1) temperature samples, de-duplicated."""
    out = [gec_generate(model, tok, prompt, cfg['max_new_tokens'])]
    for _ in range(max(cfg.get('n_samples', 1) - 1, 0)):
        out.append(gec_generate(model, tok, prompt, cfg['max_new_tokens'], do_sample=True,
                                temperature=cfg.get('sample_temperature', 1.0),
                                top_p=cfg.get('sample_top_p', 1.0)))
    return list(dict.fromkeys(t for t in out if t))


def gec_score_candidate(text, ems, W, lex, ngram, device):
    """Score `text` exactly as a pool candidate is scored (CTC evidence + tuned KenLM
    + word count). -> (score, is_oov). An OOV text has no lexicon pronunciation, so
    acoustic_scores() returns its -1e9 floor and the acoustic term is dropped."""
    am = float(acoustic_scores([text], ems, lex, device).mean(1)[0])
    is_oov = am <= -1e8
    ng = ngram.ln(text) if ngram else 0.0
    return (0.0 if is_oov else am) + W['ng'] * ng + W['nw'] * len(text.split()), is_oov


def gec_correct(model, tok, prompt, ems, base_text, W, cfg, lex, ngram, device):
    """Best-of-n generation, kept only if it does not score below the base prediction.
    -> (chosen_text, used_gec, best_is_oov)"""
    cands = gec_generate_candidates(model, tok, prompt, cfg)
    if not cands:
        return base_text, False, False
    scored = [(gec_score_candidate(t, ems, W, lex, ngram, device), t) for t in cands]
    if cfg.get('reject_oov', False):
        scored = [x for x in scored if not x[0][1]]
        if not scored:
            return base_text, False, False
    (best_s, best_oov), best_t = max(scored, key=lambda x: x[0][0])
    base_s, _ = gec_score_candidate(base_text, ems, W, lex, ngram, device)
    if best_s >= base_s - cfg.get('accept_margin_nats', 0.0):
        return best_t, True, best_oov
    return base_text, False, False


def gec_paired_gain(idx, refs, pred_base, pred_gec, n_boot, seed=1337):
    """Words gained by pred_gec over pred_base on idx, and the paired-bootstrap SE of that gain."""
    if not idx:
        return 0.0, float('nan')
    eb = np.array([utt_word_errors(refs[j], pred_base[j])[0] for j in idx], dtype=np.float64)
    eg = np.array([utt_word_errors(refs[j], pred_gec[j])[0] for j in idx], dtype=np.float64)
    diff = eb - eg                      # positive -> GEC makes fewer errors
    gain = float(diff.sum())
    if not n_boot or len(idx) < 10:
        return gain, float('nan')
    I = np.random.default_rng(seed).integers(0, len(idx), (int(n_boot), len(idx)))
    return gain, float(np.std(diff[I].sum(1)))


# ---------------------------------------------------------------------------
# the whole block
# ---------------------------------------------------------------------------
def run_selective_gec(cfg, llm_name, pool_val, pool_test, flu_val, flu_test, gate, pred_val,
                      all_idx, tune_idx, verify_idx, oof, val_refs, utt_ems_val, utt_ems_test,
                      lex, ngram, device, wer_on, seed=1337, human_time=None, log=print):
    """-> {'adopted', 'val_pred', 'test_pred', 'summary'}. `val_pred` is pred_val itself
    unless GEC was adopted; `test_pred` holds only the rewritten test trials."""
    ht = human_time or (lambda s: f'{s:.0f}s')
    n_val, n_test = len(pool_val), len(pool_test)
    res = {'adopted': False, 'val_pred': pred_val, 'test_pred': {}, 'summary': {}}
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        log(f'GPU mem free for GEC: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB')

    # ---- selective trigger: low fluency (gate tau) UNION near-tie pool margin -------------
    tau = gate['tau']
    val_margin = np.array([pool_top2_margin(pool_val, j, gate_weight_for(j, flu_val, gate)) for j in range(n_val)])
    test_margin = np.array([pool_top2_margin(pool_test, j, gate_weight_for(j, flu_test, gate)) for j in range(n_test)])
    fin = val_margin[np.isfinite(val_margin)]
    m_tau = float(np.percentile(fin, cfg['margin_percentile'])) if len(fin) else -np.inf
    val_sel = sorted(j for j in all_idx if flu_val[j] < tau or val_margin[j] <= m_tau)
    test_sel = sorted(j for j in range(n_test) if flu_test[j] < tau or test_margin[j] <= m_tau)
    n_low = int((flu_val[all_idx] < tau).sum()) if len(all_idx) else 0
    log(f'GEC selective trigger: {len(val_sel)}/{len(all_idx)} val trials ({n_low} low-fluency + '
        f'{max(len(val_sel) - n_low, 0)} extra near-tie, margin tau={m_tau:.3f}), {len(test_sel)}/{n_test} '
        f'test trials')
    res['summary'].update({'val_triggered': len(val_sel), 'test_triggered': len(test_sel),
                           'margin_tau': m_tau, 'fluency_tau': tau})
    if not val_sel:
        log('no trials below the fluency tau or margin tau -> GEC has nothing to do')
        return res

    # ---- corrector: fold A u B only, C stays held out ----------------------------------------
    k = cfg['max_candidates']
    train_ex = [{'prompt': gec_prompt_for_pool(pool_val, j, flu_val, gate, k), 'target': val_refs[j]}
                for j in tune_idx if oof[j]]
    log(f'GEC training pairs: {len(train_ex)} (fold A u B only)')
    t0 = time.time()
    model, tok = finetune_llm_gec(llm_name, train_ex, cfg, log=log)
    log(f'GEC fine-tune took {ht(time.time() - t0)}')

    val_gec = dict(pred_val)
    used_v = oov_v = 0
    for j in val_sel:
        prompt = gec_prompt_for_pool(pool_val, j, flu_val, gate, k)
        W = gate_weight_for(j, flu_val, gate)
        chosen, used, oov = gec_correct(model, tok, prompt, utt_ems_val[j], pred_val[j], W, cfg,
                                        lex, ngram, device)
        val_gec[j] = remove_punctuation(chosen)
        used_v += int(used)
        oov_v += int(used and oov)
    log(f'  best-of-{cfg["n_samples"]} GEC candidate accepted on {used_v}/{len(val_sel)} val trials '
        f'({oov_v} of those OOV) -- the rest kept the fluency-gate text')

    v_base = wer_on(verify_idx, pred_val) if verify_idx else 0.0
    v_gec = wer_on(verify_idx, val_gec) if verify_idx else 0.0
    gain, se = gec_paired_gain(verify_idx, val_refs, pred_val, val_gec, cfg['n_boot'], seed)
    log(f'fold C: fluency-gate {v_base * 100:.2f}% vs +selective-GEC {v_gec * 100:.2f}% '
        f'(gain {gain:.1f} words, paired bootstrap SE {se:.1f})')
    res['summary'].update({'val_accepted': used_v, 'val_accepted_oov': oov_v,
                           'verify_base_wer_%': v_base * 100, 'verify_gec_wer_%': v_gec * 100,
                           'gain_words': gain, 'gain_se_words': se})

    wins_anchor = (se == se and gain > cfg['anchor_z'] * se)
    wins_strict = (se != se and v_gec < v_base)          # no bootstrap -> require a strict win
    if verify_idx and not (wins_anchor or wins_strict):
        log(f'GEC gain on fold C ({gain:.1f} words) does not clear {cfg["anchor_z"]} x paired SE '
            f'({se:.1f}) -> not adopted, the submission is unaffected')
    else:
        test_base, _ = predict_texts(pool_test, gate=gate, flu=flu_test, sel=test_sel)
        tmp, used_t = {}, 0
        for j in test_sel:
            prompt = gec_prompt_for_pool(pool_test, j, flu_test, gate, k)
            W = gate_weight_for(j, flu_test, gate)
            chosen, used, _ = gec_correct(model, tok, prompt, utt_ems_test[j], test_base[j], W, cfg,
                                          lex, ngram, device)
            tmp[j] = remove_punctuation(chosen)
            used_t += int(used)
        # commit only after the whole test pass succeeded
        res.update({'adopted': True, 'val_pred': val_gec, 'test_pred': tmp})
        res['summary']['test_accepted'] = used_t
        log(f'GEC adopted: accepted on {used_t}/{len(test_sel)} test trials | all-OOF WER now '
            f'{wer_on(all_idx, val_gec) * 100:.2f}%')
    res['summary']['adopted'] = res['adopted']
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res
