"""
src/decoding.py — everything between an acoustic model's emissions and a line of
text. This is the "llm" half of the pipeline, in four stages:

  1. CANDIDATE POOL   flashlight lexicon beam search (used directly, not through
                      torchaudio.models.decoder.ctc_decoder, which is not
                      guaranteed on new torch images; output verified identical)
                      run in a CPU process pool, over every out-of-fold acoustic
                      model x several KenLM weights, text-deduplicated.
  2. EXPANSION        single-word phonetic-neighbour substitutions (phoneme edit
                      distance <= 1, homophones included) of the top hypotheses.
                      ~95% of remaining errors are SUBSTITUTIONS, and the LLM can
                      only rerank what is in the list — so putting the right word
                      into the pool matters more than a smarter reranker.
  3. EXACT FEATURES   acoustic log P(phones | x) by CTC forward — alignment-free,
                      so ensembling folds is an average of exact sequence
                      log-likelihoods. NOT posterior averaging, which destroyed
                      WER in v2 (66%) because the CTC spikes of independently
                      trained models are not time-aligned.
                      Plus KenLM ln P(text), task-adapted LLM ln P(text), #words.
  4. TUNING           log-linear weights + a fluency gate chosen on out-of-fold
                      validation by EXACT corpus WER over a vectorised grid, with
                      flat-region smoothing and an optional 1-SE tie-break.

Ported from the notebook module `b2t_decode.py` with the import path adapted;
the algorithms are unchanged.
"""

import os, sys, json, math, time, contextlib, threading, copy, random, gc
import numpy as np
import torch
import torch.nn.functional as F
import editdistance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_CLASSES, BLANK_ID, SIL_ID                      # noqa: F401
from src.metrics import (PHONE2ID, load_lexicon, official_wer, remove_punctuation,
                         utt_word_errors)

LN10 = math.log(10.0)


@contextlib.contextmanager
def silence_fd():
    sys.stdout.flush(); sys.stderr.flush()
    so, se = os.dup(1), os.dup(2)
    nul = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(nul, 1); os.dup2(nul, 2)
        yield
    finally:
        os.dup2(so, 1); os.dup2(se, 2); os.close(nul); os.close(so); os.close(se)


# -----------------------------------------------------------------------------
# 1) flashlight beam search in a process pool (CPU)
# -----------------------------------------------------------------------------
class FLDecoder:
    """flashlight-text lexicon beam search used directly (same construction as
    torchaudio.models.decoder.ctc_decoder, which is not guaranteed on new torch images).
    ONE KenLM + ONE trie are shared by all lm_weight decoders -> a single LM copy per process."""
    def __init__(self, lexicon, tokens, lm, lm_weights, beam, nbest, word_score=0.0, beam_threshold=50.0):
        from flashlight.lib.text.decoder import (CriterionType, LexiconDecoder, LexiconDecoderOptions, Trie,
                                                 SmearingMode, ZeroLM)
        from flashlight.lib.text.dictionary import create_word_dict, Dictionary, load_words
        KenLM = None
        try:
            from flashlight.lib.text.decoder.kenlm import KenLM
        except Exception:
            try:
                from flashlight.lib.text.decoder import KenLM
            except Exception:
                KenLM = None
        self.tokens = Dictionary(tokens)
        lex = load_words(lexicon)
        self.word_dict = create_word_dict(lex)
        if lm and KenLM is not None:
            self.lm = KenLM(lm, self.word_dict)
        else:
            if lm:
                print('WARNING: flashlight built without KenLM -> beam search without n-gram LM', flush=True)
            self.lm = ZeroLM()
        sil = self.tokens.get_index('|')
        blank = self.tokens.get_index('BLANK')
        trie = Trie(self.tokens.index_size(), sil)
        start = self.lm.start(False)
        for word, spellings in lex.items():
            wi = self.word_dict.get_index(word)
            _, sc = self.lm.score(start, wi)
            for sp in spellings:
                trie.insert([self.tokens.get_index(t) for t in sp], wi, sc)
        trie.smear(SmearingMode.MAX)
        self.trie = trie
        unk = self.word_dict.get_index('<unk>')
        self.nbest = nbest
        self.decoders = {}
        for lw in lm_weights:
            opts = LexiconDecoderOptions(beam_size=beam, beam_size_token=self.tokens.index_size(),
                                         beam_threshold=beam_threshold, lm_weight=lw, word_score=word_score,
                                         unk_score=float('-inf'), sil_score=0.0, log_add=False,
                                         criterion_type=CriterionType.CTC)
            self.decoders[lw] = LexiconDecoder(opts, trie, self.lm, sil, blank, unk, [], False)

    def decode(self, em):
        """em: np [T, C] -> {lm_weight: [unique word strings, best first]}"""
        em = np.ascontiguousarray(em, dtype=np.float32)
        T, N = em.shape
        out = {}
        for lw, dec in self.decoders.items():
            texts, seen = [], set()
            if T > 0:
                for r in dec.decode(em.ctypes.data, T, N)[:self.nbest]:
                    t = ' '.join(self.word_dict.get_entry(x) for x in r.words if x >= 0)
                    if t not in seen:
                        seen.add(t); texts.append(t)
            out[lw] = texts
        return out


_DEC = None


def _init_beam_worker(spec):
    global _DEC
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    with silence_fd():
        _DEC = FLDecoder(spec['lexicon'], spec['tokens'], spec['lm'], spec['lm_weights'], spec['beam'],
                         spec['nbest'], spec.get('word_score', 0.0))


def _beam_task(task):
    key, em = task
    try:
        return key, _DEC.decode(em)
    except Exception as e:
        print(f'beam task {key} failed: {e!r}', flush=True)
        return key, {lw: [] for lw in _DEC.decoders}


def safe_n_proc(requested, lm_path, log=print):
    """never start more LM-holding processes than host RAM allows."""
    n = max(1, int(requested))
    try:
        import psutil
        avail = psutil.virtual_memory().available
        lm_bytes = os.path.getsize(lm_path) if lm_path and os.path.exists(lm_path) else 0
        per_proc = 1.3 * lm_bytes + 1.5e9
        fit = int((avail - 5e9) // per_proc)
        if os.environ.get('B2T_FORCE_NPROC'):                       # local smoke test only
            fit = int(os.environ['B2T_FORCE_NPROC'])
        n = max(1, min(n, fit, os.cpu_count() or 1))
        if os.environ.get('B2T_FORCE_NPROC'):
            n = int(os.environ['B2T_FORCE_NPROC'])
        log(f'  beam processes: {n} (requested {requested}; RAM available {avail / 1e9:.1f} GB, '
            f'KenLM file {lm_bytes / 1e9:.2f} GB, budget {per_proc / 1e9:.1f} GB/process)')
    except Exception as e:
        log(f'  beam processes: {n} (RAM check unavailable: {e!r})')
    return n


def _beam_chunk(chunk):
    if os.environ.get('B2T_KILL_WORKER') and random.random() < 0.3:   # local smoke test only
        os._exit(9)
    return [_beam_task(t) for t in chunk]


def _ram_str():
    try:
        import psutil
        return f' | RAM {psutil.virtual_memory().percent:.0f}%'
    except Exception:
        return ''


def run_beam_pool(tasks, spec, n_proc=2, log=print, desc='beam', chunk=16):
    """tasks: list of (key, np.array[T,C]) -> {key: {lm_weight: [texts...]}}
    ProcessPoolExecutor detects dead workers (BrokenProcessPool); anything not finished is
    decoded serially in this process, so a killed worker can never hang or abort the run."""
    results, t0 = {}, time.time()
    n = len(tasks)
    if n == 0:
        return results
    report = max(n // 10, 1)
    last = [0]

    def progress():
        if len(results) - last[0] >= report or len(results) == n:
            last[0] = len(results)
            log(f'  [{desc}] {len(results)}/{n}  {time.time() - t0:.0f}s{_ram_str()}')

    n_proc = safe_n_proc(n_proc, spec.get('lm'), log) if n_proc > 1 else 1
    if n_proc > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            chunks = [tasks[a:a + chunk] for a in range(0, n, chunk)]
            with ProcessPoolExecutor(max_workers=n_proc, mp_context=mp.get_context('spawn'),
                                     initializer=_init_beam_worker, initargs=(spec,)) as ex:
                futs = [ex.submit(_beam_chunk, c) for c in chunks]
                for f in as_completed(futs):
                    for key, out in f.result():
                        results[key] = out
                    progress()
        except Exception as e:
            log(f'  [{desc}] process pool stopped ({e!r}); {n - len(results)} tasks left -> serial')
    todo = [t for t in tasks if t[0] not in results]
    if todo:
        _init_beam_worker(spec)
        for t in todo:
            key, out = _beam_task(t)
            results[key] = out
            progress()
    return results


# -----------------------------------------------------------------------------
# n-gram scorer (KenLM python binding)
# -----------------------------------------------------------------------------
class NgramScorer:
    """KenLM python binding; the model can be released (m=None) and is reloaded on demand."""
    def __init__(self, path):
        self.path = path
        self.m = None
        self._load()
        self.cache, self.uni = {}, {}

    def _load(self):
        import kenlm
        self.m = kenlm.Model(self.path)

    def ln(self, text):
        v = self.cache.get(text)
        if v is None:
            if self.m is None:
                self._load()
            v = self.m.score(text, bos=True, eos=True) * LN10
            self.cache[text] = v
        return v

    def unigram(self, w):
        v = self.uni.get(w)
        if v is None:
            if self.m is None:
                self._load()
            v = self.m.score(w, bos=False, eos=False) * LN10
            self.uni[w] = v
        return v


# -----------------------------------------------------------------------------
# 2) pronunciation lexicon, CTC targets, phonetic neighbours
# -----------------------------------------------------------------------------
class PronLexicon:
    def __init__(self, lexicon_path, sil_trailing=True, max_variants=2):
        self.word2prons, self.pron2words = load_lexicon(lexicon_path)
        self.word2ids = {w: [[PHONE2ID[p] for p in pr] for pr in prons] for w, prons in self.word2prons.items()}
        self.sil_trailing = sil_trailing
        self.max_variants = max_variants
        self._index = None
        self._nb_cache = {}

    def has(self, w):
        return w in self.word2ids

    def targets(self, text):
        """list of phoneme-id target variants (first pron + single alt-pron swaps), None if OOV."""
        words = text.split()
        if not words:
            return [[]]
        prons = []
        for w in words:
            p = self.word2ids.get(w)
            if p is None:
                return None
            prons.append(p)

        def assemble(choice):
            seq = []
            for k, (w_prons, c) in enumerate(zip(prons, choice)):
                seq += w_prons[c]
                if k < len(prons) - 1 or self.sil_trailing:
                    seq.append(SIL_ID)
            return seq
        base = [0] * len(prons)
        out = [assemble(base)]
        for k, w_prons in enumerate(prons):
            for c in range(1, len(w_prons)):
                if len(out) >= self.max_variants:
                    return out
                ch = list(base); ch[k] = c
                out.append(assemble(ch))
        return out

    def _build_index(self):
        idx = {}
        for pr in self.pron2words:
            keys = {pr} | {pr[:i] + pr[i + 1:] for i in range(len(pr))}
            for k in keys:
                idx.setdefault(k, []).append(pr)
        self._index = idx

    def neighbours(self, w, max_n, ngram=None):
        """words whose pronunciation is within phoneme edit distance <= 1 of any pron of w."""
        key = (w, max_n)
        if key in self._nb_cache:
            return self._nb_cache[key]
        if self._index is None:
            self._build_index()
        cands = set()
        for pr in self.word2prons.get(w, []):
            keys = {pr} | {pr[:i] + pr[i + 1:] for i in range(len(pr))}
            for k in keys:
                for q in self._index.get(k, ()):
                    if q == pr or editdistance.eval(pr, q) <= 1:
                        cands.update(self.pron2words[q])
        cands.discard(w)
        cands = list(cands)
        if ngram is not None and len(cands) > max_n:
            cands.sort(key=lambda x: -ngram.unigram(x))
        cands = cands[:max_n]
        self._nb_cache[key] = cands
        return cands


def detect_sil_convention(train_meta, lex_path, n_max=2000):
    """Does the ground-truth phoneme sequence end with ' | ' after the last word?"""
    w2p, _ = load_lexicon(lex_path)
    n = end_sil = sil_eq_words = sil_eq_words_m1 = exact_t = exact_nt = 0
    for m in train_meta[:n_max]:
        words = remove_punctuation(m['sentence']).split()
        ph = [int(p) for p in m['phonemes']]
        if not words or not ph or any(w not in w2p for w in words):
            continue
        n += 1
        end_sil += int(ph[-1] == SIL_ID)
        ns = sum(1 for p in ph if p == SIL_ID)
        sil_eq_words += int(ns == len(words)); sil_eq_words_m1 += int(ns == len(words) - 1)
        seq = []
        for k, w in enumerate(words):
            seq += [PHONE2ID[p] for p in w2p[w][0]]
            seq.append(SIL_ID)
        exact_t += int(seq == ph); exact_nt += int(seq[:-1] == ph)
    trailing = end_sil >= 0.5 * max(n, 1)
    return {'n_checked': n, 'frac_end_with_sil': end_sil / max(n, 1),
            'frac_nsil_eq_nwords': sil_eq_words / max(n, 1), 'frac_nsil_eq_nwords_minus1': sil_eq_words_m1 / max(n, 1),
            'exact_match_trailing': exact_t / max(n, 1), 'exact_match_no_trailing': exact_nt / max(n, 1),
            'sil_trailing': bool(trailing)}


# -----------------------------------------------------------------------------
# 3) exact acoustic score: log P(phones | x) via CTC forward on GPU
# -----------------------------------------------------------------------------
@torch.no_grad()
def ctc_logprob(em, targets, device, chunk=1024):
    """em: np [T,C] log-probs; targets: list of id lists -> np.float64 [K] (-1e9 if impossible)."""
    K = len(targets)
    out = np.full(K, -1e9, dtype=np.float64)
    if K == 0:
        return out
    E = torch.from_numpy(np.ascontiguousarray(em, dtype=np.float32)).to(device)
    T = E.shape[0]
    for a in range(0, K, chunk):
        tg = targets[a:a + chunk]
        k = len(tg)
        Lmax = max(1, max(len(t) for t in tg))
        tgt = torch.zeros(k, Lmax, dtype=torch.long)
        for i, t in enumerate(tg):
            if t:
                tgt[i, :len(t)] = torch.tensor(t)
        tl = torch.tensor([len(t) for t in tg], dtype=torch.long)
        lp = E[:, None, :].expand(T, k, E.shape[1]).contiguous()
        loss = F.ctc_loss(lp, tgt.to(device), torch.full((k,), T, dtype=torch.long, device=device),
                          tl.to(device), blank=BLANK_ID, reduction='none', zero_infinity=False)
        v = (-loss).double().cpu().numpy()
        v[~np.isfinite(v)] = -1e9
        out[a:a + k] = v
    return out


def acoustic_scores(texts, ems, lex, device):
    """texts -> [K, n_models] best-variant CTC log-likelihood under each emission."""
    flat, owner = [], []
    for ci, t in enumerate(texts):
        vs = lex.targets(t)
        if vs is None:
            continue
        for v in vs:
            flat.append(v); owner.append(ci)
    owner = np.asarray(owner, dtype=np.int64)
    res = np.full((len(texts), len(ems)), -1e9, dtype=np.float64)
    if not flat:
        return res
    for mi, em in enumerate(ems):
        s = ctc_logprob(em, flat, device)
        np.maximum.at(res[:, mi], owner, s)
    return res


# -----------------------------------------------------------------------------
# candidate pool container
# -----------------------------------------------------------------------------
class Pool:
    """per-utterance candidate lists with features."""
    def __init__(self):
        self.texts = []      # list[list[str]]
        self.src = []        # list[list[str]] provenance tags
        self.am = []         # list[np [K]] mean acoustic ll over the utterance's models
        self.am_min = []     # list[np [K]] min over models (disagreement-aware)
        self.ng = []         # list[np [K]]
        self.nw = []         # list[np [K]]
        self.llm = []        # list[np [K]] (nan = not scored)
        self.ntok = []

    def add_utt(self, texts, src):
        self.texts.append(list(texts)); self.src.append(list(src))
        K = len(texts)
        self.am.append(np.zeros(K)); self.am_min.append(np.zeros(K)); self.ng.append(np.zeros(K))
        self.nw.append(np.array([len(t.split()) for t in texts], dtype=np.float64))
        self.llm.append(np.full(K, np.nan)); self.ntok.append(np.zeros(K))

    def extend_utt(self, u, new_texts, tag):
        have = set(self.texts[u])
        add = [t for t in dict.fromkeys(new_texts) if t not in have]
        if not add:
            return []
        self.texts[u] += add; self.src[u] += [tag] * len(add)
        k = len(add)
        self.am[u] = np.concatenate([self.am[u], np.zeros(k)])
        self.am_min[u] = np.concatenate([self.am_min[u], np.zeros(k)])
        self.ng[u] = np.concatenate([self.ng[u], np.zeros(k)])
        self.nw[u] = np.concatenate([self.nw[u], [len(t.split()) for t in add]])
        self.llm[u] = np.concatenate([self.llm[u], np.full(k, np.nan)])
        self.ntok[u] = np.concatenate([self.ntok[u], np.zeros(k)])
        return add

    def __len__(self):
        return len(self.texts)


def fill_features(pool, utt_ems, lex, ngram, device, only_new_from=None, log=print):
    """compute am/ng for all candidates (or those at index >= only_new_from[u])."""
    t0 = time.time()
    for u in range(len(pool)):
        start = 0 if only_new_from is None else only_new_from[u]
        texts = pool.texts[u][start:]
        if not texts:
            continue
        A = acoustic_scores(texts, utt_ems[u], lex, device)
        pool.am[u][start:] = A.mean(1)
        pool.am_min[u][start:] = A.min(1)
        if ngram is not None:
            pool.ng[u][start:] = [ngram.ln(t) for t in texts]
        if (u + 1) % 500 == 0:
            log(f'  features {u + 1}/{len(pool)}  {time.time() - t0:.0f}s')


def expand_pool(pool, lex, ngram, weights, top_e=3, max_nb=12, max_new=600):
    """single-word phonetic-neighbour substitutions of each utterance's top_e candidates."""
    starts = []
    a, g = weights['ng'], weights['nw']
    for u in range(len(pool)):
        starts.append(len(pool.texts[u]))
        S = pool.am[u] + a * pool.ng[u] + g * pool.nw[u]
        order = np.argsort(-S)[:top_e]
        new = []
        for ci in order:
            words = pool.texts[u][ci].split()
            for j, w in enumerate(words):
                for nb in lex.neighbours(w, max_nb, ngram):
                    new.append(' '.join(words[:j] + [nb] + words[j + 1:]))
                if len(new) >= max_new:
                    break
        pool.extend_utt(u, new[:max_new], 'expand')
    return starts


# -----------------------------------------------------------------------------
# 4) LLM: next-word-prediction LoRA fine-tune + batched multi-GPU scoring
# -----------------------------------------------------------------------------
def _llm_ids(tok, text):
    eos = tok.eos_token_id
    return [eos] + tok(text, add_special_tokens=False)['input_ids'] + [eos]


def finetune_llm_nwp(model_name, sentences, cfg, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dev0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    use_amp = dev0 != 'cpu'
    hf_token = os.environ.get('HF_TOKEN')  # meta-llama/Llama-3.1-8B is gated -- needs an accepted-license HF token
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if not cfg.get('llm_finetune', True):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if use_amp else torch.float32, token=hf_token).to(dev0).eval()
        return model, tok, {'finetuned': False}

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # ---- QLoRA: 4-bit base + fp16 LoRA adapters (fits Qwen2.5-7B on a single T4) ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16 if use_amp else torch.float32,   # T4 (Turing) has no native bf16 tensor cores -- fp16 is the correct compute dtype here
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_cfg if use_amp else None,
        torch_dtype=torch.float32 if not use_amp else None,
        device_map={'': 0} if use_amp else None,
        token=hf_token,
    )
    if not use_amp:
        model = model.to(dev0)
    model.config.use_cache = False
    if use_amp:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()

    lcfg = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)
    rng = random.Random(0)
    sents = sorted(set(s for s in sentences if s))
    rng.shuffle(sents)
    n_dev = max(int(0.05 * len(sents)), 50)
    dev, trn = sents[:n_dev], sents[n_dev:]
    data = [_llm_ids(tok, s) for s in trn]
    dev_ids = [_llm_ids(tok, s) for s in dev]
    bs, epochs = cfg['llm_ft_batch'], cfg['llm_ft_epochs']
    accum = max(int(cfg.get('llm_ft_grad_accum', 1)), 1)     # QLoRA: gradient accumulation over small micro-batches
    steps_total = epochs * math.ceil(math.ceil(len(data) / bs) / accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['llm_ft_lr'], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s / max(steps_total, 1), 1.0))))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32   # matches bnb_4bit_compute_dtype; pairs correctly with GradScaler below

    def batchify(seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); lab[i, 1:len(s)] = torch.tensor(s[1:]); att[i, :len(s)] = 1
        return ids.to(dev0), lab.to(dev0), att.to(dev0)

    @torch.no_grad()
    def dev_nll():
        model.eval()
        tot, n = 0.0, 0
        dev_bs = max(bs, 8)   # keep eval batch close to the train micro-batch, not a fixed 64
        for a in range(0, len(dev_ids), dev_bs):
            ids, lab, att = batchify(dev_ids[a:a + dev_bs])
            with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=att).logits
            l = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                ignore_index=-100, reduction='sum')
            tot += float(l); n += int((lab[:, 1:] != -100).sum())
        model.train()
        return tot / max(n, 1)

    hist = {'dev_nll_before': dev_nll()}
    log(f'  [llm-ft] {model_name}: {len(trn)} train / {len(dev)} dev sentences, dev NLL/token before = '
        f'{hist["dev_nll_before"]:.3f} (QLoRA 4-bit, batch={bs} x accum={accum})')
    best_nll, best_state = hist['dev_nll_before'], None
    step = 0
    model.train()
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        n_batches = math.ceil(len(order) / bs)
        for bi, a in enumerate(range(0, len(order), bs)):
            ids, lab, att = batchify([data[i] for i in order[a:a + bs]])
            with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                   ignore_index=-100) / accum
            scaler.scale(loss).backward()
            is_last_in_epoch = (bi == n_batches - 1)
            if (bi + 1) % accum == 0 or is_last_in_epoch:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        nll = dev_nll()
        hist[f'dev_nll_ep{ep + 1}'] = nll
        log(f'  [llm-ft] epoch {ep + 1}/{epochs}: dev NLL/token {nll:.3f} ({time.time() - t0:.0f}s)')
        if nll < best_nll:
            best_nll = nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if 'lora_' in k}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    del opt, sched, scaler                      # free optimizer state BEFORE the merge memory spike
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    model.eval()                                # NOTE: no merge_and_unload() -- merging a 4-bit base needs a
                                                 # full fp16 dequant on top of what's already resident, which
                                                 # caused a CUDA OOM on the 7B run. LoRA adapters stay active
                                                 # and apply automatically on every forward pass through this
                                                 # PeftModel, so scoring is correct without merging -- and the
                                                 # model keeps its compact 4-bit footprint.
    model.config.use_cache = True
    hist.update({'finetuned': True, 'best_dev_nll': best_nll})
    return model, tok, hist

class LLMScorer:
    def __init__(self, model, tok, n_gpus):
        self.tok = tok
        self.replicas = [(model, 'cuda:0' if torch.cuda.is_available() else 'cpu')]
        for g in range(1, n_gpus):
            self.replicas.append((copy.deepcopy(model).to(f'cuda:{g}').eval(), f'cuda:{g}'))
        self.cache = {}

    @torch.no_grad()
    def _run(self, model, dev, seqs, out, idxs, bs):
        pad = self.tok.pad_token_id
        for a in range(0, len(seqs), bs):
            chunk = seqs[a:a + bs]
            L = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, s in enumerate(chunk):
                ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
            ids, att = ids.to(dev), att.to(dev)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
            lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            lp = (lp * att[:, 1:].float()).sum(1)
            for i, v in enumerate(lp.tolist()):
                out[idxs[a + i]] = v

    def score(self, texts, bs=32):
        """-> (sum ln P incl. final EOS, n scored tokens) for each text."""
        uniq = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if uniq:
            seqs = [_llm_ids(self.tok, t) for t in uniq]
            order = sorted(range(len(uniq)), key=lambda i: len(seqs[i]))
            res = {}
            parts = [order[g::len(self.replicas)] for g in range(len(self.replicas))]
            threads = []
            for (model, dev), part in zip(self.replicas, parts):
                th = threading.Thread(target=self._run, args=(model, dev, [seqs[i] for i in part], res, part, bs))
                th.start(); threads.append(th)
            for th in threads:
                th.join()
            for i, t in enumerate(uniq):
                self.cache[t] = (res[i], len(seqs[i]) - 1)
        vals = [self.cache[t] for t in texts]
        return np.array([v[0] for v in vals]), np.array([v[1] for v in vals], dtype=np.float64)


# -----------------------------------------------------------------------------
# 5) vectorised tuning by exact corpus WER
# -----------------------------------------------------------------------------
def pad_matrix(lists, fill, K=None, sel=None):
    sel = range(len(lists)) if sel is None else sel
    sel = list(sel)
    K = K or max(len(lists[u]) for u in sel)
    M = np.full((len(sel), K), fill, dtype=np.float64)
    for r, u in enumerate(sel):
        v = np.asarray(lists[u], dtype=np.float64)
        M[r, :len(v)] = v
    return M


def errors_lists(pool, refs):
    errs, nref = [], []
    for u in range(len(pool)):
        rw = remove_punctuation(refs[u] or '').split()
        errs.append(np.array([editdistance.eval(rw, remove_punctuation(t).split()) for t in pool.texts[u]], dtype=np.float64))
        nref.append(len(rw))
    return errs, np.asarray(nref, dtype=np.float64)


class Tuner:
    """Feature matrices for a subset of utterances; score = am + a*ng + b*llm + g*nw."""
    def __init__(self, pool, errs, nref, sel, topk_mask=None):
        self.sel = list(sel)
        K = max(len(pool.texts[u]) for u in self.sel)
        self.AM = pad_matrix(pool.am, -1e18, K, self.sel)
        self.NG = pad_matrix(pool.ng, 0.0, K, self.sel)
        self.NW = pad_matrix(pool.nw, 0.0, K, self.sel)
        L = pad_matrix(pool.llm, np.nan, K, self.sel)
        self.LLM = np.nan_to_num(L, nan=0.0)
        self.E = pad_matrix(errs, 1e6, K, self.sel)
        self.N = nref[self.sel].sum()
        self.valid = self.AM > -1e17
        if topk_mask is not None:
            self.valid &= pad_matrix(topk_mask, 0.0, K, self.sel) > 0
        self.AM = np.where(self.valid, self.AM, -1e18)

    def errors(self, a, b, g, rows=None):
        S = self.AM + a * self.NG + b * self.LLM + g * self.NW
        E = self.E
        if rows is not None:
            S, E = S[rows], E[rows]
        pick = S.argmax(1)
        return E[np.arange(len(pick)), pick].sum(), pick

    def grid(self, A, Bs, G, rows=None):
        AM, NG, LL, NW, E = self.AM, self.NG, self.LLM, self.NW, self.E
        if rows is not None:
            AM, NG, LL, NW, E = AM[rows], NG[rows], LL[rows], NW[rows], E[rows]
        err = np.full((len(A), len(Bs), len(G)), np.inf)
        if AM.shape[0] == 0:
            return np.zeros((len(A), len(Bs), len(G)))
        ar = np.arange(AM.shape[0])
        for i, a in enumerate(A):
            base_a = AM + a * NG
            for j, b in enumerate(Bs):
                base_ab = base_a + b * LL if b != 0 else base_a
                for k, g in enumerate(G):
                    pick = (base_ab + g * NW).argmax(1)
                    err[i, j, k] = E[ar, pick].sum()
        return err


def subpool(pool, masks, errs=None):
    """keep only candidates with mask>0 (per utterance); returns (Pool, errs_subset)."""
    sp, se = Pool(), []
    for u in range(len(pool)):
        keep = np.where(np.asarray(masks[u]) > 0)[0]
        if len(keep) == 0:
            keep = np.array([0])
        sp.texts.append([pool.texts[u][i] for i in keep]); sp.src.append([pool.src[u][i] for i in keep])
        for name in ('am', 'am_min', 'ng', 'nw', 'llm', 'ntok'):
            getattr(sp, name).append(getattr(pool, name)[u][keep].copy())
        if errs is not None:
            se.append(errs[u][keep].copy())
    return sp, (se if errs is not None else None)


def utt_scores(pool, u, W):
    llm = np.nan_to_num(pool.llm[u], nan=0.0)
    return pool.am[u] + W['ng'] * pool.ng[u] + W.get('llm', 0.0) * llm + W['nw'] * pool.nw[u]


def topk_masks(pool, W, k):
    masks = []
    for u in range(len(pool)):
        S = utt_scores(pool, u, W)
        m = np.zeros(len(S))
        m[np.argsort(-S)[:k]] = 1
        for i, s in enumerate(pool.src[u]):
            if s.endswith('#0'):
                m[i] = 1                      # every generator's 1-best always survives
        masks.append(m)
    return masks


def fluency(pool, W):
    """per-utterance LLM log-prob per token of the utterance's best candidate under W (no LLM term)."""
    W0 = dict(W, llm=0.0)
    f = np.zeros(len(pool))
    for u in range(len(pool)):
        i = int(np.argmax(utt_scores(pool, u, W0)))
        f[u] = pool.llm[u][i] / max(pool.ntok[u][i], 1) if np.isfinite(pool.llm[u][i]) else 0.0
    return f


def predict_texts(pool, W=None, gate=None, flu=None, sel=None):
    """W: single weight dict; or gate={'tau','hi','lo'} with per-utterance fluency flu."""
    sel = range(len(pool)) if sel is None else sel
    out, regimes = {}, {}
    for u in sel:
        if gate is not None:
            use_lo = gate.get('lo') is not None and flu[u] < gate['tau']
            Wu = gate['lo'] if use_lo else gate['hi']
            regimes[u] = 'lo' if use_lo else 'hi'
        else:
            Wu = W
        out[u] = pool.texts[u][int(np.argmax(utt_scores(pool, u, Wu)))]
    return out, regimes


def pick_from_grid(err, A, Bs, G, smooth=True, tol_rel=0.0):
    """Pick (ng, llm, nw) weights from the 3-D tune-fold error grid.

    Plain argmin picks whichever grid cell has the single lowest word-error count on the
    TUNE fold -- with a grid this dense that's often a point that's only marginally ahead
    of many neighbours by tune-fold noise, not by a real generalizable margin. That cell
    can carry a much larger LLM weight than cells that tie with it, and the larger weight
    is exactly the part that fails to transfer to the verify fold / the Kaggle test set.

    Fix (1-SE-style tolerance rule): among all cells within `tol_rel` (relative, on word-
    error COUNT) of the smoothed grid's minimum, pick the one with the SMALLEST |llm|
    weight. This only pulls the LLM weight down when the larger weight wasn't actually
    earning a meaningfully better tune-fold score in the first place.
    """
    from scipy.ndimage import uniform_filter
    A, Bs, G = np.asarray(A, dtype=float), np.asarray(Bs, dtype=float), np.asarray(G, dtype=float)  # stage-1 passes plain
                                                                                                       # Python lists (e.g. [0.0])
                                                                                                       # for the unused llm axis --
                                                                                                       # fancy-indexing those below
                                                                                                       # needs real ndarrays.
    raw = np.unravel_index(np.argmin(err), err.shape)
    base = err
    if smooth and err.size > 1:
        base = uniform_filter(err, size=3, mode='nearest') + 1e-6 * err
    best = float(base[np.unravel_index(np.argmin(base), base.shape)])
    if tol_rel > 0:
        tol = tol_rel * best                            # opt-in only: tol_rel=0 (default) reproduces plain argmin exactly
        ii, jj, kk = np.where(base <= best + tol)
        order = np.argsort(np.abs(Bs[jj]))              # among near-ties, prefer the smallest LLM weight
        idx = (ii[order[0]], jj[order[0]], kk[order[0]])
    else:
        idx = np.unravel_index(np.argmin(base), base.shape)   # exact argmin -- no tie-break bias applied
    return ({'ng': float(A[idx[0]]), 'llm': float(Bs[idx[1]]), 'nw': float(G[idx[2]]), 'err': float(err[idx])},
            {'ng': float(A[raw[0]]), 'llm': float(Bs[raw[1]]), 'nw': float(G[raw[2]]), 'err': float(err[raw])})