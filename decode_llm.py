"""
Full decoding pipeline used for the final (~15% WER) submission:

    trained CTC model -> beam search (pyctcdecode + KenLM n-gram LM)
                       -> rescore each beam's hypotheses with a causal LLM
                       -> pick the highest combined-score hypothesis per sample

This assumes:
  1. You already have a trained checkpoint (see train.py), and
  2. You've built the KenLM command-line binaries once via:
         bash scripts/build_kenlm.sh

Usage:
    python decode_llm.py --checkpoint checkpoints/best_model.pt
    python decode_llm.py --checkpoint checkpoints/best_model.pt \
        --llm_name Qwen/Qwen2.5-7B --beam_width 100 --llm_weight 1.0

Outputs submission_llm.csv (id, text) in the current directory.
"""

import argparse
import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import CONFIG
from src.dataset import load_split, load_test_samples
from src.model import HybridLSTMTransformerCTC
from src.utils import get_session2idx


def parse_args():
    p = argparse.ArgumentParser(description="KenLM + causal-LM rescoring decode pipeline")
    p.add_argument('--data_dir', default=CONFIG['data_dir'])
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--output', default='submission_llm.csv')

    p.add_argument('--kenlm_bin', default='kenlm/build/bin',
                    help='Directory containing the lmplz/build_binary executables '
                         '(built via scripts/build_kenlm.sh)')
    p.add_argument('--corpus_file', default=None,
                    help='Optional pre-built text corpus (one sentence per line) for the '
                         'n-gram LM. If omitted, the training split is loaded and its '
                         'transcripts are used, matching the original pipeline.')
    p.add_argument('--ngram_order', type=int, default=CONFIG['kenlm_ngram_order'])

    p.add_argument('--beam_width', type=int, default=CONFIG['beam_width'])
    p.add_argument('--kenlm_alpha', type=float, default=CONFIG['kenlm_alpha'])
    p.add_argument('--kenlm_beta', type=float, default=CONFIG['kenlm_beta'])
    p.add_argument('--llm_name', default=CONFIG['llm_name'])
    p.add_argument('--llm_weight', type=float, default=CONFIG['llm_weight'])
    p.add_argument('--emission_batch_size', type=int, default=32)
    return p.parse_args()


def build_corpus(corpus_path, data_dir):
    """Writes one lower-cased training sentence per line, for LM training."""
    train_data = load_split(data_dir, 'train')
    with open(corpus_path, 'w') as f:
        for sent in train_data['sentence']:
            if sent:
                f.write(sent.lower().strip() + '\n')
    return corpus_path


def train_ngram_lm(kenlm_bin, corpus_path, ngram_order, work_dir='.'):
    """Runs lmplz to train an ARPA n-gram LM, patches in an </s> symbol
    (pyctcdecode/KenLM expects both <s> and </s> to be present), then
    packs it into a binary with build_binary."""
    arpa_path = os.path.join(work_dir, 'lm.arpa')
    arpa_fixed_path = os.path.join(work_dir, 'lm_fixed.arpa')
    binary_path = os.path.join(work_dir, 'lm.binary')

    os.system(f'{kenlm_bin}/lmplz -o {ngram_order} --discount_fallback < {corpus_path} > {arpa_path}')

    has_added_eos = False
    with open(arpa_path, 'r') as r_file, open(arpa_fixed_path, 'w') as w_file:
        for line in r_file:
            if not has_added_eos and 'ngram 1=' in line:
                count = line.strip().split('=')[-1]
                w_file.write(line.replace(f'{count}', f'{int(count) + 1}'))
            elif not has_added_eos and '<s>' in line:
                w_file.write(line)
                w_file.write(line.replace('<s>', '</s>'))
                has_added_eos = True
            else:
                w_file.write(line)

    os.system(f'{kenlm_bin}/build_binary {arpa_fixed_path} {binary_path}')
    assert os.path.exists(binary_path), "KenLM binary build failed"
    return binary_path


def extract_emissions(model, samples, device, batch_size=32):
    model.eval()
    all_emissions, all_ids = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc='Extracting emissions'):
            batch = samples[i:i + batch_size]
            features = [s['features'] for s in batch]
            lengths = torch.LongTensor([len(f) for f in features])
            day_idx = torch.LongTensor([s['day_idx'] for s in batch]).to(device)

            features_padded = pad_sequence(features, batch_first=True).to(device)
            sorted_lengths, sorted_idx = lengths.sort(descending=True)

            log_probs, output_lengths = model(features_padded[sorted_idx], sorted_lengths, day_idx[sorted_idx])
            log_probs = log_probs.cpu().numpy()

            batch_emissions = [None] * len(batch)
            for b, orig_pos in enumerate(sorted_idx.tolist()):
                t_len = int(output_lengths[b])
                batch_emissions[orig_pos] = log_probs[:t_len, b, :]

            all_emissions.extend(batch_emissions)
            all_ids.extend([s['id'] for s in batch])
    return all_emissions, all_ids


def main():
    args = parse_args()
    device = CONFIG['device']

    print("=" * 80)
    print("BRAIN-TO-TEXT: KenLM + LLM RESCORING DECODE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load trained checkpoint + rebuild model
    # ------------------------------------------------------------------
    session2idx = get_session2idx(args.data_dir)
    n_days = len(session2idx)

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    char2idx = checkpoint['char2idx']
    idx2char = {v: k for k, v in char2idx.items()}
    config = checkpoint.get('config', CONFIG)

    model = HybridLSTMTransformerCTC.from_config(config, n_days=n_days, vocab_size=len(char2idx)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_samples = load_test_samples(args.data_dir, session2idx)
    print(f"Loaded {len(test_samples)} test samples")

    # ------------------------------------------------------------------
    # 2. Build / locate the n-gram LM
    # ------------------------------------------------------------------
    assert os.path.exists(f'{args.kenlm_bin}/lmplz'), (
        f"KenLM binaries not found at {args.kenlm_bin}. Run `bash scripts/build_kenlm.sh` first."
    )

    corpus_path = args.corpus_file or build_corpus('lm_corpus.txt', args.data_dir)
    binary_path = train_ngram_lm(args.kenlm_bin, corpus_path, args.ngram_order)
    print(f"KenLM binary ready at: {binary_path}")

    # ------------------------------------------------------------------
    # 3. Beam-search decoder (acoustic model + n-gram LM)
    # ------------------------------------------------------------------
    from pyctcdecode import build_ctcdecoder

    labels = ['' if idx2char[i] == '<BLANK>' else idx2char[i] for i in range(len(idx2char))]
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=binary_path,
        alpha=args.kenlm_alpha,
        beta=args.kenlm_beta,
    )

    # ------------------------------------------------------------------
    # 4. Causal LLM for fluency rescoring
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading rescoring LLM: {args.llm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm_model.eval()

    def compute_llm_score(sentence):
        """Pseudo log-likelihood of `sentence` under the LLM (higher = more fluent)."""
        if not sentence.strip():
            return -999.0
        inputs = tokenizer(sentence, return_tensors="pt").to(llm_model.device)
        with torch.no_grad():
            outputs = llm_model(**inputs, labels=inputs["input_ids"])
            return -outputs.loss.item()

    # ------------------------------------------------------------------
    # 5. Extract emissions, beam search, rescore, pick best hypothesis
    # ------------------------------------------------------------------
    test_emissions, test_ids = extract_emissions(model, test_samples, device, args.emission_batch_size)

    final_predictions = []
    print("Executing beam search & LLM rescoring...")
    for emission in tqdm(test_emissions, desc="Rescoring sequences"):
        beams = decoder.decode_beams(emission, beam_width=args.beam_width)

        best_text = ""
        best_combined_score = float('-inf')

        for beam in beams:
            text = beam[0]
            if not text:
                continue

            scores = [item for item in beam if isinstance(item, float)]
            am_score = scores[0] if len(scores) > 0 else 0.0
            lm_score = scores[1] if len(scores) > 1 else 0.0
            llm_score = compute_llm_score(text)

            combined_score = am_score + lm_score + (args.llm_weight * llm_score)
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_text = text

        if not best_text:
            best_text = beams[0][0] if len(beams) > 0 else ""

        final_predictions.append(best_text)

    # ------------------------------------------------------------------
    # 6. Write submission
    # ------------------------------------------------------------------
    df = pd.DataFrame({'id': test_ids, 'text': final_predictions})
    df = df.sort_values('id').reset_index(drop=True)
    df['id'] = range(len(df))
    df.to_csv(args.output, index=False)

    print(f"\n\u2713 Wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
