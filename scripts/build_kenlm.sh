#!/usr/bin/env bash
# ===========================================================================
# build_kenlm.sh — OPTIONAL. Build the KenLM CLI tools (lmplz, build_binary)
# from source so you can train a fresh n-gram binary from a text corpus.
#
# The default pipeline does NOT need this: decode_llm.py / evaluate.py load a
# prebuilt 4-gram KenLM *.bin (plus lexicon.txt / tokens.txt) that is resolved
# from an attached Kaggle dataset (see KENLM_DATASET_SLUG in config.py) or from
# the --kenlm_binary / --lexicon / --tokens flags. Only run this if you want to
# rebuild that binary yourself.
#
# Usage:
#   bash scripts/build_kenlm.sh
#   # then, e.g., build a 4-gram model from a one-sentence-per-line corpus:
#   ./kenlm/build/bin/lmplz -o 4 < corpus.txt > model.arpa
#   ./kenlm/build/bin/build_binary model.arpa model.bin
# ===========================================================================
set -euo pipefail

echo "Installing KenLM build dependencies (Debian/Ubuntu) ..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y build-essential cmake libboost-all-dev \
       libbz2-dev liblzma-dev zlib1g-dev
else
  echo "apt-get not found; install cmake + boost + bzip2/lzma/zlib headers manually."
fi

if [ ! -d kenlm ]; then
  echo "Cloning KenLM ..."
  git clone https://github.com/kpu/kenlm.git
fi

echo "Building KenLM ..."
mkdir -p kenlm/build
cd kenlm/build
cmake ..
make -j"$(nproc)"

echo ""
echo "Done. Binaries are in kenlm/build/bin/ (lmplz, build_binary)."
echo "Also install the python bindings if you want to query models in Python:"
echo "    pip install https://github.com/kpu/kenlm/archive/master.zip"
