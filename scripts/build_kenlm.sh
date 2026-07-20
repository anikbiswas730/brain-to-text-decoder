#!/usr/bin/env bash
# Builds the KenLM command-line binaries (lmplz, build_binary) used to train
# and package the n-gram language model for pyctcdecode.
#
# Run this once before using decode_llm.py:
#   bash scripts/build_kenlm.sh
#
# Requires: apt (Debian/Ubuntu), cmake, a C++ toolchain. On Kaggle/Colab this
# just works out of the box; on a bare Linux box you may need `sudo` in
# front of the apt-get commands below.

set -euo pipefail

echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake \
    libboost-all-dev libboost-program-options-dev libboost-thread-dev \
    libbz2-dev liblzma-dev

echo "Fetching KenLM source..."
rm -rf kenlm
wget -q -O - https://kheafield.com/code/kenlm.tar.gz | tar xz

echo "Building KenLM (lmplz, build_binary)..."
mkdir -p kenlm/build
cd kenlm/build
cmake .. > /dev/null
make -j"$(nproc)" > /dev/null

KENLM_BIN="$(pwd)/bin"
if [ ! -f "${KENLM_BIN}/lmplz" ]; then
    echo "ERROR: lmplz build failed - check CMake/make output above." >&2
    exit 1
fi

echo "KenLM binaries ready at: ${KENLM_BIN}"
