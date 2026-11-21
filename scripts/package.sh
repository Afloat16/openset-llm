#!/usr/bin/env bash
# Pack a portable build (./install.sh --portable) into a tarball that ./install.sh --bin-url can fetch.
#   bash scripts/package.sh [out.tar.gz]
# The tarball runs on the distribution it was built on or a newer one (glibc, libgomp, OpenSSL), so build it on the
# oldest one you want to support. install.sh --bin-url checks that the engine starts before it downloads the model.
set -euo pipefail
PREFIX=${OPENSET_LLM_HOME:-$HOME/.openset-llm}
OUT=${1:-openset-llm-linux-$(uname -m).tar.gz}
ls "$PREFIX"/bin/libggml-cpu-*.so >/dev/null 2>&1 || { echo "not a portable build: run ./install.sh --portable" >&2; exit 1; }
tar -czf "$OUT" -C "$PREFIX" --transform 's#^bin#openset-llm#' bin
echo "$OUT ($(du -h "$OUT" | cut -f1))"
