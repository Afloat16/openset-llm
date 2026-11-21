#!/usr/bin/env bash
# Cross-build the patched tree for aarch64 Linux (a manual check that the patches do not break the ARM build).
#   bash scripts/cross-aarch64.sh [workdir]      # needs g++-aarch64-linux-gnu, cmake, git (Debian/Ubuntu: apt)
# If qemu-aarch64 is installed it also runs `llama-server --version` under emulation.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=${1:-$HOME/cross-aarch64}
SHA=e3de715222b2667c54a267441159a5729f11155e
[ -d "$W/.git" ] || git clone -q https://github.com/PrismML-Eng/llama.cpp "$W"
git -C "$W" checkout -q "$SHA"
sh "$HERE/patches/apply.sh" "$W"
for arch in armv8-a armv8.2-a+dotprod; do
  B=$W/build-${arch//+/-}
  cmake -S "$W" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DGGML_CPU_ARM_ARCH="$arch" >/dev/null
  cmake --build "$B" -j "$(nproc)" --target llama-server llama-bench >/dev/null
  file "$B/bin/llama-server"
  if command -v qemu-aarch64 >/dev/null; then
    qemu-aarch64 -L /usr/aarch64-linux-gnu "$B/bin/llama-server" --version 2>&1 | head -3
  fi
  echo "aarch64 ($arch): build OK"
done
