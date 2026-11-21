#!/bin/bash
# Box setup for the launch benchmark (2026-11-11): bare install of this package, the two rivals, optionally a earlier build.
# Runs ON the box: a VM (Ice Lake, 8 vCPU / 8 GB, user + sudo) or a root container with a CPU quota (EPYC Zen 4,
# 8 vCPU / 16 GB cgroup / 10 GB disk, root).
# Layout on the box: ~/release = this package (a clean checkout of the commit under test), bench/ inside it,
#         optionally ~/release-prev (an older build of this package, for the new-vs-earlier arms).
#   bash ~/release/bench/box.sh setup [rivals] [prev] [ppl]
#     1. ./install.sh exactly as a user runs it (VM: nothing staged; root container: base + sidecar staged in a RAM tmpfs and
#        symlinked in, the documented 10 GB-disk workaround), log in ~/res/install.log (apply.sh prints "applied:")
#     2. rivals: ik_llama.cpp + PrismML llama.cpp at IK_SHA / PRISM_SHA, native, static (llama-server, llama-bench)
#     3. prev:   the earlier build into ~/.prev (same model file hard-linked, same fork checkout)
#     4. ppl:    llama-perplexity for the release build + WikiText-2 test
set -u
R=~/res; mkdir -p $R
IK_SHA=${IK_SHA:-c260cbde5766408a4a52f882407100d0d6843ca4}       # ik_llama.cpp HEAD 2026-11-10
PRISM_SHA=${PRISM_SHA:-6447906bf96c1a023980d3453fb59daa6af3bb23}  # PrismML llama.cpp HEAD 2026-11-10
HF=https://huggingface.co
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
ROOT_CONTAINER=0; [ "$(id -u)" = 0 ] && [ -f /sys/fs/cgroup/cpu.max ] && ROOT_CONTAINER=1
REL=~/.openset-llm; SRC=$REL/llama.cpp
CM="-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF"

setup() {
  exec > $R/setup.log 2>&1
  echo "== start $(date -u +%T) args: $*"
  if [ $ROOT_CONTAINER = 1 ]; then   # 10 GB disk < install.sh's 15 GB peak: stage the two downloads in RAM
    $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq curl procps >/dev/null
    mkdir -p /models $REL/models; mountpoint -q /models || mount -t tmpfs -o size=8g tmpfs /models
    ( curl -sSfL --retry 5 -o /models/base.gguf $HF/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf || echo FETCH_FAIL base ) &
    ( curl -sSfL --retry 5 -o /models/mtp.gguf $HF/unsloth/Qwen3.8-27B-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Q4_0.gguf || echo FETCH_FAIL sidecar ) &
    wait
    ln -sf /models/base.gguf $REL/models/base.gguf; ln -sf /models/mtp.gguf $REL/models/mtp-sidecar.gguf
  fi
  echo "== install.sh $(date -u +%T)"
  t0=$(date +%s)
  ( cd ~/release && ./install.sh ) > $R/install.log 2>&1; echo "INSTALL_RC=$? $(( $(date +%s) - t0 )) s"
  grep -c '^applied:' $R/install.log | sed 's/^/patches applied: /'
  if [ $ROOT_CONTAINER = 1 ]; then
    rm -f /models/*.gguf; umount /models
    du -k $REL/models/model.gguf; df -h / | tail -1   # overlayfs block use of the grafted file
  fi
  $SUDO apt-get install -y -qq python3-pip procps iproute2 >/dev/null 2>&1
  for x in "$@"; do $x; done
  rm -rf $SRC/build/{CMakeFiles,common,src,ggml,tools,examples,vendor} 2>/dev/null
  {
    echo "date: $(date -u +%FT%TZ)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), nproc $(nproc), $(grep MemTotal /proc/meminfo | awk '{print $2}') kB MemTotal$( [ -f /sys/fs/cgroup/memory.max ] && echo ", cgroup memory.max $(cat /sys/fs/cgroup/memory.max), cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)")"
    echo "flags: $(grep -m1 '^flags' /proc/cpuinfo | grep -o -w 'avx512[a-z0-9_]*\|amx[a-z0-9_]*' | sort -u | tr '\n' ' ')"
    echo "ours: fork $(git -C $SRC rev-parse HEAD 2>/dev/null), stamp $(cat $REL/bin/.stamp), release tree sha256 $(cd ~/release && cat serve install.sh patches/*.patch kernels/* | sha256sum | cut -c1-16)"
    [ -f ~/.prev/bin/.stamp ] && echo "prev: stamp $(cat ~/.prev/bin/.stamp)"
    for e in ik prism; do [ -f /opt/$e.sha ] && echo "$e: $(cat /opt/$e.sha)"; done
    echo "gcc: $(gcc --version | head -1)"
    ls -la $REL/models $REL/bin; df -h /
  } > $R/box.txt
  echo "SETUP_DONE $(date -u +%T)"
}

rivals() {
  for e in ik prism; do
    echo "== build $e $(date -u +%T)"
    if [ $e = prism ]; then URL=https://github.com/PrismML-Eng/llama.cpp; SHA=$PRISM_SHA; X=; else URL=https://github.com/ikawrakow/ik_llama.cpp; SHA=$IK_SHA; X='-include cstddef'; fi
    $SUDO mkdir -p /opt && $SUDO chmod 777 /opt
    git init -q /opt/$e && git -C /opt/$e fetch -q --depth 1 $URL $SHA && git -C /opt/$e checkout -q FETCH_HEAD
    git -C /opt/$e log -1 --format='%H %cs' > /opt/$e.sha
    cmake -S /opt/$e -B /opt/$e/build $CM -DCMAKE_CXX_FLAGS="$X" > $R/build-$e.log 2>&1 && cmake --build /opt/$e/build -j 8 --target llama-server llama-bench >> $R/build-$e.log 2>&1 || echo BUILD_FAIL $e
    mkdir -p /opt/$e-bin && cp /opt/$e/build/bin/llama-server /opt/$e/build/bin/llama-bench /opt/$e-bin/ && echo "BUILD_OK $e"
    rm -rf /opt/$e
    df -h / | tail -1
  done
}

prev() {   # earlier build (0001-0005) in its own home, same model file and fork checkout
  echo "== prev $(date -u +%T)"
  mkdir -p ~/.prev/models
  ln -f $REL/models/model.gguf ~/.prev/models/model.gguf
  [ -d ~/.prev/llama.cpp ] || cp -a $SRC ~/.prev/llama.cpp
  git -C ~/.prev/llama.cpp checkout -q -- . && git -C ~/.prev/llama.cpp clean -fdq ggml/src/ggml-cpu
  if [ $ROOT_CONTAINER = 1 ]; then
    # small disk: the earlier build's install.sh cannot pass its free-space check next to the installed model, and
    # its -j $(nproc) would start one compiler per host thread; build the same thing by hand, 8 jobs
    git -C ~/.prev/llama.cpp checkout -q e3de715222b2667c54a267441159a5729f11155e
    sh ~/release-prev/patches/apply.sh ~/.prev/llama.cpp > $R/install-prev.log 2>&1
    S=~/.prev/llama.cpp; rm -rf $S/build
    cmake -S $S -B $S/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF >> $R/install-prev.log 2>&1 && \
      cmake --build $S/build -j 8 --target llama-server llama-bench llama-cli >> $R/install-prev.log 2>&1; echo "INSTALL_PREV_RC=$?"
    mkdir -p ~/.prev/bin && cp $S/build/bin/llama-server $S/build/bin/llama-bench $S/build/bin/llama-cli ~/.prev/bin/
    echo "$(cd ~/release-prev && echo manual) $(cat ~/release-prev/patches/*.patch ~/release-prev/kernels/* | cksum | cut -d' ' -f1)" > ~/.prev/bin/.stamp
    rm -rf $S/.git
  else
    ( cd ~/release-prev && OPENSET_LLM_HOME=~/.prev ./install.sh ) > $R/install-prev.log 2>&1; echo "INSTALL_PREV_RC=$?"
  fi
  grep '^applied:' $R/install-prev.log
  rm -rf ~/.prev/llama.cpp/build/{CMakeFiles,common,src,ggml,tools,examples,vendor}
}

ppl() {
  echo "== ppl $(date -u +%T)"
  cmake --build $SRC/build -j 8 --target llama-perplexity > $R/build-ppl.log 2>&1 && cp $SRC/build/bin/llama-perplexity $REL/bin/ && echo BUILD_OK ppl
  if [ ! -f ~/wiki.test.raw ]; then
    curl -sSfL --retry 3 -o ~/wt2.zip $HF/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    (cd ~ && python3 -m zipfile -e wt2.zip wt2 && mv wt2/wikitext-2-raw/wiki.test.raw ~/wiki.test.raw && rm -rf wt2.zip wt2)
  fi
  ls -la ~/wiki.test.raw
}

cmd=${1:-}; shift || true
case $cmd in
  setup) setup "$@";;
  rivals|prev|ppl) exec >> $R/setup.log 2>&1; $cmd;;
  *) echo "usage: box.sh setup [rivals] [prev] [ppl]"; exit 2;;
esac
