#!/bin/bash
# jobs.sh <phase> [...]: the measurement phases, run ON the box after box.sh setup, under setsid nohup. Each phase writes
# ~/res/<phase>/ and a DONE marker; a finished phase is skipped on re-run (the runners also resume row by row).
# Nothing here touches oom_score_adj. Spill / state dirs are under $HOME (disk), never /tmp (often a tmpfs).
#   lb                      llama-bench tg128 + pp512, 3 shuffled rounds, prism / ik -rtr / ours, -t 8, the grafted model file
#   short SEED              prompts24 (all 24), gen 256, greedy, thinking off, blocks of 8 = 3 rounds, arms reshuffled per block:
#                           ours default, ours --no-spec, ik -rtr --jinja, PrismML --jinja (the user-facing launch number)
#   ctx PLAN                long-context decode/prefill (ctxrun.py, plan file plan_<PLAN>.jsonl)
#   replay STEPS ARMS       agent replay (rrun.py, 16k ctx, 64 greedy tokens per step), arms comma-separated, STEPS 0 = all
#   short2 SEED             prompts24 rep 1: ours, --no-spec, earlier build (both), interleaved per block
#   lbprev                  llama-bench new vs earlier build, 3 shuffled rounds
#   multi                   serve --multi 4, new vs earlier: aggregate tok/s + per-stream identity (multi.py)
#   lbmore                  5 more llama-bench new-vs-earlier rounds, for cells that 3 rounds did not settle
#   kld2                    the kld runs again with the decode graph fusions off
#   prevbuild               box.sh prev (the earlier build, side by side in ~/.prev)
#   ident24                 prompts24 identity arms: ours, --no-spec, GGML_ATTN=0 (both), earlier build (both)
#   broad K/N               broad prompt set (BROAD=file, not shipped) shard K/N (gen capped at 256): ours, --no-spec, GGML_ATTN=0, earlier build
#   diag                    repeats / variants of ours --no-spec and ours on 4 prompts (arms_diag.json)
#   lp                      top-2 logprobs per token on the diag prompts (lp.py)
#   quality                 GSM8K-40 + 160 coding problems, new default vs the earlier build's default (task-quality gate for 0008)
#   kld                     KL divergence of every attention path (-ub 1/4/512, GGML_ATTN 1/0) vs the fork's tiled path
#   ppl                     WikiText-2 PPL, 32 x 512: the release build, and GGML_ATTN=0
set -u
L=${L:-$(cd "$(dirname "$0")" && pwd)}; R=~/res; mkdir -p $R   # L = this bench/ folder
REPLAY=${REPLAY:-$L/replay.json}   # agent-replay input (not shipped; see README.md)
BROAD=${BROAD:-$L/prompts_broad.jsonl}
M=~/.openset-llm/models/model.gguf
export LD_LIBRARY_PATH=~/.openset-llm/bin
log() { echo "$(date -u +%T) $*" >> $R/jobs.log; }

lb() {
  O=$R/lb; mkdir -p $O
  for round in 1 2 3; do
    for arm in $(printf 'prism\nik\nours\n' | shuf); do
      case $arm in prism) B=/opt/prism-bin/llama-bench; x=;; ik) B=/opt/ik-bin/llama-bench; x="-rtr 1";; ours) B=~/.openset-llm/bin/llama-bench; x=;; esac
      for t in tg128 pp512; do
        [ $t = tg128 ] && a="-p 0 -n 128 -r 1" || a="-p 512 -n 0 -r 1"
        echo "@@ $arm $round $t" >> $O/lb.txt
        $B -m $M -t 8 $x $a -o json 2>>$O/lb.err | tr -d '\n' >> $O/lb.txt; echo >> $O/lb.txt
      done
    done
  done
}

short() {
  O=$R/short; mkdir -p $O; cd $L
  python3 bench2.py --arms arms_short.json --only-arms ours,ours_ns,ik,prism --prompts prompts24.jsonl --out $O/results.jsonl \
    --rep 0 --gen 256 --block 8 --seed ${1:-0} > $O/run.log 2>&1
}

ctx() {
  O=$R/ctx; mkdir -p $O; cd $L
  python3 ctxrun.py --arms arms_ctx.json --prompts ctx_prompts.json --plan plan_$1.jsonl --out $O/rows.jsonl >> $O/run.log 2>&1
}

replay() {
  O=$R/replay; mkdir -p $O; cd $L
  st=""; [ "$1" != 0 ] && st="--steps $1"
  for arm in ${2//,/ }; do
    [ -f $O/steps_$arm.done ] && continue
    log "replay $arm start"
    python3 rrun.py --arms arms_replay.json --arm $arm --replay $REPLAY ${REPLAY_TOKENS:+--tokens $REPLAY_TOKENS} --ctx 16384 \
      --gen 64 $st --out $O > $O/$arm.out 2>&1 && touch $O/steps_$arm.done
    log "replay $arm end"
  done
}

short2() {  # rep 1: the new tree vs the earlier build (0001-0005), interleaved per block, same prompts
  O=$R/short; mkdir -p $O; cd $L
  python3 bench2.py --arms arms_short.json --only-arms ours,ours_ns,prev,prev_ns --prompts prompts24.jsonl --out $O/results.jsonl \
    --rep 1 --gen 256 --block 8 --seed ${1:-0}1 > $O/run_rep1.log 2>&1
}

lbprev() {  # llama-bench, new build vs the earlier build, 3 shuffled rounds
  O=$R/lb; mkdir -p $O
  for round in ${LBR:-1 2 3}; do
    for arm in $(printf 'ours\nprev\n' | shuf); do
      B=~/.openset-llm/bin/llama-bench; [ $arm = prev ] && B=~/.prev/bin/llama-bench
      for t in tg128 pp512; do
        [ $t = tg128 ] && a="-p 0 -n 128 -r 1" || a="-p 512 -n 0 -r 1"
        echo "@@ ${arm}2 $round $t" >> $O/lb.txt
        $B -m $M -t 8 $a -o json 2>>$O/lb.err | tr -d '\n' >> $O/lb.txt; echo >> $O/lb.txt
      done
    done
  done
}

lbmore() {  # 5 more llama-bench new-vs-prev rounds where 3 did not settle a cell
  LBR="4 5 6 7 8" lbprev
}

multi() {
  cd $L; python3 multi.py --out $R/multi --rounds 3 > $R/multi.log 2>&1
}

prevbuild() {
  bash $L/box.sh prev
}

ident24() {
  O=$R/ident24; mkdir -p $O; cd $L
  python3 bench2.py --arms arms_short.json --only-arms ours,ours_ns,ours_a0,ours_a0_ns,prev,prev_ns --prompts prompts24.jsonl \
    --out $O/results.jsonl --rep 0 --gen 256 --block 8 --seed 11 > $O/run.log 2>&1
}

broad() {
  k=${1%/*}; O=$R/broad$k; mkdir -p $O; cd $L
  python3 bench2.py --arms arms_short.json --only-arms ours,ours_ns,ours_a0,prev --prompts $BROAD --out $O/results.jsonl \
    --shard $1 --rep 0 --gen 256 --max-gen 256 --block 8 --seed 21 > $O/run.log 2>&1
}

diag() {   # repeats / variants of ours --no-spec and ours on the 4 diag prompts
  O=$R/diag; mkdir -p $O; cd $L
  python3 bench2.py --arms arms_diag.json --prompts prompts_diag.jsonl --out $O/results.jsonl --rep 0 --gen 256 --block 4 --seed 5 > $O/run.log 2>&1
}

lp() {     # top-2 logprobs at every generated token: how near a tie is each divergence
  cd $L; python3 lp.py --arms arms_short.json --only ours,ours_ns,ours_a0_ns,ours_a0 --prompts prompts_diag.jsonl --out $R/lp > $R/lp.log 2>&1
}

kld() {    # accuracy of each attention path: KL divergence of next-token distributions vs the fork's tiled f32 path
  # (GGML_ATTN=0, -ub 512, the >= 64-row path), 4 x 512 WikiText tokens; -ub 4 = the MTP-verify row count, -ub 1 = decode
  O=$R/kld; mkdir -p $O; P=~/.openset-llm/bin/llama-perplexity; W="-f $HOME/wiki.test.raw -c 512 --chunks 4 -t 8 -fa on -b 512"
  [ -f $O/base.kld ] || GGML_ATTN=0 $P -m $M $W -ub 512 --kl-divergence-base $O/base.kld > $O/base.log 2>&1
  for ub in 1 4 512; do for at in 1 0; do
    [ $ub = 512 ] && [ $at = 0 ] && continue
    t=ub${ub}_attn${at}; [ -f $O/$t.log ] && grep -q 'Mean    KLD' $O/$t.log && continue
    GGML_ATTN=$at $P -m $M $W -ub $ub --kl-divergence-base $O/base.kld --kl-divergence > $O/$t.log 2>&1
    echo "$t $(grep -E 'Mean    KLD|Maximum KLD|Same top p|RMS Δp' $O/$t.log | tr -s ' ' | tr '\n' ';')" >> $O/summary.txt
  done; done
  rm -f $O/base.kld
}

quality() {  # task-quality gate for 0008 (2026-11-11): new default serve vs the earlier build, interleaved
  # GSM8K-40 (768 tokens) and 160 of the 421 HumanEval + MBPP-sanitized problems (seeded subset, 512 tokens)
  cd $L
  python3 bench2.py --arms arms_quality.json --prompts gsm8k.jsonl --out $R/q_gsm/results.jsonl --rep 0 --gen 768 --block 10 --seed 7 > $R/q_gsm.log 2>&1
  python3 bench2.py --arms arms_quality.json --prompts problems_q160.jsonl --out $R/q_code/results.jsonl --rep 0 --gen 512 --block 12 --seed 8 > $R/q_code.log 2>&1
  python3 evaluate.py problems_q160.jsonl $R/q_code > $R/q_code/graded.jsonl 2> $R/q_code/eval.err
}

kld2() {   # is the M = 1 difference our kernel or the decode-only K6 graph fusions? the same runs with GGML_K6=0
  O=$R/kld; P=~/.openset-llm/bin/llama-perplexity; W="-f $HOME/wiki.test.raw -c 512 --chunks 4 -t 8 -fa on -b 512"
  [ -f $O/base.kld ] || GGML_ATTN=0 $P -m $M $W -ub 512 --kl-divergence-base $O/base.kld > $O/base2.log 2>&1
  for t in ub1_attn1_k60 ub4_attn1_k60 ub1_attn0_k60 ub4_attn0_k60; do
    ub=${t#ub}; ub=${ub%%_*}; at=${t#*attn}; at=${at%%_*}
    GGML_K6=0 GGML_ATTN=$at $P -m $M $W -ub $ub --kl-divergence-base $O/base.kld --kl-divergence > $O/$t.log 2>&1
    echo "$t $(grep -E 'Mean    KLD|Maximum KLD|Same top p|RMS Δp' $O/$t.log | tr -s ' ' | tr '\n' ';')" >> $O/summary.txt
  done
  rm -f $O/base.kld
}

ppl() {
  O=$R/ppl; mkdir -p $O
  if [ ! -x ~/.prev/bin/llama-perplexity ]; then   # the earlier build tree was pruned after install
    S=~/.prev/llama.cpp; cmake -S $S -B $S/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF > $O/build-prev.log 2>&1 && \
      cmake --build $S/build -j 8 --target llama-perplexity >> $O/build-prev.log 2>&1 && cp $S/build/bin/llama-perplexity ~/.prev/bin/
  fi
  for a in new attn0 prev; do
    E=; [ $a = attn0 ] && E="GGML_ATTN=0"; B=~/.openset-llm/bin; [ $a = prev ] && B=~/.prev/bin
    t0=$(date +%s)
    env $E $B/llama-perplexity -m $M -f ~/wiki.test.raw -c 512 -b 512 --chunks 32 -t 8 -fa on > $O/ppl.$a.log 2>&1
    echo "$a $(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' $O/ppl.$a.log) $(( $(date +%s) - t0 )) s" >> $O/summary.txt
  done
}

while [ $# -gt 0 ]; do
  ph=$1; shift; arg=; arg2=
  case $ph in short|short2|ctx|broad) arg=$1; shift;; replay) arg=$1; arg2=$2; shift 2;; esac
  tag=$ph${arg:+_${arg//\//-}}
  [ -f $R/DONE.$tag ] && continue
  log "start $tag"
  $ph $arg $arg2
  touch $R/DONE.$tag; log "end $tag"
done
log ALL_DONE
