#!/usr/bin/env python3
"""`serve --multi 4`: aggregate throughput and per-stream identity, new tree vs the earlier build, interleaved.

Per round (arm order shuffled): for each arm a fresh server; 4 prompts of prompts24 sent at once (256 tokens each,
ignore_eos, greedy), aggregate tok/s = generated tokens / wall from first send to last reply; then each of the 4 prompts
alone on the same server (per-stream identity: a stream's reply must equal the same request alone). Arms "one" are
single-stream `--no-spec` servers given the first prompt alone (the per-stream baseline). Memory sampled per request.
    python3 multi.py --out ~/res/multi --rounds 3
"""
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor
from bench2 import start, stop, post, Sampler

ARMS = {
    "new_m4": {"env": {}, "cmd": "@H@/release/serve --threads 8 --multi 4"},
    "prev_m4": {"env": {"OPENSET_LLM_HOME": "@H@/.prev"}, "cmd": "@H@/release-prev/serve --threads 8 --multi 4"},
    "new_one": {"env": {}, "cmd": "@H@/release/serve --threads 8 --no-spec"},
    "prev_one": {"env": {"OPENSET_LLM_HOME": "@H@/.prev"}, "cmd": "@H@/release-prev/serve --threads 8 --no-spec"},
}


def req(port, p, out, i):
    t0 = time.time()
    r = post(port, {"messages": p["messages"], "max_tokens": 256, "temperature": 0, "seed": 0, "cache_prompt": False,
                    "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}})
    out[i] = (r, time.time() - t0, time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = [json.loads(l) for l in open("prompts24.jsonl")]
    P = [p for p in P if not p["id"].startswith("code-")][:12]   # short prompts: the 2k-per-stream context fits
    f = open(os.path.join(a.out, "multi.jsonl"), "a")
    for rd in range(a.rounds):
        ps = P[rd * 4:(rd + 1) * 4]
        order = list(ARMS)
        random.Random(f"m{rd}").shuffle(order)
        for arm in order:
            port, proc, log = start(arm, ARMS[arm], a.out, f"r{rd}")
            smp = None
            try:
                post(port, {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}})
                smp = Sampler(proc.pid); smp.start()
                if arm.endswith("_m4"):
                    res = [None] * 4
                    t0 = time.time()
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        requests = [pool.submit(req, port, p, res, i) for i, p in enumerate(ps)]
                        for request in requests:
                            request.result()  # propagate a failed stream before reading partial results
                    wall = max(x[2] for x in res) - t0
                    ntok = sum(x[0]["timings"]["predicted_n"] for x in res)
                    smp.halt = True
                    alone = [None] * 4
                    for i, p in enumerate(ps):
                        req(port, p, alone, i)
                    same = [res[i][0]["choices"][0]["message"]["content"] == alone[i][0]["choices"][0]["message"]["content"] for i in range(4)]
                    row = {"round": rd, "arm": arm, "agg_tok_s": ntok / wall, "tokens": ntok, "wall_s": wall,
                           "per_stream_tok_s": [x[0]["timings"]["predicted_per_second"] for x in res],
                           "alone_tok_s": [x[0]["timings"]["predicted_per_second"] for x in alone], "identical_to_alone": same,
                           "texts": [x[0]["choices"][0]["message"]["content"] for x in res]}
                else:
                    res = [None]
                    req(port, ps[0], res, 0)
                    smp.halt = True
                    row = {"round": rd, "arm": arm, "agg_tok_s": res[0][0]["timings"]["predicted_per_second"],
                           "tokens": res[0][0]["timings"]["predicted_n"], "texts": [res[0][0]["choices"][0]["message"]["content"]]}
                row.update({"prompts": [p["id"] for p in ps], "rss_anon_peak_kb": smp.anon, "mem_avail_min_kb": smp.avail})
                f.write(json.dumps(row) + "\n"); f.flush()
                print(arm, rd, round(row["agg_tok_s"], 2), row.get("identical_to_alone"), flush=True)
            finally:
                if smp is not None:
                    smp.halt = True
                    smp.join()
                stop(proc)
    print("MULTI_DONE", flush=True)


if __name__ == "__main__":
    main()
