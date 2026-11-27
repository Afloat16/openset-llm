#!/usr/bin/env python3
"""Launch-benchmark analysis: python3 analyze.py raw > tables.md   (raw/<box>/{short,lb,ctx,replay,ident24,broad*,ppl}/)

Boxes: e1 = identity box (Ice Lake), e2 / e3 = Ice Lake bench boxes, d1 = Zen 4 bench box. Every ratio is paired on one
box (per prompt, per round, or per replay step) and pooled within a CPU class; CIs are percentile bootstraps of the
median (10,000 resamples of the paired units). Nothing is pooled across CPU classes.
"""
import glob, json, os, random, re, statistics as st, sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw"
CLASS = {"e2": "Ice Lake", "e3": "Ice Lake", "d1": "Zen 4", "e1": "Ice Lake"}
rng = random.Random(2026)
OUT = {}


def med(x):
    return st.median(x) if x else float("nan")


def ci(x, f=med, n=10000):
    if len(x) < 2:
        return (float("nan"), float("nan"))
    b = sorted(f([x[rng.randrange(len(x))] for _ in x]) for _ in range(n))
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]


def fmt(x, ci_=None, nd=2, suf=""):
    s = f"{x:.{nd}f}{suf}"
    if ci_ and ci_[0] == ci_[0]:
        s += f" [{ci_[0]:.{nd}f}, {ci_[1]:.{nd}f}]"
    return s


def rows(path):
    out = []
    for f in glob.glob(path):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l)
                r["_box"] = f.split(os.sep)[-3] if "/" in f else "?"
                out.append(r)
    return out


def box_of(f):
    return os.path.relpath(f, RAW).split(os.sep)[0]


def load(sub, name):
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "*", sub, name))):
        b = box_of(f)
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                r["_box"] = b
                out.append(r)
    return out


# ------------------------------------------------------------------ short context (prompts24)
def short():
    R = [r for r in load("short", "results.jsonl") if "error" not in r and r.get("rep", 0) == 0]
    print("## 1. Short context: prompts24, 256 tokens, greedy, thinking off (llama-server `predicted_per_second`)\n")
    print("Per box: arm medians [95 % CI] over prompts; ratios paired per prompt. Pooled rows: all prompts of that CPU class.\n")
    print("| box | n | ours `./serve` | ours `--no-spec` | ik_llama `-rtr` | PrismML | ours / ik | ours / PrismML | ours / `--no-spec` | `--no-spec` / ik | acceptance (ours) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for b in sorted({r["_box"] for r in R}):
        groups[b] = [r for r in R if r["_box"] == b]
    for cls in ("Ice Lake", "Zen 4"):
        bs = [b for b in groups if CLASS.get(b) == cls]
        if len(bs) > 1:
            groups[f"pooled {cls}"] = [r for b in bs for r in groups[b]]
    for g, rr in groups.items():
        by = {}
        for r in rr:
            by.setdefault((r["_box"], r["prompt_id"]), {})[r["arm"]] = r
        def arm(a):
            return [v[a]["decode_tok_s"] for v in by.values() if a in v and v[a].get("decode_tok_s")]
        def ratio(a, b):
            return [v[a]["decode_tok_s"] / v[b]["decode_tok_s"] for v in by.values() if a in v and b in v
                    and v[a].get("decode_tok_s") and v[b].get("decode_tok_s")]
        cells = []
        for a in ("ours", "ours_ns", "ik", "prism"):
            x = arm(a)
            cells.append(fmt(med(x), ci(x)) if x else "–")
        rat = {}
        for a, b in (("ours", "ik"), ("ours", "prism"), ("ours", "ours_ns"), ("ours_ns", "ik")):
            x = ratio(a, b)
            rat[(a, b)] = (med(x), ci(x), len(x))
            cells.append(fmt(med(x), ci(x), suf="×") if x else "–")
        dn = sum(v["ours"].get("draft_n") or 0 for v in by.values() if "ours" in v)
        da = sum(v["ours"].get("draft_accepted") or 0 for v in by.values() if "ours" in v)
        cells.append(f"{da / dn:.2f}" if dn else "–")
        n = len(arm("ours"))
        print(f"| {g} | {n} | " + " | ".join(cells) + " |")
        OUT.setdefault("short", {})[g] = {"n": n, "ours": med(arm("ours")), "ours_ci": ci(arm("ours")),
                                          "ours_ns": med(arm("ours_ns")), "ik": med(arm("ik")), "prism": med(arm("prism")),
                                          "ratios": {f"{a}/{b}": v for (a, b), v in rat.items()},
                                          "accept": da / dn if dn else None}
    # comparability: thinking, output lengths
    print("\n**Comparability of replies (every box):**\n")
    print("| arm | rows | replies with `<think>` in content | reasoning_content non-empty | tokens generated (sum) | finish=length |")
    print("|---|---|---|---|---|---|")
    for a in ("ours", "ours_ns", "ik", "prism"):
        x = [r for r in R if r["arm"] == a]
        print(f"| {a} | {len(x)} | {sum(1 for r in x if r.get('think_in_content'))} | {sum(1 for r in x if r.get('reasoning_chars'))} "
              f"| {sum(r.get('predicted_n') or 0 for r in x)} | {sum(1 for r in x if r.get('finish') == 'length')} |")
    print("\n**Memory during the short-context pass** (server RssAnon peak / box MemAvailable min over the requests, MiB; "
          "in a container MemAvailable is the host's, so only RssAnon counts there):\n")
    print("| box | arm | peak RssAnon | min MemAvailable |")
    print("|---|---|---|---|")
    for b in sorted({r["_box"] for r in R}):
        for a in ("ours", "ours_ns", "ik", "prism"):
            x = [r for r in R if r["_box"] == b and r["arm"] == a and r.get("rss_anon_peak_kb")]
            if x:
                print(f"| {b} | {a} | {max(r['rss_anon_peak_kb'] for r in x) // 1024} | {min(r['mem_avail_min_kb'] for r in x) // 1024} |")
    # spec == no-spec on the launch prompts
    same = tot = 0
    for b in {r["_box"] for r in R}:
        by = {}
        for r in R:
            if r["_box"] == b:
                by.setdefault(r["prompt_id"], {})[r["arm"]] = r["sha256"]
        for v in by.values():
            if "ours" in v and "ours_ns" in v:
                tot += 1
                same += v["ours"] == v["ours_ns"]
    print(f"\nours default vs `--no-spec`, same box, byte-identical replies: **{same}/{tot}**\n")
    OUT.setdefault("ident", {})["short ours vs ours_ns"] = (same, tot)


REG = []   # (metric, class/box, ratio new/prev, CI, n): every cell where new is > 2 % below the earlier build


def flag(metric, where, x, higher_better=True):
    """x = list of paired new/prev ratios (for lower-is-better metrics pass prev/new so that < 1 is always worse)"""
    if not x:
        return "–"
    m, c = med(x), ci(x)
    if m < 0.98:
        REG.append((metric, where, m, c, len(x)))
    return fmt(m, c, nd=3, suf="×") + f" (n={len(x)})" + (" **< 0.98**" if m < 0.98 else "")


def newprev():
    R = [r for r in load("short", "results.jsonl") if "error" not in r and r.get("rep") == 1]
    if not R:
        return
    print("## 1b. New tree vs the earlier build (0001–0005), prompts24 rep 1, interleaved per block\n")
    print("| class | box | n | new `./serve` | prev `./serve` | new / prev | new `--no-spec` | prev `--no-spec` | new / prev (`--no-spec`) | accept new / prev | identical new vs prev (default / `--no-spec`) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cls in ("Ice Lake", "Zen 4"):
        boxes = sorted({r["_box"] for r in R if CLASS[r["_box"]] == cls})
        for b in boxes + (["pooled"] if len(boxes) > 1 else []):
            rr = [r for r in R if CLASS[r["_box"]] == cls and (b == "pooled" or r["_box"] == b)]
            by = {}
            for r in rr:
                by.setdefault((r["_box"], r["prompt_id"]), {})[r["arm"]] = r
            def arm(a):
                return [v[a]["decode_tok_s"] for v in by.values() if a in v]
            def rat(a, c):
                return [v[a]["decode_tok_s"] / v[c]["decode_tok_s"] for v in by.values() if a in v and c in v]
            def acc(a):
                dn = sum(v[a].get("draft_n") or 0 for v in by.values() if a in v)
                return sum(v[a].get("draft_accepted") or 0 for v in by.values() if a in v) / dn if dn else float("nan")
            same = sum(v["ours"]["sha256"] == v["prev"]["sha256"] for v in by.values() if "ours" in v and "prev" in v)
            same_ns = sum(v["ours_ns"]["sha256"] == v["prev_ns"]["sha256"] for v in by.values() if "ours_ns" in v and "prev_ns" in v)
            print(f"| {cls} | {b} | {len(arm('ours'))} | {fmt(med(arm('ours')))} | {fmt(med(arm('prev')))} | "
                  f"{flag('short-context user tok/s (default)', f'{cls} {b}', rat('ours','prev'))} | {fmt(med(arm('ours_ns')))} | "
                  f"{fmt(med(arm('prev_ns')))} | {flag('short-context tok/s (--no-spec)', f'{cls} {b}', rat('ours_ns','prev_ns'))} | "
                  f"{acc('ours'):.3f} / {acc('prev'):.3f} | {same}/{len(arm('prev'))} / {same_ns}/{len(arm('prev_ns'))} |")
            if b in ("pooled",) or len(boxes) == 1:
                OUT.setdefault("newprev_short", {})[cls] = {"def": (med(rat("ours", "prev")), ci(rat("ours", "prev"))),
                                                           "ns": (med(rat("ours_ns", "prev_ns")), ci(rat("ours_ns", "prev_ns")))}
    print("\nMemory (rep 1), peak server RssAnon MiB / min MemAvailable MiB (VM boxes only):\n")
    print("| box | new default | prev default | new `--no-spec` | prev `--no-spec` |")
    print("|---|---|---|---|---|")
    for b in sorted({r["_box"] for r in R}):
        cells = []
        for a in ("ours", "prev", "ours_ns", "prev_ns"):
            x = [r for r in R if r["_box"] == b and r["arm"] == a and r.get("rss_anon_peak_kb")]
            cells.append(f"{max(r['rss_anon_peak_kb'] for r in x)//1024} / {min(r['mem_avail_min_kb'] for r in x)//1024 if CLASS[b] != 'Zen 4' else '–'}" if x else "–")
            if x and CLASS[b] != "Zen 4":
                OUT.setdefault("mem", {})[(b, a)] = (max(r["rss_anon_peak_kb"] for r in x) // 1024, min(r["mem_avail_min_kb"] for r in x) // 1024)
        print(f"| {b} | " + " | ".join(cells) + " |")
    print()


# ------------------------------------------------------------------ llama-bench
def lb():
    print("## 2. llama-bench, plain decode tg128 and prefill pp512, `-t 8`, 3 shuffled rounds per box\n")
    print("| box | test | ours | ik `-rtr 1` | PrismML | ours / ik (per round) | ours / PrismML |")
    print("|---|---|---|---|---|---|---|")
    allr = {}
    for f in sorted(glob.glob(os.path.join(RAW, "*", "lb", "lb.txt"))):
        b = box_of(f)
        cur = None
        for l in open(f):
            l = l.strip()
            if l.startswith("@@"):
                _, arm, rnd, test = l.split()
                cur = (arm, int(rnd), test)
            elif l and cur:
                try:
                    j = json.loads(l)
                    ts = j[0]["avg_ts"]
                except Exception:
                    continue
                allr.setdefault(b, {}).setdefault(cur[2], {}).setdefault(cur[1], {})[cur[0]] = ts
    for b, tests in allr.items():
        for t in ("tg128", "pp512"):
            rd = tests.get(t, {})
            def a(x):
                return [v[x] for v in rd.values() if x in v]
            def r(x, y):
                return [v[x] / v[y] for v in rd.values() if x in v and y in v]
            print(f"| {b} | {t} | {fmt(med(a('ours')))} | {fmt(med(a('ik')))} | {fmt(med(a('prism')))} | "
                  f"{fmt(med(r('ours','ik')), suf='×')} ({', '.join(f'{v:.2f}' for v in r('ours','ik'))}) | "
                  f"{fmt(med(r('ours','prism')), suf='×')} ({', '.join(f'{v:.2f}' for v in r('ours','prism'))}) |")
            OUT.setdefault("lb", {}).setdefault(b, {})[t] = {"ours": med(a("ours")), "ik": med(a("ik")), "prism": med(a("prism")),
                                                             "ours/ik": r("ours", "ik"), "ours/prism": r("ours", "prism")}
    for cls in ("Ice Lake", "Zen 4"):
        for t in ("tg128", "pp512"):
            for y in ("ik", "prism"):
                x = [v for b in allr if CLASS.get(b) == cls for v in OUT["lb"][b][t][f"ours/{y}"]]
                if x:
                    OUT.setdefault("lb_pooled", {})[(cls, t, y)] = (med(x), ci(x), len(x))
    print("\nNew build vs the earlier build (llama-bench, interleaved, per round):\n")
    print("| box | test | new | prev | new / prev |")
    print("|---|---|---|---|---|")
    for b, tests in allr.items():
        for t in ("tg128", "pp512"):
            rd = tests.get(t, {})
            x = [v["ours2"] / v["prev2"] for v in rd.values() if "ours2" in v and "prev2" in v]
            if x:
                print(f"| {b} | {t} | {fmt(med([v['ours2'] for v in rd.values() if 'ours2' in v]))} | "
                      f"{fmt(med([v['prev2'] for v in rd.values() if 'prev2' in v]))} | {flag(f'llama-bench {t}', b, x)} |")
    for cls in ("Ice Lake", "Zen 4"):
        for t in ("tg128", "pp512"):
            x = [v["ours2"] / v["prev2"] for b, tests in allr.items() if CLASS.get(b) == cls for v in tests.get(t, {}).values() if "ours2" in v and "prev2" in v]
            if x:
                print(f"| **pooled {cls}** | {t} | | | {flag(f'llama-bench {t} (pooled)', cls, x)} |")
                OUT.setdefault("lb_np", {})[(cls, t)] = (med(x), ci(x), len(x))
    if OUT.get("lb_pooled"):
        print("\nPooled per class (rounds × boxes): " + "; ".join(
            f"{c} {t} ours/{y} {fmt(m, c_, suf='×')} (n={n})" for (c, t, y), (m, c_, n) in OUT["lb_pooled"].items()) + "\n")


# ------------------------------------------------------------------ long context
def ctx():
    R = load("ctx", "rows.jsonl")
    if not R:
        return
    print("## 3. Long context: one WikiText haystack per depth (2026-11-09 harness), 128 greedy tokens, same text for every engine\n")
    probes = [r for r in R if r.get("mode") == "load"]
    if probes:
        print("**Load-only fit probes** (no generation or throughput measurement):\n")
        print("| box | round | depth | arm | status |")
        print("|---|---|---|---|---|")
        for r in probes:
            print(f"| {r['_box']} | {r['round']} | {r['ctx_n'] // 1024}k | {r['arm']} | {r['status']} |")
        print()
    R = [r for r in R if r.get("mode", "fill") == "fill"]
    if not R:
        return
    print("Decode = reply tok/s after the prompt; prefill = prompt tok/s. Medians over rounds; ratios paired per (box, round).\n")
    print("| class | depth | arm | n | decode tok/s | prefill tok/s | acceptance | peak RssAnon MiB | min MemAvailable MiB | status | recall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(CLASS[r["_box"]], r["ctx_n"], r["arm"]) for r in R}, key=lambda k: (k[0], k[1], ["ours", "ours_ns", "prev", "prev_ns", "ik", "prism"].index(k[2])))
    for cls, c, a in keys:
        x = [r for r in R if CLASS[r["_box"]] == cls and r["ctx_n"] == c and r["arm"] == a]
        ok = [r for r in x if r["status"] == "ok"]
        dn = sum(r.get("draft_n") or 0 for r in ok)
        da = sum(r.get("draft_accepted") or 0 for r in ok)
        rec = sum(sum(r.get("recall", [])) for r in ok)
        print(f"| {cls} | {c // 1024}k | {a} | {len(ok)} | {fmt(med([r['decode_tok_s'] for r in ok]))} | "
              f"{fmt(med([r['prefill_tok_s'] for r in ok]), nd=1)} | {f'{da/dn:.2f}' if dn else '–'} | "
              f"{max([r.get('peak_anon_kB', 0) for r in ok] or [0]) // 1024} | "
              f"{min([r.get('min_avail_kB', 1 << 40) for r in ok] or [0]) // 1024 if ok else '–'} | "
              f"{','.join(sorted({r['status'] for r in x}))} | {rec}/{5 * len(ok)} |")
        OUT.setdefault("ctx", {})[(cls, c, a)] = {"n": len(ok), "dec": med([r["decode_tok_s"] for r in ok]),
                                                 "pre": med([r["prefill_tok_s"] for r in ok]), "acc": da / dn if dn else None}
    print("\nPaired ratios (per box and round), median [min, max]:\n")
    print("| class | depth | ours / ik decode | ours / PrismML decode | ours / `--no-spec` decode | ours / ik prefill | ours / PrismML prefill | new / prev decode (default) | new / prev decode (`--no-spec`) | new default / prev `--no-spec` | new / prev prefill (default) | new / prev prefill (`--no-spec`) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cls, c in sorted({(k[0], k[1]) for k in keys}):
        pr = {}
        for r in R:
            if CLASS[r["_box"]] == cls and r["ctx_n"] == c and r["status"] == "ok":
                pr.setdefault((r["_box"], r["round"]), {})[r["arm"]] = r
        def rr(a, b, f):
            x = [v[a][f] / v[b][f] for v in pr.values() if a in v and b in v]
            return f"{med(x):.2f}× [{min(x):.2f}, {max(x):.2f}] (n={len(x)})" if x else "–"
        def fl(a, b, f, name):
            return flag(f"{name} at {c // 1024}k", cls, [v[a][f] / v[b][f] for v in pr.values() if a in v and b in v])
        print(f"| {cls} | {c // 1024}k | {rr('ours','ik','decode_tok_s')} | {rr('ours','prism','decode_tok_s')} | "
              f"{rr('ours','ours_ns','decode_tok_s')} | {rr('ours','ik','prefill_tok_s')} | {rr('ours','prism','prefill_tok_s')} | "
              f"{fl('ours','prev','decode_tok_s','decode, default')} | {fl('ours_ns','prev_ns','decode_tok_s','decode, --no-spec')} | "
              f"{fl('ours','prev_ns','decode_tok_s','decode, new default vs prev --no-spec')} | {fl('ours','prev','prefill_tok_s','prefill, default')} | "
              f"{fl('ours_ns','prev_ns','prefill_tok_s','prefill, --no-spec')} |")
    # identity at depth
    same = tot = 0
    for r in R:
        pass
    by = {}
    for r in R:
        if r["status"] == "ok" and r["arm"] in ("ours", "ours_ns"):
            by.setdefault((r["_box"], r["round"], r["ctx_n"]), {})[r["arm"]] = r["sha256"]
    for v in by.values():
        if len(v) == 2:
            tot += 1
            same += v["ours"] == v["ours_ns"]
    print(f"\nLong-context replies, ours default vs `--no-spec` (where both ran with speculation actually on, i.e. Zen 4 and Ice Lake 4k): **{same}/{tot}** byte-identical\n")
    OUT.setdefault("ident", {})["ctx ours vs ours_ns"] = (same, tot)


# ------------------------------------------------------------------ agent replay
def replay():
    print("## 4. Agent replay: seconds per agent step (11-request coding agent with tail rewrites, 16k ctx, 64 greedy tokens per step)\n")
    print("Step time = client wall time of the request (prompt read + 64 tokens). Rivals ran steps 0–5; the per-engine ratio "
          "pairs the same steps on the same box.\n")
    for b in sorted({box_of(f) for f in glob.glob(os.path.join(RAW, "*", "replay", "steps_*.jsonl"))}):
        S = {}
        for f in glob.glob(os.path.join(RAW, b, "replay", "steps_*.jsonl")):
            arm = os.path.basename(f)[6:-6]
            S[arm] = {(r.get("session", 1), r["step"]): r for r in map(json.loads, filter(str.strip, open(f)))}
        arms = [a for a in ("ours", "ours_nr", "ours_ns", "ours_ns_nr", "prev", "prev_ns", "ik", "prism") if a in S]
        steps = sorted({k for a in arms for k in S[a] if k[0] == 1})
        print(f"### {b} ({CLASS.get(b)})\n")
        print("| step | n_prompt | " + " | ".join(f"{a}: read / wall s" for a in arms) + " |")
        print("|---|---|" + "---|" * len(arms))
        for k in steps:
            n = next((S[a][k]["n_prompt"] for a in arms if k in S[a]), "")
            cells = []
            for a in arms:
                r = S[a].get(k)
                cells.append(f"{r['prompt_n']} / {r['wall_s']}" if r and r.get("wall_s") else ("died" if r else "–"))
            print(f"| {k[1]} | {n} | " + " | ".join(cells) + " |")
        ok = lambda a, k: k in S[a] and S[a][k].get("wall_s")
        if "ours" in S:
            print()
            print("| arm | steps (paired with ours) | sum wall s | s per step | ours, same steps | arm / ours (sum) | per-step ratio, steps ≥ 1: median [min, max] |")
            print("|---|---|---|---|---|---|---|")
            tot = {}
            for a in arms:
                ks = [k for k in steps if ok(a, k) and ok("ours", k)]
                if not ks:
                    continue
                ta = sum(S[a][k]["wall_s"] for k in ks); to = sum(S["ours"][k]["wall_s"] for k in ks)
                x = [S[a][k]["wall_s"] / S["ours"][k]["wall_s"] for k in ks if k[1] >= 1]
                tot[a] = (ta, len(ks))
                rng_ = f"{med(x):.2f}× [{min(x):.2f}, {max(x):.2f}]" if x else "–"
                print(f"| {a} | {ks[0][1]}–{ks[-1][1]} | {ta:.0f} | {ta/len(ks):.0f} | {to:.0f} | {ta/to:.2f}× | {rng_} |")
                if a != "ours":
                    OUT.setdefault("replay", {}).setdefault(b, {})[a] = {"sum_ratio": ta / to, "step_med": med(x) if x else None,
                                                                        "steps": len(ks), "s_per_step": ta / len(ks), "ours_s_per_step": to / len(ks)}
                    if a.startswith("prev") and x:
                        flag(f"agent step seconds vs {a} (prev/new)", b, x)
            OUT.setdefault("replay", {}).setdefault(b, {})["_tot"] = tot
            full = [k for k in steps if k in S["ours"] and S["ours"][k].get("wall_s")]
            print(f"- ours over all {len(full)} steps: {sum(S['ours'][k]['wall_s'] for k in full):.0f} s, "
                  f"{sum(S['ours'][k]['wall_s'] for k in full)/len(full):.0f} s/step; prompt tokens read "
                  f"{sum(S['ours'][k]['prompt_n'] or 0 for k in full if k[1] >= 1)} over steps 1–{full[-1][1]}")
            acc = [(S["ours"][k].get("draft_n") or 0, S["ours"][k].get("draft_n_accepted") or 0) for k in full]
            if sum(d for d, _ in acc):
                print(f"- draft acceptance on the replay (ours): {sum(a for _, a in acc) / sum(d for d, _ in acc):.2f}")
            for other in ("ours_nr", "ours_ns_nr"):
                if other in S:
                    base = "ours" if other == "ours_nr" else "ours_ns"
                    if base not in S:
                        continue
                    ks = [k for k in S[base] if k in S[other]]
                    same = sum(1 for k in ks if S[base][k]["text"] == S[other][k]["text"] and S[base][k]["tool_calls"] == S[other][k]["tool_calls"])
                    print(f"- identity {base} (resume) vs {other} (`--no-resume`): **{same}/{len(ks)}** requests byte-identical (content + tool calls)")
                    OUT.setdefault("ident", {})[f"replay {b} {base} vs {other}"] = (same, len(ks))
        print()


# ------------------------------------------------------------------ identity sets
def ident():
    print("## 5. Greedy identity on the combined tree\n")
    print("| set (box) | comparison | byte-identical |")
    print("|---|---|---|")
    for sub in ("ident24", "broad0", "broad1"):
        R = [r for r in load(sub, "results.jsonl") if "error" not in r]
        if not R:
            continue
        for b in sorted({r["_box"] for r in R}):
            by = {}
            for r in R:
                if r["_box"] == b:
                    by.setdefault(r["prompt_id"], {})[r["arm"]] = r["sha256"]
            for x, y in (("ours", "ours_ns"), ("ours_a0", "prev"), ("ours_a0_ns", "prev_ns"), ("ours_ns", "prev_ns"),
                         ("ours", "prev"), ("ours_a0", "ours_a0_ns"), ("prev", "prev_ns"), ("ours_ns", "ours_a0_ns")):
                v = [p for p in by.values() if x in p and y in p]
                if v:
                    s = sum(p[x] == p[y] for p in v)
                    print(f"| {sub} ({b}) | {x} vs {y} | {s}/{len(v)} |")
                    OUT.setdefault("ident", {})[f"{sub} {x} vs {y}"] = (s, len(v))
            if sub.startswith("broad"):
                acc = {}
                for r in R:
                    if r["_box"] == b and r["arm"] in ("ours", "ours_a0", "prev"):
                        a = acc.setdefault(r["arm"], [0, 0])
                        a[0] += r.get("draft_n") or 0
                        a[1] += r.get("draft_accepted") or 0
                print(f"| {sub} ({b}) | draft acceptance | " + ", ".join(f"{k} {v[1]/v[0]:.2f}" for k, v in acc.items() if v[0]) + " |")
                sp = {}
                for r in R:
                    if r["_box"] == b and r.get("decode_tok_s"):
                        sp.setdefault(r["prompt_id"], {})[r["arm"]] = r["decode_tok_s"]
                x = [p["ours"] / p["ours_ns"] for p in sp.values() if "ours" in p and "ours_ns" in p]
                if x:
                    print(f"| {sub} ({b}) | speed ours / `--no-spec` (paired) | {fmt(med(x), ci(x), suf='×')} |")
                    OUT.setdefault("broad_speed", {})[b] = (med(x), ci(x), len(x))
    print()


def multi():
    R = load("multi", "multi.jsonl")
    if not R:
        return
    print("## 7. `serve --multi 4`: aggregate tok/s and per-stream identity, new vs earlier build (interleaved rounds)\n")
    print("| box | round | new multi4 agg | prev multi4 agg | new / prev | new one-stream `--no-spec` | prev one-stream | new multi4 / one | streams identical to alone (new / prev) | new = prev texts | RssAnon peak MiB new / prev | min MemAvailable new / prev |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    by_class = {}
    for b in sorted({r["_box"] for r in R}):
        for rd in sorted({r["round"] for r in R if r["_box"] == b}):
            v = {r["arm"]: r for r in R if r["_box"] == b and r["round"] == rd}
            if not all(k in v for k in ("new_m4", "prev_m4", "new_one", "prev_one")):
                continue
            ratio = v["new_m4"]["agg_tok_s"] / v["prev_m4"]["agg_tok_s"]
            by_class.setdefault(CLASS.get(b, b), []).append(ratio)
            print(f"| {b} | {rd} | {v['new_m4']['agg_tok_s']:.2f} | {v['prev_m4']['agg_tok_s']:.2f} | {ratio:.3f}× | "
                  f"{v['new_one']['agg_tok_s']:.2f} | {v['prev_one']['agg_tok_s']:.2f} | {v['new_m4']['agg_tok_s']/v['new_one']['agg_tok_s']:.2f}× | "
                  f"{sum(v['new_m4']['identical_to_alone'])}/4 / {sum(v['prev_m4']['identical_to_alone'])}/4 | "
                  f"{sum(a == c for a, c in zip(v['new_m4']['texts'], v['prev_m4']['texts']))}/4 | "
                  f"{v['new_m4']['rss_anon_peak_kb']//1024} / {v['prev_m4']['rss_anon_peak_kb']//1024} | {v['new_m4']['mem_avail_min_kb']//1024} / {v['prev_m4']['mem_avail_min_kb']//1024} |")
    for cls, ratios in sorted(by_class.items()):
        print(f"\nnew / prev aggregate ({cls}): {flag('--multi 4 aggregate tok/s', cls, ratios)}\n")


def kld():
    for f in glob.glob(os.path.join(RAW, "*", "kld", "summary.txt")):
        print(f"## 8. Attention-path accuracy: KL divergence vs the fork's tiled f32 path (`GGML_ATTN=0 -ub 512`), 4 × 512 WikiText ({box_of(f)})\n")
        for l in open(f):
            print(f"- {l.strip()}")
        print()


def mcnemar(b, c):
    """exact two-sided McNemar p on discordant counts b (new right, prev wrong) and c (reverse)"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def quality():
    import re
    G = {json.loads(l)["id"]: json.loads(l)["gold"] for l in open(os.path.join(os.path.dirname(RAW) or ".", "gsm8k.jsonl"))}
    def ans(t):
        m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", t) or re.findall(r"-?\d[\d,]*\.?\d*", t)
        try:
            return float(m[-1].replace(",", "")) if m else None
        except ValueError:
            return None
    rows = []
    for f in glob.glob(os.path.join(RAW, "*", "q_gsm", "results.jsonl")):
        d = os.path.dirname(f)
        res = {}
        for r in map(json.loads, filter(str.strip, open(f))):
            if "error" in r:
                continue
            t = open(os.path.join(d, "outputs", r["arm"], f"{r['prompt_id']}.r0.txt")).read()
            a = ans(t)
            res.setdefault(r["prompt_id"], {})[r["arm"]] = (a is not None and abs(a - float(G[r["prompt_id"]])) < 1e-6, r["sha256"])
        rows.append(("GSM8K-40", res))
    for f in glob.glob(os.path.join(RAW, "*", "q_code", "graded.jsonl")):
        res = {}
        for r in map(json.loads, filter(str.strip, open(f))):
            res.setdefault(r["prompt_id"], {})[r["arm"]] = (bool(r["pass"]), None)
        rows.append(("HumanEval + MBPP subset, pass@1 (sanitized)", res))
    if not rows:
        return
    print("## 9. Task-quality gate: new default `serve` vs the earlier build's default, same box, interleaved\n")
    print("| task | n | new | prev | Δ points | new right / prev wrong | reverse | McNemar p | identical texts |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, res in rows:
        v = [x for x in res.values() if "new" in x and "prev" in x]
        if not v:
            continue
        n = len(v); a = sum(x["new"][0] for x in v); b_ = sum(x["prev"][0] for x in v)
        b = sum(x["new"][0] and not x["prev"][0] for x in v); c = sum(x["prev"][0] and not x["new"][0] for x in v)
        same = sum(x["new"][1] == x["prev"][1] for x in v) if v[0]["new"][1] else "–"
        p = mcnemar(b, c)
        OUT.setdefault("quality", {})[name] = {"n": n, "new": a, "prev": b_, "b": b, "c": c, "p": p}
        print(f"| {name} | {n} | {a} ({100*a/n:.1f} %) | {b_} ({100*b_/n:.1f} %) | {100*(a-b_)/n:+.1f} | {b} | {c} | {p:.3f} | {same} |")
    print()


def regressions():
    print("## 0. Regressions: every cell where the new tree is more than 2 % below the earlier build (median of paired ratios)\n")
    if not REG:
        print("None.\n")
        return
    print("| metric | where | new / prev (or prev / new for seconds) | 95 % CI | n |")
    print("|---|---|---|---|---|")
    for m, w, r, c, n in REG:
        print(f"| {m} | {w} | {r:.3f} | [{c[0]:.3f}, {c[1]:.3f}] | {n} |")
    print()


def ppl():
    for f in glob.glob(os.path.join(RAW, "*", "ppl", "summary.txt")):
        print(f"## 6. WikiText-2 PPL, 32 × 512 ({box_of(f)})\n")
        for l in open(f):
            print(f"- {l.strip()}")
        print()


if __name__ == "__main__":
    print("# Benchmark tables (generated by analyze.py)\n")
    for fn in (short, newprev, lb, ctx, replay, ident, ppl, multi, kld, quality, regressions):
        try:
            fn()
        except Exception as e:
            print(f"\n(**{fn.__name__} failed: {type(e).__name__}: {e}**)\n")
    json.dump({k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in OUT.items()},
              open(os.path.join(RAW, "..", "headline.json"), "w"), indent=1, default=str)
