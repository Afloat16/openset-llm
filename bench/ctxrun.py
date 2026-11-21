#!/usr/bin/env python3
"""Long-context runner (no oom_score_adj write): one llama-server per job, a real long prompt, 128 greedy tokens, memory sampled throughout.

    python3 ctxrun.py --arms arms.json --prompts prompts.json --plan plan.jsonl --out ~/res/rows.jsonl

arms.json: {"arm": {"env": {...}, "cmd": "... -c @C@ ..."}}  (@C@ = context size; --host/--port are appended)
plan.jsonl: one job per line {"round": r, "ctx": "16384", "arm": "ours", "mode": "fill" | "load"}, run in file order.
  fill = start, warm up ("Say hi."), snapshot, send the haystack prompt, snapshot after the reply, stop.
  load = start, warm up, snapshot, stop (cheap fit probe: KV/state/compute are allocated at load).
Resumable: (round, ctx, arm, mode) already in --out are skipped.
Per row: llama-server timings (prompt n/ms/tok-s, predicted n/ms/tok-s, drafts), wall, recall score, sha256 of the
reply, and memory: process RssAnon/RssFile/VmHWM at load / idle / after, the sampled peak RssAnon, min MemAvailable,
cgroup memory.current/peak/anon/file/shmem (in a container the 16 GB limit is the cgroup's; MemAvailable is the host's),
major faults of the server during the request, and whether the server survived (exit code / signal if not).
"""
import argparse, hashlib, json, os, random, shlex, signal, socket, subprocess, sys, threading, time, urllib.request

HOST = socket.gethostname()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kv(path, keys=None):
    out = {}
    try:
        for l in open(path):
            k, _, v = l.partition(":") if ":" in l else l.partition(" ")
            k = k.strip()
            if keys is None or k in keys:
                try:
                    out[k] = int(v.split()[0])
                except (ValueError, IndexError):
                    pass
    except OSError:
        pass
    return out


def cg():
    d = {}
    for f in ("memory.current", "memory.peak", "memory.max"):
        try:
            v = open(f"/sys/fs/cgroup/{f}").read().strip()
            d[f.split(".")[1]] = int(v) // 1024 if v.isdigit() else v
        except OSError:
            pass
    st = kv("/sys/fs/cgroup/memory.stat", {"anon", "file", "shmem", "file_mapped"})
    d.update({k: v // 1024 for k, v in st.items()})
    return d   # kB


def snap(pid):
    s = kv(f"/proc/{pid}/status", {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM"})
    m = kv("/proc/meminfo", {"MemAvailable", "MemFree", "Cached", "SwapTotal", "SwapFree"})
    return {"proc": s, "meminfo": m, "cgroup": cg(), "majflt": majflt(pid)}


def majflt(pid):
    try:
        return int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[9])
    except (OSError, IndexError, ValueError):
        return None


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.peak_anon, self.min_avail, self.peak_cg = pid, False, 0, 1 << 62, 0

    def run(self):
        while not self.stop:
            a = kv(f"/proc/{self.pid}/status", {"RssAnon"}).get("RssAnon", 0)
            self.peak_anon = max(self.peak_anon, a)
            self.min_avail = min(self.min_avail, kv("/proc/meminfo", {"MemAvailable"}).get("MemAvailable", 1 << 62))
            c = cg().get("current")
            if isinstance(c, int):
                self.peak_cg = max(self.peak_cg, c)
            time.sleep(0.25)


def wait_gone():
    for _ in range(120):
        if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode != 0:
            return
        time.sleep(1)
    subprocess.run(["pkill", "-9", "-x", "llama-server"])
    time.sleep(3)


def drop_caches():
    cmd = "sync; echo 3 > /proc/sys/vm/drop_caches"
    if os.geteuid() != 0:
        cmd = "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"
    subprocess.run(["bash", "-c", cmd], capture_output=True)


def grade(text, keys):
    t = text.lower()
    return [int(any(k.lower() in t for k in ks)) for ks in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=128)
    a = ap.parse_args()
    arms = json.load(open(a.arms))
    P = json.load(open(a.prompts))
    plan = [json.loads(l) for l in open(a.plan) if l.strip()]
    outdir = os.path.dirname(os.path.abspath(a.out))
    for d in ("logs", "outputs"):
        os.makedirs(os.path.join(outdir, d), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["round"], r["ctx"], r["arm"], r["mode"]))
    print(f"# {len(plan)} jobs, {len(done)} done", file=sys.stderr, flush=True)
    failed = {}   # jobs with "stop_on_fail": an arm that failed a fill at C skips later fills at >= C
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r["mode"] == "fill" and r["status"] != "ok":
                failed[r["arm"]] = min(failed.get(r["arm"], 1 << 30), r["ctx_n"])
    for j in plan:
        key = (j["round"], j["ctx"], j["arm"], j["mode"])
        if key in done:
            continue
        if j.get("stop_on_fail") and j["mode"] == "fill" and P["prompts"][j["ctx"]]["ctx"] >= failed.get(j["arm"], 1 << 30):
            print(f"skip {key}: {j['arm']} already failed at {failed[j['arm']]}", file=sys.stderr, flush=True)
            continue
        p = P["prompts"][j["ctx"]]
        C = p["ctx"]
        spec = arms[j["arm"]]
        wait_gone()
        drop_caches()
        time.sleep(2)
        tag = f"r{j['round']}.{j['ctx']}.{j['arm']}.{j['mode']}"
        log = os.path.join(outdir, "logs", tag + ".log")
        port = free_port()
        env = dict(os.environ, **{k: v.replace("@H@", os.path.expanduser("~")) for k, v in spec.get("env", {}).items()})
        cmd = shlex.split(spec["cmd"].replace("@C@", str(C)).replace("@H@", os.path.expanduser("~"))) + ["--host", "127.0.0.1", "--port", str(port)]
        row = {"round": j["round"], "ctx": j["ctx"], "ctx_n": C, "arm": j["arm"], "mode": j["mode"], "host": HOST,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cmd": " ".join(cmd), "env": spec.get("env", {}),
               "start_meminfo": kv("/proc/meminfo", {"MemAvailable"}), "start_cgroup": cg()}
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
        # the `serve` script execs llama-server, so proc.pid is the server's pid
        # no oom_score_adj anywhere (driver or server); the kernel's own choice under OOM is the 7 GB server
        status = "ok"
        healthy = False
        while time.time() - t0 < 2400:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(1)
        row["load_s"] = round(time.time() - t0, 1)
        samp = None
        try:
            if not healthy:
                status = "died_at_load" if proc.poll() is not None else "not_healthy"
                raise RuntimeError(status)
            row["snap_load"] = snap(proc.pid)
            samp = Sampler(proc.pid)
            samp.start()
            post(port, {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False}}, 2400)
            row["warm_s"] = round(time.time() - t0 - row["load_s"], 1)
            row["snap_idle"] = snap(proc.pid)
            if j["mode"] == "fill":
                f0 = majflt(proc.pid)
                t1 = time.time()
                r = post(port, {"messages": p["messages"], "max_tokens": a.gen, "temperature": 0, "seed": 0,
                                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}, 6 * 3600)
                row["wall_s"] = round(time.time() - t1, 2)
                text = r["choices"][0]["message"].get("content") or ""
                t = r.get("timings", {})
                row.update({"prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                            "prefill_tok_s": t.get("prompt_per_second"), "predicted_n": t.get("predicted_n"),
                            "predicted_ms": t.get("predicted_ms"), "decode_tok_s": t.get("predicted_per_second"),
                            "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
                            "finish": r["choices"][0].get("finish_reason"), "timings": t,
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "recall": grade(text, P["keys"]), "text_head": text[:400]})
                row["snap_after"] = snap(proc.pid)
                f1 = row["snap_after"]["majflt"]
                row["majflt_request"] = (f1 - f0) if (f0 is not None and f1 is not None) else None
                open(os.path.join(outdir, "outputs", tag + ".txt"), "w").write(text)
        except Exception as e:
            if status == "ok":
                status = "died" if proc.poll() is not None else f"error: {type(e).__name__}: {e}"[:300]
        finally:
            if samp:
                samp.stop = True
                row.update({"peak_anon_kB": samp.peak_anon, "min_avail_kB": samp.min_avail, "peak_cgroup_kB": samp.peak_cg})
            rc = proc.poll()
            if rc is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            else:
                row["exit_code"] = rc
            row["status"] = status
            try:
                k = subprocess.run(["bash", "-c", "dmesg 2>/dev/null | grep -i -E 'killed process|out of memory' | tail -3 || sudo -n dmesg 2>/dev/null | grep -i -E 'killed process|out of memory' | tail -3"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
                row["dmesg_oom"] = k[-600:]
            except Exception:
                pass
            if status != "ok":
                row["log_tail"] = open(log, errors="replace").read()[-2500:]
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            if j["mode"] == "fill" and status != "ok":
                failed[j["arm"]] = min(failed.get(j["arm"], 1 << 30), C)
            msg = f"{tag:40s} {status:12s}"
            if row.get("decode_tok_s"):
                msg += (f" pp {row['prefill_tok_s']:.1f} tok/s ({row['prompt_n']} tok, {row['prompt_ms']/1000:.0f} s)"
                        f" tg {row['decode_tok_s']:.2f} recall {sum(row['recall'])}/5"
                        f" anon {row['snap_after']['proc'].get('RssAnon', 0)//1024} MiB peak {row.get('peak_anon_kB', 0)//1024}"
                        f" minavail {row.get('min_avail_kB', 0)//1024} cg {row.get('peak_cgroup_kB', 0)//1024}")
            elif row.get("snap_idle"):
                msg += f" idle anon {row['snap_idle']['proc'].get('RssAnon', 0)//1024} MiB HWM {row['snap_idle']['proc'].get('VmHWM', 0)//1024}"
            print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)
    print("# CTXRUN_DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
