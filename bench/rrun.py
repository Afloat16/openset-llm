#!/usr/bin/env python3
"""Agent replay (no oom_score_adj). Replay a multi-request conversation against one fresh server and log prefill reuse + client-side TTFT per step.

Runs ON the box. Any server command (an "arm" from arms.json),
streamed requests (TTFT = time to the first generated token as the client sees it), the server's own timings,
peak RSS / min MemAvailable per step, and an optional server restart before re-sending step 0 (disk prefix test).

    python3 rrun.py --arms arms.json --arm rel_nospec --replay replay.json --ctx 16384 --gen 64 --out ~/res/agent
    python3 rrun.py ... --restart-step0     # after the replay: restart the server, send step 0 again (cold session)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

PORT = 8080
URL = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


def status_kb(pid, key):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(key):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


class Peak(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.rss, self.anon, self.minavail = pid, False, 0, 0, 1 << 30

    def run(self):
        while not self.stop:
            self.rss = max(self.rss, status_kb(self.pid, "VmRSS") // 1024)
            self.anon = max(self.anon, status_kb(self.pid, "RssAnon") // 1024)
            self.minavail = min(self.minavail, mem_available_mb())
            time.sleep(0.2)


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def stream_chat(body):
    """POST a streamed chat request; return (ttft_s, wall_s, text, tool_calls, timings)."""
    body = dict(body, stream=True)
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content, calls, timings = "", {}, {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "timings" in ev:
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                got = False
                if d.get("content"):
                    content += d["content"]
                    got = True
                if d.get("reasoning_content"):
                    got = True
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                    got = True
                if got and ttft is None:
                    ttft = time.time() - t0
    wall = time.time() - t0
    return ttft, wall, content, [calls[k] for k in sorted(calls)], timings


def start_server(arm, a, out, tag):
    cmd = arm["cmd"].replace("@H@", str(Path.home())).replace("@C@", str(a.ctx))
    env = dict(os.environ)
    for k, v in arm.get("env", {}).items():
        env[k] = v.replace("@H@", str(Path.home()))
    for _ in range(120):   # ik's server cannot bind while old connections on the port sit in TIME_WAIT
        if not subprocess.run(f"ss -tan | grep -q ':{PORT} '", shell=True).returncode == 0:
            break
        time.sleep(1)
    else:
        raise SystemExit(f"port {PORT} is still in use after 120 s; stop its owner or retry later")
    log = open(out / f"server_{a.arm}{tag}.log", "w")
    # nothing touches oom_score_adj (driver and server both at the default 0)
    srv = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        for _ in range(900):
            if srv.poll() is not None:
                raise SystemExit(f"server died at load, see {log.name}")
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"server not healthy after 900 s, see {log.name}")
    except BaseException:
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        srv.wait()
        raise
    finally:
        log.close()
    time.sleep(1)
    pid = server_pid(srv.pid)
    return srv, pid, cmd


def server_pid(shell_pid):
    """The llama-server process under the shell / serve wrapper (exec'd, so usually the same pid)."""
    try:
        out = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else shell_pid
    except Exception:
        return shell_pid


def stop_server(srv):
    try:
        os.killpg(srv.pid, signal.SIGTERM)
        srv.wait(timeout=60)
    except Exception:
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except Exception:
            pass
    subprocess.run("pkill -9 -f llama-server", shell=True)
    time.sleep(2)


def run_step(a, data, req, prev_tokens, pid, arm_name, extra):
    msgs = req["messages"]
    tools = data.get("tools") or None
    base = {"messages": msgs, "max_tokens": a.gen, "temperature": 0, "top_k": 1, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        base["tools"] = tools
    if a.tokens and not a.live:   # tokens rendered once by our server (rivals' template endpoints differ): same model, same tokenizer
        toks = a.tokens[req["step"]]
    else:
        tmpl = post("/apply-template", {k: base[k] for k in ("messages", "chat_template_kwargs") if k in base} |
                    ({"tools": tools} if tools else {}))["prompt"]
        toks = post("/tokenize", {"content": tmpl, "add_special": False, "parse_special": True})["tokens"]
    common = lcp(prev_tokens, toks)
    peak = Peak(pid)
    peak.start()
    try:
        ttft, wall, text, calls, timings = stream_chat(base)
        err = None
    except Exception as e:  # server died (OOM) or timed out
        ttft, wall, text, calls, timings, err = None, None, "", [], {}, repr(e)
    peak.stop = True
    peak.join()
    row = {"arm": arm_name, "step": req["step"], **extra, "n_prompt": len(toks), "lcp_prev": common,
           "ideal_new": len(toks) - common, "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
           "prompt_ms": timings.get("prompt_ms"), "predicted_n": timings.get("predicted_n"),
           "predicted_ms": timings.get("predicted_ms"), "draft_n": timings.get("draft_n"),
           "draft_n_accepted": timings.get("draft_n_accepted"), "ttft_s": None if ttft is None else round(ttft, 3),
           "wall_s": None if wall is None else round(wall, 2), "peak_rss_mb": peak.rss, "peak_anon_mb": peak.anon,
           "min_mem_available_mb": peak.minavail, "text": text, "tool_calls": calls, "error": err}
    return row, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--steps", type=int, default=0, help="only the first N requests (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--restart-step0", action="store_true")
    ap.add_argument("--live", action="store_true", help="chat replay: assistant turns = this run's own replies")
    ap.add_argument("--tokens", default="", help="json {step: [token ids]} (from --dump-tokens)")
    ap.add_argument("--dump-tokens", default="", help="render + tokenize every request, write json, exit")
    a = ap.parse_args()
    a.tokens = {int(k): v for k, v in json.load(open(a.tokens)).items()} if a.tokens else None
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    arm = json.load(open(a.arms))[a.arm]
    data = json.load(open(a.replay))
    name = a.arm + a.tag
    subprocess.run("pkill -9 -f llama-server", shell=True)
    time.sleep(1)
    srv, pid, cmd = start_server(arm, a, out, a.tag)
    meta = {"arm": a.arm, "tag": a.tag, "cmd": cmd, "env": arm.get("env", {}), "ctx": a.ctx, "gen": a.gen,
            "replay": a.replay, "t_start": time.strftime("%FT%TZ", time.gmtime()),
            "rss_after_load_mb": status_kb(pid, "VmRSS") // 1024, "anon_after_load_mb": status_kb(pid, "RssAnon") // 1024,
            "mem_available_after_load_mb": mem_available_mb()}
    if a.dump_tokens:
        dump = {}
        for req in data["requests"]:
            tools = data.get("tools") or None
            tmpl = post("/apply-template", {"messages": req["messages"], "chat_template_kwargs": {"enable_thinking": False}} |
                        ({"tools": tools} if tools else {}))["prompt"]
            dump[req["step"]] = post("/tokenize", {"content": tmpl, "add_special": False, "parse_special": True})["tokens"]
        Path(a.dump_tokens).write_text(json.dumps(dump))
        stop_server(srv)
        return
    rows_f = open(out / f"steps_{name}.jsonl", "w")
    prev = []
    reqs = data["requests"][: a.steps] if a.steps else data["requests"]
    replies = []
    for req in reqs:
        if a.live:   # closed loop: the k-th assistant message is what this server answered at step k
            req = dict(req, messages=[dict(m) for m in req["messages"]])
            k = 0
            for m in req["messages"]:
                if m["role"] == "assistant":
                    m["content"] = replies[k]
                    k += 1
        row, prev = run_step(a, data, req, prev, pid, name, {"session": 1})
        replies.append(row["text"])
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()
        print(json.dumps({k: v for k, v in row.items() if k not in ("text", "tool_calls")}), flush=True)
        if row["error"] or srv.poll() is not None:
            meta["died_at_step"] = req["step"]
            break
    meta["anon_end_mb"] = status_kb(pid, "RssAnon") // 1024
    meta["mem_available_end_mb"] = mem_available_mb()
    if a.restart_step0 and "died_at_step" not in meta:
        stop_server(srv)
        subprocess.run("sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null", shell=True)
        srv, pid, _ = start_server(arm, a, out, a.tag + "_restart")
        row, _ = run_step(a, data, data["requests"][0], [], pid, name, {"session": 2})
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()
        print(json.dumps({k: v for k, v in row.items() if k not in ("text", "tool_calls")}), flush=True)
    stop_server(srv)
    meta["t_end"] = time.strftime("%FT%TZ", time.gmtime())
    (out / f"meta_{name}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
