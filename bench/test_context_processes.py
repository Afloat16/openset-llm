import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import ctxrun


def process_stopped(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        return True


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux process groups and /proc")
class ContextProcessTests(unittest.TestCase):
    def run_job(self, root, *, leader_exits=False, interrupt_health=False, interrupt_warmup=False):
        child_file = root / "child.pid"
        source = (
            "import os,signal,time\n"
            "read_end,write_end=os.pipe()\n"
            "pid=os.fork()\n"
            "if pid == 0:\n"
            " os.close(read_end)\n"
            " signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
            " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            " os.write(write_end,b'1')\n"
            " os.close(write_end)\n"
            " while True: time.sleep(1)\n"
            "os.close(write_end)\n"
            "os.read(read_end,1)\n"
            "os.close(read_end)\n"
            f"open({str(child_file)!r},'w').write(str(pid))\n"
            + ("os._exit(0)\n" if leader_exits else "while True: time.sleep(1)\n")
        )
        arms = root / "arms.json"
        prompts = root / "prompts.json"
        plan = root / "plan.jsonl"
        out = root / "rows.jsonl"
        arms.write_text(json.dumps({"test": {"cmd": shlex.join([sys.executable, "-c", source])}}))
        prompts.write_text(json.dumps({"prompts": {"small": {"ctx": 16, "messages": []}}, "keys": []}))
        plan.write_text(json.dumps({"round": 0, "ctx": "small", "arm": "test", "mode": "load"}) + "\n")
        processes, samplers = [], []
        real_popen, real_sampler = subprocess.Popen, ctxrun.Sampler

        def spawn(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            processes.append(proc)
            deadline = time.monotonic() + 5
            while not child_file.exists() and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            self.assertTrue(child_file.exists())
            if leader_exits:
                proc.wait(timeout=5)
            return proc

        def sample(pid):
            sampler = real_sampler(pid)
            samplers.append(sampler)
            return sampler

        health = SimpleNamespace(status=200)
        response = MagicMock()
        response.__enter__.return_value = health
        argv = ["ctxrun.py", "--arms", str(arms), "--prompts", str(prompts), "--plan", str(plan), "--out", str(out)]
        try:
            with (
                patch.object(sys, "argv", argv),
                patch.object(ctxrun, "wait_gone"),
                patch.object(ctxrun, "drop_caches"),
                patch.object(ctxrun.time, "sleep"),
                patch.object(ctxrun.subprocess, "Popen", side_effect=spawn),
                patch.object(ctxrun.subprocess, "run", return_value=SimpleNamespace(stdout="")),
                patch.object(ctxrun.urllib.request, "urlopen", side_effect=KeyboardInterrupt if interrupt_health else None,
                             return_value=response),
                patch.object(ctxrun, "post", side_effect=KeyboardInterrupt if interrupt_warmup else None,
                             return_value={}),
                patch.object(ctxrun, "Sampler", side_effect=sample),
            ):
                if interrupt_health or interrupt_warmup:
                    with self.assertRaises(KeyboardInterrupt):
                        ctxrun.main()
                else:
                    ctxrun.main()
            child = int(child_file.read_text())
            deadline = time.monotonic() + 1
            while not process_stopped(child) and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            self.assertIsNotNone(processes[0].poll(), "the owned server leader was left running")
            self.assertTrue(process_stopped(child), "an owned server child was left running")
            self.assertTrue(all(not sampler.is_alive() for sampler in samplers))
            if not interrupt_health and not interrupt_warmup:
                rows = [json.loads(line) for line in out.read_text().splitlines()]
                self.assertEqual(rows[0]["status"], "died_at_load" if leader_exits else "ok")
            else:
                self.assertFalse(out.exists(), "an interrupted job must not be recorded as completed")
        finally:
            for sampler in samplers:
                sampler.stop = True
                sampler.join(timeout=2)
            for proc in processes:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)

    def test_interrupt_during_health_check_cleans_owned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_job(Path(directory), interrupt_health=True)

    def test_dead_leader_still_cleans_its_children(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_job(Path(directory), leader_exits=True)

    def test_success_cleans_signal_resistant_children_and_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_job(Path(directory))

    def test_interrupt_during_warmup_stops_sampler_without_completed_row(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_job(Path(directory), interrupt_warmup=True)

    def test_cleanup_preserves_an_unrelated_process_group(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as directory:
                self.run_job(Path(directory))
            self.assertIsNone(unrelated.poll())
        finally:
            os.killpg(unrelated.pid, signal.SIGKILL)
            unrelated.wait(timeout=5)

    def test_busy_external_server_is_not_killed(self):
        with patch.object(ctxrun.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with patch.object(ctxrun.time, "sleep"):
                with self.assertRaisesRegex(SystemExit, "still running"):
                    ctxrun.wait_gone()
        self.assertTrue(all(call.args[0][0] == "pgrep" for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
