import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import bench2


class SamplerCleanupTest(unittest.TestCase):
    def run_benchmark(self, fail):
        samplers = []
        sampler_type = bench2.Sampler

        def sample(pid):
            sampler = sampler_type(pid)
            samplers.append(sampler)
            return sampler

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "arms.json").write_text(json.dumps({"test": {"cmd": "unused"}}))
            (root / "prompts.jsonl").write_text(json.dumps({"id": "one", "task": "test", "messages": []}) + "\n")
            argv = ["bench2.py", "--arms", str(root / "arms.json"), "--prompts",
                    str(root / "prompts.jsonl"), "--out", str(root / "results.jsonl")]
            proc = Mock(pid=os.getpid())
            proc.poll.return_value = None
            reply = {"choices": [{"message": {"content": "ok"}}], "timings": {}}
            try:
                with patch.object(bench2.sys, "argv", argv), \
                     patch.object(bench2, "start", return_value=(1, proc, "unused")), \
                     patch.object(bench2, "stop"), patch.object(bench2, "post"), \
                     patch.object(bench2, "tokenize", return_value=[]), \
                     patch.object(bench2, "majfaults", return_value=0), \
                     patch.object(bench2, "Sampler", side_effect=sample), \
                     patch.object(bench2, "request", side_effect=RuntimeError("request failed") if fail else None,
                                  return_value=(reply, "ok", 0.1)), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    bench2.main()
                self.assertEqual(len(samplers), 1)
                self.assertFalse(samplers[0].is_alive(), "request sampler must stop before the next request")
                row = json.loads((root / "results.jsonl").read_text())
                self.assertEqual("error" in row, fail)
            finally:
                for sampler in samplers:
                    sampler.halt = True
                    sampler.join(timeout=2)

    def test_failed_request_stops_sampler(self):
        self.run_benchmark(True)

    def test_successful_request_joins_sampler(self):
        self.run_benchmark(False)


if __name__ == "__main__":
    unittest.main()
