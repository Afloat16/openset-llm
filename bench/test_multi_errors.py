import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import multi


class MultiRequestErrorTest(unittest.TestCase):
    def run_round(self, failure):
        sampler = Mock(halt=False, anon=1, avail=1)
        proc = Mock(pid=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = [{"id": str(i), "messages": []} for i in range(4)]
            (root / "prompts24.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
            out = root / "results"
            old = os.getcwd()

            def request(port, prompt, result, index):
                if failure and index == 2:
                    raise RuntimeError("request for stream 2 failed")
                reply = {"timings": {"predicted_n": 1, "predicted_per_second": 1},
                         "choices": [{"message": {"content": prompt["id"]}}]}
                result[index] = (reply, 0.1, time.time())

            try:
                os.chdir(root)
                with patch.object(multi, "ARMS", {"test_m4": {}}), \
                     patch.object(multi, "start", return_value=(1, proc, "unused")), \
                     patch.object(multi, "stop") as stop, patch.object(multi, "post"), \
                     patch.object(multi, "req", side_effect=request), \
                     patch.object(multi, "Sampler", return_value=sampler), \
                     patch("threading.excepthook"), \
                     patch("sys.argv", ["multi.py", "--out", str(out), "--rounds", "1"]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "stream 2 failed"):
                            multi.main()
                    else:
                        multi.main()
                    stop.assert_called_once_with(proc)
                self.assertTrue(sampler.halt)
                sampler.join.assert_called_once()
                rows = (out / "multi.jsonl").read_text().splitlines()
                self.assertEqual(len(rows), 0 if failure else 1)
            finally:
                os.chdir(old)

    def test_worker_failure_reaches_caller_and_cleans_up(self):
        self.run_round(True)

    def test_successful_round_also_joins_sampler(self):
        self.run_round(False)


if __name__ == "__main__":
    unittest.main()
