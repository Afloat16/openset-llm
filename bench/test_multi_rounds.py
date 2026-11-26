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


class MultiRoundTest(unittest.TestCase):
    def run_rounds(self, rounds, count=12):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = [{"id": str(i), "messages": []} for i in range(count)]
            (root / "prompts24.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
            out = root / "results"
            old = os.getcwd()

            def request(port, prompt, result, index):
                reply = {"timings": {"predicted_n": 1, "predicted_per_second": 1},
                         "choices": [{"message": {"content": prompt["id"]}}]}
                result[index] = (reply, 0.1, time.time())

            try:
                os.chdir(root)
                with patch.object(multi, "ARMS", {"test_m4": {}}), \
                     patch.object(multi, "start", return_value=(1, Mock(pid=1), "unused")), \
                     patch.object(multi, "stop"), patch.object(multi, "post"), \
                     patch.object(multi, "req", side_effect=request), \
                     patch.object(multi, "Sampler", return_value=Mock(anon=1, avail=1)), \
                     patch("sys.argv", ["multi.py", "--out", str(out), "--rounds", str(rounds)]), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    multi.main()
                return [json.loads(line) for line in (out / "multi.jsonl").read_text().splitlines()]
            finally:
                os.chdir(old)

    def test_more_than_three_rounds_keep_four_streams(self):
        rows = self.run_rounds(4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(r["prompts"]) == 4 for r in rows))
        self.assertEqual(rows[3]["prompts"], rows[0]["prompts"])
        self.assertTrue(all(all(r["identical_to_alone"]) for r in rows))

    def test_nonpositive_round_count_is_rejected(self):
        with self.assertRaises(SystemExit) as error:
            self.run_rounds(0)
        self.assertEqual(error.exception.code, 2)

    def test_too_few_prompts_is_rejected_before_start(self):
        with self.assertRaises(SystemExit) as error:
            self.run_rounds(1, count=3)
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
