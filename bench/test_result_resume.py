import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench2
import ctxrun


class ResultResumeTest(unittest.TestCase):
    def check_resume(self, runner, tail, terminated=True, invalid=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if runner is bench2:
                row = {"prompt_id": "one", "arm": "test", "rep": 0}
                (root / "arms.json").write_text('{"test": {"cmd": "unused"}}')
                (root / "prompts.json").write_text('{"id":"one","task":"test","messages":[]}\n')
            else:
                row = {"round": 0, "ctx": "8192", "ctx_n": 8192, "arm": "test", "mode": "fill", "status": "ok"}
                (root / "arms.json").write_text('{}')
                (root / "prompts.json").write_text('{"prompts": {}}')
                (root / "plan.jsonl").write_text(json.dumps(row) + "\n")
            path = root / "results.jsonl"
            complete = json.dumps(row).encode()
            original = complete + (b"\n" if terminated else b"") + tail
            path.write_bytes(original)
            argv = ["runner.py", "--arms", str(root / "arms.json"), "--prompts", str(root / "prompts.json"),
                    "--out", str(path)]
            if runner is ctxrun:
                argv += ["--plan", str(root / "plan.jsonl")]
            with patch("sys.argv", argv), patch.object(runner.subprocess, "Popen") as spawn, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if invalid:
                    with self.assertRaises(json.JSONDecodeError):
                        runner.main()
                else:
                    runner.main()
            spawn.assert_not_called()
            if invalid:
                self.assertEqual(path.read_bytes(), original)
            else:
                self.assertEqual(path.read_bytes(), complete + b"\n")
                with path.open("ab") as stream:
                    stream.write(b'{"next":true}\n')
                self.assertEqual(len([json.loads(line) for line in path.read_bytes().splitlines()]), 2)

    def test_server_benchmark_recovers_truncated_tail(self):
        self.check_resume(bench2, b'{"prompt_id":')

    def test_context_runner_recovers_truncated_tail(self):
        self.check_resume(ctxrun, b'{"round":')

    def test_truncated_utf8_tail_is_recoverable(self):
        self.check_resume(bench2, b'{"text":"\xe4\xb8')

    def test_complete_row_without_newline_is_preserved(self):
        self.check_resume(bench2, b"", terminated=False)

    def test_corrupt_complete_record_is_not_discarded(self):
        self.check_resume(bench2, b'{broken}\n', invalid=True)


if __name__ == "__main__":
    unittest.main()
