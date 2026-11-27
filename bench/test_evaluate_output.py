import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluate import run


class EvaluationOutputTest(unittest.TestCase):
    def test_binary_stdout_does_not_change_a_passing_exit_status(self):
        self.assertEqual(run("import os\nos.write(1, b'\\xff')\nassert 2 + 2 == 4"), (True, ""))

    def test_binary_stderr_keeps_a_readable_failure(self):
        self.assertEqual(run("import os, sys\nos.write(2, b'error: \\xff\\n')\nsys.exit(7)"),
                         (False, "error: \ufffd"))

    def test_binary_stderr_does_not_change_a_passing_exit_status(self):
        self.assertEqual(run("import os\nos.write(2, b'\\xff')"), (True, ""))

    def test_text_diagnostics_and_empty_stderr_keep_their_meaning(self):
        self.assertEqual(run("raise ValueError('caf\u00e9')"), (False, "ValueError: caf\u00e9"))
        self.assertEqual(run("import sys\nsys.exit(7)"), (False, "rc 7"))

    def test_grading_continues_after_a_reply_writes_binary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problems = root / "problems.jsonl"
            problems.write_text("\n".join(json.dumps({
                "id": pid, "entry_point": "answer", "src_prompt": "def answer():\n",
                "test": "assert answer() == 42",
            }) for pid in ("first", "second")))
            outputs = root / "outputs" / "arm"
            outputs.mkdir(parents=True)
            (outputs / "first.r0.txt").write_text(
                "import os\ndef answer():\n    os.write(1, b'\\xff')\n    return 42\n")
            (outputs / "second.r0.txt").write_text("def answer():\n    return 42\n")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate.py")),
                                     str(problems), str(root)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([row["prompt_id"] for row in rows], ["first", "second"])
            for row in rows:
                self.assertTrue(row["pass"])
                self.assertTrue(row["pass_raw"])
                self.assertEqual(row["err"], "")
                self.assertEqual(row["err_raw"], "")


if __name__ == "__main__":
    unittest.main()
