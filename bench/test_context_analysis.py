import contextlib
import io
import unittest
from unittest.mock import patch

import analyze


def row(mode, **values):
    return dict({"_box": "e2", "round": 0, "ctx": "8192", "ctx_n": 8192,
                 "arm": "ours", "mode": mode, "status": "ok"}, **values)


class ContextAnalysisTest(unittest.TestCase):
    def analyze_rows(self, rows):
        output = io.StringIO()
        with patch.object(analyze, "load", return_value=rows), \
             patch.object(analyze, "OUT", {}), patch.object(analyze, "REG", []), \
             contextlib.redirect_stdout(output):
            analyze.ctx()
            summary = analyze.OUT.copy()
        return output.getvalue(), summary

    def test_load_only_probe_is_reported_without_throughput(self):
        output, summary = self.analyze_rows([row("load")])
        self.assertIn("Load-only fit probes", output)
        self.assertIn("| e2 | 0 | 8k | ours | ok |", output)
        self.assertNotIn("ctx", summary)

    def test_load_rows_do_not_replace_fill_measurements(self):
        fill = row("fill", decode_tok_s=12.0, prefill_tok_s=30.0, sha256="reply", recall=[1] * 5)
        output, summary = self.analyze_rows([fill, row("load")])
        self.assertIn("Load-only fit probes", output)
        self.assertEqual(summary["ctx"][("Ice Lake", 8192, "ours")]["n"], 1)
        self.assertEqual(summary["ctx"][("Ice Lake", 8192, "ours")]["dec"], 12.0)

    def test_failed_load_is_not_counted_as_failed_generation(self):
        fill = row("fill", decode_tok_s=12.0, prefill_tok_s=30.0, sha256="reply", recall=[1] * 5)
        output, summary = self.analyze_rows([fill, row("load", status="died_at_load")])
        self.assertIn("| e2 | 0 | 8k | ours | died_at_load |", output)
        self.assertEqual(summary["ctx"][("Ice Lake", 8192, "ours")]["n"], 1)


if __name__ == "__main__":
    unittest.main()
