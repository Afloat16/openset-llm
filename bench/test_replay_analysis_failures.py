import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze


def row(step, text, error=None):
    return {'step': step, 'session': 1, 'n_prompt': 8, 'prompt_n': None if error else 8,
            'wall_s': None if error else 1., 'text': text, 'tool_calls': [], 'error': error}


class ReplayAnalysisFailureTest(unittest.TestCase):
    def analyze(self, ours, no_resume):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'e2/replay'
            output.mkdir(parents=True)
            for name, rows in [('ours', ours), ('ours_nr', no_resume)]:
                (output / f'steps_{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            stdout = io.StringIO()
            with patch.object(analyze, 'RAW', directory), patch.object(analyze, 'OUT', {}), \
                 contextlib.redirect_stdout(stdout):
                analyze.replay()
                headline = dict(analyze.OUT)
            return stdout.getvalue(), headline

    def test_two_failed_requests_are_not_identical_answers(self):
        failed = row(1, '', 'connection lost')
        text, headline = self.analyze([row(0, 'A'), failed], [row(0, 'B'), failed])
        self.assertEqual(headline['ident']['replay e2 ours vs ours_nr'], (0, 1))
        self.assertIn('**0/1** requests byte-identical', text)
        self.assertIn('died', text)

    def test_all_failed_run_has_no_identity_claim_and_does_not_crash(self):
        failed = row(0, '', 'connection lost')
        text, headline = self.analyze([failed], [failed])
        self.assertIn('no successful replay steps', text)
        self.assertNotIn('requests byte-identical', text)
        self.assertNotIn('ident', headline)

    def test_successful_pairs_keep_original_identity_and_timing(self):
        text, headline = self.analyze([row(0, 'same'), row(1, 'A')],
                                      [row(0, 'same'), row(1, 'B')])
        self.assertEqual(headline['ident']['replay e2 ours vs ours_nr'], (1, 2))
        self.assertEqual(headline['replay']['e2']['ours_nr']['steps'], 2)
        self.assertIn('over all 2 steps', text)

    def test_rounded_zero_duration_still_has_a_valid_answer(self):
        completed = dict(row(0, 'same'), wall_s=0.)
        text, headline = self.analyze([completed], [completed])
        self.assertEqual(headline['ident']['replay e2 ours vs ours_nr'], (1, 1))
        self.assertIn('over all 1 steps', text)


if __name__ == '__main__':
    unittest.main()
