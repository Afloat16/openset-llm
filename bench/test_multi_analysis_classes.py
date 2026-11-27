import contextlib
import io
import unittest
from unittest.mock import patch

import analyze


def round_rows(box, ratio, rd=0):
    return [{'_box': box, 'round': rd, 'arm': arm,
             'agg_tok_s': ratio if arm == 'new_m4' else 1.,
             'identical_to_alone': [True] * 4, 'texts': ['answer'] * 4,
             'rss_anon_peak_kb': 1024, 'mem_avail_min_kb': 1024}
            for arm in ('new_m4', 'prev_m4', 'new_one', 'prev_one')]


class MultiAnalysisClassesTest(unittest.TestCase):
    def analyze(self, rows):
        output = io.StringIO()
        with patch.object(analyze, 'load', return_value=rows), \
             patch.object(analyze, 'REG', []), contextlib.redirect_stdout(output):
            analyze.multi()
            regressions = list(analyze.REG)
        return output.getvalue(), regressions

    def test_other_cpu_cannot_hide_regression(self):
        output, regressions = self.analyze(round_rows('e2', 2.) + round_rows('d1', .5))
        self.assertIn('aggregate (Ice Lake): 2.000× (n=1)', output)
        self.assertIn('aggregate (Zen 4): 0.500× (n=1)', output)
        self.assertEqual([(row[1], row[2], row[4]) for row in regressions], [('Zen 4', .5, 1)])

    def test_same_class_rounds_still_pool(self):
        output, regressions = self.analyze(round_rows('e2', 1.2) + round_rows('e3', 1.4))
        self.assertIn('1.300×', output)
        self.assertIn('(n=2)', output)
        self.assertEqual(regressions, [])

    def test_unknown_boxes_remain_separate(self):
        output, regressions = self.analyze(round_rows('custom-a', .6) + round_rows('custom-b', 1.4))
        self.assertIn('aggregate (custom-a): 0.600× (n=1)', output)
        self.assertIn('aggregate (custom-b): 1.400× (n=1)', output)
        self.assertEqual([(row[1], row[2]) for row in regressions], [('custom-a', .6)])

    def test_incomplete_round_does_not_enter_summary(self):
        output, regressions = self.analyze(round_rows('e2', 1.2) + round_rows('d1', .1)[:-1])
        self.assertIn('1.200× (n=1)', output)
        self.assertNotIn('aggregate (Zen 4)', output)
        self.assertEqual(regressions, [])


if __name__ == '__main__':
    unittest.main()
