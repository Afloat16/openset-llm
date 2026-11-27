import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class JobsFailureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'tools').mkdir()
        (self.root / 'results/q_code').mkdir(parents=True)
        # Redirect only the driver's fixed local paths; execute its actual phase/marker logic.
        source = Path(__file__).with_name('jobs.sh').read_text()
        source = source.replace('R=~/res', 'R="$JOB_TEST_ROOT/results"')
        for path in ('~/.openset-llm/bin/llama-bench', '~/.prev/bin/llama-bench'):
            source = source.replace(path, '"$JOB_TEST_ROOT/tools/llama-bench"')
        source = source.replace('~/.prev', '"$JOB_TEST_ROOT/prev"')
        (self.root / 'jobs.sh').write_text(source)
        self.command('python3', '''printf '%s\\n' "$1" >> "$JOB_TEST_ROOT/calls"
[ "$JOB_TEST_FAIL" = "$1" ] && exit 37
printf '{}\\n'
''')
        self.command('cmake', '[ "$JOB_TEST_FAIL" = cmake ] && exit 37\nexit 0\n')
        self.command('llama-bench', '''[ "$JOB_TEST_FAIL" = llama-bench ] && exit 37
printf '[{"avg_ts":1}]\\n'
''')

    def command(self, name, body):
        path = self.root / 'tools' / name
        path.write_text('#!/bin/sh\n' + body)
        path.chmod(0o755)

    def run_jobs(self, *phases, failure=''):
        env = dict(os.environ, JOB_TEST_ROOT=str(self.root), JOB_TEST_FAIL=failure,
                   L=str(Path(__file__).parent.resolve()),
                   PATH=str(self.root / 'tools') + os.pathsep + os.environ['PATH'])
        return subprocess.run(['bash', str(self.root / 'jobs.sh'), *phases], env=env,
                              capture_output=True, text=True, timeout=15)

    def assert_failed(self, result, tag):
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertFalse((self.root / 'results' / ('DONE.' + tag)).exists())
        self.assertNotIn('ALL_DONE', (self.root / 'results/jobs.log').read_text())

    def test_failed_simple_phase_is_not_marked_complete(self):
        self.assert_failed(self.run_jobs('short', '1', failure='bench2.py'), 'short_1')

    def test_failed_pipeline_is_not_marked_complete(self):
        self.assert_failed(self.run_jobs('lbprev', failure='llama-bench'), 'lbprev')

    def test_failed_replay_arm_is_not_marked_complete(self):
        self.assert_failed(self.run_jobs('replay', '0', 'ours,prism', failure='rrun.py'), 'replay_0')
        self.assertFalse((self.root / 'results/replay/steps_ours.done').exists())
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['rrun.py'])

    def test_quality_stops_before_grading_after_runner_failure(self):
        self.assert_failed(self.run_jobs('quality', failure='bench2.py'), 'quality')
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['bench2.py'])

    def test_failed_perplexity_build_stops_phase(self):
        self.assert_failed(self.run_jobs('ppl', failure='cmake'), 'ppl')
        self.assertFalse((self.root / 'results/ppl/summary.txt').exists())

    def test_successful_phase_is_skipped_on_retry(self):
        for _ in range(2):
            result = self.run_jobs('short', '1')
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'results/DONE.short_1').exists())
        self.assertEqual((self.root / 'calls').read_text().splitlines(), ['bench2.py'])


if __name__ == '__main__':
    unittest.main()
