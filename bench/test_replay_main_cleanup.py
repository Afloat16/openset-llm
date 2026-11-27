import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import rrun


class ReplayMainCleanupTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.processes = []
        self.addCleanup(self.cleanup)
        (self.root / 'arms.json').write_text(json.dumps({'fixture': {'cmd': 'fixture'}}))
        (self.root / 'replay.json').write_text(json.dumps({'requests': [
            {'step': 0, 'messages': [{'role': 'user', 'content': 'hi'}]}]}))
        self.argv = ['rrun.py', '--arms', str(self.root / 'arms.json'), '--arm', 'fixture',
                     '--replay', str(self.root / 'replay.json'), '--out', str(self.root / 'out')]

    def cleanup(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def start(self, *_):
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                                start_new_session=True)
        self.processes.append(proc)
        return proc, proc.pid, 'fixture'

    def run_main(self, *extra, post_error=None, step=None):
        step_patch = patch.object(rrun, 'run_step', side_effect=step) if step else patch.object(rrun, 'run_step', wraps=rrun.run_step)
        with patch.object(sys, 'argv', self.argv + list(extra)), \
             patch.object(rrun, 'start_server', side_effect=self.start), \
             patch.object(rrun, 'post', side_effect=post_error), \
             step_patch, \
             patch.object(rrun.time, 'sleep'):
            rrun.main()

    def assert_all_stopped(self):
        self.assertTrue(self.processes)
        self.assertTrue(all(p.poll() is not None for p in self.processes), 'replay left a server running')

    def test_dump_template_failure_stops_server(self):
        with self.assertRaisesRegex(RuntimeError, 'template unavailable'):
            self.run_main('--dump-tokens', str(self.root / 'tokens.json'),
                          post_error=RuntimeError('template unavailable'))
        self.assert_all_stopped()

    def test_replay_tokenization_failure_stops_server(self):
        with self.assertRaisesRegex(RuntimeError, 'tokenizer unavailable'):
            self.run_main(post_error=RuntimeError('tokenizer unavailable'))
        self.assert_all_stopped()

    def test_output_open_failure_stops_server(self):
        (self.root / 'out' / 'steps_fixture.jsonl').mkdir(parents=True)
        with self.assertRaises(IsADirectoryError):
            self.run_main()
        self.assert_all_stopped()

    def test_interrupt_after_restart_stops_replacement_server(self):
        row = {'text': 'done', 'error': None, 'tool_calls': []}
        with self.assertRaises(KeyboardInterrupt), patch.object(rrun.subprocess, 'run'):
            self.run_main('--restart-step0', step=[(row, []), KeyboardInterrupt()])
        self.assertEqual(len(self.processes), 2)
        self.assert_all_stopped()

    def test_success_writes_results_and_stops_server(self):
        row = {'text': 'done', 'error': None, 'tool_calls': []}
        self.run_main(step=[(row, [1, 2])])
        self.assert_all_stopped()
        self.assertEqual(json.loads((self.root / 'out' / 'steps_fixture.jsonl').read_text()), row)
        self.assertIn('t_end', json.loads((self.root / 'out' / 'meta_fixture.json').read_text()))


if __name__ == '__main__':
    unittest.main()
