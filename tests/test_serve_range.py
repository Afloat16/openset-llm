import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServeIntegerRangeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name)
        (self.prefix / 'bin').mkdir()
        (self.prefix / 'models').mkdir()
        (self.prefix / 'models/model.gguf').touch()
        server = self.prefix / 'bin/llama-server'
        server.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        server.chmod(0o755)

    def serve(self, *args):
        return subprocess.run(['bash', str(ROOT / 'serve'), '--no-spec', '--no-resume', *args],
                              env={**os.environ, 'OPENSET_LLM_HOME': str(self.prefix)},
                              capture_output=True, text=True, timeout=10)

    def test_unrepresentable_numbers_fail_before_server_execution(self):
        for option in ('--ctx', '--multi', '--threads', '--port', '--spec-depth'):
            for value in ('2147483648', '9223372036854775808', '18446744073709551616'):
                with self.subTest(option=option, value=value):
                    result = self.serve(option, value)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('2147483647', result.stderr)
                    self.assertEqual(result.stdout, '')

    def test_total_context_cannot_overflow_server_integer(self):
        result = self.serve('--ctx', '1073741824', '--multi', '2')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('--ctx', result.stderr)
        self.assertIn('--multi', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_decimal_and_representable_boundary_remain_intact(self):
        for context, streams, expected in [('0008', '02', '16'), ('536870911', '4', '2147483644')]:
            with self.subTest(context=context):
                result = self.serve('--ctx', context, '--multi', streams)
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = json.loads(result.stdout)
                self.assertEqual(argv[argv.index('-c') + 1], expected)

    def test_zero_context_keeps_the_engine_default(self):
        result = self.serve('--ctx', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(result.stdout)
        self.assertEqual(argv[argv.index('-c') + 1], '0')


if __name__ == '__main__':
    unittest.main()
