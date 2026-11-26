import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DecimalServeOptions(unittest.TestCase):
    def test_zero_padded_context_and_streams_use_decimal_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            (prefix / 'bin').mkdir()
            (prefix / 'models').mkdir()
            (prefix / 'models/model.gguf').touch()
            server = prefix / 'bin/llama-server'
            server.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            server.chmod(0o755)
            result = subprocess.run(['bash', str(ROOT / 'serve'), '--no-spec', '--no-resume',
                '--ctx', '0008', '--multi', '02', '--threads', '0002', '--port', '08080'],
                env={**os.environ, 'OPENSET_LLM_HOME': tmp}, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index('-c') + 1], '16')
            self.assertEqual(args[args.index('-np') + 1], '2')
            self.assertEqual(args[args.index('-t') + 1], '2')
            self.assertEqual(args[args.index('--port') + 1], '8080')


if __name__ == '__main__':
    unittest.main()
