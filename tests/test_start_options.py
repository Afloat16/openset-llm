import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StartBindOptions(unittest.TestCase):
    def run_start(self, *options):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app, tools, prefix = root / 'app', root / 'tools', root / 'prefix'
            for p in (app, tools, prefix / 'bin', prefix / 'models'):
                p.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'start', app / 'start')
            def executable(p, content):
                p.write_text(content)
                p.chmod(0o755)
            for name in ('dirname', 'bash', 'seq'):
                (tools / name).symlink_to(shutil.which(name))
            executable(prefix / 'bin/llama-server', '#!/bin/sh\nexit 0\n')
            (prefix / 'models/model.gguf').touch()
            executable(app / 'serve', '#!/bin/sh\n/bin/sleep 0.2\n')
            executable(tools / 'sleep', '#!/bin/sh\nexit 0\n')
            executable(tools / 'curl', '#!/bin/sh\nprintf "%s\\n" "$@" >> "$PROBE_LOG"\nexit 0\n')
            env = {**os.environ, 'OPENSET_LLM_HOME': str(prefix), 'PATH': str(tools),
                   'PROBE_LOG': str(root / 'probes'), 'DISPLAY': '', 'WAYLAND_DISPLAY': ''}
            result = subprocess.run(['bash', str(app / 'start'), *options], env=env,
                                    text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = root / 'probes'
            return log.read_text() if log.exists() else '', result.stdout

    def test_explicit_loopback_bind_is_used_for_readiness(self):
        log, output = self.run_start('--host', '127.0.0.2', '--port', '9000')
        self.assertIn('http://127.0.0.2:9000/health', log)
        self.assertIn('Open http://127.0.0.2:9000', output)

    def test_ipv6_wildcard_uses_bracketed_loopback(self):
        log, _ = self.run_start('--host', '::', '--port', '9000')
        self.assertIn('http://[::1]:9000/health', log)

    def test_api_only_mode_does_not_open_or_probe_the_chat_page(self):
        log, output = self.run_start('--no-webui')
        self.assertEqual(log, '')
        self.assertNotIn('Ready:', output)


if __name__ == '__main__':
    unittest.main()
