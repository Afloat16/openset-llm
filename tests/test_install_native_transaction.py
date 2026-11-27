import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeInstallTransactionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prefix = self.root / 'installation'
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        (self.prefix / 'bin').mkdir(parents=True)
        (self.prefix / 'models').mkdir()
        (self.prefix / 'llama.cpp/.git').mkdir(parents=True)
        (self.prefix / 'models/model.gguf').write_bytes(b'cached model')
        self.previous = b'#!/bin/sh\nexit 0\n# working engine\n'
        self.command(self.prefix / 'bin/llama-server', self.previous.decode())
        (self.prefix / 'bin/.stamp').write_text('previous build')
        for name in ('dirname', 'uname', 'grep', 'cut', 'sed', 'awk', 'cat', 'mkdir', 'id',
                     'mv', 'rm', 'mktemp', 'head', 'nproc', 'cksum', 'chmod', 'ninja'):
            executable = shutil.which(name)
            if executable:
                (self.tools / name).symlink_to(executable)
        self.command(self.tools / 'df', "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\\nfixture 999999999 0 999999999 0%% /\\n'\n")
        self.command(self.tools / 'du', "#!/bin/sh\nprintf '0 fixture\\n'\n")
        for name in ('curl', 'python3', 'git', 'c++', 'sh'):
            self.command(self.tools / name, '#!/bin/sh\nexit 0\n')
        self.command(self.tools / 'cmake', '''#!/bin/sh
if [ "$1" = --build ]; then
  mkdir -p "$2/bin"
  printf '#!/bin/sh\\nexit %s\\n# replacement engine\\n' "$ENGINE_EXIT" > "$2/bin/llama-server"
  printf '#!/bin/sh\\nexit 0\\n' > "$2/bin/llama-bench"
  printf '#!/bin/sh\\nexit 0\\n' > "$2/bin/llama-cli"
  chmod +x "$2/bin/llama-server" "$2/bin/llama-bench" "$2/bin/llama-cli"
fi
''')
        self.command(self.tools / 'cp', '''#!/bin/sh
for arg in "$@"; do destination=$arg; done
if [ "$COPY_FAIL" = 1 ]; then
  printf truncated > "$destination/llama-server"
  exit 37
fi
exec "''' + shutil.which('cp') + '''" "$@"
''')
        self.env = dict(os.environ, OPENSET_LLM_HOME=str(self.prefix), PATH=str(self.tools),
                        ENGINE_EXIT='0', COPY_FAIL='0')
        self.env.pop('OPENSET_LLM_BIN_URL', None)

    def command(self, path, source):
        path.write_text(source)
        path.chmod(0o755)

    def install(self):
        return subprocess.run(['/bin/bash', str(ROOT / 'install.sh'), '--no-mtp'], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def assert_previous_engine(self):
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), self.previous)
        self.assertEqual((self.prefix / 'bin/.stamp').read_text(), 'previous build')
        self.assertEqual(list(self.prefix.glob('.bin.*')), [])

    def test_partial_copy_failure_preserves_working_engine(self):
        self.env['COPY_FAIL'] = '1'
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assert_previous_engine()

    def test_unusable_built_server_preserves_working_engine(self):
        self.env['ENGINE_EXIT'] = '11'
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assert_previous_engine()

    def test_success_replaces_engine_and_stamp_together(self):
        (self.prefix / 'bin/obsolete.so').write_bytes(b'old backend')
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'replacement engine', (self.prefix / 'bin/llama-server').read_bytes())
        self.assertNotEqual((self.prefix / 'bin/.stamp').read_text(), 'previous build')
        self.assertFalse((self.prefix / 'bin/obsolete.so').exists())
        self.assertEqual(list(self.prefix.glob('.bin.*')), [])


if __name__ == '__main__':
    unittest.main()
