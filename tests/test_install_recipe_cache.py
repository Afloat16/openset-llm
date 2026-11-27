import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BuildRecipeCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prefix = self.root / 'installation'
        self.source = self.root / 'source'
        self.tools = self.root / 'tools'
        for path in (self.tools, self.source / 'patches', self.source / 'kernels', self.prefix / 'bin',
                     self.prefix / 'models', self.prefix / 'llama.cpp/.git'):
            path.mkdir(parents=True)
        shutil.copy(ROOT / 'install.sh', self.source / 'install.sh')
        (self.source / 'patches/0001.patch').write_text('patch bytes')
        (self.source / 'kernels/kernel.h').write_text('kernel bytes')
        (self.source / 'patches/apply.sh').write_text('printf recipe-v1 > "$1/recipe.txt"\n')
        (self.prefix / 'models/model.gguf').write_bytes(b'cached model')
        self.command('git', 'exit 0\n')
        self.command('df', "printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\nfixture 999999999 0 999999999 0%% /\\n'\n")
        self.command('du', "printf '0 fixture\\n'\n")
        self.command('cmake', '''printf '%s\\n' "$*" >> "$RECIPE_TEST_ROOT/configure.log"
if [ "$1" = --build ]; then
  printf 'build\\n' >> "$RECIPE_TEST_ROOT/builds"
  mkdir -p "$2/bin"
  for name in llama-server llama-bench llama-cli; do
    printf '#!/bin/sh\\nexit 0\\n# ' > "$2/bin/$name"
    cat "$OPENSET_LLM_HOME/llama.cpp/recipe.txt" >> "$2/bin/$name"
    chmod +x "$2/bin/$name"
  done
fi
''')
        self.env = dict(os.environ, OPENSET_LLM_HOME=str(self.prefix), RECIPE_TEST_ROOT=str(self.root),
                        PATH=str(self.tools) + os.pathsep + os.environ['PATH'])
        self.env.pop('OPENSET_LLM_BIN_URL', None)

    def command(self, name, source):
        path = self.tools / name
        path.write_text('#!/bin/sh\n' + source)
        path.chmod(0o755)

    def install(self):
        result = subprocess.run(['bash', str(self.source / 'install.sh'), '--no-mtp'], env=self.env,
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def build_count(self):
        return len((self.root / 'builds').read_text().splitlines())

    def test_changed_cmake_recipe_rebuilds_cached_engine(self):
        self.install()
        script = self.source / 'install.sh'
        script.write_text(script.read_text().replace('-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_BUILD_TYPE=RelWithDebInfo'))
        self.install()
        self.assertEqual(self.build_count(), 2)
        self.assertIn('-DCMAKE_BUILD_TYPE=RelWithDebInfo', (self.root / 'configure.log').read_text())

    def test_changed_patch_application_rebuilds_generated_source(self):
        self.install()
        (self.source / 'patches/apply.sh').write_text('printf recipe-v2 > "$1/recipe.txt"\n')
        self.install()
        self.assertEqual(self.build_count(), 2)
        self.assertIn('recipe-v2', (self.prefix / 'bin/llama-server').read_text())

    def test_unchanged_recipe_keeps_cached_engine(self):
        self.install()
        result = self.install()
        self.assertEqual(self.build_count(), 1)
        self.assertIn('skipping the build', result.stdout)


if __name__ == '__main__':
    unittest.main()
