from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResumeModelIdentity(unittest.TestCase):
    def test_weight_replacement_invalidates_the_environment_key(self):
        compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('needs a C++ compiler')
        patch = (ROOT / 'patches/0007-resume.patch').read_text()
        additions = '\n'.join(line[1:] for line in patch.splitlines()
                              if line.startswith('+') and not line.startswith('+++'))
        start = additions.index('static uint64_t rsm_fnv(')
        end = additions.index('static std::string rsm_path(')
        helpers = additions[start:end]
        source = r'''
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
using llama_tokens = std::vector<int32_t>;
static uint64_t rsm_env_hash = 0;
'''
        source += helpers
        source += r'''
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string path = argv[1];
    const auto before = rsm_env_desc(path, 1, 1, 1, false);
    if (before != rsm_env_desc(path, 1, 1, 1, false)) return 3;
    const auto stamp = std::filesystem::last_write_time(path);
    const auto size = std::filesystem::file_size(path);
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp((1 << 20) + 128);
        file.put('x');
    }
    std::filesystem::last_write_time(path, stamp + std::chrono::seconds(1));
    if (std::filesystem::file_size(path) != size) return 4;
    const auto after = rsm_env_desc(path, 1, 1, 1, false);
    if (before == after) {
        std::fprintf(stderr, "changed model weights reuse the original prefix-cache key\n");
        return 1;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, executable, model = root / 'identity.cpp', root / 'identity', root / 'model.gguf'
            code.write_text(source)
            model.write_bytes(bytes((1 << 20) + 256))
            build = subprocess.run([compiler, '-std=c++17', '-O2', str(code), '-o', str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable), str(model)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
