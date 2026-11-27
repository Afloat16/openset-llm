from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResumePublication(unittest.TestCase):
    def test_failed_publication_removes_the_temporary_state(self):
        compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None or shutil.which('git') is None:
            self.skipTest('needs a C++ compiler and git')
        patch = (ROOT / 'patches/0007-resume.patch').read_text()
        additions = '\n'.join(line[1:] for line in patch.splitlines()
                              if line.startswith('+') and not line.startswith('+++'))
        start = additions.index('    void rsm_save_prefix(')
        method = additions[start:additions.index('\n    }', start) + len('\n    }')] + '\n'
        patch10 = (ROOT / 'patches/0010-multi-resume.patch').read_text()
        hunks = re.split(r'(?=^@@ )', patch10, flags=re.MULTILINE)[1:]
        save_hunks = [h for h in hunks if 'const size_t n_t =' in h or 'rename(tmp_t, ft, ec)' in h]
        additions10 = '\n'.join(line[1:] for line in patch10.splitlines()
                                if line.startswith('+') and not line.startswith('+++'))
        seal = additions10[additions10.index('static const char RSM_SEAL'):
                           additions10.index('static bool rsm_sealed(')]
        harness = r'''
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
using llama_tokens = std::vector<int32_t>;
#define SLT_INF(...) ((void) 0)
static std::string directory;
static bool fail_write = false;
static uint64_t rsm_hash(const llama_tokens &) { return 1; }
static std::string rsm_path(uint64_t, const char * part) { return directory + "/prefix." + part; }
static int64_t ggml_time_us() { return 0; }
static void rsm_drop_cache(const std::string &) {}
static void rsm_prune() {}
static bool common_speculative_get_state(int *, int, std::vector<uint8_t> &) { return false; }
static size_t llama_state_seq_save_file(void *, const char * path, int, const int32_t *, size_t) {
    std::ofstream file(path, std::ios::binary);
    file.write("state", 5);
    return fail_write ? 0 : 5;
}
struct input_tokens { llama_tokens values = llama_tokens(65, 0); const llama_tokens & get_tokens() const { return values; } };
struct server_task { input_tokens tokens; };
struct server_slot { int rsm_prefix = 64, id = 0; server_task * task; };
'''
        harness += seal
        harness += r'''
struct server {
    void * ctx_tgt = nullptr, * ctx_dft = nullptr;
    std::unique_ptr<int> spec;
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialized = subprocess.run(['git', 'init', '-q', str(root)], capture_output=True, text=True)
            self.assertEqual(initialized.returncode, 0, initialized.stderr)
            # Reconstruct the actual save method after patch 0010, using git's patch application.
            fragment = root / 'save.cpp'
            fragment.write_text(method)
            partial = 'diff --git a/save.cpp b/save.cpp\n--- a/save.cpp\n+++ b/save.cpp\n' + ''.join(save_hunks)
            applied = subprocess.run(['git', 'apply', '--recount', '-'], cwd=root, input=partial,
                                     capture_output=True, text=True)
            self.assertEqual(applied.returncode, 0, applied.stderr)
            harness += fragment.read_text()
            harness += r'''
};
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    directory = argv[1];
    const std::string mode = argv[2];
    const auto final = rsm_path(1, "tgt"), temporary = final + ".tmp";
    if (mode == "publish-failure") std::filesystem::create_directory(final);
    fail_write = mode == "write-failure";
    server model;
    server_task request;
    server_slot slot{64,0,&request};
    model.rsm_save_prefix(slot);
    if (std::filesystem::exists(temporary)) {
        std::fprintf(stderr, "temporary state survived %s\n", mode.c_str());
        return 1;
    }
    if (mode == "publish-failure" && !std::filesystem::is_directory(final)) return 3;
    if (mode == "write-failure" && std::filesystem::exists(final)) return 4;
    if (mode == "success" && !std::filesystem::is_regular_file(final)) return 5;
    return 0;
}
'''
            code, executable = root / 'publication.cpp', root / 'publication'
            code.write_text(harness)
            build = subprocess.run([compiler, '-std=c++17', '-O2', str(code), '-o', str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for mode in ('publish-failure', 'write-failure', 'success'):
                with self.subTest(mode=mode):
                    case = root / mode
                    case.mkdir()
                    result = subprocess.run([str(executable), str(case), mode],
                                            capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
