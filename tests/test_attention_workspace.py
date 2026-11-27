from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AttentionWorkspace(unittest.TestCase):
    def test_worker_exit_releases_its_cached_workspace(self):
        compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('needs a C++ compiler')
        # Compile the production workspace implementation without requiring a full engine build.
        source = (ROOT / 'kernels/attn-fast.inc').read_text()
        workspace = source[:source.index('static const int8_t attnf_zero_row')]
        harness = r'''
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <stdlib.h>
#include <string.h>
#include <math.h>
enum ggml_type { GGML_TYPE_F16, GGML_TYPE_Q8_K, GGML_TYPE_KV4, GGML_TYPE_KV5, GGML_TYPE_KV6 };
static std::atomic<int> live_allocations{0};
static void * tracked_aligned_alloc(size_t alignment, size_t size) {
    void * p = ::aligned_alloc(alignment, size);
    if (p) ++live_allocations;
    return p;
}
static void tracked_free(void * p) {
    if (p) --live_allocations;
    ::free(p);
}
#define aligned_alloc tracked_aligned_alloc
#define free tracked_free
'''
        harness += workspace
        harness += r'''
int main() {
    for (int i = 0; i < 8; ++i) {
        std::thread worker([] {
            attnf_ws * first = attnf_ws_get();
            if (!first || attnf_ws_get() != first) std::abort();
            first->Q[0] = 1.0f;
            if (attnf_ws_get()->Q[0] != 1.0f) std::abort();
        });
        worker.join();
        if (live_allocations != 0) {
            std::fprintf(stderr, "%d workspace allocations remain after worker exit\n", live_allocations.load());
            return 1;
        }
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            source_file, executable = Path(tmp) / 'workspace.cpp', Path(tmp) / 'workspace'
            source_file.write_text(harness)
            build = subprocess.run([compiler, '-std=c++17', '-O2', '-pthread', '-I', str(ROOT / 'kernels'),
                                    str(source_file), '-o', str(executable)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
