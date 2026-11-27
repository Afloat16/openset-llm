from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AttentionConcurrentGraphs(unittest.TestCase):
    def test_independent_graphs_keep_their_attention_partials(self):
        compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('needs a C++ compiler')
        harness = r'''
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#define GGML_ABORT(...) std::abort()
#define GGML_ASSERT(condition) do { if (!(condition)) std::abort(); } while (0)
using ggml_fp16_t = uint16_t;
#define GGML_CPU_FP16_TO_FP32(value) ((float) (value))
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_K, GGML_TYPE_KV4, GGML_TYPE_KV5, GGML_TYPE_KV6 };
struct ggml_tensor {
    ggml_type type;
    int64_t ne[4];
    size_t nb[4];
    void * data;
    ggml_tensor * src[5] = {};
    float op_params[3] = {1, 0, 0};
};
struct ggml_threadpool { bool first; int barriers = 0; int chunk = 0; };
struct ggml_compute_params { int ith, nth; void * wdata; size_t wsize; ggml_threadpool * threadpool; };
static std::mutex lock;
static std::condition_variable changed;
static bool first_ready = false, second_finished = false;
static void ggml_barrier(ggml_threadpool * pool) {
    if (pool->first && ++pool->barriers == 2) {
        std::unique_lock<std::mutex> guard(lock);
        first_ready = true;
        changed.notify_all();
        changed.wait(guard, [] { return second_finished; });
    }
}
static void ggml_threadpool_chunk_set(ggml_threadpool * pool, int value) { pool->chunk = value; }
static int ggml_threadpool_chunk_add(ggml_threadpool * pool, int value) { int old = pool->chunk; pool->chunk += value; return old; }
static size_t ggml_type_size(ggml_type type) {
    return type == GGML_TYPE_Q8_K ? 292 : type == GGML_TYPE_KV4 ? 132 : type == GGML_TYPE_KV5 ? 164 : 196;
}
#include "attn-fast.inc"
static bool run_graph(bool first, int8_t value) {
    attnf_q8k keys{}, values{};
    keys.d = values.d = 1.0f;
    for (int i = 0; i < 256; ++i) values.qs[i] = value;
    float query[256] = {}, output[256] = {};
    ggml_tensor q{GGML_TYPE_F32, {256,1,1,1}, {4,1024,1024,1024}, query};
    ggml_tensor k{GGML_TYPE_Q8_K, {256,1,1,1}, {292,292,292,292}, &keys};
    ggml_tensor v{GGML_TYPE_Q8_K, {256,1,1,1}, {292,292,292,292}, &values};
    ggml_tensor dst{GGML_TYPE_F32, {256,1,1,1}, {4,1024,1024,1024}, output};
    dst.src[0] = &q; dst.src[1] = &k; dst.src[2] = &v;
    std::vector<uint8_t> scratch(4096);
    ggml_threadpool pool{first};
    ggml_compute_params params{0,1,scratch.data(),scratch.size(),&pool};
    if (!attnf_forward(&params, &dst)) return false;
    for (float actual : output) {
        if (actual != value) {
            std::fprintf(stderr, "graph with V=%d produced %g\n", value, actual);
            return false;
        }
    }
    return true;
}
int main() {
    bool a = false, b = false;
    std::thread first([&] { a = run_graph(true, 1); });
    std::thread second([&] {
        {
            std::unique_lock<std::mutex> guard(lock);
            changed.wait(guard, [] { return first_ready; });
        }
        b = run_graph(false, 9);
        {
            std::lock_guard<std::mutex> guard(lock);
            second_finished = true;
        }
        changed.notify_all();
    });
    first.join(); second.join();
    return a && b ? 0 : 1;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            code, executable = Path(tmp) / 'concurrent.cpp', Path(tmp) / 'concurrent'
            code.write_text(harness)
            build = subprocess.run([compiler, '-std=c++17', '-O2', '-pthread', '-I', str(ROOT / 'kernels'),
                                    str(code), '-o', str(executable)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
