from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResumeLora(unittest.TestCase):
    def test_active_adapters_do_not_use_the_base_model_disk_cache(self):
        compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('needs a C++ compiler')
        patch = (ROOT / 'patches/0007-resume.patch').read_text()
        source = '\n'.join(line[1:] for line in patch.splitlines()
                           if line.startswith('+') and not line.startswith('+++'))
        method = source[source.index('    bool rsm_load_prefix('):
                        source.index('    // 0007: write the prefix state;')]
        harness = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
using llama_tokens = std::vector<int32_t>;
using llama_token = int32_t;
constexpr int LLAMA_TOKEN_NULL = -1;
constexpr int COMMON_CHAT_ROLE_SYSTEM = 0, COMMON_CHAT_ROLE_UNKNOWN = 1, USER = 2;
#define SRV_INF(...) ((void) 0)
#define SLT_INF(...) ((void) 0)
#define SLT_WRN(...) ((void) 0)
struct { std::string dir; } rsm;
static uint64_t rsm_env_hash = 1;
static int loads = 0;
static std::string rsm_env_desc(const std::string &, int, int, int, bool) { return "env"; }
static uint64_t rsm_fnv(uint64_t, const void *, size_t) { return 1; }
static uint64_t rsm_hash(const llama_tokens &) { return 1; }
static std::string rsm_path(uint64_t, const char * part) { return rsm.dir + "/prefix." + part; }
static int64_t ggml_time_us() { return 0; }
static bool rsm_read_bytes(const std::string &, std::vector<uint8_t> &) { return false; }
static void common_speculative_set_state(int *, int, const std::vector<uint8_t> &) {}
static void rsm_drop_cache(const std::string &) {}
static size_t llama_state_seq_load_file(void *, const char *, int, int32_t * tokens, size_t count, size_t * got) {
    ++loads;
    std::fill(tokens, tokens + count, 0);
    *got = count;
    return 1;
}
struct input_tokens {
    llama_tokens values = llama_tokens(65, 0);
    const llama_tokens & get_tokens() const { return values; }
};
struct span { int role, pos; };
struct task_params { struct { std::vector<span> spans = {{COMMON_CHAT_ROLE_SYSTEM,0},{USER,64}}; } message_spans; };
struct server_task { input_tokens tokens; task_params params; int32_t n_tokens() const { return 65; } };
struct adapter { float scale; };
struct server_slot {
    server_task * task;
    int id = 0, rsm_prefix = -1;
    std::vector<adapter> lora;
    struct { void seq_rm(int, int, int) {} } mem;
    struct { llama_tokens tokens; void clear() { tokens.clear(); } } prompt;
};
struct server {
    struct { struct { std::string path; } model; int cache_type_k=0, cache_type_v=0, flash_attn_type=0; } params_base;
    void * ctx_tgt = nullptr, * ctx_dft = nullptr;
    std::unique_ptr<int> spec;
'''
        harness += method
        harness += r'''
};
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    rsm.dir = argv[1];
    std::ofstream(rsm_path(1, "tgt")).put('x');
    server model;
    server_task request;
    for (float scale : {1.0f, -0.5f, 0.0f}) {
        server_slot slot{&request}; slot.lora = {{scale}};
        int32_t n_past = 0; loads = 0;
        const bool loaded = model.rsm_load_prefix(slot, n_past);
        const bool expected = scale == 0.0f;
        if (loaded != expected || loads != (expected ? 1 : 0) || n_past != (expected ? 64 : 0)) {
            std::fprintf(stderr, "scale=%g: loaded=%d calls=%d n_past=%d\n", scale, loaded, loads, n_past);
            return 1;
        }
    }
    std::filesystem::remove(rsm_path(1, "tgt"));
    server_slot active{&request}; active.lora = {{1.0f}};
    int32_t n_past = 0;
    if (model.rsm_load_prefix(active, n_past) || active.rsm_prefix != -1) {
        std::fprintf(stderr, "active LoRA request scheduled a base-namespace cache write\n");
        return 1;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, executable = root / 'lora.cpp', root / 'lora'
            code.write_text(harness)
            build = subprocess.run([compiler, '-std=c++17', '-O2', str(code), '-o', str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable), str(root)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
