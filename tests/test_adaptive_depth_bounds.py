from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveDepthBounds(unittest.TestCase):
    def test_controller_bounds_remain_inside_the_depth_tables(self):
        compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('needs a C++ compiler')
        patch = (ROOT / 'patches/0005-adaptive-speculation.patch').read_text().split('@@ -347,')[0]
        controller = '\n'.join(line[1:] for line in patch.splitlines()
                               if line.startswith('+') and not line.startswith('+++'))
        source = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
static int64_t ggml_time_us() { return 1000000; }
struct controller {
    int id = 0;
'''
        source += controller
        source += r'''
};
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    setenv("LLAMA_SPEC_ADAPT_MIN", argv[1], 1);
    controller state;
    state.adapt_dmax = std::atoi(argv[2]);
    const int limit = std::max(0, std::min(state.adapt_dmax, 8));
    for (int i = 0; i < 32; ++i) {
        const int depth = state.adapt_choose();
        if (depth < 0 || depth > limit) {
            std::fprintf(stderr, "invalid draft depth %d for limit %d\n", depth, limit);
            return 1;
        }
        state.adapt_d_step = depth;
        state.adapt_t_step = 0;
        state.adapt_record(depth, depth);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            code, executable = Path(tmp) / 'adaptive.cpp', Path(tmp) / 'adaptive'
            code.write_text(source)
            build = subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=undefined',
                                    '-fno-sanitize-recover=undefined', str(code), '-o', str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for minimum, maximum in ((-1, 3), (-128, 3), (0, -1), (0, 0), (99, 3), (0, 64), (2, 3)):
                with self.subTest(minimum=minimum, maximum=maximum):
                    result = subprocess.run([str(executable), str(minimum), str(maximum)],
                                            capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
