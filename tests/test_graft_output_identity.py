import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

try:
    import gguf
    import numpy as np
except ImportError:
    gguf = np = None

ROOT = Path(__file__).resolve().parents[1]


def disable_core_dumps():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class GraftOutputIdentity(unittest.TestCase):
    def write_model(self, path, block_count, name, values):
        writer = gguf.GGUFWriter(str(path), 'qwen35')
        writer.add_uint32('qwen35.block_count', block_count)
        writer.add_tensor(name, np.asarray(values, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def run_graft(self, base, side, output):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
            '--gguf-py', str(Path(gguf.__file__).parent.parent), str(base), str(side), str(output)],
            text=True, capture_output=True, timeout=15, preexec_fn=disable_core_dumps)

    def test_input_aliases_are_rejected_before_loading_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, side = root / 'base.gguf', root / 'side.gguf'
            base.write_bytes(b'base model contents')
            side.write_bytes(b'sidecar contents')
            symlink, hardlink = root / 'alias.gguf', root / 'hardlink.gguf'
            symlink.symlink_to(base)
            os.link(side, hardlink)
            for output in (base, side, symlink, hardlink):
                with self.subTest(output=output.name):
                    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
                        '--gguf-py', str(root / 'unused-reader'), str(base), str(side), str(output)],
                        text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('output must be different from both input files', result.stderr)
                    self.assertEqual(base.read_bytes(), b'base model contents')
                    self.assertEqual(side.read_bytes(), b'sidecar contents')

    @unittest.skipIf(gguf is None, 'install gguf and numpy to exercise real GGUF files')
    def test_real_input_models_survive_all_output_aliases(self):
        for alias in ('base', 'side', 'symlink', 'hardlink'):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                base, side = root / 'base.gguf', root / 'side.gguf'
                self.write_model(base, 1, 'blk.0.attn_norm.weight', [1, 2, 3])
                self.write_model(side, 2, 'blk.1.ffn_norm.weight', [4, 5, 6, 7, 8])
                before_base, before_side = base.read_bytes(), side.read_bytes()
                output = base if alias == 'base' else side
                if alias == 'symlink':
                    output = root / 'alias.gguf'
                    output.symlink_to(base)
                elif alias == 'hardlink':
                    output = root / 'hardlink.gguf'
                    os.link(side, output)
                result = self.run_graft(base, side, output)
                self.assertEqual(base.read_bytes(), before_base, 'grafting changed the base input model')
                self.assertEqual(side.read_bytes(), before_side, 'grafting changed the sidecar input model')
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('output must be different from both input files', result.stderr)

    @unittest.skipIf(gguf is None, 'install gguf and numpy to exercise real GGUF files')
    def test_distinct_output_grafts_real_tensors_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, side, output = (root / name for name in ('base.gguf', 'side.gguf', 'out.gguf'))
            self.write_model(base, 1, 'blk.0.attn_norm.weight', [1, 2, 3])
            self.write_model(side, 2, 'blk.1.ffn_norm.weight', [4, 5, 6, 7, 8])
            before_base, before_side = base.read_bytes(), side.read_bytes()
            result = self.run_graft(base, side, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(base.read_bytes(), before_base)
            self.assertEqual(side.read_bytes(), before_side)
            reader = gguf.GGUFReader(str(output))
            self.assertEqual({tensor.name: tensor.data.tolist() for tensor in reader.tensors},
                             {'blk.0.attn_norm.weight': [1, 2, 3],
                              'blk.1.ffn_norm.weight': [4, 5, 6, 7, 8]})


if __name__ == '__main__':
    unittest.main()
