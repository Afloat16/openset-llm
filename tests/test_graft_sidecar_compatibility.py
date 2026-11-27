from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import gguf
    import numpy as np
except ImportError:
    raise unittest.SkipTest('install gguf and numpy to exercise real GGUF files')

ROOT = Path(__file__).resolve().parents[1]


class GraftSidecarCompatibility(unittest.TestCase):
    def write_model(self, path, arch, layers, name, embedding_length):
        writer = gguf.GGUFWriter(str(path), arch)
        writer.add_uint32(f'{arch}.block_count', layers)
        writer.add_uint32(f'{arch}.embedding_length', embedding_length)
        writer.add_tensor(name, np.ones(embedding_length, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def run_graft(self, root):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
            '--gguf-py', str(Path(gguf.__file__).parent.parent), str(root / 'base.gguf'),
            str(root / 'side.gguf'), str(root / 'out.gguf')],
            capture_output=True, text=True, timeout=15)

    def test_incompatible_sidecars_fail_before_output_is_replaced(self):
        for arch, dimensions, diagnostic in (('llama', 4, 'architecture'),
                                               ('qwen35', 8, 'embedding length')):
            with self.subTest(arch=arch, dimensions=dimensions), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_model(root / 'base.gguf', 'qwen35', 1, 'blk.0.attn_norm.weight', 4)
                self.write_model(root / 'side.gguf', arch, 2, 'blk.1.ffn_norm.weight', dimensions)
                output = root / 'out.gguf'
                output.write_bytes(b'previous model')
                result = self.run_graft(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(output.read_bytes(), b'previous model')

    def test_compatible_sidecar_keeps_model_metadata_and_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_model(root / 'base.gguf', 'qwen35', 1, 'blk.0.attn_norm.weight', 4)
            self.write_model(root / 'side.gguf', 'qwen35', 2, 'blk.1.ffn_norm.weight', 4)
            result = self.run_graft(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            reader = gguf.GGUFReader(str(root / 'out.gguf'))
            self.assertEqual(reader.fields['qwen35.embedding_length'].parts[-1][0], 4)
            self.assertEqual(len(reader.tensors), 2)
            for tensor in reader.tensors:
                np.testing.assert_array_equal(tensor.data, [1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
