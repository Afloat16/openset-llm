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


class GraftAlignment(unittest.TestCase):
    def write_model(self, path, block_count, name, values, alignment):
        writer = gguf.GGUFWriter(str(path), 'qwen35')
        writer.add_uint32('qwen35.block_count', block_count)
        writer.add_custom_alignment(alignment)
        writer.add_tensor(name, np.asarray(values, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_grafted_tensors_follow_the_alignment_declared_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for alignment in (64, 256):
                with self.subTest(alignment=alignment):
                    base, side, output = (root / f'{label}-{alignment}.gguf' for label in ('base', 'side', 'out'))
                    self.write_model(base, 1, 'blk.0.attn_norm.weight', [1, 2, 3], alignment)
                    self.write_model(side, 2, 'blk.1.ffn_norm.weight', [4, 5, 6, 7, 8], 32)
                    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
                        '--gguf-py', str(Path(gguf.__file__).parent.parent), str(base), str(side), str(output)],
                        capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    reader = gguf.GGUFReader(str(output))
                    self.assertEqual(reader.alignment, alignment)
                    tensors = {tensor.name: tensor.data.tolist() for tensor in reader.tensors}
                    self.assertEqual(tensors, {'blk.0.attn_norm.weight': [1, 2, 3],
                                               'blk.1.ffn_norm.weight': [4, 5, 6, 7, 8]})


if __name__ == '__main__':
    unittest.main()
