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


class GraftByteOrder(unittest.TestCase):
    def write_model(self, path, block_count, name, endian):
        writer = gguf.GGUFWriter(str(path), 'qwen35', endianess=endian)
        writer.add_uint32('qwen35.block_count', block_count)
        writer.add_tensor(name, np.asarray([1, -2, 3.25], dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def run_graft(self, root):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
            '--gguf-py', str(Path(gguf.__file__).parent.parent), str(root / 'base.gguf'),
            str(root / 'side.gguf'), str(root / 'out.gguf')],
            capture_output=True, text=True, timeout=15)

    def test_matching_inputs_retain_their_byte_order_and_tensor_values(self):
        for endian in (gguf.GGUFEndian.LITTLE, gguf.GGUFEndian.BIG):
            with self.subTest(endian=endian), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_model(root / 'base.gguf', 1, 'blk.0.attn_norm.weight', endian)
                self.write_model(root / 'side.gguf', 2, 'blk.1.ffn_norm.weight', endian)
                result = self.run_graft(root)
                self.assertEqual(result.returncode, 0, result.stderr)
                reader = gguf.GGUFReader(str(root / 'out.gguf'))
                for tensor in reader.tensors:
                    np.testing.assert_array_equal(tensor.data, [1, -2, 3.25])
                self.assertEqual(reader.endianess, endian)

    def test_mixed_byte_orders_are_rejected_before_overwriting_output(self):
        for base_endian, side_endian in ((gguf.GGUFEndian.LITTLE, gguf.GGUFEndian.BIG),
                                         (gguf.GGUFEndian.BIG, gguf.GGUFEndian.LITTLE)):
            with self.subTest(base_endian=base_endian), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_model(root / 'base.gguf', 1, 'blk.0.attn_norm.weight', base_endian)
                self.write_model(root / 'side.gguf', 2, 'blk.1.ffn_norm.weight', side_endian)
                output = root / 'out.gguf'
                output.write_bytes(b'previous model')
                result = self.run_graft(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('same byte order', result.stderr)
                self.assertEqual(output.read_bytes(), b'previous model')


if __name__ == '__main__':
    unittest.main()
