from pathlib import Path
import resource
import signal
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


def limit_output_file():
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class GraftAtomicOutput(unittest.TestCase):
    def write_model(self, path, block_count, name):
        writer = gguf.GGUFWriter(str(path), 'qwen35')
        writer.add_uint32('qwen35.block_count', block_count)
        writer.add_tensor(name, np.arange(1024, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def run_graft(self, root, limited=False):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
            '--gguf-py', str(Path(gguf.__file__).parent.parent), str(root / 'base.gguf'),
            str(root / 'side.gguf'), str(root / 'out.gguf')],
            capture_output=True, text=True, timeout=15,
            preexec_fn=limit_output_file if limited else None)

    def prepare(self, root):
        self.write_model(root / 'base.gguf', 1, 'blk.0.attn_norm.weight')
        self.write_model(root / 'side.gguf', 2, 'blk.1.ffn_norm.weight')

    def test_failed_write_preserves_an_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            output = root / 'out.gguf'
            output.write_bytes(b'previous working model')
            names = set(root.iterdir())
            result = self.run_graft(root, limited=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), b'previous working model')
            self.assertEqual(set(root.iterdir()), names)

    def test_failed_write_leaves_no_output_or_staging_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            names = set(root.iterdir())
            result = self.run_graft(root, limited=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(set(root.iterdir()), names)

    def test_success_replaces_output_with_a_complete_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            output = root / 'out.gguf'
            output.write_bytes(b'previous working model')
            names = set(root.iterdir())
            result = self.run_graft(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(root.iterdir()), names)
            reader = gguf.GGUFReader(str(output))
            self.assertEqual([t.name for t in reader.tensors],
                             ['blk.0.attn_norm.weight', 'blk.1.ffn_norm.weight'])
            for tensor in reader.tensors:
                np.testing.assert_array_equal(tensor.data, np.arange(1024, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
