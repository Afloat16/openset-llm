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


class GraftMetadataTypes(unittest.TestCase):
    def write_model(self, path, block_count, name, metadata=()):
        writer = gguf.GGUFWriter(str(path), 'qwen35')
        writer.add_uint32('qwen35.block_count', block_count)
        for key, value, kind, subtype in metadata:
            writer.add_key_value(key, value, kind, sub_type=subtype)
        writer.add_tensor(name, np.asarray([1, 2, 3], dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def graft(self, root, metadata):
        base, side, output = (root / name for name in ('base.gguf', 'side.gguf', 'out.gguf'))
        self.write_model(base, 1, 'blk.0.attn_norm.weight', metadata)
        self.write_model(side, 2, 'blk.1.ffn_norm.weight')
        result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/graft_mtp.py'),
            '--gguf-py', str(Path(gguf.__file__).parent.parent), str(base), str(side), str(output)],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return gguf.GGUFReader(str(base)), gguf.GGUFReader(str(output))

    def test_numeric_arrays_retain_their_declared_type_and_values(self):
        value_type = gguf.GGUFValueType
        cases = [
            (value_type.UINT32, [16, 24, 24]),
            (value_type.INT8, [-128, 127]),
            (value_type.UINT64, [2 ** 63 + 17]),
            (value_type.FLOAT64, [1.0000000000001, -2.0000000000003]),
            (value_type.BOOL, [True, False]),
        ]
        for subtype, values in cases:
            with self.subTest(subtype=subtype), tempfile.TemporaryDirectory() as tmp:
                before, after = self.graft(Path(tmp), [
                    ('qwen35.rope.dimension_sections', values, value_type.ARRAY, subtype)])
                original = before.fields['qwen35.rope.dimension_sections']
                copied = after.fields['qwen35.rope.dimension_sections']
                self.assertEqual(copied.types, original.types)
                self.assertEqual([copied.parts[i][0].item() for i in copied.data],
                                 [original.parts[i][0].item() for i in original.data])

    def test_empty_string_metadata_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = self.graft(Path(tmp), [
                ('tokenizer.chat_template', '', gguf.GGUFValueType.STRING, None)])
            self.assertIn('tokenizer.chat_template', before.fields)
            self.assertIn('tokenizer.chat_template', after.fields)
            self.assertEqual(bytes(after.fields['tokenizer.chat_template'].parts[-1]), b'')

    def test_string_metadata_retains_its_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = self.graft(Path(tmp), [
                ('tokenizer.ggml.tokens', [b'hello', b'\xff'],
                 gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING),
                ('general.description', b'\xff', gguf.GGUFValueType.STRING, None)])
            for key in ('tokenizer.ggml.tokens', 'general.description'):
                original, copied = before.fields[key], after.fields[key]
                self.assertEqual(copied.types, original.types)
                self.assertEqual([bytes(copied.parts[i]) for i in copied.data],
                                 [bytes(original.parts[i]) for i in original.data])


if __name__ == '__main__':
    unittest.main()
