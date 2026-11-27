import unittest

from evaluate import program, run


PROBLEM = {'entry_point': 'answer', 'src_prompt': 'def answer():\n', 'test': 'assert answer() == 42'}


class MainGuardGradingTest(unittest.TestCase):
    def assert_passes(self, source):
        for clean in (False, True):
            with self.subTest(clean=clean):
                self.assertEqual(run(program(PROBLEM, source, clean)), (True, ''))

    def test_main_guard_in_docstring_is_preserved(self):
        self.assert_passes('''def answer():
    """Example:
if __name__ == "__main__":
    print(answer())
    """
    return 42
''')

    def test_main_guard_in_string_value_is_preserved(self):
        self.assert_passes('''EXAMPLE = """
if __name__ == '__main__':
    print(answer())
"""
def answer():
    return 42 if 'print(answer())' in EXAMPLE else 0
''')

    def test_actual_main_guard_tail_is_removed(self):
        self.assert_passes('''def answer():
    return 42
if __name__ == "__main__":
    raise RuntimeError("model demo is not the test")
''')

    def test_truncated_main_demo_is_still_removed(self):
        self.assert_passes('def answer():\n    return 42\nif __name__ == "__main__":\n    print("unfinished')

    def test_nested_main_guard_is_not_a_top_level_tail(self):
        self.assert_passes('''def answer():
    if __name__ == "__main__":
        return 42
    return 0
''')


if __name__ == '__main__':
    unittest.main()
