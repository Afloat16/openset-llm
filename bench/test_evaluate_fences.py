import unittest

from evaluate import extract, program, run


PROBLEM = {"entry_point": "answer", "src_prompt": "def answer():\n", "test": "assert answer() == 42"}


class FencedCodeGradingTest(unittest.TestCase):
    def assert_passes(self, reply, problem=PROBLEM):
        for clean in (False, True):
            with self.subTest(clean=clean):
                self.assertEqual(run(program(problem, reply, clean)), (True, ""))

    def test_backticks_inside_a_string_do_not_end_the_block(self):
        self.assert_passes('```python\ndef answer():\n    return "```"\n```',
                           {**PROBLEM, "test": 'assert answer() == "```"'})

    def test_backticks_in_a_comment_do_not_end_the_block(self):
        self.assert_passes('```python\n# Return the answer, not the ``` delimiter.\ndef answer():\n    return 42\n```')

    def test_shorter_fence_inside_a_longer_block_is_preserved(self):
        self.assert_passes('````python\nEXAMPLE = """\n```\n"""\ndef answer():\n'
                           '    return 42 if EXAMPLE.strip() == "```" else 0\n````\n')

    def test_backticks_followed_by_text_are_not_a_closing_fence(self):
        self.assert_passes('```python\nEXAMPLE = """\n``` this is part of a string\n"""\n'
                           'def answer():\n    return 42\n```\n')

    def test_longer_closing_fence_and_surrounding_prose(self):
        self.assert_passes("Here is the solution:\n```python\ndef answer():\n    return 42\n````\nExplanation follows.")

    def test_closing_fence_allows_indentation_and_trailing_whitespace(self):
        self.assert_passes("```python\ndef answer():\n    return 42\n   ``` \t\nExplanation follows.")

    def test_language_aliases_and_crlf(self):
        for language in ("", "python", "PY", "python3"):
            with self.subTest(language=language):
                self.assert_passes(f"```{language}\r\ndef answer():\r\n    return 42\r\n```\r\n")

    def test_block_defining_entry_point_is_still_selected(self):
        self.assert_passes("```python\nx = 0\n```\n```text\nnot Python\n```\n"
                           "```py\ndef answer():\n    return 42\n```\n")

    def test_first_python_block_remains_the_fallback(self):
        self.assertEqual(extract("```text\nignore\n```\n```python\nx = 1\n```\n```py\nx = 2\n```", "answer"),
                         "x = 1\n")

    def test_unclosed_block_keeps_truncated_reply_for_sanitization(self):
        source = "def answer():\n    return 42\ndef unfinished("
        reply = "```python\n" + source
        self.assertEqual(extract(reply, "answer"), source)
        self.assertEqual(run(program(PROBLEM, reply, True)), (True, ""))
        self.assertFalse(run(program(PROBLEM, reply, False))[0])

    def test_unfenced_code_is_unchanged(self):
        source = 'def answer():\n    return 42  # ``` is just a comment\n'
        self.assertEqual(extract(source, "answer"), source)
        self.assert_passes(source)


if __name__ == "__main__":
    unittest.main()
