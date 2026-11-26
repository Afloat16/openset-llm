import unittest

from evaluate import sanitize


class DecoratorSanitizationTest(unittest.TestCase):
    def execute(self, source):
        namespace = {}
        exec(sanitize(source), namespace)
        return namespace

    def test_function_decorator_remains_effective(self):
        namespace = self.execute('''
def increment(function):
    return lambda: function() + 1
@increment
def answer():
    return 41
''')
        self.assertEqual(namespace["answer"](), 42)

    def test_class_decorator_remains_effective(self):
        namespace = self.execute('''
from dataclasses import dataclass
@dataclass
class Point:
    x: int
    y: int
''')
        self.assertEqual(namespace["Point"](2, 3).y, 3)

    def test_multiline_decorator_is_complete(self):
        namespace = self.execute('''
@(
    lambda function: lambda: function() + 1
)
def answer():
    return 41
''')
        self.assertEqual(namespace["answer"](), 42)

    def test_truncated_tail_and_example_asserts_are_still_removed(self):
        namespace = self.execute('''
def answer():
    return 42
assert False
def incomplete(
''')
        self.assertEqual(namespace["answer"](), 42)


if __name__ == "__main__":
    unittest.main()
