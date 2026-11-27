import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rrun


def events(*payloads):
    return ''.join('data: ' + (x if isinstance(x, str) else json.dumps(x)) + '\n\n'
                   for x in payloads).encode()


class ReplayStreamCompletionTest(unittest.TestCase):
    def stream(self, payload):
        with patch.object(rrun.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
            return rrun.stream_chat({'messages': []})

    def test_partial_content_followed_by_eof_is_an_error(self):
        payload = events({'choices': [{'delta': {'content': 'partial'}}]})
        with self.assertRaisesRegex(RuntimeError, 'before.*DONE'):
            self.stream(payload)

    def test_empty_eof_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, 'before.*DONE'):
            self.stream(b'')

    def test_explicit_error_is_not_hidden_by_done(self):
        payload = events({'error': {'message': 'context exhausted'}}, '[DONE]')
        with self.assertRaisesRegex(RuntimeError, 'context exhausted'):
            self.stream(payload)

    def test_complete_stream_keeps_content_tool_calls_and_timings(self):
        payload = events(
            {'choices': [{'delta': {'reasoning_content': 'thinking'}}]},
            {'choices': [{'delta': {'content': 'answer', 'tool_calls': [
                {'index': 0, 'function': {'name': 'lookup', 'arguments': '{"id":'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [
                {'index': 0, 'function': {'arguments': '7}'}}]}}]},
            {'choices': [], 'timings': {'predicted_n': 3}}, '[DONE]')
        ttft, wall, content, calls, timings = self.stream(payload)
        self.assertIsNotNone(ttft)
        self.assertGreaterEqual(wall, ttft)
        self.assertEqual(content, 'answer')
        self.assertEqual(calls, [{'name': 'lookup', 'arguments': '{"id":7}'}])
        self.assertEqual(timings, {'predicted_n': 3})

    def test_run_step_records_truncation_as_failed_measurement(self):
        args = SimpleNamespace(gen=16, tokens={0: [1, 2]}, live=False)
        request = {'step': 0, 'messages': [{'role': 'user', 'content': 'hi'}]}
        with patch.object(rrun.urllib.request, 'urlopen', return_value=io.BytesIO(events(
                {'choices': [{'delta': {'content': 'partial'}}]}))):
            row, tokens = rrun.run_step(args, {}, request, [], 99999999, 'fixture', {})
        self.assertIsNotNone(row['error'])
        self.assertIn('DONE', row['error'])
        self.assertIsNone(row['wall_s'])
        self.assertIsNone(row['predicted_n'])
        self.assertEqual(tokens, [1, 2])


if __name__ == '__main__':
    unittest.main()
