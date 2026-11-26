from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import rrun


class ReplayReadinessTest(unittest.TestCase):
    def start(self, proc, health, occupied=False):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(rrun.subprocess, "run", return_value=Mock(returncode=0 if occupied else 1)), \
             patch.object(rrun.subprocess, "Popen", return_value=proc) as spawn, \
             patch.object(rrun.urllib.request, "urlopen", side_effect=health), \
             patch.object(rrun.time, "sleep"), patch.object(rrun, "server_pid", return_value=proc.pid):
            args = SimpleNamespace(ctx=4096, arm="test")
            try:
                return rrun.start_server({"cmd": "unused"}, args, Path(directory), "")
            finally:
                if occupied:
                    self.assertFalse(spawn.called, "a busy port must be rejected before launching a server")

    def test_health_deadline_raises_and_reaps_server(self):
        proc = Mock(pid=400)
        proc.poll.return_value = None
        with patch.object(rrun.os, "killpg") as killpg:
            with self.assertRaises(SystemExit) as error:
                self.start(proc, OSError("not ready"))
        self.assertIn("not healthy", str(error.exception))
        killpg.assert_called_once_with(400, signal.SIGKILL)
        proc.wait.assert_called_once()

    def test_dead_server_is_not_accepted_as_ready(self):
        proc = Mock(pid=400)
        proc.poll.return_value = 1
        response = Mock()
        response.__enter__ = Mock(return_value=Mock(status=200))
        response.__exit__ = Mock(return_value=False)
        with patch.object(rrun.os, "killpg"):
            with self.assertRaises(SystemExit):
                self.start(proc, lambda *args, **kwargs: response)
        proc.wait.assert_called_once()

    def test_busy_port_times_out_before_spawn(self):
        proc = Mock(pid=400)
        proc.poll.return_value = None
        with self.assertRaises(SystemExit):
            self.start(proc, OSError("not ready"), occupied=True)


if __name__ == "__main__":
    unittest.main()
