import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

import rrun


class ReplayProcessTest(unittest.TestCase):
    def test_pid_lookup_is_limited_to_the_started_process_group(self):
        def pgrep(command, **kwargs):
            # A global lookup finds another user's server before our own child.
            owned = "-g" in command and command[command.index("-g") + 1] == "400"
            return Mock(stdout="401\n" if owned else "99\n401\n")

        with patch.object(rrun.subprocess, "run", side_effect=pgrep):
            self.assertEqual(rrun.server_pid(400), 401)

    def test_shutdown_only_signals_the_owned_group(self):
        proc = Mock(pid=400)
        with patch.object(rrun.os, "killpg") as killpg, \
             patch.object(rrun.subprocess, "run") as run, patch.object(rrun.time, "sleep"):
            rrun.stop_server(proc)
        self.assertFalse(run.called, "cleanup must not run global pkill")
        self.assertTrue(killpg.called)
        self.assertTrue(all(call.args[0] == 400 for call in killpg.call_args_list))
        proc.wait.assert_called()

    def test_forced_shutdown_reaps_the_child(self):
        proc = Mock(pid=400)
        proc.wait.side_effect = [subprocess.TimeoutExpired("server", 60), 0]
        with patch.object(rrun.os, "killpg") as killpg, \
             patch.object(rrun.subprocess, "run"), patch.object(rrun.time, "sleep"):
            rrun.stop_server(proc)
        killpg.assert_any_call(400, signal.SIGKILL)
        self.assertEqual(proc.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
