"""Local regression checks for zombie-aware custody; no signals or remote work."""
import copy
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

import finish_handover as f


IDENTITY = {"pid": 1234, "uid": 1000, "start_ticks": 4567, "boot_id": "boot"}


def stat_text(state="Z", start_ticks=4567):
    return "1234 (name (with spaces)) " + " ".join([state, *["0"] * 18, str(start_ticks)])


class ProcessStateTests(unittest.TestCase):
    def test_matching_zombie_and_dead_states_count_as_exited(self):
        for state in ("Z", "X"):
            with self.subTest(state=state), patch.object(f.c, "identity", return_value=IDENTITY), \
                 patch.object(Path, "read_text", return_value=stat_text(state)):
                self.assertEqual(f.process_state(IDENTITY), state)
                self.assertTrue(f.original_exited(IDENTITY))

    def test_matching_running_sleeping_stopped_states_are_not_exited(self):
        for state in ("R", "S", "D", "T", "I"):
            with self.subTest(state=state), patch.object(f.c, "identity", return_value=IDENTITY), \
                 patch.object(Path, "read_text", return_value=stat_text(state)):
                self.assertFalse(f.original_exited(IDENTITY))

    def test_missing_pid_counts_as_exited_without_second_proc_read(self):
        with patch.object(f.c, "identity", side_effect=FileNotFoundError), \
             patch.object(Path, "read_text") as read:
            self.assertEqual(f.process_state(IDENTITY), "absent_or_reused")
            read.assert_not_called()

    def test_reused_pid_does_not_read_or_signal_replacement_process(self):
        for field, value in (("start_ticks", 4568), ("uid", 1001), ("boot_id", "new-boot")):
            changed = dict(IDENTITY, **{field: value})
            with self.subTest(field=field), patch.object(f.c, "identity", return_value=changed), \
                 patch.object(Path, "read_text") as read, \
                 patch.object(f.os, "pidfd_open", create=True) as opened:
                self.assertEqual(f.process_state(IDENTITY), "absent_or_reused")
                self.assertFalse(f.stop_exact_watcher(IDENTITY, Path("/request.json")))
                read.assert_not_called()
                opened.assert_not_called()

    def test_exit_between_identity_and_stat_read_is_allowed(self):
        with patch.object(f.c, "identity", return_value=IDENTITY), \
             patch.object(Path, "read_text", side_effect=ProcessLookupError):
            self.assertTrue(f.original_exited(IDENTITY))

    def test_changed_start_time_between_reads_fails_closed(self):
        with patch.object(f.c, "identity", return_value=IDENTITY), \
             patch.object(Path, "read_text", return_value=stat_text("Z", 9999)), \
             self.assertRaisesRegex(ValueError, "Identity changed"):
            f.original_exited(IDENTITY)

    def test_unreadable_or_malformed_proc_state_fails_closed(self):
        with patch.object(f.c, "identity", return_value=IDENTITY):
            with patch.object(Path, "read_text", side_effect=PermissionError), self.assertRaises(PermissionError):
                f.original_exited(IDENTITY)
            with patch.object(Path, "read_text", return_value="1234 (name) Z 0"), \
                 self.assertRaisesRegex(ValueError, "Malformed"):
                f.original_exited(IDENTITY)


class DrainedTests(unittest.TestCase):
    def fixture(self):
        return ({"old_owner": IDENTITY, "pane": {"id": "%1"}},
                {"owner": IDENTITY, "phase": "stopped", "active_fit_ids": [],
                 "counts": {"complete": 10, "pending": 100}},
                {"id": "%1", "dead": True})

    def test_exited_owner_stopped_queue_and_dead_same_pane_are_required(self):
        request, status, pane = self.fixture()
        with patch.object(f, "original_exited", return_value=True):
            f.require_drained(request, status, pane)
        with patch.object(f, "original_exited", return_value=False), self.assertRaises(ValueError):
            f.require_drained(request, status, pane)
        bad_statuses = [dict(status, phase="full"), dict(status, owner=dict(IDENTITY, start_ticks=9999)),
                        dict(status, active_fit_ids=["fit"]), dict(status, counts={"running": 1}),
                        dict(status, counts={"failed": 1}), dict(status, error="failure")]
        with patch.object(f, "original_exited", return_value=True):
            for changed in bad_statuses:
                with self.subTest(status=changed), self.assertRaises(ValueError):
                    f.require_drained(request, changed, pane)
            for changed in (dict(pane, dead=False), dict(pane, id="%2")):
                with self.subTest(pane=changed), self.assertRaises(ValueError):
                    f.require_drained(request, status, changed)


class WatcherTests(unittest.TestCase):
    def cmdline(self, request):
        return b"\0".join([b"python", str(f.handover.HERE / "handover.py").encode(),
                           b"watch", b"--request", str(request).encode(), b""])

    def test_exact_watcher_is_signalled_through_pidfd_only(self):
        request = Path("/project/request.json")
        with patch.object(f, "original_exited", side_effect=[False, True, True]), \
             patch.object(f.c, "same_process", return_value=True), \
             patch.object(Path, "read_bytes", return_value=self.cmdline(request)), \
             patch.object(f.os, "pidfd_open", return_value=17, create=True) as opened, \
             patch.object(f.signal, "pidfd_send_signal", create=True) as sent, \
             patch.object(f.os, "close") as closed, patch.object(f.os, "kill") as unsafe:
            self.assertTrue(f.stop_exact_watcher(IDENTITY, request))
        opened.assert_called_once_with(IDENTITY["pid"])
        sent.assert_called_once_with(17, signal.SIGTERM)
        closed.assert_called_once_with(17)
        unsafe.assert_not_called()

    def test_identity_reuse_after_pidfd_open_prevents_signal_and_closes_fd(self):
        with patch.object(f, "original_exited", return_value=False), \
             patch.object(f.c, "same_process", return_value=False), \
             patch.object(f.os, "pidfd_open", return_value=17, create=True), \
             patch.object(f.signal, "pidfd_send_signal", create=True) as sent, \
             patch.object(f.os, "close") as closed, \
             self.assertRaisesRegex(ValueError, "Watcher identity changed"):
            f.stop_exact_watcher(IDENTITY, Path("/request.json"))
        sent.assert_not_called()
        closed.assert_called_once_with(17)

    def test_unrelated_command_or_different_request_is_never_signalled(self):
        request = Path("/project/request.json")
        wrong_commands = [b"python\0training.py\0", self.cmdline(Path("/different/request.json"))]
        for cmdline in wrong_commands:
            with self.subTest(cmdline=cmdline), patch.object(f, "original_exited", return_value=False), \
                 patch.object(f.c, "same_process", return_value=True), \
                 patch.object(Path, "read_bytes", return_value=cmdline), \
                 patch.object(f.os, "pidfd_open", return_value=17, create=True), \
                 patch.object(f.signal, "pidfd_send_signal", create=True) as sent, \
                 patch.object(f.os, "close") as closed, \
                 self.assertRaisesRegex(ValueError, "not the exact"):
                f.stop_exact_watcher(IDENTITY, request)
            sent.assert_not_called()
            closed.assert_called_once_with(17)


if __name__ == "__main__":
    unittest.main()
