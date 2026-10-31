"""Scheduling risks: duplicate spawn, memory pairing, parents and fixed custody."""
import unittest

from launcher import schedule_ready


class LauncherTests(unittest.TestCase):
    def fixture(self):
        runs = {key: {"run": {"config": {"reservation_mib": reserve, "estimated_seconds": cost}}}
                for key, reserve, cost in (("a", 4, 10), ("b", 4, 9), ("c", 4, 8), ("d", 4, 7))}
        states = {key: "registered" for key in runs}
        placement = {key: {"worker": "allocation/gpu0"} for key in runs}
        dependencies = {key: [] for key in runs}
        return runs, states, placement, dependencies

    def test_two_slots_and_no_duplicate_before_child_claim(self):
        rows = self.fixture()
        active = [{"key": "a", "worker": "allocation/gpu0", "reserve_mib": 4}]
        self.assertEqual(schedule_ready(*rows, active, {"allocation/gpu0": 20}, "allocation"), ["b"])
        self.assertEqual(schedule_ready(*rows, [], {"allocation/gpu0": 20}, "allocation"), ["a", "b"])

    def test_memory_admission_pairs_small_jobs_and_waits_for_large(self):
        rows = self.fixture()
        self.assertEqual(schedule_ready(*rows, [], {"allocation/gpu0": 6}, "allocation"), ["a"])
        self.assertEqual(schedule_ready(*rows, [], {"allocation/gpu0": 3}, "allocation"), [])

    def test_failed_or_running_parent_never_unlocks_child(self):
        runs, states, placement, dependencies = self.fixture()
        dependencies["b"] = ["a"]
        for parent_state in ("failed", "running"):
            states["a"] = parent_state
            result = schedule_ready(runs, states, placement, dependencies, [],
                                    {"allocation/gpu0": 20}, "allocation")
            self.assertNotIn("b", result)
        states["a"] = "complete"
        self.assertIn("b", schedule_ready(runs, states, placement, dependencies, [],
                                         {"allocation/gpu0": 20}, "allocation"))

    def test_no_cross_host_fallback(self):
        rows = self.fixture()
        rows[2]["a"]["worker"] = "server77/gpu1"
        self.assertNotIn("a", schedule_ready(*rows, [], {"allocation/gpu0": 20}, "allocation"))

if __name__ == "__main__":
    unittest.main()
