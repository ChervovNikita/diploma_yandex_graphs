"""Native reference orchestration checks; project-local deliberate fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from native_driver import BLOCKS
from runtime import ROOT, file_descriptor, load_runtime, verify_descriptor
from schedule import BankClock, Selector, stage_at


class Contracts(unittest.TestCase):
    def test_fixed_blocks_four_pass_cost_and_native_stage_clock(self):
        self.assertEqual(BLOCKS, ((0, 17), (1, 29), (2, 43)))
        self.assertEqual(sum(stage_at(u) == "local" for u in range(1, 2701)), 200)
        self.assertEqual(sum(stage_at(u) == "global" for u in range(1, 2701)), 2500)
        self.assertEqual(4 * 2700, 10800)
        clock = BankClock(200)
        clock.require_transition()
        self.assertEqual(clock.advance(), 201)

    def test_one_pooled_selector_earliest_ties_and_local_winner(self):
        selector = Selector()
        self.assertTrue(selector.observe(7, 10, 1))
        self.assertFalse(selector.observe(7, 10, 201))
        self.assertEqual(selector.selected_update, 1)
        self.assertTrue(selector.observe(8, 10, 202))

    def test_default_no_numerical_entry(self):
        with self.assertRaises(RuntimeError):
            load_runtime()
        result = subprocess.run([sys.executable, "-B", str(ROOT / "native_driver.py")],
                                check=True, capture_output=True, text=True)
        self.assertFalse(json.loads(result.stdout)["scientific_training"])
        self.assertFalse(any(n in sys.modules for n in ("torch", "numpy", "torch_geometric")))

    def test_portable_in_project_descriptor(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            phase = Path(directory)
            package = phase / "package"
            package.mkdir()
            source = phase / "synthetic_source.py"
            source.write_text("# project-local fixture\n")
            binding = dict(file_descriptor(source), path=source.name)
            with patch("runtime.ROOT", package):
                self.assertEqual(verify_descriptor(binding)["path"], str(source.resolve()))


if __name__ == "__main__":
    unittest.main()
