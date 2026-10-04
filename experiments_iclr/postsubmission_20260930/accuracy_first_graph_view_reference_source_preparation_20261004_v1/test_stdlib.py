"""Reference orchestration checks only; deliberate fixtures remain in project."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contract import BLOCKS, Selector, fit_spec, independent_inventory, pass_plan, stage_at
from runtime import ROOT, descriptor, load_runtime, verify


def result(member):
    return {"bindings": {"fit": fit_spec("native_member", 0, member), "context": "same"},
            "completed_actual_updates": 2700,
            "selection": {"actual_update": member + 1, "global": bool(member % 2)}}


class ReferenceContracts(unittest.TestCase):
    def test_coefficients_passes_and_seed_policy(self):
        self.assertEqual(pass_plan("view_augmented_single"), (("native", 1.0), ("equal", 0.5), ("different", 0.5)))
        self.assertEqual(pass_plan("native_member"), (("native", 1.0),))
        for split, seed in BLOCKS:
            self.assertEqual(fit_spec("view_augmented_single", split)["seed"], seed)
            self.assertEqual([fit_spec("native_member", split, m)["seed"] for m in range(4)],
                             [seed + 1009 * m for m in range(4)])
        with self.assertRaises(ValueError):
            fit_spec("native_single", 0)  # Alias, never a fifth physical native fit.

    def test_stage_exposure_and_strict_local_final_ties(self):
        self.assertEqual(sum(not stage_at(u) for u in range(1, 2701)), 200)
        self.assertEqual(sum(stage_at(u) for u in range(1, 2701)), 2500)
        self.assertEqual(2700 * len(pass_plan("view_augmented_single")), 8100)
        selector = Selector()
        self.assertTrue(selector.observe(5, 10, 1))
        self.assertFalse(selector.observe(5, 10, 201))
        self.assertEqual(selector.selected_update, 1)
        with self.assertRaises(ValueError):
            selector.observe(5, 10, 0)

    def test_four_independent_selectors_and_member0_alias(self):
        ordered = independent_inventory([result(m) for m in (3, 1, 0, 2)])
        self.assertEqual([r["bindings"]["fit"]["member"] for r in ordered], list(range(4)))
        self.assertEqual(ordered[0]["bindings"]["fit"]["seed"], 17)
        self.assertEqual(len({r["selection"]["actual_update"] for r in ordered}), 4)
        with self.assertRaises(ValueError):
            independent_inventory([result(m) for m in (0, 0, 2, 3)])
        altered = [result(m) for m in range(4)]
        altered[1]["bindings"]["context"] = "different labels"
        with self.assertRaises(ValueError):
            independent_inventory(altered)

    def test_default_entry_imports_no_numerics(self):
        with self.assertRaises(RuntimeError):
            load_runtime()
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "driver.py")], check=True,
                                   capture_output=True, text=True)
        self.assertFalse(json.loads(completed.stdout)["scientific_training"])
        self.assertFalse(any(name in sys.modules for name in ("numpy", "torch", "torch_geometric")))

    def test_portable_descriptor_resolution(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            phase = Path(directory)
            package = phase / "package"
            package.mkdir()
            source = phase / "synthetic_source.py"
            source.write_text("# in-project descriptor fixture\n")
            record = dict(descriptor(source), path=source.name)
            with patch("runtime.ROOT", package):
                self.assertEqual(verify(record)["path"], str(source.resolve()))


if __name__ == "__main__":
    unittest.main()
