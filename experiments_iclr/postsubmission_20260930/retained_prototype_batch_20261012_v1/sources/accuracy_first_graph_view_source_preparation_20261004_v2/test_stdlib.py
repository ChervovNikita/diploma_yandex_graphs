"""Synthetic implementation checks only. Never imports numerical packages."""
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from driver import verify_freeze
from bank import resolve_device, device_provenance, seed_all, rng_state, restore_rng
from runtime import ROOT, file_descriptor, load_runtime, verify_descriptor
from schedule import BankClock, CONDITIONS, Selector, assignment, pass_plan, stage_at
from views import TrainRole, construct_views, matched_null, native_edges, save_coverage, validate_native, verify_bundle


def fixture():
    role = TrainRole(30, 0, tuple(range(24)), (0,) * 12 + (1,) * 12,
                     (24, 25, 26), (27, 28, 29))
    edges = native_edges(30, ((a, b) for a in range(30) for b in range(a + 1, 30)))
    return construct_views(role, edges, 17)


class ViewsTests(unittest.TestCase):
    def test_train_domain_excludes_validation_and_test_labels(self):
        with self.assertRaises(ValueError):
            TrainRole.from_label_map(3, 0, (0,), {0: 1, 1: 2}, (1,), (2,))
        with self.assertRaises(ValueError):
            TrainRole(3, 0, (0, 1), (0,), (), (2,))
        with self.assertRaises(ValueError):
            TrainRole(3, 0, (0,), (0,), (0,), (1, 2))

    def test_fixed_pair_deletion_loops_order_and_coverage(self):
        bundle = fixture()
        self.assertEqual(bundle["coverage"]["categories"]["equal"]["eligible_pairs"], 132)
        self.assertEqual(bundle["coverage"]["categories"]["different"]["eligible_pairs"], 144)
        self.assertEqual(len(bundle["deleted_pairs"]["equal"]), 13)
        self.assertEqual(len(bundle["deleted_pairs"]["different"]), 14)
        for name, edges in bundle["views"].items():
            validate_native(30, edges)
            self.assertTrue(all(a in range(24) and b in range(24) for a, b in bundle["deleted_pairs"][name]))
            self.assertEqual(tuple(e for e in bundle["native"] if e in set(edges)), edges)
        self.assertEqual(bundle["coverage"], fixture()["coverage"])

    def test_validation_test_mask_exchange_cannot_change_views(self):
        original = fixture()
        role = original["role"]
        swapped = TrainRole(role.nodes, role.split, role.train_ids, role.train_labels, role.test_ids, role.val_ids)
        other = construct_views(swapped, original["native"], 17)
        self.assertEqual(original["deleted_pairs"], other["deleted_pairs"])

    def test_live_masks_must_match_pre_freeze_coverage(self):
        bundle = fixture()
        verify_bundle(bundle)
        bundle["views"]["equal"] = bundle["native"]
        with self.assertRaises(ValueError):
            verify_bundle(bundle)

    def test_null_exact_strata_and_deterministic_shortfall(self):
        for rows in fixture()["coverage"]["null_strata"].values():
            self.assertTrue(all(r["requested"] == r["selected"] and r["shortfall"] == 0 for r in rows))
        pool, strata = ((0, 1),), {(0, 1): (1, 2)}
        chosen, records = matched_null(pool, {(1, 2): 3, (2, 2): 1}, strata, 7, "undersupply")
        self.assertEqual(chosen, pool)
        self.assertEqual([r["shortfall"] for r in records], [2, 1])
        self.assertEqual((chosen, records), matched_null(pool, {(1, 2): 3, (2, 2): 1}, strata, 7, "undersupply"))

    def test_small_categories_are_ineligible_and_fixture_cannot_freeze(self):
        role = TrainRole(3, 0, (0, 1), (0, 1), (), (2,))
        bundle = construct_views(role, native_edges(3, [(0, 1)]), 1)
        self.assertFalse(bundle["coverage"]["coverage_eligible"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.json"
            record = save_coverage(fixture(), path, origin="synthetic_fixture")
            self.assertFalse(record["scientific_freeze_eligible"])
            freeze = {"schema": "accuracy-first-graph-view-freeze-v1", "source_manifest_sha256": "a",
                      "protocol_sha256": "b", "conditions": list(CONDITIONS),
                      "official_TRAIN_coverage": file_descriptor(path)}
            with self.assertRaises(ValueError):
                verify_freeze(SimpleNamespace(manifest_sha256="a", protocol_sha256="b"), fixture(), freeze)
            with self.assertRaises(RuntimeError):
                save_coverage(fixture(), Path(directory) / "official.json", origin="official_TRAIN")


class ClocksTests(unittest.TestCase):
    def test_all_conditions_equal_passes_and_stage_exposure(self):
        for condition in CONDITIONS:
            member_counts = {stage: [Counter() for _ in range(4)] for stage in ("local", "global")}
            for update in range(1, 2701):
                plan = pass_plan(condition, 17, update)
                self.assertEqual(plan[:4], tuple((m, "native") for m in range(4)))
                counts = Counter(view for _, view in plan[4:])
                self.assertEqual(sorted(counts.values()), [2, 2])
                for member, view in plan[4:]:
                    member_counts[stage_at(update)][member][view] += 1
            if condition == "tied_shuffled":
                for stage, length in (("local", 200), ("global", 2500)):
                    self.assertTrue(all(row == {"equal": length // 2, "different": length // 2}
                                        for row in member_counts[stage]))
            else:
                self.assertTrue(all(len(row) == 1 for rows in member_counts.values() for row in rows))

    def test_transition_clock_does_not_follow_rewound_optimizer(self):
        clock = BankClock(200)
        clock.require_transition()
        selected_checkpoint_update, rewound_Adam_step = 37, 37
        self.assertEqual(clock.advance(), 201)
        self.assertEqual(stage_at(clock.actual_update), "global")
        self.assertEqual(assignment("tied_shuffled", 17, clock.actual_update), assignment("tied_shuffled", 17, 201))
        self.assertNotEqual(clock.actual_update, selected_checkpoint_update)
        self.assertNotEqual(clock.actual_update, rewound_Adam_step)
        with self.assertRaises(ValueError):
            BankClock(37).require_transition()

    def test_one_selector_strict_first_tie_across_stages(self):
        selector = Selector()
        self.assertTrue(selector.observe(3, 10, 1))
        self.assertFalse(selector.observe(3, 10, 200))
        self.assertFalse(selector.observe(3, 10, 201))
        self.assertEqual(selector.selected_update, 1)
        self.assertTrue(selector.observe(4, 10, 202))
        self.assertEqual(selector.selected_update, 202)
        with self.assertRaises(ValueError):
            selector.observe(10, 10, 0)


class EntryTests(unittest.TestCase):
    def test_phase_relative_descriptors_relocate_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            phase = Path(directory)
            package = phase / "package"
            package.mkdir()
            source = phase / "immutable_source.py"
            source.write_text("# synthetic source relocation fixture\n")
            record = dict(file_descriptor(source), path="immutable_source.py")
            with patch("runtime.ROOT", package):
                actual = verify_descriptor(record)
            self.assertEqual(actual["path"], str(source.resolve()))
            self.assertEqual(actual["sha256"], record["sha256"])

    def test_default_cannot_import_numerics(self):
        with self.assertRaises(RuntimeError):
            load_runtime()
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "driver.py")], check=True, capture_output=True, text=True)
        self.assertFalse(json.loads(completed.stdout)["numerical_imports"])
        self.assertFalse(json.loads(completed.stdout)["scientific_training"])
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "torch_geometric")))

    def test_only_numerical_entry_function_imports_numerics(self):
        for name in ("views.py", "schedule.py", "runtime.py", "bank.py", "driver.py"):
            tree = ast.parse((ROOT / name).read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name.split(".")[0] not in {"torch", "numpy", "torch_geometric"}
                                        for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], {"torch", "numpy", "torch_geometric"})


class FakeDevice:
    def __init__(self, value, index=None):
        if isinstance(value, FakeDevice):
            self.type, self.index = value.type, value.index
        else:
            parts = value.split(":")
            self.type, self.index = parts[0], int(parts[1]) if len(parts) == 2 else None
        if index is not None:
            self.index = index

    def __str__(self):
        return self.type + (":" + str(self.index) if self.index is not None else "")


def fake_runtime():
    calls = []
    token = lambda name: SimpleNamespace(clone=lambda: name)
    cuda = SimpleNamespace(current_device=lambda: 2,
                           manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
                           get_rng_state=lambda device: (calls.append(("cuda_get", str(device))) or token("CUDA_STATE")),
                           set_rng_state=lambda state, device: calls.append(("cuda_set", str(device), state)))
    torch = SimpleNamespace(device=FakeDevice, cuda=cuda,
                            manual_seed=lambda seed: calls.append(("torch_seed", seed)),
                            get_rng_state=lambda: token("CPU_STATE"),
                            set_rng_state=lambda state: calls.append(("cpu_set", state)))
    np_random = SimpleNamespace(seed=lambda seed: calls.append(("numpy_seed", seed)),
                                get_state=lambda: ("STUB", SimpleNamespace(tolist=lambda: [1, 2]), 0, 0, 0.0),
                                set_state=lambda value: calls.append(("numpy_set", value)))
    numpy = SimpleNamespace(random=np_random, uint32="stub_uint32", asarray=lambda value, dtype: tuple(value))
    return SimpleNamespace(torch=torch, numpy=numpy), calls


class DeviceTests(unittest.TestCase):
    """Stub dispatch checks; no real CUDA/PyTorch execution or qualification."""
    def test_cuda_spellings_resolve_to_explicit_selected_index(self):
        rt, _ = fake_runtime()
        for value in ("cuda", FakeDevice("cuda"), "cuda:2", FakeDevice("cuda:2")):
            self.assertEqual(device_provenance(resolve_device(rt, value)),
                             {"type": "cuda", "index": 2, "canonical": "cuda:2"})
        self.assertEqual(str(resolve_device(rt, "cuda:3")), "cuda:3")
        self.assertEqual(device_provenance(resolve_device(rt, "cpu")),
                         {"type": "cpu", "index": None, "canonical": "cpu"})

    def test_seed_dispatch_includes_implicit_cuda_and_device_objects(self):
        for value in ("cuda", FakeDevice("cuda"), "cuda:3"):
            rt, calls = fake_runtime()
            seed_all(rt, 17, value)
            self.assertEqual(calls, [("numpy_seed", 17), ("torch_seed", 17), ("cuda_seed", 17)])
        rt, calls = fake_runtime()
        seed_all(rt, 17, "cpu")
        self.assertEqual(calls, [("numpy_seed", 17), ("torch_seed", 17)])

    def test_rng_capture_restore_binds_selected_device(self):
        for value in ("cuda", FakeDevice("cuda"), "cuda:2"):
            rt, calls = fake_runtime()
            state = rng_state(rt, value)
            self.assertEqual(state["cuda"], "CUDA_STATE")
            self.assertIn(("cuda_get", "cuda:2"), calls)
            restore_rng(rt, value, state)
            self.assertIn(("cuda_set", "cuda:2", "CUDA_STATE"), calls)
            with self.assertRaises(ValueError):
                restore_rng(rt, "cuda:3", state)
            with self.assertRaises(ValueError):
                restore_rng(rt, value, dict(state, cuda=None))
        rt, _ = fake_runtime()
        state = rng_state(rt, "cpu")
        self.assertIsNone(state["cuda"])
        restore_rng(rt, "cpu", state)
        with self.assertRaises(ValueError):
            restore_rng(rt, "cpu", dict(state, cuda="unexpected"))


if __name__ == "__main__":
    unittest.main()
