"""Stdlib-only source/decision checks; no tensor, data or trial execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from selector import ARMS, PAIRS, FrozenConstants, choose_pair


def constants(**changes):
    fields = dict(radius=0.01, radius_cap=0.02, rank_atol=1e-10, rank_rtol=1e-8,
                  sign_atol=1e-10, mean_logit_atol=1e-6, mean_logit_rtol=1e-5,
                  d_positive_min=1e-10, d_match_atol=1e-9, d_match_rtol=1e-5,
                  bisection_steps=32, tie_tolerance=0.01, abstention_tolerance=0.02)
    fields.update(changes)
    return FrozenConstants(**fields)


def row(index, value, status="eligible"):
    return dict(pair_index=index, post_trial_pooled_train_ce=value, status=status)


class SourceChecks(unittest.TestCase):
    def test_fixed_pair_order_and_complete_controls(self):
        self.assertEqual(PAIRS, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(len(ARMS), 5)
        self.assertIn("fixed_first_graph_pair", ARMS)

    def test_earliest_tie_against_true_minimum(self):
        winner, reason = choose_pair([row(0, 0.808), row(1, 0.800), row(2, 0.799)], 1.0, constants())
        self.assertEqual((winner, reason), (0, "selected"))

    def test_strict_common_trial_abstention(self):
        self.assertEqual(choose_pair([row(0, 0.981)], 1.0, constants()),
                         (None, "common_trial_abstention"))

    def test_failed_and_nonfinite_trials_are_retained_but_not_selected(self):
        self.assertEqual(choose_pair([row(0, 0.1, "trial_failure"), row(1, float("nan")),
                                      row(2, 0.7)], 1.0, constants())[0], 2)

    def test_no_eligible_pair_abstains(self):
        self.assertEqual(choose_pair([row(0, None, "rank_or_D0_failure")], 1.0, constants()),
                         (None, "no_eligible_pair"))

    def test_unbounded_or_invalid_constants_rejected(self):
        for change in (dict(radius=0), dict(radius_cap=0.001), dict(bisection_steps=0),
                       dict(d_match_rtol=float("inf")), dict(sign_atol=0)):
            with self.assertRaises(ValueError):
                constants(**change).validate()

    def test_compile_and_import_without_numerical_libraries(self):
        import driver
        for name in ("selector.py", "driver.py", "test_source.py"):
            path = Path(__file__).parent/name
            tree = ast.parse(path.read_text(), filename=str(path))
            compile(tree, str(path), "exec")
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                    self.assertFalse(any(name.startswith(("torch", "numpy")) for name in names))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertEqual(driver.SEEDS, (17, 29, 43))

    def test_descriptor_rejects_absolute_and_parent_paths(self):
        import driver
        for path in ("/unrelated/source.py", "../unrelated/source.py"):
            with self.assertRaises(ValueError):
                driver.source_path({"path": path})

    def test_phase_relocation_loads_bound_local_dependencies_without_bytecode(self):
        import driver
        old_here, old_bytecode, old_cache_prefix = driver.HERE, sys.dont_write_bytecode, sys.pycache_prefix
        names = ("relocation_dependency_a", "relocation_dependency_b", "relocation_native")
        try:
            with tempfile.TemporaryDirectory() as directory:
                phase = Path(directory)/"original_phase"
                packet, donors = phase/"packet", phase/"donors"
                packet.mkdir(parents=True)
                donors.mkdir()
                texts = {names[0]: "VALUE = 41\n", names[1]:
                         "from relocation_dependency_a import VALUE\nVALUE += 1\n",
                         names[2]: "from relocation_dependency_b import VALUE\n"}
                records = {}
                for name, text in texts.items():
                    path = donors/(name+".py")
                    path.write_text(text)
                    records[name] = dict(path="donors/"+path.name, bytes=len(path.read_bytes()),
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                sys.pycache_prefix = None  # Make the fixture cache local on every host.
                cache = Path(importlib.util.cache_from_source(str(donors/(names[0]+".py"))))
                cache.parent.mkdir()
                cache.write_bytes(b"stale bytecode must remain unread and unchanged")
                bindings = dict(native_source_dependencies=[records[names[0]], records[names[1]]],
                                runtime_modules={"integration": records[names[1]]},
                                native_adapter=records[names[2]])
                (packet/"SOURCE_BINDINGS.json").write_text(json.dumps(bindings))
                relocated = Path(directory)/"different_host"/"relocated_phase"
                relocated.parent.mkdir()
                shutil.move(str(phase), relocated)
                driver.HERE = relocated/"packet"
                sys.dont_write_bytecode = False  # The loader itself must avoid donor writes.
                modules = driver.load_sources()
                self.assertEqual(modules["native_adapter"].VALUE, 42)
                self.assertEqual(Path(modules["native_adapter"].__file__).resolve(),
                                 driver.source_path(bindings["native_adapter"]))
                caches = list((relocated/"donors").rglob("*.pyc"))
                self.assertEqual(len(caches), 1)
                self.assertEqual(caches[0].read_bytes(), b"stale bytecode must remain unread and unchanged")
                self.assertNotIn("torch", sys.modules)
                self.assertNotIn("numpy", sys.modules)
        finally:
            driver.HERE, sys.dont_write_bytecode, sys.pycache_prefix = old_here, old_bytecode, old_cache_prefix
            for name in names:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
