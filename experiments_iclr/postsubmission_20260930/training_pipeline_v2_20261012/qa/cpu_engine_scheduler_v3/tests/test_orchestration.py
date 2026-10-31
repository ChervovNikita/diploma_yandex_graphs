"""Small contract/registry/metric checks. No scientific training or downloads."""
import copy
from pathlib import Path
import tempfile
import unittest

from metrics import EpochMeans
from partition import assign
from registry import Registry
from spec import validate


def study():
    return {"schema_version": 1, "study_id": "test", "stage": "engineering",
            "hypothesis": "engineering fixture", "dataset": "fixture",
            "data": {"task": "classification", "preprocessing": "fixture", "prior_exposure": "fixture",
                     "roles": {key: key for key in ("train", "select", "cal", "evaluate")}, "evidence": []},
            "source": {"git_commit": "fixture", "files": [
                {"path": "fixture.py", "bytes": 0, "sha256": "fixture"}]},
            "runs": [{"run_id": "fit", "kind": "model_fit", "method": "single", "split": 0,
                      "seed": 1, "body": 0, "adapter": "fixture.py:create", "config": {}, "parents": [],
                      "budget": {"unit": "optimizer_updates", "maximum": 3, "policy": "full_budget"},
                      "selection": {"metric": "accuracy", "direction": "max", "ties": "first",
                                    "evaluation_interval": 1, "checkpoint0": True}}],
            "comparisons": [{"candidate": "fit", "reference": "fit"}],
            "release": {"policy": "whole_roster", "primary_endpoint": "selected"},
            "resources": {"wall_seconds": 100}}


class OrchestrationTests(unittest.TestCase):
    def test_sample_weighted_epoch_mean(self):
        metric = EpochMeans()
        metric.add({"ce": 1., "objective": 1.}, samples=1)
        metric.add({"ce": 3., "objective": 3.}, samples=3)
        result = metric.finish()
        self.assertEqual(result["ce"], 2.5)
        self.assertEqual(result["operation_counts"]["optimizer_updates"], 2)
        self.assertEqual(result["sample_count"], 4)

    def test_registry_prevents_duplicate_fit_and_contract_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory) / "registry.sqlite")
            spec = study()
            registry.register(spec)
            registry.register(copy.deepcopy(spec))
            registry.claim("test", "fit", {"pid": 1, "start_ticks": 2})
            with self.assertRaises(ValueError):
                registry.claim("test", "fit", {"pid": 3})
            changed = copy.deepcopy(spec)
            changed["runs"][0]["seed"] = 3
            with self.assertRaises(ValueError):
                registry.register(changed)
            registry.finish("test", "fit", {"completed_steps": 3}, success=True)
            self.assertTrue(registry.release_ready("test"))
            self.assertEqual(registry.summary("test")["counts"],
                             [{"kind": "model_fit", "state": "complete", "records": 1}])
            registry.close()

    def test_failed_hpo_is_terminal_but_not_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory) / "registry.sqlite")
            registry.register(study())
            registry.claim("test", "fit", {"pid": 1})
            registry.finish("test", "fit", {"failure": "fixture"}, success=False)
            self.assertTrue(registry.all_terminal("test"))
            self.assertFalse(registry.release_ready("test"))
            self.assertFalse(registry.required_complete("test", ["fit"]))
            registry.close()

    def test_original_paper_gnnm_is_excluded(self):
        spec = study()
        spec["source"]["files"] = [{"path": "fixture.py", "bytes": 0, "sha256": "fixture"}]
        spec["dataset"] = "Amazon Ratings"
        spec["runs"][0]["method"] = "original_gnnm"
        with self.assertRaisesRegex(ValueError, "excluded"):
            validate(spec)
        spec["dataset"] = "WikiCS"
        validate(spec)

    def test_balanced_assignment_keeps_groups_intact(self):
        groups = [{"id": str(i), "estimated_seconds": value, "maximum_cuda_bytes": 1,
                   "run_ids": [str(i) + "/a", str(i) + "/b"]}
                  for i, value in enumerate((10, 9, 8, 4, 3, 2))]
        result = assign(groups)
        self.assertEqual(result["target_parallel_slots_per_gpu"], 2)
        self.assertEqual(set(result["estimated_gpu_seconds"]),
                         {"allocation/gpu0", "server77/gpu0", "server77/gpu1"})
        self.assertEqual(len(result["groups"]), 6)
        self.assertEqual(sum(result["estimated_gpu_seconds"].values()), 36)


if __name__ == "__main__":
    unittest.main()
