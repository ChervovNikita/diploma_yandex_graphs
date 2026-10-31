"""Search closure, failed seeds and prospective-domain membership."""
import copy
import unittest

from adapters.search import select_configuration
from run_study import resolve_selected_config


class FakeRegistry:
    def __init__(self, rows):
        self.rows = rows
    def run(self, study, run):
        return self.rows[run]


class SearchTests(unittest.TestCase):
    def fixture(self):
        candidates = [{"config_id": name, "parameters": {"learning_rate": lr},
                       "run_ids": [name+"0", name+"1"]} for name, lr in (("a", .001), ("b", .01))]
        run = {"config": {"ranking_metric": "accuracy", "ranking_direction": "max", "configurations": candidates}}
        rows = {name+str(seed): {"state": "complete", "contract": {
            "config": {"stage": "search"}, "budget": {"maximum": 10}}, "result": {
                "completed_steps": 10, "selected": {"validation": {"accuracy": score,
                      "pooled_nll": loss, "objective": loss}}}}
                for name, score, loss in (("a", .8, .5), ("b", .8, .4)) for seed in range(2)}
        return run, rows

    def select(self, run, rows):
        return select_configuration(project_root=None, study={"study_id": "fixture"}, run=run,
                                    output_root=None, output_dir=None, registry=FakeRegistry(rows), device="cpu")

    def test_whole_seed_cell_and_loss_tie_break(self):
        run, rows = self.fixture()
        result = self.select(run, rows)
        self.assertEqual(result["selected_configuration"]["config_id"], "b")
        rows["a1"]["state"] = "running"
        with self.assertRaises(ValueError):
            self.select(run, rows)

    def test_failed_config_is_recorded_and_not_cherry_picked_across_seeds(self):
        run, rows = self.fixture()
        rows["b1"].update(state="failed", result={"failure": "engineering fixture"})
        result = self.select(run, rows)
        self.assertEqual(result["selected_configuration"]["config_id"], "a")
        self.assertEqual(len(result["failed_search_trajectories"]), 1)

    def test_resolved_choice_must_belong_to_the_approved_search_domain(self):
        run, rows = self.fixture()
        result = self.select(run, rows)
        rows["selection"] = {"state": "complete", "contract": {"kind": "selection", **run}, "result": result}
        fit = {"config": {"selected_config_from": "selection", "learning_rate": .123}}
        actual = resolve_selected_config({"study_id": "fixture"}, fit, FakeRegistry(rows))
        self.assertEqual(actual["config"]["learning_rate"], .01)
        self.assertEqual(fit["config"]["learning_rate"], .123)
        rows["selection"]["result"]["selected_configuration"]["parameters"]["learning_rate"] = 99
        with self.assertRaises(ValueError):
            resolve_selected_config({"study_id": "fixture"}, fit, FakeRegistry(rows))

if __name__ == "__main__":
    unittest.main()
