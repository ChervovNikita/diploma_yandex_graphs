"""Tiny synthetic CPU boundary-source/optimizer/RNG/export engineering checks."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import test_polyformer as fixture
from test_polyformer_matrix import fixture_inputs
from adapters import polyformer, polyformer_boundary as boundary
from adapters.polyformer_endpoints import export_bank, independent_contracts
from compile_polyformer_matrix import make_study
from compile_polyformer_boundary_matrix import append_boundary, boundary_domain
from engine import capture_checkpoint, restore_checkpoint, run_trajectory


class BoundaryDomainTests(unittest.TestCase):
    def test_40_effective_points_complete_controls_and_actual_reference_cohorts(self):
        custody, recipe, source, profile = fixture_inputs()
        root = Path.cwd()
        old, counts, banks, domains = make_study(root, custody, root / "synthetic-data", recipe, source, profile,
                                               "native_fixed", "old/runtime")
        previous = {"counts": counts, "served_banks": banks, "search_domains": domains}
        result, total, served, newdomains = append_boundary(old, previous, "fresh/runtime", [])
        self.assertEqual(len(boundary_domain(old)), 40)
        self.assertEqual(total["initial_training_segments"] - counts["initial_training_segments"], 175)
        self.assertEqual(total["all_run_records"] - counts["all_run_records"], 204)
        self.assertEqual(total["full_budget_optimizer_updates"] - counts["full_budget_optimizer_updates"], 350000)
        self.assertEqual(total["planned_TRAIN_route_forwards"] - counts["planned_TRAIN_route_forwards"], 848000)
        self.assertEqual(total["planned_validation_route_forwards"] - counts["planned_validation_route_forwards"], 848000)
        for prior, actual in zip(old["runs"], result["runs"]):
            prior = copy.deepcopy(prior); actual = copy.deepcopy(actual)
            prior.pop("adapter"); actual.pop("adapter")
            self.assertEqual(prior, actual)
        for mode in (boundary.M4, boundary.M1):
            self.assertEqual(newdomains[mode]["configuration_count"], 40)
        cohort = {(r["split"], r["seed"]) for r in result["runs"] if r["kind"] == "ensemble_assembly"}
        for comp in result["comparisons"]:
            for method in comp["references"] + comp["candidates"]:
                self.assertEqual({(r["split"], r["seed"]) for r in result["runs"]
                                  if r["kind"] == "ensemble_assembly" and r["method"] == method}, cohort)
        group = next(b for b in served if b["arm"] == boundary.M1 + "_I4")
        fits = {r["run_id"]: r for r in result["runs"]}
        independent_contracts(result, [fits[i] for i in group["fit_parents"]])
        self.assertEqual(group["routes"], 4)


@unittest.skipIf(fixture.torch is None or not fixture.SOURCES_PRESENT or not fixture.PYG_PRESENT,
                 "Existing allocation Torch/PyG and pinned sources required")
class BoundaryNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = fixture.torch
        cls.previous_threads = cls.t.get_num_threads(); cls.t.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        cls.t.set_num_threads(cls.previous_threads)

    def bank(self, members=4, seed=1009, native=None):
        t = self.t; x = t.arange(24, dtype=t.float32).reshape(8, 3) / 24
        return boundary.build_bank(t, fixture.PROJECT, [x, x * .5, x * .25],
            native=native or fixture.SMALL_NATIVE, classes=2, dataset="synthetic-engineering", members=members,
            initialization_seed=seed, dropout_seeds=[301010 + i * 1009 for i in range(members)])

    def adapter(self, directory, members=4, body=0, group=None):
        t = self.t; root = fixture.PROJECT
        inputs = {}
        for key, value in {"graph": {"x": t.arange(24, dtype=t.float32).reshape(8, 3) / 24,
            "edge_index": t.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])},
            "train": {"ids": t.tensor([0, 1, 2, 3]), "y": t.tensor([0, 0, 1, 1])},
            "select": {"ids": t.tensor([4, 5]), "y": t.tensor([0, 1])}}.items():
            path = directory / (key + ".pt"); t.save(value, path); inputs[key] = str(path.relative_to(root))
        mode = boundary.M4 if members == 4 else boundary.M1
        config = {"native": dict(fixture.SMALL_NATIVE), "classes": 2, "node_inputs": inputs,
            "polyformer_mode": mode, "members": members, "fusion": "mean_logits", "train_views": 1,
            "initialization_seed": 1009 + 1000003 * body,
            "dropout_seeds": [301010 + 10000019 * body + i * 1009 for i in range(members)],
            "preprocessing": "pinned_mono_base", "learning_rate": .005, "attn_lr": .0005,
            "weight_decay": 0, "attn_weight_decay": 0, "loss": {"variant": "own", "coefficients": {}},
            "factor_initialization": "source_rsb_xavier", "factor_seed": None,
            "boundary_source": copy.deepcopy(boundary.BOUNDARY_SOURCE), "is_original_gnnm": members == 4,
            "optimizer": {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8,
                          "grouping": "attnmodule_vs_other", "weight_decay": 0}}
        if group: config["independence_group"] = group
        row = {"run_id": "body" + str(body) if members == 1 else "shared4", "kind": "model_fit",
            "method": mode, "split": 0, "seed": 101, "body": body, "config": config, "parents": [],
            "adapter": str((fixture.PACKAGE / "adapters/polyformer_boundary.py").relative_to(root)) + ":make_adapter",
            "budget": {"maximum": 2, "unit": "optimizer_updates", "policy": "full_budget"},
            "selection": {"metric": "accuracy", "direction": "max", "ties": "first", "save_every": 1}}
        study = {"study_id": "synthetic-boundary", "dataset": "synthetic-engineering", "runs": [],
                 "data": {"roles": "tiny synthetic disjoint four roles"}, "source": {}}
        return boundary.make_adapter(torch=t, project_root=root, study=study, run=row, device="cpu"), row, study

    def test_exact_source_construction_two_replacements_and_plain_interior(self):
        t = self.t; outside = t.get_rng_state().clone(); bank = self.bank()
        self.assertTrue(t.equal(t.get_rng_state(), outside))
        with polyformer._constructor_seed(t, 1009, t.device("cpu")):
            args = polyformer._arguments(fixture.SMALL_NATIVE, 3, 2, "synthetic-engineering")
            body = polyformer.native_class(t, fixture.PROJECT)(None, args)
            prior = dict(body.named_parameters())
            block = boundary.source_class(t, fixture.PROJECT)
            first = block(3, 8, True, "cpu", 4); last = block(8, 2, False, "cpu", 4)
        for a, b in ((bank.body.lin1.block, first), (bank.body.lin3.block, last)):
            self.assertTrue(all(t.equal(a.state_dict()[k], v) for k, v in b.state_dict().items()))
            self.assertIsNone(a.W.bias); self.assertTrue(t.equal(a.S, t.ones_like(a.S)))
            self.assertTrue(t.equal(a.B, t.zeros_like(a.B)))
        self.assertTrue(set(bank.body.lin1.block.R.detach().flatten().tolist()) <= {-1., 1.})
        self.assertTrue(t.equal(bank.body.lin3.block.R, t.ones_like(bank.body.lin3.block.R)))
        current = dict(bank.body.named_parameters())
        self.assertTrue(all(t.equal(current[name], prior[name]) for name in bank.interior_parameter_names))
        dense = sum(isinstance(m, t.nn.Linear) for m in bank.body.modules())
        self.assertEqual(dense, sum(isinstance(m, t.nn.Linear) for m in body.modules()))
        self.assertEqual(bank.boundary_sites, ("lin1", "lin3")); self.assertEqual(bank.factor_sites, 0)
        self.assertFalse(any(type(m).__name__ == "FactorLinear" for m in bank.body.modules()))
        self.assertEqual(len({id(current[n]) for n in bank.interior_parameter_names}), len(bank.interior_parameter_names))
        bank.train(); logits, hidden = bank.forward_route(2)
        hidden.retain_grad(); (t.nn.functional.cross_entropy(logits[:4], t.tensor([0, 0, 1, 1])) + hidden.square().mean()).backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(bank.body.lin2.weight.grad)
        for name in ("lin1", "lin3"):
            block = getattr(bank.body, name).block
            for parameter in (block.R, block.S, block.B):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(t.equal(parameter.grad[:2], t.zeros_like(parameter.grad[:2])))
                self.assertTrue(t.equal(parameter.grad[3:], t.zeros_like(parameter.grad[3:])))

    def test_native_training_function_exact_update_and_grouped_AdamW_ownership(self):
        t = self.t
        with tempfile.TemporaryDirectory(prefix="boundary-update-", dir=fixture.PROJECT) as name:
            adapter, row, _ = self.adapter(Path(name)); other, _, _ = self.adapter(Path(name))
            env = {"torch": t, "F": t.nn.functional, "TABM_INITS": 4}
            polyformer._definitions(fixture.PROJECT / boundary.TRAIN_SOURCE["path"], ("train_step_tabm",), env)
            class NativeInterface(t.nn.Module):
                def __init__(self, bank): super().__init__(); self.bank = bank
                def forward(self, data, x, tabm_seed): return self.bank.forward_route(tabm_seed)[0]
            data = SimpleNamespace(x=t.empty(0), y=t.tensor([0, 0, 1, 1, 0, 1, 0, 1]),
                                   train_mask=t.tensor([True, True, True, True, False, False, False, False]))
            value = env["train_step_tabm"](NativeInterface(other.model), data, other.optimizer, False)
            record = adapter.train_step(1)
            self.assertAlmostEqual(record["objective"], value, places=7)
            self.assertTrue(all(t.equal(a, b) for a, b in zip(adapter.model.parameters(), other.model.parameters())))
            self.assertEqual(record["operation_counts"]["backward_calls"], 4)
            self.assertEqual(record["operation_counts"]["optimizer_updates"], 1)
            self.assertIsInstance(adapter.optimizer, t.optim.AdamW)
            grouped = [p for group in adapter.optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(grouped), len({id(p) for p in grouped}))
            self.assertEqual({id(p) for p in grouped}, {id(p) for p in adapter.model.parameters()})
            for (name, _), group in zip(adapter.model.named_parameters(), adapter.optimizer.param_groups):
                self.assertEqual(group["weight_decay"], 0)
                self.assertEqual(group["lr"], row["config"]["attn_lr"] if "attnmodule" in name else row["config"]["learning_rate"])

    def test_M1_unscaled_own_CE_and_private_RNG_checkpoint_replay(self):
        t = self.t
        with tempfile.TemporaryDirectory(prefix="boundary-state-", dir=fixture.PROJECT) as name:
            directory = Path(name); adapter, _, _ = self.adapter(directory, members=1)
            other, _, _ = self.adapter(directory, members=1)
            other.model.train(); other.optimizer.zero_grad(set_to_none=True)
            logits = other.model()[0]
            loss = t.nn.functional.cross_entropy(logits[other.train_ids], other.train_y); loss.backward(); other.optimizer.step()
            outside = t.get_rng_state().clone(); record = adapter.train_step(1)
            self.assertTrue(t.equal(outside, t.get_rng_state()))
            self.assertEqual(record["operation_counts"]["backward_calls"], 1)
            self.assertTrue(all(t.equal(a, b) for a, b in zip(adapter.model.parameters(), other.model.parameters())))
            metrics = adapter.validate(1); decision = adapter.observe_validation(1, metrics)
            path = directory / "own.pt"; t.save(capture_checkpoint(t, adapter, fit_id="own", step=1,
                validation=metrics, decision=decision), path)
            state = adapter.model.capture_runtime_state(); first = adapter.train_step(2)
            expected = {k: v.clone() for k, v in adapter.model.state_dict().items()}
            restore_checkpoint(t, path, adapter, expected_fit_id="own")
            actual = adapter.model.capture_runtime_state()
            self.assertTrue(t.equal(state["streams"][0]["cpu"], actual["streams"][0]["cpu"]))
            self.assertTrue(all(t.equal(v, actual["poly_attn_bias"][k]) for k, v in state["poly_attn_bias"].items()))
            second = adapter.train_step(2)
            self.assertEqual(first, second)
            self.assertTrue(all(t.equal(v, adapter.model.state_dict()[k]) for k, v in expected.items()))
            raw = t.tensor([[[8., 0.]], [[0., 2.]]])
            self.assertTrue(t.equal(adapter.probabilities(raw), raw.mean(0).softmax(-1)))
            self.assertFalse(t.allclose(adapter.probabilities(raw), raw.softmax(-1).mean(0)))

    def test_own_selected_M4_M1_I4_target_absent_export_and_body0_reuse(self):
        t = self.t; root = fixture.PROJECT
        with tempfile.TemporaryDirectory(prefix="boundary-export-", dir=root) as name:
            directory = Path(name); output = directory / "runs"; rows = {}; study = None
            for body, members in [(0, 4)] + [(i, 1) for i in range(4)]:
                adapter, row, current = self.adapter(directory, members=members, body=body, group="paid-i4" if members == 1 else None)
                if study is None: study = current
                study["runs"].append(row)
                result = run_trajectory(t, adapter, {"fit_id": row["run_id"], "budget_steps": 2, "budget_unit": "optimizer_updates",
                    "validation_interval": 1, "validate_initial": False, "prefixes": [], "inherited_fit_ids": [],
                    "adapter_spec": {"synthetic_fixture": True}}, output / study["study_id"] / row["run_id"])
                rows[row["run_id"]] = {"state": "complete", "contract": row, "result": result}
            independent_contracts(study, [rows["body" + str(i)]["contract"] for i in range(4)])
            endpoint = {}
            for role, identifier in (("cal", 6), ("evaluate", 7)):
                path = directory / (role + "_ids.pt"); t.save(t.tensor([identifier]), path)
                endpoint[role + "_ids"] = str(path.relative_to(root))
                endpoint[role + "_targets"] = str((directory / (role + "_targets.pt")).relative_to(root))
            class Registry:
                def run(self, study_id, identifier): return rows[identifier]
            outputs = {}
            for label, fits in (("M4", ["shared4"]), ("M1", ["body0"]), ("I4", ["body" + str(i) for i in range(4)])):
                out = directory / label; out.mkdir()
                run = {"parents": fits, "config": {"fit_parents": fits, "fusion": "mean_logits", "endpoint_inputs": endpoint,
                    "singleton_reference": "fixed_body0" if label == "M1" else None}}
                result = export_bank(project_root=root, study=study, run=run, output_root=output, output_dir=out,
                                     registry=Registry(), device="cpu")
                self.assertEqual(result["fusion"], "mean_logits")
                self.assertFalse(result["CAL_targets_read"]); self.assertFalse(result["evaluation_targets_read"])
                self.assertEqual(len(result["ancestry"]), len(fits))
                outputs[label] = t.load(root / result["files"]["cal"]["path"], weights_only=True)["logits"]
            self.assertEqual(tuple(outputs["M4"].shape), (4, 1, 2))
            self.assertEqual(tuple(outputs["I4"].shape), (4, 1, 2))
            self.assertTrue(t.equal(outputs["M1"], outputs["I4"][:1]))
            self.assertFalse((directory / "cal_targets.pt").exists()); self.assertFalse((directory / "evaluate_targets.pt").exists())
            bad = copy.deepcopy([rows["body" + str(i)]["contract"] for i in range(4)])
            bad[3]["config"]["dropout_seeds"] = bad[0]["config"]["dropout_seeds"]
            with self.assertRaisesRegex(ValueError, "dropout"):
                independent_contracts(study, bad)


if __name__ == "__main__":
    unittest.main()
