"""No torch, native construction, data load, forward, backward, or replay."""
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import session_extension as extension
import study


class FakeSession:
    def __init__(self):
        self.config = dict(extension.NATIVE_CONFIG)
        self.procedure = "gnnm_boundary_native"
        self.model = types.SimpleNamespace()
        self.optimizer = object()
        self.calls = []
        self.loaded = None

    def metadata(self):
        return {"native": "preserved"}

    def state(self):
        return {"native": "state", "model": {"route_banks_and_residual": "registered by retained parents"}}

    def load_state(self, value):
        self.loaded = value


def fake_load(root, relative):
    def transform(session, job):
        session.calls.append((relative, copy.deepcopy(job)))
        previous = session.metadata
        if relative.startswith("molecular/"):
            session.extension_metadata = {"ingredient": job["method_settings"], "method_id": job["procedure"]}
            metadata = copy.deepcopy(session.extension_metadata)
            session.train_update = lambda: "never called"
        else:
            session.model.route_bn_state = "registered fake"
            metadata = {"normalization": "private", "banks_sha256": job["method_settings"]["banks_sha256"]}
        session.metadata = lambda: {**previous(), "candidate_extension": copy.deepcopy(metadata)}
        return session
    return types.SimpleNamespace(transform=transform)


class SourceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(Path(__file__).with_name("SPEC.json").read_text())

    def test_finite_descriptors_and_native_route_unchanged(self):
        planned = study.validate_structure(self.spec)
        self.assertEqual(len(planned), 9)
        parent = json.loads(Path(__file__).parents[1].joinpath("pilot_20261015_bn_state_v1/SPEC.json").read_text())
        changes = {"id", "procedure", "method_settings", "session_transform"}
        for job in planned:
            baseline = next(j for j in parent["jobs"] if j["seed"] == job["seed"])
            self.assertEqual({k: v for k, v in job.items() if k not in changes},
                             {k: v for k, v in baseline.items() if k not in changes})

    def test_composition_metadata_and_checkpoint_identity(self):
        for method in extension.METHODS:
            with self.subTest(method=method):
                job = next(j for j in study.jobs(self.spec) if j["procedure"] == method)
                session = FakeSession()
                model, optimizer = session.model, session.optimizer
                with patch.object(extension, "_load", fake_load):
                    extension.transform(session, job)
                self.assertIs(session.model, model)
                self.assertIs(session.optimizer, optimizer)
                self.assertEqual([r[0] for r in session.calls],
                                 ["molecular/session_extension.py", "private_bn_state_v3/session_extension.py"])
                self.assertEqual(session.calls[0][1]["method_settings"], job["method_settings"]["ingredient"])
                meta = session.metadata()["candidate_extension"]
                self.assertIn("ingredient", meta)
                self.assertIn("normalization", meta)
                self.assertEqual(session.metadata()["native"], "preserved")
                state = session.state()
                session.load_state(state)
                self.assertNotIn("bn_geometry_correction_binding", session.loaded)
                self.assertIn("bn_geometry_correction_binding", state)
                for changed in ("seed", "method", "source_sha256"):
                    altered = copy.deepcopy(state)
                    altered["bn_geometry_correction_binding"][changed] = "different"
                    with self.assertRaises(ValueError):
                        session.load_state(altered)

    def test_settings_and_freshness_rejections(self):
        base = study.jobs(self.spec)[0]
        variants = []
        for key, value in (("seed", 999), ("family", "node"), ("TEST_access", True), ("procedure", "unlisted")):
            job = copy.deepcopy(base); job[key] = value; variants.append(job)
        job = copy.deepcopy(base); job["method_settings"]["ingredient"]["adapter_rank"] = 7; variants.append(job)
        for job in variants:
            with self.assertRaises(ValueError):
                extension.transform(FakeSession(), job)
        session = FakeSession(); session.model.route_bn_state = "already attached"
        with self.assertRaises(ValueError):
            extension.transform(session, base)

    def test_signed_factorial_interaction_and_missing_seed(self):
        rows = []
        for seed in (101, 202, 303):
            for n in (0, 1):
                for g in (0, 1):
                    for c in (0, 1):
                        y = 60 + 2 * n + 3 * g + 5 * c + (-4 + 7 * n) * g * c
                        rows.append({"seed": seed, "cell": [n, g, c],
                                     "values": {"AUC_percent": y, "BCE": -y / 100}})
        report = study.summarize(rows, [101, 202, 303])
        values = {(r["name"], r["metric"]): r["mean"] for r in report}
        self.assertEqual(values[("J_shared", "AUC_percent")], -4)
        self.assertEqual(values[("J_private", "AUC_percent")], 3)
        self.assertEqual(values[("J_private_minus_J_shared", "AUC_percent")], 7)
        self.assertAlmostEqual(values[("J_private_minus_J_shared", "BCE")], -.07)
        rows[-1]["values"] = None
        report = study.summarize(rows, [101, 202, 303])
        target = next(r for r in report if r["name"] == "J_private_minus_J_shared" and r["metric"] == "AUC_percent")
        self.assertIsNone(target["mean"])
        self.assertEqual(target["missing_seeds"], [303])
        self.assertEqual(study.cell_summaries(rows, [101, 202, 303])[-1]["metrics"]["BCE"]["mean"], None)

    def test_closed_valid_scalar_law(self):
        good = {"status": "complete", "selected_VALID": {"role": "VALID", "records": 151,
                "complete_source_VALID": True, "TEST_accessed": False, "loss": 0.8, "metrics": {"rocauc": .7}}}
        self.assertEqual(study.scalar_result(good), {"AUC_percent": 70.0, "BCE": .8})
        good["selected_VALID"]["role"] = "TEST"
        with self.assertRaises(ValueError):
            study.scalar_result(good)
        self.assertIsNone(study.scalar_result({"status": "failed"}))


if __name__ == "__main__":
    program = unittest.main(exit=False)
    if any(name in sys.modules for name in ("torch", "numpy", "torch_geometric", "ogb")):
        raise RuntimeError("Cheap source checks unexpectedly imported a numerical/model library")
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
