"""Meaningful metadata tests: inventory/dependency custody and dormant output guards."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dormant_builder", HERE / "build_dormant_queues.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
CONFIG = json.loads((HERE / "TARGETS_AND_WEIGHTS.json").read_text())


def fixture():
    tasks = []
    for cell in range(27):
        for seed in range(10):
            name = "profile-%02d" % cell
            stem = f"{name}/split-{seed}/seed-{seed}"
            def fit(suffix, track, method, body=None):
                tid = stem + "/" + suffix
                row = {"id": tid, "kind": "fit", "cell_id": name, "family": ["node", "molecular", "link"][cell % 3],
                       "split": seed, "cohort_seed": seed, "body": body, "protocol_track": track,
                       "scientific_method_id": method, "group_id": stem + "/" + track,
                       "dependencies": [], "output_relative": "fits/" + b.digest(tid),
                       "runtime_job_hash": b.digest([tid, "runtime"]), "canonical_runtime_job_hash": b.digest([tid, "runtime"]),
                       "science_contract_hash": b.digest([tid, "science"]),
                       "source_descriptor": {"path": "/fixture/" + b.digest(tid) + ".json", "bytes": 12, "sha256": b.digest(tid)},
                       "worker": "allocation/gpu0", "extra_canonical_field": {"must_remain_bound": True}}
                tasks.append(row)
                return row
            fit("source-native", "native_replica", "native_single", 0)
            natives = [fit(f"native-body-{body}", "long_convergence", "native_single", body) for body in range(4)]
            for arm in range(18 if cell * 10 + seed < 20 else 19):
                fit(f"shared-{arm}", "long_convergence", "shared-example")
            tid = stem + "/native-I4/aggregate-own-selected-VALID"
            tasks.append({"id": tid, "kind": "aggregate_I4_exports", "cell_id": name,
                          "protocol_track": "long_convergence", "group_id": stem + "/long_convergence",
                          "science_contract_hash": b.digest(tid), "runtime_job_hash": None,
                          "dependencies": [t["id"] for t in natives], "M1_exact_reuse": natives[0]["id"],
                          "output_relative": "banks/" + b.digest(tid), "worker": "allocation/gpu0"})
    return {"schema": b.SCHEMA, "tasks": tasks}


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = fixture()

    def test_full6730_coverage_hashes_track_merge_and_dependency_order(self):
        original = b.digest(self.plan)
        queues, summary = b.build_views(self.plan, CONFIG)
        self.assertEqual(summary["canonical_counts"], {"tasks": 6730, "fits": 6460, "exports": 270})
        self.assertEqual(summary["canonical_groups"], 540)
        self.assertEqual(summary["comparison_cohorts"], 270)
        self.assertEqual(b.digest(self.plan), original)
        index = {t["id"]: t for t in self.plan["tasks"]}
        owners = {}
        for worker, rows in queues.items():
            seen = set()
            for row in rows:
                t = row["canonical_task"]
                self.assertTrue(set(t["dependencies"]) <= seen)
                self.assertEqual(row["canonical_task_sha256"], b.digest(index[t["id"]]))
                self.assertEqual(t, {k: index[t["id"]][k] for k in b.COPY_FIELDS if k in index[t["id"]]})
                seen.add(t["id"])
                self.assertNotIn(t["id"], owners)
                owners[t["id"]] = worker
        self.assertEqual(set(owners), set(index))
        for t in self.plan["tasks"]:
            self.assertTrue(all(owners[d] == owners[t["id"]] for d in t["dependencies"]))
        for stratum in summary["strata"]:
            self.assertLessEqual(max(stratum["cohorts_by_worker"].values()) - min(stratum["cohorts_by_worker"].values()), 1)
        self.assertEqual(sum(w["relative_work_ticks"] for w in summary["workers"]), 1350 * 20 + 5110 * 80 + 270)
        self.assertFalse(summary["launch_admitted"])
        self.assertFalse(summary["time_balance_or_ETA_claim"])

    def test_deterministic_under_input_permutation(self):
        alternate = copy.deepcopy(self.plan)
        random.Random(78).shuffle(alternate["tasks"])
        self.assertEqual(b.build_views(self.plan, CONFIG), b.build_views(alternate, CONFIG))

    def reject(self, plan, text):
        with self.assertRaisesRegex(ValueError, text):
            b.build_views(plan, CONFIG)

    def test_duplicate_id_missing_task_and_unknown_dependency_reject(self):
        p = copy.deepcopy(self.plan); p["tasks"][1] = copy.deepcopy(p["tasks"][0])
        self.reject(p, "Duplicate/invalid task ID")
        p = copy.deepcopy(self.plan); p["tasks"].pop()
        self.reject(p, "inventory counts")
        p = copy.deepcopy(self.plan); p["tasks"][0]["dependencies"] = ["foreign"]
        self.reject(p, "Unknown dependency")

    def test_cross_cohort_dependency_and_cycle_reject(self):
        p = copy.deepcopy(self.plan)
        p["tasks"][0]["dependencies"] = [next(t["id"] for t in p["tasks"] if t.get("cohort_seed") == 1)]
        self.reject(p, "Dependency crosses merged cohort")
        p = copy.deepcopy(self.plan)
        p["tasks"][0]["dependencies"] = [p["tasks"][1]["id"]]
        p["tasks"][1]["dependencies"] = [p["tasks"][0]["id"]]
        self.reject(p, "Dependency cycle")

    def test_I4_order_M1_identity_and_export_cohort_mismatch_reject(self):
        for operation, message in [("reverse", "ordered I4"), ("M1", "M1 must reuse"), ("seed", "Export cohort field")]:
            p = copy.deepcopy(self.plan)
            t = next(t for t in p["tasks"] if t["kind"] != "fit")
            if operation == "reverse": t["dependencies"].reverse()
            elif operation == "M1": t["M1_exact_reuse"] = t["dependencies"][1]
            else: t["cohort_seed"] = 999
            self.reject(p, message)

    def test_input_pin_failures_and_existing_output_make_no_new_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inp = root / "canonical.json"; inp.write_bytes(b.canonical(self.plan))
            config = copy.deepcopy(CONFIG); config["canonical_plan"] = b.file_pin(inp)
            output = root / "out"
            wrong = copy.deepcopy(config); wrong["canonical_plan"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Input bytes/hash"):
                b.generate(inp, output, wrong, {})
            self.assertFalse(output.exists())
            output.mkdir(); marker = output / "keep"; marker.write_text("original")
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                b.generate(inp, output, config, {})
            self.assertEqual(list(output.iterdir()), [marker])
            self.assertEqual(marker.read_text(), "original")

    def test_cli_writes_three_dormant_lists_and_source_tamper_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); package = root / "source"; package.mkdir()
            (package / "build_dormant_queues.py").write_bytes((HERE / "build_dormant_queues.py").read_bytes())
            inp = root / "plan.json"; inp.write_bytes(b.canonical(self.plan))
            config = copy.deepcopy(CONFIG); config["canonical_plan"] = b.file_pin(inp)
            (package / "TARGETS_AND_WEIGHTS.json").write_bytes(b.canonical(config))
            entries = [dict(b.file_pin(p), path=p.name) for p in sorted(package.iterdir())]
            manifest = package / "MANIFEST.json"; manifest.write_bytes(b.canonical({"files": entries}))
            pin = b.file_pin(manifest)
            old_here = b.HERE
            b.HERE = package
            try:
                output = root / "queues"
                with contextlib.redirect_stdout(io.StringIO()):
                    b.main(["--output", str(output), "--source-manifest-sha256", pin["sha256"]])
                self.assertEqual(len(list(output.glob("QUEUE_*.json"))), 3)
                result = json.loads((output / "SUMMARY.json").read_text())
                self.assertEqual(sum(w["tasks"] for w in result["workers"]), 6730)
                for row in json.loads((output / "MANIFEST.json").read_text())["files"]:
                    actual = b.file_pin(output / row["path"])
                    self.assertEqual(actual["sha256"], row["sha256"])
                for path in output.glob("QUEUE_*.json"):
                    queue = json.loads(path.read_text())
                    self.assertFalse(queue["launch_admitted"])
                    self.assertFalse(queue["dispatch_supported"])
                (package / "TARGETS_AND_WEIGHTS.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Source member changed"):
                    b.source_check(package, pin["sha256"])
            finally:
                b.HERE = old_here


if __name__ == "__main__":
    unittest.main(verbosity=2)
