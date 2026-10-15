import tempfile
import unittest
from pathlib import Path
import admission
import dependency_plan as dep


class DependencyTests(unittest.TestCase):
    def test_deduplicate_and_conflict(self):
        r = dep.row("test/a.py", {"bytes": 1, "sha256": "a" * 64}, True)
        self.assertEqual(dep.deduplicate([r, r]), [r])
        with self.assertRaises(ValueError):
            dep.deduplicate([r, {**r, "sha256": "b" * 64}])

    def test_path_escape_rejected(self):
        for path in ("../../x", "/tmp/x", "experiments_iclr/postsubmission_20260930/../x"):
            with self.assertRaises(ValueError):
                dep.relative(path)

    def test_tampered_metadata_pin_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p = Path(tmp) / "input.json"
            p.write_text("{}")
            pin = dep.pin(p)
            p.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                dep.read(p, pin)

    def test_provider_and_final_role_scopes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p = Path(tmp)
            for name in ("dataset/raw/ind.test.index", "dataset/split/time/test.pt", "dataset/split/time/train.pt",
                         "dataset/processed/pre_filter.pt", "dataset/RELEASE_v1.txt", "dataset/results/score.json"):
                f = p / name
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"opaque")
            dirs = [dep.PHASE + "/dataset"]
            provider = dep.supplemental_inventory(p, dirs)
            role = dep.role_index_inventory(p, dirs)
            self.assertEqual(len(provider), 3)
            self.assertEqual(len(role), 2)
            self.assertTrue(any(r["source_relative"].endswith("test.pt") for r in role))
            self.assertFalse(any("score.json" in r["source_relative"] for r in provider + role))


class AdmissionTests(unittest.TestCase):
    def disk(self):
        return {"free_disk_bytes": 1000, "floor_bytes": 100, "pending_transfer_net_bytes": 100,
                "transfer_temporary_extra_bytes": 20, "peer_growth_allowance_bytes": 50,
                "current_campaign_retained_bytes": 5000, "campaign_retained_cap_bytes": 6000,
                "active": [{"id": "a", "additional_to_peak_bytes": 200, "additional_retained_bytes": 100}],
                "candidate": {"id": "b", "additional_to_peak_bytes": 300, "additional_retained_bytes": 200}}

    def test_incremental_disk_without_double_reserving_retained(self):
        r = admission.disk_admission(self.disk())
        self.assertEqual(r["recommendation"], "CAPACITY_ONLY_PASS")
        self.assertEqual(r["required_free_bytes"], 770)
        self.assertEqual(r["retained_after_active_and_candidate_bytes"], 5300)
        self.assertFalse(r["launch_admitted"])

    def test_disk_unknown_cap_and_low_free(self):
        d = self.disk()
        d["candidate"]["additional_to_peak_bytes"] = None
        self.assertEqual(admission.disk_admission(d)["recommendation"], "HOLD_FOR_PROFILE")
        d = self.disk()
        d.update(free_disk_bytes=700, campaign_retained_cap_bytes=5100)
        self.assertEqual(len(admission.disk_admission(d)["reasons"]), 2)

    def pair(self):
        return {"jobs": [{"solo_work": 100, "solo_seconds": 10, "pair_work": 70, "pair_seconds": 10},
                         {"solo_work": 20, "solo_seconds": 10, "pair_work": 14, "pair_seconds": 10}],
                **{k: True for k in ("source_and_work_identical", "lifecycle_reused_or_passed", "resource_headroom_pass",
                                     "disk_pass", "peer_identity_stable", "both_active_throughout_pair_window")}}

    def test_heterogeneous_pair_work_normalization(self):
        r = admission.pair_evaluation(self.pair())
        self.assertAlmostEqual(r["aggregate_solo_equivalent_speedup"], 1.4)
        self.assertEqual(r["recommendation"], "PAIR_PROFILE_CANDIDATE")
        p = self.pair()
        p["jobs"][0]["pair_work"] = 50
        p["jobs"][1]["pair_work"] = 10
        self.assertEqual(admission.pair_evaluation(p)["recommendation"], "KEEP_ONE_CHILD")

    def test_invalid_pair_window_and_failed_correctness(self):
        p = self.pair()
        p["jobs"][0]["pair_seconds"] = 11
        with self.assertRaises(ValueError):
            admission.pair_evaluation(p)
        p = self.pair()
        p["source_and_work_identical"] = False
        self.assertEqual(admission.pair_evaluation(p)["recommendation"], "KEEP_ONE_CHILD")


if __name__ == "__main__":
    unittest.main()
