"""Stdlib fault injection for status commits; no tensors/runtime/data imports."""
import sys
sys.dont_write_bytecode = True
import unittest
from qualification_status import (PASS_STATUS, TWIN_PASS_STATUS, commit_twin, commit_qualification,
                                  invalidate_qualification, qualification_succeeded, validate_accounting)


def accounting():
    return {"inclusive_seconds": {}, "calls": {},
        "work_counts": {"explicit_timing_synchronizations": 1}, "enumerations": [],
        "wall_seconds": 1.0, "peak_cuda_allocated_bytes": 1, "peak_cuda_reserved_bytes": 1,
        "peak_host_rss_bytes": 1, "timings_are_nested_inclusive": True,
        "profiling_and_guards_charged": True, "final_cuda_synchronization_completed": True,
        "opaque_library_transfers_and_synchronizations": "fixture metadata only"}


class FakeMeter:
    def __init__(self, error=None, value=None):
        self.error, self.value = error, accounting() if value is None else value

    def receipt(self):
        if self.error is not None:
            raise self.error
        return self.value


def candidate(stage="gpu_parity"):
    return {"stage": stage, "status": "FAILED", "all_required_checks_passed": False,
        "stage_checks_passed": True, "accounting_complete": False,
        "setup_or_GPU_parity_accounting": accounting(), "final_process_accounting": accounting()}


class QualificationStatusTests(unittest.TestCase):
    def test_accounting_commit_success_requires_valid_status(self):
        value = candidate()
        commit_qualification(value, PASS_STATUS["gpu_parity"])
        self.assertTrue(qualification_succeeded(value))
        value["status"] = "FAILED"
        self.assertFalse(qualification_succeeded(value))
        self.assertFalse(qualification_succeeded({"all_required_checks_passed": True, "accounting_complete": True}))

    def test_twin_sync_failure_never_commits_success(self):
        value = {"status": "IN_PROGRESS", "batches": [None] * 17}
        with self.assertRaisesRegex(RuntimeError, "synthetic final CUDA sync failure"):
            commit_twin(value, FakeMeter(error=RuntimeError("synthetic final CUDA sync failure")))
        self.assertEqual(value["status"], "FAILED")
        self.assertIs(value["all_required_checks_passed"], False)
        self.assertIs(value["accounting_complete"], False)
        self.assertNotIn("accounting", value)
        self.assertTrue(value["failures"])

    def test_missing_accounting_field_invalidates_twin(self):
        incomplete = accounting(); del incomplete["peak_cuda_reserved_bytes"]
        value = {"status": "IN_PROGRESS"}
        with self.assertRaises(RuntimeError):
            commit_twin(value, FakeMeter(value=incomplete))
        self.assertNotEqual(value["status"], TWIN_PASS_STATUS)
        self.assertIs(value["all_required_checks_passed"], False)

    def test_GPU_final_accounting_failure_invalidates_numerical_success(self):
        value = candidate(); value["final_process_accounting"]["final_cuda_synchronization_completed"] = False
        with self.assertRaises(RuntimeError):
            commit_qualification(value, PASS_STATUS["gpu_parity"])
        self.assertFalse(qualification_succeeded(value))
        self.assertEqual(value["status"], "FAILED")
        self.assertTrue(value["stage_checks_passed"])

    def test_family_final_failure_invalidates_both_completed_twins(self):
        value = candidate("resource_epoch")
        value["twins"] = [{"mode": mode} for mode in ("private", "pooled_after_clamp")]
        for twin in value["twins"]:
            commit_twin(twin, FakeMeter())
        commit_qualification(value, PASS_STATUS["resource_epoch"])
        self.assertTrue(qualification_succeeded(value))
        invalidate_qualification(value, RuntimeError("synthetic late final accounting failure"), "fixture")
        self.assertFalse(qualification_succeeded(value))
        for twin in value["twins"]:
            self.assertEqual(twin["status"], "NOT_QUALIFIED_FAMILY_FAILED")
            self.assertIs(twin["all_required_checks_passed"], False)
            self.assertTrue(twin["native_epoch_work_and_accounting_completed_before_family_failure"])

    def test_diagnostic_retry_cannot_rescue_failure(self):
        value = candidate()
        invalidate_qualification(value, RuntimeError("synthetic accounting failure"), "fixture")
        value["diagnostic_process_accounting_after_failure"] = accounting()
        with self.assertRaises(RuntimeError):
            commit_qualification(value, PASS_STATUS["gpu_parity"])
        self.assertFalse(qualification_succeeded(value))

    def test_nonfinite_accounting_and_absent_sync_fail(self):
        for key, replacement in (("wall_seconds", float("nan")), ("final_cuda_synchronization_completed", False)):
            value = accounting(); value[key] = replacement
            with self.assertRaises(RuntimeError):
                validate_accounting(value)

    def test_failure_retains_primary_and_secondary_conditions(self):
        value = candidate()
        try:
            try:
                raise ValueError("synthetic primary workload failure")
            except ValueError:
                raise RuntimeError("synthetic secondary sync failure")
        except RuntimeError as error:
            invalidate_qualification(value, error, "fixture")
        self.assertEqual([row["error_type"] for row in value["failures"][0]["errors"]], ["RuntimeError", "ValueError"])
        self.assertFalse(qualification_succeeded(value))


if __name__ == "__main__":
    require_no_runtime = not {"torch", "numpy", "pandas", "torch_sparse", "torch_scatter"} & set(sys.modules)
    if not require_no_runtime:
        raise RuntimeError("Fault injection imported numerical runtime")
    unittest.main()
