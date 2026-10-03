"""Stdlib fail-closed accounting/status commit for admitted runtime results."""
import math

PASS_STATUS = {"gpu_parity": "GPU_TRAIN_PARITY_PASSED",
               "resource_epoch": "BOTH_COMPLETE_NATIVE_RESOURCE_EPOCHS_FINITE"}
TWIN_PASS_STATUS = "COMPLETE_NATIVE_EPOCH_FINITE"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_accounting(accounting):
    require(type(accounting) is dict, "Required accounting receipt absent")
    for name in ("inclusive_seconds", "calls", "work_counts"):
        require(type(accounting.get(name)) is dict, "Required accounting map absent: " + name)
        for value in accounting[name].values():
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "Invalid accounting value in " + name)
    require(type(accounting.get("enumerations")) is list, "Required enumeration accounting absent")
    for name in ("wall_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_host_rss_bytes"):
        value = accounting.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Required finite timing/memory accounting absent: " + name)
    for name in ("timings_are_nested_inclusive", "profiling_and_guards_charged", "final_cuda_synchronization_completed"):
        require(accounting.get(name) is True, "Required accounting/synchronization condition absent: " + name)
    require(accounting["work_counts"].get("explicit_timing_synchronizations", 0) >= 1,
            "Final synchronized accounting boundary was not recorded")
    require(isinstance(accounting.get("opaque_library_transfers_and_synchronizations"), str)
            and accounting["opaque_library_transfers_and_synchronizations"], "Opaque-library accounting boundary absent")
    return accounting


def error_chain(error):
    entries, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        entries.append({"error_type": type(error).__name__, "error_condition": str(error)[:4000]})
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return entries


def record_failure(target, error, category):
    target["status"] = "FAILED"
    target["all_required_checks_passed"] = False
    target.setdefault("failures", []).append({"category": category, "errors": error_chain(error)})


def commit_twin(result, meter):
    # A successful status is assigned only after receipt construction, its
    # final CUDA synchronization, and all promised accounting fields succeed.
    result["all_required_checks_passed"] = False
    result["accounting_complete"] = False
    try:
        accounting = validate_accounting(meter.receipt())
    except Exception as error:
        record_failure(result, error, "twin_final_accounting_or_CUDA_sync")
        raise
    result["accounting"] = accounting
    result["accounting_complete"] = True
    result["status"] = TWIN_PASS_STATUS
    result["all_required_checks_passed"] = True


def invalidate_qualification(receipt, error, category):
    record_failure(receipt, error, category)
    receipt["accounting_complete"] = False
    receipt["fallback_or_scope_reduction_attempted"] = False
    for twin in receipt.get("twins", []):
        if twin.get("status") == TWIN_PASS_STATUS:
            twin["native_epoch_work_and_accounting_completed_before_family_failure"] = True
        # Completed work is retained, but no constituent is left qualified
        # after a family/final synchronization or accounting failure.
        twin["status"] = "NOT_QUALIFIED_FAMILY_FAILED"
        twin["all_required_checks_passed"] = False
        twin["family_qualification_failed"] = True


def commit_qualification(receipt, candidate_status):
    receipt["all_required_checks_passed"] = False
    receipt["accounting_complete"] = False
    try:
        require(receipt["stage"] in PASS_STATUS, "Unknown qualification stage")
        require(candidate_status == PASS_STATUS.get(receipt["stage"]), "Invalid qualification candidate status")
        require(receipt.get("stage_checks_passed") is True, "Numerical/work checks are incomplete")
        require(not receipt.get("failures"), "Earlier failure prevents qualification")
        validate_accounting(receipt.get("setup_or_GPU_parity_accounting"))
        validate_accounting(receipt.get("final_process_accounting"))
        if receipt["stage"] == "resource_epoch":
            twins = receipt.get("twins", [])
            require(len(twins) == 2, "Incomplete twin family")
            for twin in twins:
                require(twin.get("status") == TWIN_PASS_STATUS and twin.get("all_required_checks_passed") is True
                        and twin.get("accounting_complete") is True and not twin.get("failures"),
                        "Twin was not completely accounted and qualified")
                validate_accounting(twin.get("accounting"))
    except Exception as error:
        invalidate_qualification(receipt, error, "qualification_commit")
        raise
    receipt["accounting_complete"] = True
    receipt["status"] = candidate_status
    receipt["all_required_checks_passed"] = True


def qualification_succeeded(receipt):
    return (receipt.get("stage") in PASS_STATUS
            and receipt.get("status") == PASS_STATUS.get(receipt.get("stage"))
            and receipt.get("all_required_checks_passed") is True
            and receipt.get("accounting_complete") is True
            and not receipt.get("failures"))
