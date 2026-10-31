"""Stdlib source-readiness and decision/refusal regression checks; no real data."""
import argparse
import ast
import copy
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from admission import (HERE, PRIOR, CELLS, LOSSES, PLACEMENTS, binding_format, checked,
    choose_all_gains, prior_literal, relative_path, sources, write_new)


def rejects(operation):
    try:
        operation()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("Invalid source/reference/decision input was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = sources()
    for name in ("admission.py", "admitted_core.py", "runner.py", "verify_admission.py", "recovery.py"):
        ast.parse((HERE / name).read_text(), filename=name)
    rule = prior_literal("RULE")
    # Compile only the exact frozen stdlib mechanism function, importing no torch.
    tree = ast.parse((PRIOR / "lookahead_core.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mechanism_rule")
    scope = {"math": math, "RULE": rule, "LOSSES": LOSSES, "PLACEMENTS": PLACEMENTS}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(PRIOR / "lookahead_core.py"), "exec"), scope)
    base = {(s, f, loss, p): 0. for s, f in CELLS for loss in LOSSES for p in PLACEMENTS}
    patterns = [dict(base)]
    positive = {k: 0. if k[3] == "tied" else .002 for k in base}
    patterns.append(positive)
    hurt = dict(positive)
    hurt[0, 1., "own", "first"] = -.001
    patterns.append(hurt)
    margin = dict(positive)
    margin[1, .1, "pooled", "first"] = .001
    patterns.append(margin)
    separate = dict(base)
    for s, f in CELLS:
        separate[s, f, "own", "middle"] = .003
    patterns.append(separate)
    for gains in patterns:
        if choose_all_gains(gains) != scope["mechanism_rule"](gains):
            raise AssertionError("Decision arithmetic differs from exact v1 source")
    decision = choose_all_gains(positive)
    if any(row["candidate_for_separate_full_predictive_study"] != "first" for row in decision["loss_families"].values()):
        raise AssertionError("Tied placement ranking changed")
    if choose_all_gains(hurt)["loss_families"]["own"]["candidate_for_separate_full_predictive_study"] != "middle":
        raise AssertionError("Full-label adverse guard changed")
    if choose_all_gains(separate)["loss_families"]["pooled"]["abstain"] is not True:
        raise AssertionError("Loss families are not independent")
    from recovery import verify_technical_case, GUARD_ERROR, FAILURE_MESSAGE
    events = []
    for f in (.1, 1.):
        for s in range(3):
            for stage in ("probe", "replay"):
                label = "seed%d_label%d" % (s, int(100 * f))
                events.extend([{"UTC": "2026-09-30T10:15:45+00:00", "cell": label, "stage": stage + "_idle_guard",
                    "guard": {"status": "idle_no_compute_context", "GPU": {"memory_MiB": 0, "utilization_percent": 0, "compute_context_count": 0}}},
                    {"UTC": "2026-09-30T10:15:45+00:00", "cell": label, "stage": stage + "_started"},
                    {"UTC": "2026-09-30T10:15:46+00:00", "cell": label, "stage": stage + "_terminal", "exit_code": 0}])
    events[-1]["exit_code"] = 1
    failure = {"type": "RuntimeError", "message": FAILURE_MESSAGE,
        "remaining_cells_not_qualified": True, "retry_or_outcome_truncation": False,
        "completed_cells": [{"seed": s, "label_fraction": f} for f in (.1, 1.) for s in range(3) if (s, f) != (2, 1.)]}
    reason = {"UTC": "2026-09-30T10:24:01+00:00", "completed_probe_fits": 6, "completed_full_replays": 5,
        "blocked_cell": {"seed": 2, "label_fraction": 1.0, "stage": "replay"}, "guard_error": GUARD_ERROR,
        "child_or_replay_supervisor_created_by_failed_attempt": False, "automatic_retry": False,
        "scientific_source_or_masks_or_selection_rule_changed": False, "predictive_validation_or_test_scored": False,
        "preceding_queue_idle_snapshot": {"memory_MiB": 0, "utilization_percent": 0, "compute_context_count": 0}}
    verify_technical_case(failure, events, reason, "RuntimeError: " + GUARD_ERROR)
    other_failure = copy.deepcopy(failure)
    other_failure["message"] = "gradient replay failed"
    rejects(lambda: verify_technical_case(other_failure, events, reason, "RuntimeError: " + GUARD_ERROR))
    earlier_error = copy.deepcopy(events)
    earlier_error[2]["exit_code"] = 1
    rejects(lambda: verify_technical_case(failure, earlier_error, reason, "RuntimeError: " + GUARD_ERROR))
    child_created = dict(reason, child_or_replay_supervisor_created_by_failed_attempt=True)
    rejects(lambda: verify_technical_case(failure, events, child_created, "RuntimeError: " + GUARD_ERROR))
    predated = dict(reason, UTC="2026-09-30T10:15:40+00:00")
    rejects(lambda: verify_technical_case(failure, events, predated, "RuntimeError: " + GUARD_ERROR))
    if (HERE / "admitted_core.py").read_bytes() != (HERE.parent / "objective_lookahead_molhiv_admitted_v3/admitted_core.py").read_bytes():
        raise AssertionError("V4 changes the actual K8 scientific engine")
    missing = dict(positive)
    missing.pop((2, 1., "pooled", "last"))
    rejects(lambda: choose_all_gains(missing))
    nonfinite = dict(positive)
    nonfinite[0, .1, "own", "first"] = math.inf
    rejects(lambda: choose_all_gains(nonfinite))
    nontied = dict(positive)
    nontied[0, .1, "own", "tied"] = 1e-12
    rejects(lambda: choose_all_gains(nontied))
    rejects(lambda: binding_format({"schema": "gnnm-molhiv-lookahead-real-binding-v3"}))
    rejects(lambda: binding_format({"schema": "old_or_SOURCE_ONLY"}))
    rejects(lambda: relative_path("../outside"))
    rejects(lambda: relative_path(str(HERE / "admission.py")))
    rejects(lambda: checked({"path": "objective_lookahead_molhiv_admitted_v4/admission.py", "sha256": "0" * 64}))
    rejects(lambda: checked({"path": "objective_lookahead_molhiv_admitted_v4/admission.py", "sha256": frozen["objective_lookahead_molhiv_admitted_v4/admission.py"], "extra": True}))
    if any(n == "torch" or n.startswith("torch.") or n.startswith("ogb") for n in sys.modules):
        raise AssertionError("Stdlib admission/readiness checks imported a target runtime")
    if frozen != sources():
        raise AssertionError("Source changed during readiness check")
    write_new(args.output, {"schema": "gnnm-molhiv-admitted-GPU-recovery-source-readiness-v4", "status": "passed",
        "sources": frozen, "syntax_parsed": True, "exact_v1_stdlib_rule_patterns": len(patterns),
        "complete48_seed_finite_tied_margin_harm_family_tie_rules_checked": True,
        "malformed_binding_and_confined_hash_references_refused": True,
        "narrow_recovery_constructed_metadata_cases_passed": True,
        "v3_K8_core_byte_identical": True,
        "torch_or_provider_imported": False, "dataset_or_checkpoint_loaded": False,
        "real_route_admission_or_GPU_execution_tested": False, "source_readiness_is_not_execution_qualification": True})
    print("Stdlib syntax/hash/rule/refusal readiness passed; no real or GPU execution performed.")


if __name__ == "__main__":
    main()
