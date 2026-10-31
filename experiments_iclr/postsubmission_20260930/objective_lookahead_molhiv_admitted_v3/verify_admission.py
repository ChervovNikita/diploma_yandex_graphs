"""Stdlib source-readiness and decision/refusal regression checks; no real data."""
import argparse
import ast
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
    for name in ("admission.py", "admitted_core.py", "runner.py", "verify_admission.py"):
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
    rejects(lambda: checked({"path": "objective_lookahead_molhiv_admitted_v3/admission.py", "sha256": "0" * 64}))
    rejects(lambda: checked({"path": "objective_lookahead_molhiv_admitted_v3/admission.py", "sha256": frozen["objective_lookahead_molhiv_admitted_v3/admission.py"], "extra": True}))
    if any(n == "torch" or n.startswith("torch.") or n.startswith("ogb") for n in sys.modules):
        raise AssertionError("Stdlib admission/readiness checks imported a target runtime")
    if frozen != sources():
        raise AssertionError("Source changed during readiness check")
    write_new(args.output, {"schema": "gnnm-molhiv-admitted-GPU-source-readiness-v3", "status": "passed",
        "sources": frozen, "syntax_parsed": True, "exact_v1_stdlib_rule_patterns": len(patterns),
        "complete48_seed_finite_tied_margin_harm_family_tie_rules_checked": True,
        "malformed_binding_and_confined_hash_references_refused": True,
        "torch_or_provider_imported": False, "dataset_or_checkpoint_loaded": False,
        "real_route_admission_or_GPU_execution_tested": False, "source_readiness_is_not_execution_qualification": True})
    print("Stdlib syntax/hash/rule/refusal readiness passed; no real or GPU execution performed.")


if __name__ == "__main__":
    main()
