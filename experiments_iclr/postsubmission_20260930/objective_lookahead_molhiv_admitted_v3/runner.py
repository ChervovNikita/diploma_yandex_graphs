"""Prepare, gate, execute, replay and qualify the bounded real GPU K8 bridge."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from admission import (HERE, PHASE, PRIOR, PARENT, CELLS, PLACEMENTS, LOSSES, Admission,
    admit, binding_format, canonical, checked, choose_all_gains, inside, live_supervisor,
    manifest, prior_literal, queue_modules, read_json, reference, relative_path,
    sha, sources, write_new)


def heavy():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Require CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch")
    for path in (PRIOR, PARENT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    import admitted_core as core
    import lookahead_core as base
    return torch, core, base


def parent_module():
    spec = importlib.util.spec_from_file_location("gnnm_exact_parent_P25_probe_v6", PARENT / "probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import common
    if Path(common.__file__).resolve() != PARENT / "common.py":
        raise ValueError("Conflicting parent probe import")
    return module


def output_directory(path):
    p = inside(path, HERE)
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_manifest(run):
    write_new(run / "artifact_manifest.json", {str(f.relative_to(run)): sha(f)
        for f in sorted(run.rglob("*")) if f.is_file() and f.name != "artifact_manifest.json"})


def preparation():
    return {"schema": "gnnm-molhiv-admitted-GPU-bridge-source-preparation-v3",
        "status": "SOURCE_PREPARED_NOT_ADMITTED_NOT_EXECUTED", "sources": sources(),
        "recipe": prior_literal("RECIPE"), "rule": prior_literal("RULE"),
        "cells": [{"seed": s, "label_fraction": f, "source_epoch": 25} for s, f in CELLS],
        "trajectories": 48, "real_input_binding": None, "gates_executed_here": False,
        "previous_SOURCE_ONLY_sources_unchanged": True, "CPU_molecular_sibling_required": False,
        "source_preparation_is_not_execution_approval": True,
        "binding_template": {"schema": "gnnm-molhiv-lookahead-real-binding-v3", "source_sha256": sources(),
            "coordinator_approval": None, "queue_qualification": None, "queue_result": None,
            "queue_supervisor_directory": None, "v2_gpu_gate": None, "v3_gpu_gate": None}}


def synthetic_gate(args):
    # Route/current-supervisor checks precede the CUDA context; no data/checkpoint input exists.
    live = live_supervisor(args.supervisor, sys.argv)
    qc, _, _ = queue_modules()
    idle = qc.idle_guard()
    frozen = sources()
    torch, core, base = heavy()
    from gpu_lookahead_core_v2 import paired_gpu_lookahead
    from verify_cpu import units, gradient_gate, rule_gate
    torch.manual_seed(6019)
    module = base.source_module()
    common = module.MolecularProjectorEnsemble(module.gine_options(output_dim=1,
        hidden_dim=128, num_layers=3, dropout=.1, hidden_dim_multiplier=2), members=4).to("cuda:0")
    common.train()
    modes = [(m, m.training) for m in common.modules()]
    gradients = []
    for parameter in common.parameters():
        parameter.grad = torch.full_like(parameter, .03125)
        gradients.append((parameter, parameter.grad, parameter.grad.detach().clone()))
    update, screen = units("Q-update", 8, 0), units("Q-screen", 6, 20)
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    before = base.state_hash(common)
    old, _ = paired_gpu_lookahead(common, update, screen, {"P:synthetic-group"}, 0,
                                   fixture_mode=True, include_null=True)
    snapshot_hashes = {}
    def first_capture(name, stage, value):
        snapshot_hashes[name, stage] = core.tree_hash(value)
    report, artifacts = core.run_cell(common, update, screen, {"P:synthetic-group"}, 0,
                                     synthetic=True, include_null=True, capture=first_capture)
    replay_hashes = {}
    def replay_capture(name, stage, value):
        replay_hashes[name, stage] = core.tree_hash(value)
    repeated, _ = core.run_cell(common, update, screen, {"P:synthetic-group"}, 0,
                               synthetic=True, include_null=True, capture=replay_capture)
    if report != repeated or snapshot_hashes != replay_hashes or len(snapshot_hashes) != 100:
        raise ValueError("V3 full forward/state/gradient/optimizer/RNG snapshot replay failed")
    if (not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
        or base.state_hash(common) != before):
        raise ValueError("V3 synthetic engine mutated source or RNG")
    if any(m.training != original for m, original in modes) or any(
            p.grad is not identity or not torch.equal(p.grad, value) for p, identity, value in gradients):
        raise ValueError("V3 synthetic engine mutated common source gradient identities or modes")
    trace_keys = ("step", "unit_ids", "dropout_seed", "own_loss", "pooled_loss")
    for name, row in report["trajectories"].items():
        reference_row = old["trajectories"][name]
        for key in ("training_loss", "initial_screen", "final_screen", "final_state_sha256",
                    "parameter_count", "private_block_parameter_count", "optimizer_step_counts",
                    "paired_pooled_bce_gain_vs_actual_tied_after_K", "paired_graph_gains_vs_actual_tied_after_K"):
            if row[key] != reference_row[key]:
                raise ValueError("V3 changes the sealed v2 scientific recurrence: " + key)
        if [{k: t[k] for k in trace_keys} for t in row["steps"]] != reference_row["steps"]:
            raise ValueError("V3 changes the sealed v2 update schedule or loss trace")
    initial_count = sum(p.numel() for p in common.parameters())
    block_count = report["trajectories"]["own_tied"]["private_block_parameter_count"]
    for family in LOSSES:
        tied, null = [report["trajectories"][family + "_" + p] for p in ("tied", "null_tied_copy")]
        if tied["steps"] != null["steps"] or tied["final_screen"] != null["final_screen"]:
            raise ValueError("V3 paired null failed")
        for placement, index in (("first", 0), ("middle", 1), ("last", 2)):
            model, optimizer, _ = artifacts[family + "_" + placement]
            copies = model.private_blocks[str(index)]
            if len(copies) != 4 or not isinstance(model.blocks[index], torch.nn.Identity):
                raise ValueError("V3 does not remove the unused shared block")
            if sum(p.numel() for p in model.parameters()) != initial_count + 3 * block_count:
                raise ValueError("Wrong complete-block capacity")
            original = common.blocks[index].state_dict()
            for name, source in original.items():
                values = [c.state_dict()[name] for c in copies]
                if any(v.shape != source.shape or v.dtype != source.dtype for v in values) or len({v.data_ptr() for v in values}) != 4:
                    raise ValueError("Full private block storage check failed")
            actual = list(model.parameters())
            opt = [p for g in optimizer.param_groups for p in g["params"]]
            if len(actual) != len(opt) or {id(p) for p in actual} != {id(p) for p in opt}:
                raise ValueError("Optimizer identity check failed")
    try:
        core.run_cell(common, update, screen, {"P:synthetic-group"}, 0)
    except ValueError:
        pass
    else:
        raise ValueError("Missing admission was accepted")
    # Cross-check the copied stdlib selection against the frozen v1 function.
    gains = {(s, f, loss, p): 0. if p == "tied" else .002 for s, f in CELLS for loss in LOSSES for p in PLACEMENTS}
    if choose_all_gains(gains) != base.mechanism_rule(gains):
        raise ValueError("Stdlib finalizer changes the sealed decision rule")
    if sources() != frozen:
        raise ValueError("Source changed during v3 synthetic gate")
    write_new(args.output, {"schema": "gnnm-GPU-lookahead-admitted-sibling-synthetic-gate-v3",
        "passed": True, "sources": frozen, "live_supervisor": live, "idle_guard": idle,
        "dataset_loaded": False, "checkpoint_loaded": False, "exact_GPU_paired_null": True,
        "full_block_storage_optimizer_identity_checked": True, "exact_v2_scientific_mirror": True,
        "full_snapshot_and_forward_trace_replay": True, "snapshot_count_per_gate_run": 100,
        "logit_autograd": gradient_gate(), "fixed_rule": rule_gate(), "environment": report["environment"],
        "actual_optimizer_steps_total_across_v2_v3_and_v3_replay": 240,
        "CPU_molecular_sibling_required": False, "real_input_admission": False,
        "predictive_accuracy_claimed": False, "novelty_claimed": False})


def runtime_units(admitted, descriptor, parent, base, torch):
    """Reconstruct exactly admitted original masks/folds and a seed704 Q split."""
    bundle = checked(admitted.evidence["bundle"])
    protocol = read_json(checked(admitted.evidence["protocol"]))
    contract = read_json(checked(descriptor["contract"]))
    original, folds, mask, _ = parent.prepare(bundle, descriptor["label_fraction"])
    if folds != contract["folds"] or mask != contract["mask"]:
        raise ValueError("Native graph labels/mask/folds differ from the qualified source")
    from recurrence_collector import unit_fingerprint, state_hash as parent_state_hash
    record = read_json(checked(descriptor["record"]))
    if [unit_fingerprint(u) for u in sorted(original, key=lambda u: u.unit_id)] != record["training_unit_fingerprints"]:
        raise ValueError("Native original TRAIN tensor fingerprints differ from full P25 replay")
    from group_folds import balance, validate_grouped
    validate_grouped(folds)
    mapping = folds["unit_to_group"]
    counts = dict(Counter(mapping[u] for u in folds["score_unit_ids"]))
    (ug, sg), graph_counts = balance(folds["Q_groups"], counts, 704)
    if min(len(ug), len(sg)) < prior_literal("RULE")["minimum_groups_per_update_and_screen_fold"]:
        raise ValueError("Source has insufficient Q update/screen groups")
    by_id = {u.unit_id: u for u in original}
    def make(groups):
        selected = set(groups)
        return [base.Unit(uid, mapping[uid], by_id[uid].graph, "train", False)
                for uid in sorted(folds["score_unit_ids"]) if mapping[uid] in selected]
    update, screen = make(ug), make(sg)
    if len(update) + len(screen) != len(folds["score_unit_ids"]) or [len(update), len(screen)] != graph_counts:
        raise ValueError("Q split lost or duplicated a graph")
    native = parent.environment()
    for key in ("torch", "compiled_cuda", "packages", "runtime_class_source_sha256", "default_dtype"):
        if native[key] != protocol["environment"][key]:
            raise ValueError("Native model runtime differs from the freshly authorized P25 freeze: " + key)
    module = base.source_module()
    common = module.MolecularProjectorEnsemble(module.gine_options(output_dim=1, hidden_dim=128,
        num_layers=3, dropout=.1, hidden_dim_multiplier=2), members=4)
    checkpoint = torch.load(checked(descriptor["checkpoint"]), map_location="cpu", weights_only=False)
    if (checkpoint["epoch"] != 25 or checkpoint["seed"] != descriptor["seed"]
        or checkpoint["label_fraction"] != descriptor["label_fraction"]
        or checkpoint["protocol_sha256"] != contract["protocol_sha256"]):
        raise ValueError("Deserialized P25 source checkpoint metadata changed")
    common.load_state_dict(checkpoint["state_dict"], strict=True)
    if parent_state_hash(common) != record["model_state_sha256"]:
        raise ValueError("Physical common source differs from its replayed P25 state")
    return common.to("cuda:0"), update, screen, set(folds["P_groups"]), {
        "P_Q_folds_sha256": __import__("hashlib").sha256(canonical(folds)).hexdigest(),
        "Q_group_split_seed": 704, "Q_update_groups": ug, "Q_screen_groups": sg,
        "Q_update_unit_ids": [u.unit_id for u in update], "Q_screen_unit_ids": [u.unit_id for u in screen],
        "source_mask": mask, "source_descriptor": descriptor,
        "P25_runtime_flags": {k: protocol["environment"][k] for k in ("deterministic_algorithms", "matmul_tf32", "cudnn_tf32")},
        "K8_runtime_flags": "deterministic_algorithms=True; TF32=False; AMP=False; fixed by the separate v2/v3 GPU gates"}


def execute(args):
    live = live_supervisor(args.supervisor, sys.argv)
    admitted = admit(args.binding)
    frozen = sources()
    torch, core, base = heavy()
    parent = parent_module()
    run = output_directory(args.output)
    lock = {"schema": "gnnm-molhiv-admitted-GPU-lookahead-run-lock-v3", "admission": admitted.evidence,
        "live_supervisor": live, "source_sha256": frozen, "recipe": prior_literal("RECIPE"),
        "rule": prior_literal("RULE"), "cells": list(admitted.cells), "trajectories": 48,
        "outcome_scheduling": False, "automatic_retry": False, "validation_or_test_loaded": False,
        "start_UTC": datetime.now(timezone.utc).isoformat()}
    write_new(run / "run_lock.json", lock)
    completed, start = [], time.perf_counter()
    try:
        for descriptor in admitted.cells:
            s, f = descriptor["seed"], descriptor["label_fraction"]
            cell = run / ("seed%d_label%d" % (s, int(100 * f)))
            cell.mkdir()
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(30131 + 1000 * s + int(100 * f))
                common, update, screen, p_groups, data_lock = runtime_units(admitted, descriptor, parent, base, torch)
                write_new(cell / "input_lock.json", data_lock)
                def capture(name, stage, value):
                    target = cell / "snapshots" / name / (stage + ".pt")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise ValueError("Refuse overwriting a trajectory snapshot")
                    torch.save(value, target)
                begin = time.perf_counter()
                report, artifacts = core.run_cell(common, update, screen, p_groups, s,
                    admission=admitted, fraction=f, capture=capture)
                write_new(cell / "cell_report.json", report)
                snapshots = list((cell / "snapshots").rglob("*.pt"))
                if len(report["trajectories"]) != 8 or len(snapshots) != 80:
                    raise ValueError("Incomplete eight-arm full snapshot coverage")
                completed.append({"seed": s, "label_fraction": f, "report": reference(cell / "cell_report.json"),
                    "input_lock": reference(cell / "input_lock.json"), "snapshot_count": 80,
                    "snapshot_bytes": sum(p.stat().st_size for p in snapshots), "wall_seconds": time.perf_counter() - begin})
                del common, artifacts
            torch.cuda.empty_cache()
        if sources() != frozen:
            raise ValueError("Source changed during admitted execution")
        summary = {"schema": "gnnm-molhiv-admitted-GPU-lookahead-execution-summary-v3",
            "status": "all_48_completed_pending_authorization_and_independent_replay", "cells": completed,
            "run_lock": reference(run / "run_lock.json"), "source_sha256": frozen, "binding": reference(args.binding),
            "live_supervisor": live, "all_six_cells_used": True, "trajectories": 48,
            "optimizer_steps": 384, "full_snapshot_count": 480, "wall_seconds": time.perf_counter() - start,
            "validation_or_test_loaded": False, "mechanism_decision_emitted": False,
            "predictive_accuracy_claimed": False, "novelty_claimed": False,
            "cost_scope": "K8 execution and artifact capture; all P25/export/gate/replay overhead retained separately"}
        write_new(run / "summary.json", summary)
        write_manifest(run)
    except Exception as exc:
        write_new(run / "failure.json", {"type": type(exc).__name__, "message": str(exc),
            "completed_cells": completed, "mechanism_lock_qualified": False, "no_retry": True})
        write_manifest(run)
        raise


def completed_execution(run, supervisor, admitted):
    qc, qualifier, _ = queue_modules()
    run = inside(run, HERE)
    record = qc.authorization(inside(supervisor), admitted.qualified_gpu_uuid, "--output", run)
    command = read_json(inside(supervisor) / "command.json")["argv"]
    if Path(command[1]).resolve() != HERE / "runner.py" or command[2] != "execute":
        raise ValueError("Execution receipt used another bridge stage")
    qualifier.command_input_path(command, "--binding", checked(admitted.evidence["binding"]))
    run_manifest = manifest(run)
    summary = read_json(run / "summary.json")
    if (summary.get("status") != "all_48_completed_pending_authorization_and_independent_replay"
        or summary.get("source_sha256") != sources() or summary.get("binding") != admitted.evidence["binding"]
        or summary.get("all_six_cells_used") is not True or summary.get("trajectories") != 48
        or summary.get("optimizer_steps") != 384 or summary.get("full_snapshot_count") != 480
        or summary.get("validation_or_test_loaded") is not False or summary.get("mechanism_decision_emitted") is not False
        or len(summary.get("cells", [])) != 6
        or {(r["seed"], r["label_fraction"]) for r in summary["cells"]} != set(CELLS)):
        raise ValueError("Execution does not provide a complete qualified fixed48 design")
    lock = read_json(checked(summary["run_lock"]))
    if (lock.get("admission") != admitted.evidence or lock.get("source_sha256") != sources()
        or lock.get("cells") != list(admitted.cells) or lock.get("recipe") != prior_literal("RECIPE")
        or lock.get("rule") != prior_literal("RULE") or lock.get("trajectories") != 48
        or lock.get("validation_or_test_loaded") is not False
        or lock.get("automatic_retry") is not False or lock.get("outcome_scheduling") is not False):
        raise ValueError("Execution source/input/scientific run lock changed")
    stages = {"initial", "final", *("step%02d" % i for i in range(1, 9))}
    expected_names = {l + "_" + p for l in LOSSES for p in PLACEMENTS}
    for cell in summary["cells"]:
        directory = checked(cell["report"]).parent
        if (directory.parent != run or cell.get("snapshot_count") != 80
            or checked(cell["input_lock"]).parent != directory
            or {str(p.relative_to(directory / "snapshots")) for p in (directory / "snapshots").rglob("*.pt")}
               != {n + "/" + stage + ".pt" for n in expected_names for stage in stages}):
            raise ValueError("Complete initial/K8/final artifact coverage required for every arm")
    return summary, {"summary": reference(run / "summary.json"), "artifact_manifest": run_manifest,
        "supervisor_directory": str(inside(supervisor).relative_to(PHASE)), **record}


def replay(args):
    live = live_supervisor(args.supervisor, sys.argv)
    admitted = admit(args.binding)
    summary, execute_record = completed_execution(args.run, args.execute_supervisor, admitted)
    frozen = sources()
    torch, core, base = heavy()
    parent = parent_module()
    errors, total = [], 0
    for descriptor in admitted.cells:
        s, f = descriptor["seed"], descriptor["label_fraction"]
        row = next(r for r in summary["cells"] if (r["seed"], r["label_fraction"]) == (s, f))
        original = read_json(checked(row["report"]))
        cell = checked(row["report"]).parent
        snapshot_count = 0
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(30131 + 1000 * s + int(100 * f))
            common, update, screen, p_groups, data_lock = runtime_units(admitted, descriptor, parent, base, torch)
            if data_lock != read_json(checked(row["input_lock"])):
                raise ValueError("Independent replay inputs/folds/masks/source differ")
            def compare_snapshot(name, stage, value):
                nonlocal snapshot_count
                stored = torch.load(cell / "snapshots" / name / (stage + ".pt"), map_location="cpu", weights_only=False)
                if core.tree_hash(stored) != core.tree_hash(value):
                    raise ValueError("Exact full tensor/state/optimizer/gradient/RNG replay differs: " + name + "/" + stage)
                snapshot_count += 1
            repeated, artifacts = core.run_cell(common, update, screen, p_groups, s,
                admission=admitted, fraction=f, capture=compare_snapshot)
            if repeated != original or snapshot_count != 80:
                raise ValueError("Exact full forward/screening/update trace replay differs")
            del common, artifacts
        torch.cuda.empty_cache()
        errors.append({"seed": s, "label_fraction": f, "status": "exact_pass", "snapshots": snapshot_count,
                       "all_forward_losses_logits_gradients_states_optimizers_RNG_equal": True})
        total += snapshot_count
    if sources() != frozen:
        raise ValueError("Sources changed during replay")
    write_new(args.output, {"schema": "gnnm-molhiv-admitted-GPU-lookahead-full-replay-v3",
        "status": "exact_pass_pending_replay_authorization", "binding": reference(args.binding),
        "execute": execute_record, "source_sha256": frozen, "live_supervisor": live, "cells": errors,
        "all_six_cells_used": True, "all_48_trajectories_replayed": True, "snapshots_compared": total,
        "optimizer_steps_replayed": 384, "validation_or_test_loaded": False,
        "tolerance": "exact tensor-value hashes and exact JSON equality; no post-outcome tolerance relaxation",
        "mechanism_decision_emitted": False, "predictive_accuracy_claimed": False})


def finalize(args):
    live = live_supervisor(args.supervisor, sys.argv)
    admitted = admit(args.binding)
    summary, executed = completed_execution(args.run, args.execute_supervisor, admitted)
    qc, qualifier, _ = queue_modules()
    replay_path = inside(args.replay)
    replayed = qc.authorization(inside(args.replay_supervisor), admitted.qualified_gpu_uuid, "--output", replay_path)
    argv = read_json(inside(args.replay_supervisor) / "command.json")["argv"]
    if Path(argv[1]).resolve() != HERE / "runner.py" or argv[2] != "replay":
        raise ValueError("Replay receipt used another bridge stage")
    for flag, path in (("--binding", args.binding), ("--run", args.run), ("--execute-supervisor", args.execute_supervisor)):
        qualifier.command_input_path(argv, flag, path)
    replay_report = read_json(replay_path)
    if (replay_report.get("status") != "exact_pass_pending_replay_authorization"
        or replay_report.get("binding") != admitted.evidence["binding"] or replay_report.get("execute") != executed
        or replay_report.get("source_sha256") != sources() or replay_report.get("all_six_cells_used") is not True
        or replay_report.get("all_48_trajectories_replayed") is not True
        or replay_report.get("snapshots_compared") != 480 or replay_report.get("optimizer_steps_replayed") != 384
        or replay_report.get("validation_or_test_loaded") is not False):
        raise ValueError("All48 exact authorized replay required before a decision")
    gains = {}
    for cell in summary["cells"]:
        report = read_json(checked(cell["report"]))
        if (report["screen_scored_completely"] is not True or report["source_and_inputs_preserved"] is not True
            or report["validation_or_test_loaded"] is not False
            or set(report["trajectories"]) != {l + "_" + p for l in LOSSES for p in PLACEMENTS}):
            raise ValueError("Missing complete screening endpoint")
        for name, trajectory in report["trajectories"].items():
            loss, placement = name.split("_", 1)
            gains[cell["seed"], cell["label_fraction"], loss, placement] = trajectory["paired_pooled_bce_gain_vs_actual_tied_after_K"]
    mechanism = choose_all_gains(gains)
    families = {}
    for loss in LOSSES:
        chosen = mechanism["loss_families"][loss]["candidate_for_separate_full_predictive_study"]
        families[loss] = {"training_loss": loss, "candidate": chosen,
            "private_indices": [] if chosen is None else [PLACEMENTS.index(chosen) - 1], "abstain": chosen is None,
            "all_six_cells_used": True, "validation_or_test_used": False}
    write_new(args.output, {"schema": "gnnm-molhiv-qualified-lookahead-mechanism-lock-v3",
        "status": "qualified_training_only_mechanism_screen", "all_six_cells_used": True,
        "all_48_trajectories_replayed": True, "validation_or_test_used": False,
        "binding": admitted.evidence["binding"], "source_sha256": sources(), "execute": executed,
        "replay": {"report": reference(replay_path), "supervisor_directory": str(inside(args.replay_supervisor).relative_to(PHASE)), **replayed},
        "finalizer_live_supervisor": live, "loss_families": families, "mechanism": mechanism,
        "CPU_molecular_sibling_required": False, "accuracy_promotion_allowed": False,
        "predictive_accuracy_or_novelty_or_superiority_claimed": False,
        "cost_accounting": {"P25_sources_and_all_prerequisites": admitted.evidence["queue_result"],
            "lookahead_execution_seconds": executed["seconds"], "full_replay_seconds": replayed["seconds"],
            "adaptive_search_overhead_must_be_counted": True},
        "finalizer_completed_authorization_required_before_downstream_use": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("synthetic-gate")
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--supervisor", type=Path, required=True)
    for name in ("execute", "replay", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--binding", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--supervisor", type=Path, required=True)
        if name != "execute":
            p.add_argument("--run", type=Path, required=True)
            p.add_argument("--execute-supervisor", type=Path, required=True)
        if name == "finalize":
            p.add_argument("--replay", type=Path, required=True)
            p.add_argument("--replay-supervisor", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        write_new(args.output, preparation())
    else:
        {"synthetic-gate": synthetic_gate, "execute": execute, "replay": replay, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
