"""Compile local metadata-only source-bound random-state probe requests.

No numerics, network, execution, role tensor decoding or outcome access.
This creates a proposal only; it has no launch entry point or GO action.
"""
from pathlib import Path
import gzip
import hashlib
import json


HERE = Path(__file__).resolve().parent
V = HERE.parents[1]
P = V.parent
ORIGINAL_FIVE = {"roman-empire", "amazon-ratings", "minesweeper", "questions", "tolokers"}


def receipt(path):
    path = Path(path).resolve()
    if not path.is_relative_to(P) or path.is_symlink():
        raise ValueError("Local project receipt only")
    raw = path.read_bytes()
    return {"path": str(path.relative_to(P)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_study(path):
    raw = Path(path).read_bytes()
    value = json.loads(gzip.decompress(raw))
    if value["dataset"].lower().replace("_", "-") in ORIGINAL_FIVE:
        raise ValueError("Original-paper dataset excluded for every method")
    return value


def node_request(path, study, run, arm, family):
    c = run["config"]
    if run["kind"] != "model_fit" or c["learning_rate"] != .002 or c["dropout"] != .4:
        raise ValueError("Exact fixed final-shape search point required")
    return {
        "case_id": family + "-" + study["dataset"] + "-" + c["backbone"].lower() + "-" + arm,
        "family": family, "dataset": study["dataset"], "backbone": c["backbone"],
        "arm": arm, "study_receipt": receipt(path), "study_id": study["study_id"],
        "source_run_id": run["run_id"], "source_run_hash": hashlib.sha256(canonical(run).encode()).hexdigest(),
        "constructor": "exact frozen source builder/native constructor; no full SELECT-reading factory",
        "model_graph_shape": study["data"]["evidence"]["model_graph"],
        "train_rows": 580 if study["dataset"] == "wikics" else 160,
        "readable_role_inputs": {k: v for k, v in c["node_inputs"].items() if k in {"graph", "train"}},
        "fixed_weight_input": c.get("bootstrap_weights"),
        "source_bindings": c.get("endpoint_sources", c.get("bootstrap_sources")),
        "shape_config": {k: c[k] for k in ("depth", "width", "classes", "dropout", "learning_rate", "members")},
        "seed": run["seed"], "body": run["body"],
        "cold_cycles": 1, "warm_cycles": 3, "timeout_seconds": 90,
        "source_callback": "unchanged frozen custom_train closure AST; optimizer step intercepted",
        "expected_actions_per_cycle": {
            "model_forwards": c["members"], "optimizer_attempts": 1,
            "backward_calls": 0 if family == "private_bootstrap" else 1,
            "autograd_grad_VJP_calls": 2 if family == "private_bootstrap" else 0,
            "actual_optimizer_updates": 0, "optimizer_kernels": 0,
        },
        "extra_proposed_checks": ["no-grad full-graph serving shape/timing", "disposable random-state save/restore"],
        "scientific_fit": False, "emit_losses_logits_predictions_or_scores": False,
    }


def main():
    requests = []
    metadata = []
    original_baselines = 0
    endpoint = P / "endpoint_value_packet_20261012_v3/matrix/studies"
    for dataset in ("wikics", "amazon-photo"):
        path = endpoint / ("endpoint-value-standardized-v3-" + dataset + ".json.gz")
        study = read_study(path)
        original_baselines += sum(r["kind"] == "model_fit" and r["config"].get("endpoint_arm") == "original_gnnm" for r in study["runs"])
        for arm in ("BASE", "E4", "N4", "equipped_M1", "ordinary_M1", "original_gnnm"):
            run = next(r for r in study["runs"] if r["kind"] == "model_fit" and r["config"].get("endpoint_arm") == arm
                       and r["config"]["stage"] == "search" and r["seed"] == 76101
                       and r["config"]["learning_rate"] == .002 and r["config"]["dropout"] == .4)
            requests.append(node_request(path, study, run, arm, "endpoint_value"))
        metadata.append(receipt(path))
    bootstrap = V / "preparation/private_bootstrap_allocation_20261012_v1/allocation_artifacts/massive_private_bootstrap_packet_20261012_v1"
    for backbone in ("GAT", "SAGE"):
        for dataset in ("wikics", "amazon-photo"):
            path = bootstrap / "matrix/studies" / ("private-bootstrap-standardized-v1-" + backbone.lower() + "-" + dataset + ".json.gz")
            study = read_study(path)
            original_baselines += sum(r["kind"] == "model_fit" and r["config"].get("bootstrap_arm") == "original_gnnm" for r in study["runs"])
            for arm in ("shared4_coherent_bootstrap", "shared4_paired_graph_bootstrap", "shared4_rank1_lora_graph_bootstrap"):
                run = next(r for r in study["runs"] if r["kind"] == "model_fit" and r["config"].get("bootstrap_arm") == arm
                           and r["config"]["stage"] == "search" and r["seed"] == 77101
                           and r["config"]["learning_rate"] == .002 and r["config"]["dropout"] == .4)
                requests.append(node_request(path, study, run, arm, "private_bootstrap"))
            metadata.append(receipt(path))
    weights = json.loads((bootstrap / "fixed_TRAIN_weights/MANIFEST.json").read_text())
    shape_sets = {ds: sorted({r["source_record"]["shape"][1] for r in weights["tables"] if r["dataset"] == ds}) for ds in ("wikics", "amazon-photo")}
    if shape_sets != {"wikics": [580], "amazon-photo": [160]}:
        raise ValueError("Mask0 count no longer covers the packet's TRAIN shapes")
    path = V / "preparation/ogb_author_family_implementation_20261012_v1/allocation_artifacts/ogb_batchnorm_author_cli_packet_20261012_v1/studies/ogb-batchnorm-declared-seed-author-20261012-v1.json.gz"
    author = read_study(path)
    metadata.append(receipt(path))
    for dataset in ("ogbg-molhiv", "ogbg-molbace"):
        for model in ("gin", "gin-virtual", "gcn", "gcn-virtual"):
            run = next(r for r in author["runs"] if r["config"]["dataset"] == dataset and r["config"]["source_model"] == model and r["seed"] == 73101)
            requests.append({
                "case_id": "ogb_author-" + dataset + "-" + model,
                "family": "ogb_author", "dataset": dataset, "backbone": model,
                "study_receipt": receipt(path), "study_id": author["study_id"],
                "source_run_id": run["run_id"], "source_run_hash": hashlib.sha256(canonical(run).encode()).hexdigest(),
                "constructor": {"class": "GNN", "gnn_type": model.split("-")[0], "virtual_node": model.endswith("-virtual"), "num_tasks": 1,
                                "num_layer": 5, "emb_dim": 300, "drop_ratio": .5,
                                "omitted_defaults_preserved": {"residual": False, "JK": "last", "graph_pooling": "mean"}},
                "seed": 73101, "native_source_files": run["cli"]["source_files"],
                "raw_provider_custody": author["data"]["dataset_custody"][dataset]["receipt"],
                "TRAIN_pack_policy": {"pack_count": 3, "batch_size": 32,
                                      "packs": ["top32 TRAIN num_nodes; graph-ID tie", "top32 TRAIN num_edges; graph-ID tie", "deterministic official TRAIN include-tail"],
                                      "tail_graphs": 5 if dataset == "ogbg-molhiv" else 26,
                                      "actual_pack_bindings_status": "PREPARATION_REQUIRED_BEFORE_CUDA"},
                "cold_cycles_per_pack": 1, "warm_cycles_per_pack": 3,
                "timeout_seconds": 120, "native_callback": "exact pinned main_pyg.train AST; one-batch iterable; step intercepted",
                "BatchNorm_buffer_policy": "native train-mode changes recorded; initial state restored after every cycle",
                "native_optimizer": {"class": "Adam", "lr": .001, "weight_decay": 0, "actual_optimizer_updates": 0},
                "prohibited_calls": ["main", "eval", "Evaluator", "full provider decoding of heldout y", "optimizer kernel", "scientific fit", "selected epoch/checkpoint"],
                "emit_losses_logits_predictions_or_scores": False,
            })
    if len(requests) != 32 or len({r["case_id"] for r in requests}) != 32 or original_baselines != 378:
        raise ValueError("Exact finite proposal/source baseline inventory changed")
    proof = V / "preparation/ogb_native_source_readonly_root_20261012_v1"
    source_receipts = json.loads((proof / "SOURCE_RECEIPTS.json").read_text())
    for row in source_receipts["files"]:
        local = proof / "source" / Path(row["path"]).name
        observed = receipt(local)
        if (observed["bytes"], observed["sha256"]) != (row["bytes"], row["sha256"]):
            raise ValueError("Pinned author mirror changed")
    proposal = {
        "schema": "remaining-resource-admission-local-proposal-v1", "scientific_GO": False,
        "launch_enabled": False, "root_review_required": True, "local_only_preparation": True,
        "requested_cases": requests, "case_counts": {"endpoint_value": 12, "private_bootstrap": 12, "ogb_author": 8},
        "single_process_callback_count": 192, "source_step_attempts_if_all_callbacks_reach_step": 192,
        "actual_optimizer_updates_authorized": 0, "scientific_fits_authorized": 0,
        "child_deadline_sum_seconds": sum(r["timeout_seconds"] for r in requests), "runtime_ETA": None,
        "conditional_two_process_pair_count": 2, "pair_timeout_seconds": 120,
        "pair_condition": "root source approval and measured aggregate one/two-slot memory/RAM/disk admission; no automatic launch",
        "single_plus_pair_deadline_sum_seconds": sum(r["timeout_seconds"] for r in requests) + 240,
        "preparation_and_cleanup_overhead_not_priced": True,
        "excluded_datasets_all_methods": sorted(ORIGINAL_FIVE),
        "original_GNNM_fit_records_preserved_in_audited_six_node_studies": original_baselines,
        "original_GNNM_whole_v8_coverage": "root v8 audit:1,822 fit records across34 standardized studies; not independently recomputed here",
        "fixed_TRAIN_weight_table_shape_sets": shape_sets,
        "input_metadata_receipts": metadata + [receipt(bootstrap / "fixed_TRAIN_weights/MANIFEST.json"), receipt(proof / "SOURCE_RECEIPTS.json")],
        "not_whole_master_admission": True, "new_accuracy_or_convergence_claim": False,
        "notes": ["No historical outcomes read for case choice.", "No repeated synthetic test or remote execution.", "Every failure retained; no automatic retry.",
                  "Weighted M4 reservations may transfer only after exact source/layout containment; controls are not individually measured.",
                  "Random-state save/restore is engineering evidence only; native author CLI has no selected model checkpoint."]}
    target = HERE / "PROPOSAL.json"
    target.write_text(json.dumps(proposal, indent=2, sort_keys=True, allow_nan=False) + "\n")
    inputs = [receipt(HERE / "README.md"), receipt(Path(__file__)), receipt(target)]
    (HERE / "MANIFEST.json").write_text(json.dumps({"schema": "local-proposal-seal-v1", "files": inputs, "scientific_fits": 0, "CUDA_executions": 0, "remote_access": 0,
                                                   "numeric_tests_repeated": 0, "GO_created": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": 32, "source_baseline_fit_records": original_baselines, "deadline_sum_seconds": proposal["child_deadline_sum_seconds"],
                      "proposal_sha256": receipt(target)["sha256"], "launch_enabled": False}, sort_keys=True))


if __name__ == "__main__":
    main()
