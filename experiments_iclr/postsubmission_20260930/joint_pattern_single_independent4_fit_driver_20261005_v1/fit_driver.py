"""Fresh native structured-single fits and independently selected raw-logit pool.

Preparation only: this driver has not been executed. It has no resume/donor,
TEST, joint-member optimization, automatic retry, or remote dispatch path.
"""
import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
import traceback

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CB = PHASE / "exact_cb_support_bucket_paired_predictive_preparation_20261004_v1"
SINGLE = PHASE / "joint_pattern_capable_single_control_source_20261005_v1"
NATIVE = PHASE / "graph_ncNC_member_completion_qualification_preparation_20261003_v2"
DESIGN = PHASE / "graph_ncNC_collab_predictive_pilot_design_20261003_v1"
CARDINALITY = PHASE / "ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1"


def dependencies():
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["sources"]:
        path = PHASE / row["path"]
        value = path.read_bytes()
        if len(value) != row["bytes"] or sha256(value).hexdigest() != row["sha256"]:
            raise RuntimeError("Source binding changed: " + row["path"])
    # Authenticate the native helpers actually imported by modules(context).
    native_manifest = json.loads((NATIVE / "MANIFEST.json").read_text())
    needed = {"graph_ops.py", "prototype.py", "native_reference.py", "selected_state.py"}
    rows = [row for row in native_manifest["files"] if row["path"] in needed]
    if {row["path"] for row in rows} != needed:
        raise RuntimeError("Native helper bindings are incomplete")
    for row in rows:
        path = (NATIVE / row["path"]).resolve()
        if not path.is_relative_to(NATIVE.resolve()):
            raise RuntimeError("Native source escaped its bound directory")
        value = path.read_bytes()
        if len(value) != row.get("bytes", row.get("size")) or sha256(value).hexdigest() != row["sha256"]:
            raise RuntimeError("Native manifest source changed: " + row["path"])
    sys.path[:0] = [str(CB), str(SINGLE), str(NATIVE), str(CARDINALITY)]
    names = ("pilot_common", "pilot_accounting", "pilot_model", "pilot_data", "pilot_state",
             "pilot_evaluate", "paired_train", "pattern_teacher", "conditional_loss", "single_control", "cardinality_model")
    loaded = {name: importlib.import_module(name) for name in names}
    for name, module in loaded.items():
        expected = SINGLE if name == "single_control" else CARDINALITY if name == "cardinality_model" else CB
        if Path(module.__file__).resolve() != expected / (name + ".py"):
            raise RuntimeError("Module shadowed: " + name)
    return loaded


def score_valid_scalar(model, data, mods, helpers):
    """Complete TRAIN context and native VALID batches; one raw scalar/query."""
    import torch
    model.eval()
    torch.cuda.synchronize(0)
    started = perf_counter()
    graph = mods["graph_ops"].Graph.from_pairs(data["pairs"], len(data["x"]))
    outputs, batch_counts = [], []
    with torch.no_grad():
        h = model.encode(data["x"], graph)
        for queries in (data["valid_positive"], data["valid_negative"]):
            scores, count = [], 0
            for indices in mods["native_utils"].PermIterator(queries.device, len(queries), 131072, False):
                current, _ = model.query_forward(h, graph, queries[indices], emit_auxiliary=False)
                helpers["pilot_common"].require(current.shape == (len(indices),) and bool(torch.isfinite(current).all()),
                                                "Incomplete/nonfinite native scalar VALID batch")
                scores.append(model.serve(current).detach().cpu())
                count += len(indices)
            helpers["pilot_common"].require(count == len(queries), "VALID rows omitted")
            outputs.append(torch.cat(scores))
            batch_counts.append(len(scores))
    torch.cuda.synchronize(0)
    tensor_sha = helpers["pilot_data"].tensor_sha
    return outputs[0], outputs[1], {"wall_seconds": perf_counter() - started,
        "positive_queries": len(outputs[0]), "negative_queries": len(outputs[1]), "query_batches": batch_counts,
        "encoder_calls": 1, "graph": "complete_TRAIN_only", "serving": "native_scalar_raw_logit",
        "score_digests": {"positive": tensor_sha(outputs[0]), "negative": tensor_sha(outputs[1])}}


def train_epoch(model, optimizer, data, mods, sampler, teacher, helpers, auxiliary, attempts):
    import torch
    state, common = helpers["pilot_state"], helpers["pilot_common"]
    model.train()
    torch.cuda.synchronize(0)
    started = perf_counter()
    negatives, iterator, stream = helpers["pilot_data"].epoch_stream(data, mods, sampler)
    batches = []
    for batch, record_ids in enumerate(iterator, start=1):
        common.require(len(record_ids) == 65536, "Native full batch changed")
        attempts.progress({"phase": "record_mask_forward_backward_Adam", "attempted_batch": batch, "completed_batches": batch - 1})
        before = state.rng_digest(state.rng_state())
        optimizer.zero_grad(set_to_none=True)
        graph = mods["graph_ops"].Graph.mask_train_batch(data["pairs"], record_ids, len(data["x"]))
        h = model.encode(data["x"], graph)
        loss = helpers["single_control"].objective(model, h, graph, data["pairs"][record_ids],
            negatives[record_ids], teacher, helpers["conditional_loss"], arm=auxiliary)
        common.require(bool(torch.isfinite(loss["total"])), "Nonfinite native objective")
        loss["total"].backward()
        helpers["paired_train"].finite(model, optimizer, gradients=True)
        optimizer.step()
        helpers["paired_train"].finite(model, optimizer, gradients=False)
        batches.append({"batch": batch, "record_ids_sha256": helpers["pilot_data"].tensor_sha(record_ids),
                        "start_rng_sha256": before, "end_rng_sha256": state.rng_digest(state.rng_state())})
        attempts.progress({"completed_batches": batch})
    common.require(len(batches) == 17, "Incomplete native epoch")
    torch.cuda.synchronize(0)
    return {"wall_seconds": perf_counter() - started, "stream": stream, "batch_receipts": batches,
            "optimizer_steps": 17, "encoder_calls": 17, "target_query_groups": 34,
            "positive_records": 1114112, "negative_records": 1114112, "dropped_positive_tail": 64940,
            "full_node_decoder_xlin_calls": 102, "candidate_depth_zero_calls": 68,
            "target_trajectories_per_query": 1, "internal_density_components": 4}


def fit_member(context, mods, data, sampler, metric, device, output, seed, auxiliary, helpers, attempts):
    """One fresh100-epoch/1700-update fit; own first-best native VALID selector."""
    import torch
    common, state = helpers["pilot_common"], helpers["pilot_state"]
    output = Path(output)
    output.mkdir()  # A previous failure/result is never overwritten.
    unit = "structured_single_" + auxiliary
    model, optimizer = helpers["single_control"].make_control(mods, seed, device, mods["graph_ops"],
        helpers["cardinality_model"].native_adjacency, helpers["pilot_model"].make_native)
    initial = state.snapshot(model, optimizer)
    record = {"epoch": 0, "current": initial, "best": None, "best_state": None, "epochs": [],
              "initial_state_sha256": state.state_digest(initial), "initial_rng_sha256": state.rng_digest(initial["rng"])}
    state.write_journal(output, context, unit, seed, record)
    teacher = helpers["pattern_teacher"].ObservationTeacher.from_train(data["pairs"], len(data["x"]))
    for epoch in range(1, 101):
        attempts.phase("native_TRAIN", member_seed=seed, epoch=epoch, completed_batches=0)
        start_rng = state.rng_digest(state.rng_state())
        train = train_epoch(model, optimizer, data, mods, sampler, teacher, helpers, auxiliary, attempts)
        train_rng = state.rng_state()
        before_valid = state.rng_digest(train_rng)
        attempts.phase("complete_VALID_individual_selection", member_seed=seed, epoch=epoch)
        positive, negative, valid = score_valid_scalar(model, data, mods, helpers)
        common.require(state.rng_digest(state.rng_state()) == before_valid, "VALID consumed training RNG")
        quality = helpers["pilot_evaluate"].hits50(metric, positive, negative)
        current = state.snapshot(model, optimizer, rng=train_rng)
        best, replace = mods["design"].select_validation_candidate(record["best"],
            candidate_id="epoch_" + str(epoch), hits50=quality, order=epoch)
        record.update(epoch=epoch, current=current, best=best)
        if replace:
            record["best_state"] = current
        row = {"epoch": epoch, "start_rng_sha256": start_rng, "end_rng_sha256": before_valid,
               "TRAIN": train, "VALID": valid, "private_hits50": quality}
        record["epochs"].append(row)
        state.write_journal(output, context, unit, seed, record)
        with (output / "EPOCHS.jsonl").open("a") as stream:
            json.dump(row, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print("EPOCH_COMPLETE seed=" + str(seed) + " epoch=" + str(epoch) + "/100", flush=True)
    attempts.phase("selected_member_replay", member_seed=seed)
    state.restore_snapshot(model, optimizer, record["best_state"])
    before = state.rng_digest(state.rng_state())
    positive, negative, reference = score_valid_scalar(model, data, mods, helpers)
    common.require(helpers["pilot_evaluate"].hits50(metric, positive, negative) == record["best"]["hits50"]
                   and state.rng_digest(state.rng_state()) == before, "Selected member VALID replay differs")
    selected = state.write_selected(output, context, unit, unit, seed, record["best"], [record["best_state"]])
    common.require(common.file_sha(output / selected["path"]) == selected["sha256"], "Own selected checkpoint differs")
    payload = torch.load(output / selected["path"], map_location="cpu", weights_only=False)
    common.require(payload["identity"] == context["identity"] and payload["seed"] == seed
                   and payload["unit"] == unit and len(payload["states"]) == 1,
                   "Own selected checkpoint provenance differs")
    replay, replay_optimizer = helpers["single_control"].make_control(mods, seed, device, mods["graph_ops"],
        helpers["cardinality_model"].native_adjacency, helpers["pilot_model"].make_native)
    state.restore_snapshot(replay, replay_optimizer, payload["states"][0])
    before = state.rng_digest(state.rng_state())
    rp, rn, replay_receipt = score_valid_scalar(replay, data, mods, helpers)
    common.require(torch.equal(rp, positive) and torch.equal(rn, negative)
                   and state.rng_digest(state.rng_state()) == before, "Serialized own selected logits/RNG differ")
    logits = state.atomic_torch(output / "VALID_LOGITS.pt", {"positive": positive, "negative": negative,
        "selection": record["best"], "checkpoint": selected, "graph": "complete_TRAIN_only", "TEST_read": False})
    receipt = {"seed": seed, "epochs": 100, "optimizer_steps": 1700, "selection_candidates": 100,
        "selected_checkpoint": selected, "selected_VALID_logits": logits, "complete_VALID_evaluations": 102,
        "initial_state_sha256": record["initial_state_sha256"], "initial_rng_sha256": record["initial_rng_sha256"],
        "selected_replay": reference, "serialized_replay": replay_receipt, "fresh_initialization": True,
        "total_parameters": sum(p.numel() for p in model.parameters()), "teacher": teacher.receipt()}
    common.atomic_json(output / "COMPLETE.json", receipt)
    return positive, negative, receipt


def fit_and_pool(context, mods, data, sampler, metric, device, output, control, base_seed, auxiliary, helpers, attempts):
    """Train independently; pool only each member's individually selected logits."""
    helpers["pilot_common"].require(control in ("single", "independent4") and base_seed in mods["design"].BASE_SEEDS
        and auxiliary in ("joint", "separate"), "Outside existing seed recipe or source objectives")
    count = 1 if control == "single" else 4
    seeds = [mods["design"].native_member_seed(base_seed, member) for member in range(count)]
    rows = [fit_member(context, mods, data, sampler, metric, device, Path(output) / ("member_" + str(member)),
                      seed, auxiliary, helpers, attempts) for member, seed in enumerate(seeds)]
    positive, negative = (rows[0][0], rows[0][1]) if count == 1 else helpers["pilot_evaluate"].mean_native_scores(rows)
    # One final measurement; this pool is not another winner-search candidate.
    quality = helpers["pilot_evaluate"].hits50(metric, positive, negative)
    logits = helpers["pilot_state"].atomic_torch(Path(output) / "SELECTED_POOL_VALID_LOGITS.pt", {
        "positive": positive, "negative": negative, "member_seeds": seeds,
        "members": [row[2] for row in rows], "private_hits50": quality, "serving": "mean_selected_native_raw_logits",
        "ensemble_checkpoint_search": False, "TEST_read": False})
    return {"control": control, "auxiliary": auxiliary, "base_seed": base_seed, "member_seeds": seeds,
            "unique_fresh_fits": count, "optimizer_steps": count * 1700,
            "internal_density_components": count * 4, "complete_VALID_evaluations": count * 102,
            "pool_additional_graph_traversals": 0, "members": [row[2] for row in rows], "selected_pool_logits": logits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("single", "independent4"), required=True)
    parser.add_argument("--auxiliary", choices=("joint", "separate"), default="joint")
    parser.add_argument("--base-seed", type=int, choices=range(5), default=0)
    parser.add_argument("--data-authority", type=Path, required=True)
    parser.add_argument("--runtime-authority", type=Path, required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = perf_counter(), process_time()
    output = args.output.resolve()
    if not output.is_relative_to(PHASE.resolve()):
        raise ValueError("Output must remain inside this project's research repository")
    for authority in (args.data_authority, args.runtime_authority):
        if not authority.resolve().is_relative_to(PHASE.resolve()):
            raise ValueError("Use the existing authority files inside this research repository")
    output.mkdir(parents=True, exist_ok=False)
    attempts, helpers = None, None
    try:
        if "torch" in sys.modules:
            raise RuntimeError("Use a fresh process; configure CUBLAS before Torch imports")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        helpers = dependencies()
        common = helpers["pilot_common"]
        context = {"authority": common.read_bound_json(args.data_authority, common.DATA_AUTHORITY_SHA),
            "runtime": common.read_bound_json(args.runtime_authority, common.RUNTIME_AUTHORITY_SHA),
            "release": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, "release_sha256": None,
            "paths": {"prototype_root": NATIVE, "design_root": DESIGN},
            "identity": {"driver_source_sha256": common.file_sha(__file__),
                "source_bindings_sha256": common.file_sha(HERE / "SOURCE_BINDINGS.json"),
                "data_authority_sha256": common.DATA_AUTHORITY_SHA, "runtime_authority_sha256": common.RUNTIME_AUTHORITY_SHA,
                "control": args.control, "auxiliary": args.auxiliary, "base_seed": args.base_seed,
                "authorization_reference": args.authorization_reference}}
        attempts = helpers["pilot_accounting"].Attempts(output, context, "fit", args.control, args.base_seed, started=started)
        attempts.phase("existing_runtime_and_native_source_acquisition")
        mods = helpers["pilot_model"].modules(context)
        device, sampler = helpers["pilot_model"].ordinary_runtime(context)
        import torch
        profile_before = helpers["pilot_model"].runtime_settings()
        torch.use_deterministic_algorithms(True, warn_only=False)  # Existing exact-CB transition.
        torch.cuda.reset_peak_memory_stats(0)
        common.atomic_json(output / "ACQUISITIONS.json", {"source_bindings": context["identity"]["source_bindings_sha256"],
            "data_authority": str(args.data_authority.resolve()), "runtime_authority": str(args.runtime_authority.resolve()),
            "runtime_profile_before": profile_before, "runtime_profile": helpers["pilot_model"].runtime_settings()})
        attempts.phase("complete_TRAIN_raw_VALID_acquisition")
        data = helpers["pilot_data"].load_data(context, device)
        metric = helpers["pilot_evaluate"].evaluator(context)
        common.atomic_json(output / "DATA_RECEIPT.json", data["digests"])
        result = fit_and_pool(context, mods, data, sampler, metric, device, output, args.control,
                              args.base_seed, args.auxiliary, helpers, attempts)
        torch.cuda.synchronize(0)
        result.update(cpu_seconds=process_time() - cpu_started, wall_seconds=perf_counter() - started,
                      CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        result["accounting"] = attempts.finish("COMPLETE")
        common.atomic_json(output / "COMPLETE.json", result)
    except BaseException as error:
        (output / "FAILURE.txt").write_text(traceback.format_exc())
        if attempts is not None:
            attempts.finish("FAILED", error)
        cost = {"wall_seconds": perf_counter() - started, "cpu_seconds": process_time() - cpu_started,
                "automatic_retry": False}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            cost.update(CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        (output / "FAILURE_COST.json").write_text(json.dumps(cost, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
