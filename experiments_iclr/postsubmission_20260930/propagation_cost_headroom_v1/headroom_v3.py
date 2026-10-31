"""Complete-graph cost diagnostic; no scientific fits or quality measurements.

The oracle removes both residual sparse propagation and its centering work.
Its outputs are a different model and must never be scored or promoted.
"""
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if root not in path.parents:
        raise ValueError("Require path strictly inside declared packet")
    return path


def write(path, value):
    with within(path, HERE).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    from propagation_cost_engineering_v1.contract import live_context, verify_own_seal, verify_impl
    live = live_context(args.supervisor, Path(__file__))
    request_path = within(args.request, HERE)
    request = json.loads(request_path.read_text())
    if (request["schema"] != "gnnm-complete-graph-cost-headroom-v1"
            or request["source_sha256"] != sha(__file__)
            or request["scientific_training_authorized"] is not False
            or request["quality_scoring_authorized"] is not False
            or request["tasks"] != ["Reddit2", "ogbn-products"]):
        raise ValueError("Explicit source-bound two-complete-graph diagnostic required")
    if request["GPU_wall_cap_seconds"] > 600 or request["GPU_wall_cap_seconds"] <= 0:
        raise ValueError("Bounded diagnostic cap required")
    verify_own_seal()
    verify_impl()
    output = within(PROJECT / request["output_relative_to_phase"], HERE)
    if output.exists():
        raise FileExistsError("Preserve every previous diagnostic generation")
    output.mkdir()
    started = time.monotonic()
    write(output / "START.json", {"UTC": datetime.now(timezone.utc).isoformat(),
          "request_sha256": sha(request_path), "source_sha256": sha(__file__), "live": live,
          "scope": "timing only; oracle deletes work and changes predictions; no fitted weights or target scores"})
    try:
        import torch
        from propagation_cost_engineering_v1.contract import TASKS, check_gpu, check_peak
        from propagation_cost_engineering_v1.data_contract import load_acquired
        from propagation_cost_engineering_v1.prepare import audit_prepared, load_family
        from propagation_cost_engineering_v1.complete_models import (
            CompleteEnsemble, construct_source_arm, inference_validation_work)

        class MeanOnlyCostOracle(CompleteEnsemble):
            def _propagate(self, family, normalized, color):
                # An intentionally optimistic timing scenario. This is not
                # equivalent to the candidate or any source ensemble.
                common = family.full(normalized.mean(dim=1))
                return common[:, None, :].expand_as(normalized)

        def check():
            if time.monotonic() - started >= request["GPU_wall_cap_seconds"]:
                raise RuntimeError("Complete diagnostic wall cap reached")

        def timed(model, family, features, indices, execution):
            check()
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            inference_validation_work(model, family, features, indices, execution)
            torch.cuda.synchronize(0)
            check()
            return time.perf_counter() - start

        reports = []
        for task in request["tasks"]:
            ref = request["engineering_phases"][task]
            path = within(PROJECT / ref["path"], PROJECT)
            if sha(path) != ref["sha256"]:
                raise ValueError("Qualified complete engineering phase changed")
            phase = json.loads(path.read_text())
            if phase["task"] != task:
                raise ValueError("Complete task identity changed")
            check_gpu(phase)
            export, data_binding = load_acquired(phase)
            check()
            metadata, directory = audit_prepared(
                Path(phase["repo_root"]) / phase["prepared_manifest_relative_to_repo"], phase, data_binding)
            check()
            features = export["x"].to("cuda:0")
            indices = export["official_split_node_ids"]["valid"].to("cuda:0")
            task_report = {"task": task, "nodes": metadata["nodes"],
                           "raw_edges": metadata["raw_edges"], "data_binding": data_binding, "variants": []}
            kinds = ("exact_member", "candidate", "mean_only_cost_oracle")
            resident = {}
            records = {kind: {"kind": kind, "warm_samples_seconds": []} for kind in kinds}
            for kind in kinds:
                check()
                arm = "exact_member" if kind in ("exact_member", "mean_only_cost_oracle") else "candidate"
                family = load_family(metadata, directory, arm, "lazy_two_colors")
                source = construct_source_arm(TASKS[task], "candidate" if kind == "mean_only_cost_oracle" else kind)
                if kind == "mean_only_cost_oracle":
                    model = MeanOnlyCostOracle(TASKS[task]["features"], TASKS[task]["classes"], arm="candidate")
                    model.load_state_dict(source.state_dict())
                else:
                    model = source
                resident[kind] = (model.to("cuda:0").eval(), family)
                check_peak()
                del model, family, source
            task_report["setup_and_load_peak"] = check_peak()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            orders = []
            order_rng = random.Random(20260930 + TASKS[task]["index"])
            with torch.no_grad():
                for kind in kinds:
                    model, family = resident[kind]
                    records[kind]["cold_seconds"] = timed(model, family, features, indices, "packed")
                    check_peak()
                for round_index in range(60):
                    order = list(kinds)
                    order_rng.shuffle(order)
                    orders.append(order)
                    for kind in order:
                        model, family = resident[kind]
                        value = timed(model, family, features, indices, "packed")
                        check_peak()
                        if round_index >= 10:
                            records[kind]["warm_samples_seconds"].append(value)
            resident_peak = check_peak()
            for kind in kinds:
                samples = records[kind]["warm_samples_seconds"]
                row = {**records[kind], "median_seconds": statistics.median(samples),
                       "p90_seconds": sorted(samples)[44], "execution": "packed", "dtype": "float32",
                       "quality_scores_exist": False, "scientific_fit": False,
                       "own_arm_peak_not_isolated": True, "three_arm_resident_peak": resident_peak}
                task_report["variants"].append(row)
                write(output / (("reddit2" if task == "Reddit2" else "products") + "__" + kind + ".json"), row)
            task_report["paired_shuffled_orders"] = orders
            task_report["timing_policy"] = "one cold per arm,10paired shuffled warm rounds,50paired shuffled measured rounds; allthree models resident"
            del resident, model, family
            baseline, candidate, oracle = task_report["variants"]
            task_report["candidate_exact_median_ratio"] = candidate["median_seconds"] / baseline["median_seconds"]
            task_report["optimistic_oracle_exact_median_ratio"] = oracle["median_seconds"] / baseline["median_seconds"]
            task_report["optimistic_25percent_headroom_measured"] = task_report["optimistic_oracle_exact_median_ratio"] <= .75
            task_report["scope"] = "Complete initialized models and identical label-free validation proxy; optimistic deletion scenario, not a proved lower bound on all possible implementations"
            reports.append(task_report)
            del features, indices, export
            gc.collect()
            torch.cuda.empty_cache()
        result = {"schema": request["schema"], "status": "completed_cost_diagnostic",
                  "request_sha256": sha(request_path), "source_sha256": sha(__file__), "live": live,
                  "whole_wall_seconds": time.monotonic() - started, "tasks": reports,
                  "scientific_training_admitted": False, "predictive_superiority": False,
                  "oracle_outputs_scored_or_saved": False, "heldout_targets_loaded": False}
        write(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "ratios": {
            r["task"]: r["optimistic_oracle_exact_median_ratio"] for r in reports}}), flush=True)
    except BaseException as error:
        write(output / "FAILURE.json", {"type": type(error).__name__, "message": str(error),
              "whole_wall_seconds": time.monotonic() - started, "partial_artifacts_preserved": True,
              "scientific_training_admitted": False})
        raise


if __name__ == "__main__":
    main()
