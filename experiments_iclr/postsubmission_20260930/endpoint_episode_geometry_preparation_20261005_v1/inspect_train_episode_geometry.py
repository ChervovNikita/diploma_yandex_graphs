"""Separately gated TRAIN-only full-cycle geometry measurement; no fit.

Prepared source only. Numerical imports occur after source/input/review gates.
The candidate sizes, paired mask, and ordinary fallback remain unadopted.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import socket
import time

NODES, FEATURES = 3327, 3703
RAW_ROWS, SELF_ROWS, TRAIN_ROWS = 3918, 48, 3870
OUTER_SIZE, INNER_SIZE, ROUTES, SEED = 64, 256, 4, 20261005
TRAIN_SHA = "1e97ad3a73ecfeb69489aa6d92d02dc44c2925b4056a8e4cd4b548f1c0ebaa27"
FEATURE_AUTHORITY_SHA = "cf1f3c2d66735838b8acb860e466fe459ed8e35b15c559878233e9ff7bb62a68"
RUNTIME_AUTHORITY_SHA = "2dfa804a4761c559b6fa2f954f98300b81dabd4ac7f8b2da97e5301b9d253496"
RUNTIME = {"torch": "2.1.2+cu118", "numpy": "1.26.4", "torch_geometric": "2.7.0"}
LIMITS = {"cpu_threads": 2, "cpu_interop_threads": 1, "CUDA_visible_devices": "",
          "soft_budget_seconds": 300, "external_hard_budget_seconds": 360, "attempts": 1}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("Geometry module shadowed")
    return module


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    for key in ("source_review_approved", "TRAIN_geometry_measurement_authorized",
                "geometry_only", "external_360_second_hard_bound_confirmed", "feature_authority_verified"):
        if job.get(key) is not True:
            raise ValueError("Separate root geometry review required: " + key)
    for key in ("fits_authorized", "heldout_access_authorized", "feature_payload_access",
                "query_payload_export", "candidate_sizes_adopted", "paired_common_mask_adopted",
                "outer_only_fallback_adopted"):
        if job.get(key) is not False:
            raise ValueError("Geometry scope differs: " + key)
    if job.get("expected_hostname") != socket.gethostname() or job.get("execution_limits") != LIMITS:
        raise ValueError("Reviewed CPU host/bounds differ")
    if job.get("root_review_reference", "").startswith("PENDING") or not job.get("root_review_reference"):
        raise ValueError("Require concrete root review reference")
    candidate = {"outer_size": OUTER_SIZE, "inner_per_class_per_route": INNER_SIZE,
                 "routes": ROUTES, "seed": SEED, "cycle": 0, "cycles": 1}
    if job.get("candidate_geometry") != candidate:
        raise ValueError("Prospectively fixed candidate geometry differs; no outcome-adaptive size search")
    if str(args.output.resolve()) != job.get("output_directory") or args.output.exists():
        raise ValueError("Require the reviewed fresh geometry output directory")
    source = Path(__file__).resolve().parent
    manifest_path = source / "SOURCE_MANIFEST.json"
    if sha(manifest_path) != job.get("source_manifest_sha256"):
        raise ValueError("Reviewed source packet manifest differs")
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        if Path(row["path"]).name != row["path"] or sha(source / row["path"]) != row["sha256"]:
            raise ValueError("Sealed geometry packet changed")
    script_names = {Path(__file__).name, "episode_geometry.py", "native_episode_cycle.py"}
    if set(job.get("source_sha256", {})) != script_names:
        raise ValueError("Require exact three source hash bindings")
    for name, digest in job["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("Geometry source changed: " + name)
    if job.get("expected_runtime_versions") != RUNTIME:
        raise ValueError("Reviewed sampler runtime versions differ")
    for key, digest in (("feature_authority_metadata", FEATURE_AUTHORITY_SHA),
                        ("runtime_authority_metadata", RUNTIME_AUTHORITY_SHA)):
        authority = job[key]
        if authority["sha256"] != digest or sha(authority["path"]) != digest:
            raise ValueError("Pinned authority receipt metadata differs")
    if job["feature_authority_metadata"].get("declared_feature") != {
        "key": "entity_embedding", "dtype": "float32", "shape": [NODES, FEATURES],
        "sha256": "14b7d9de27cf0775147737126899c15151e82bcd4dde67afa31f3650f2609944"}:
        raise ValueError("Feature population authority differs")
    inputs = job["train_only_inputs"]
    if set(inputs) != {"train_pos.txt"}:
        raise ValueError("Only authenticated TRAIN may enter geometry inspection")
    record = inputs["train_pos.txt"]
    train_path = Path(record["path"]).resolve(strict=True)
    if train_path.name != "train_pos.txt" or record["sha256"] != TRAIN_SHA or sha(train_path) != TRAIN_SHA:
        raise ValueError("Pinned full TRAIN bytes differ")
    if record["counts"] != {"raw_rows": RAW_ROWS, "self_loops": SELF_ROWS, "native_nonself_rows": TRAIN_ROWS}:
        raise ValueError("Pinned full TRAIN count binding differs")
    args.output.mkdir(exist_ok=False)
    result = {"scope": "TRAIN_only_candidate_geometry_measurement", "fits": 0,
              "actual_optimizer_updates": 0, "VALID_TEST_access": False, "feature_payload_access": False,
              "accuracy_or_ranking_metrics": False, "query_payload_export": False,
              "source_manifest_sha256": sha(manifest_path), "job_sha256": sha(args.job),
              "candidate_geometry": candidate, "candidate_sizes_adopted": False,
              "paired_common_mask_adopted": False, "outer_only_fallback_adopted": False,
              "stages": []}

    def stage(name, **details):
        entry = {"name": name, "elapsed_seconds": time.monotonic()-start, **details}
        result["stages"].append(entry)
        print(json.dumps(entry), flush=True)
        if time.monotonic()-start > LIMITS["soft_budget_seconds"]:
            raise TimeoutError("TRAIN-only geometry soft CPU budget exceeded")

    failure = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import torch
        import torch_geometric
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        versions = {"torch": torch.__version__, "numpy": np.__version__,
                    "torch_geometric": torch_geometric.__version__}
        if versions != RUNTIME:
            raise ValueError("Installed native sampler runtime differs; no backend substitute")
        result["runtime"] = versions
        geometry = load_source("reviewed_episode_geometry", source / "episode_geometry.py")
        sampler = load_source("reviewed_native_episode_cycle", source / "native_episode_cycle.py")
        train, raw_rows, self_rows = [], 0, 0
        with train_path.open() as stream:
            for line in stream:
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    raise ValueError("Native TRAIN requires two tab-separated endpoints")
                u, v = map(int, fields)
                raw_rows += 1
                if not 0 <= u < NODES or not 0 <= v < NODES:
                    raise ValueError("TRAIN endpoint outside authorized population")
                if u == v:
                    self_rows += 1
                else:
                    train.append((u, v))
        if (raw_rows, self_rows, len(train)) != (RAW_ROWS, SELF_ROWS, TRAIN_ROWS):
            raise ValueError("Full TRAIN native filtering differs")
        full_graph = geometry.neighbors(train, NODES)
        py_state, np_state = sampler.random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state().clone()
        negative, order = sampler.native_cycle(train, NODES, SEED, cycle=0)
        np_after = np.random.get_state()
        restored = (py_state == sampler.random.getstate() and torch.equal(cpu_state, torch.get_rng_state())
                    and np_state[0] == np_after[0] and np.array_equal(np_state[1], np_after[1])
                    and np_state[2:] == np_after[2:])
        if not restored:
            raise ValueError("Prospective data draw changed global/model CPU RNG")
        geometry.validate_negative_bank(train, negative, NODES)
        batches = geometry.outer_batches(order, len(train), OUTER_SIZE)
        if len(batches) != 61 or len(batches[-1]) != 30 or sum(map(len, batches)) != TRAIN_ROWS:
            raise ValueError("Complete outer coverage/final partial episode differs")
        result["full_TRAIN"] = {"sha256": TRAIN_SHA, "raw_rows": raw_rows, "self_loops_removed": self_rows,
            "nonself_unique_undirected_rows": len(train), "nodes": NODES,
            "degree_all_nodes": geometry.distribution([len(row) for row in full_graph]),
            "positive_degree_CN_strata": geometry.strata_counts(train, full_graph),
            "negative_degree_CN_strata": geometry.strata_counts(negative, full_graph)}
        result["data_draw"] = {"negative_bank_rows": len(negative),
            "negative_bank_unique_equivalent_facts": len({geometry.canonical(pair) for pair in negative}),
            "negative_bank_sha256": object_sha(negative), "outer_order_sha256": object_sha(order),
            "CPU_Python_NumPy_RNG_restored": restored, "CUDA_RNG_seeded_or_touched": False,
            "prospective_stream": "independent data stream; no frozen-fit global sequence parity claim"}
        endpoint_streams = geometry.route_streams(SEED, ROUTES)
        control_streams = geometry.route_streams(SEED, ROUTES, control=True)
        endpoints, controls, rows, rejected, masks = [], [], [], [], Counter()
        for index, outer_ids in enumerate(batches):
            reference = geometry.endpoint_episode(train, negative, outer_ids, endpoint_streams, INNER_SIZE)
            control = geometry.matched_random_episode(train, negative, reference, control_streams, full_graph)
            endpoints.append(reference); controls.append(control)
            own_mask = geometry.mask_indices(reference)
            _, own_graph = geometry.support(train, NODES, own_mask)
            row = {"episode": index, "outer_positive_queries": len(outer_ids),
                   "outer_negative_queries": len(outer_ids), "tail_episode": index == len(batches)-1,
                   "outer_endpoint_count": len(reference["outer_endpoints"]),
                   "eligible_positive": reference["eligible_positive"],
                   "eligible_negative": reference["eligible_negative"],
                   "endpoint_feasible": reference["feasible"], "control_feasible": control["feasible"],
                   "endpoint_own_mask": geometry.describe_episode(train, negative, reference, full_graph, own_graph, own_mask)}
            if reference["feasible"] and control["feasible"]:
                for ref_route, ctrl_route in zip(reference["inner"], control["inner"]):
                    for pairs, key in ((train, "pos_ids"), (negative, "neg_ids")):
                        if geometry.strata_counts([pairs[i] for i in ref_route[key]], full_graph) != geometry.strata_counts([pairs[i] for i in ctrl_route[key]], full_graph):
                            raise ValueError("Per-route pre-mask degree/CN strata mismatch")
                common_mask = geometry.mask_indices(reference, control)
                _, common_graph = geometry.support(train, NODES, common_mask)
                masks.update(common_mask)
                endpoint_description = geometry.describe_episode(train, negative, reference, full_graph, common_graph, common_mask)
                if endpoint_description["inner_queries_touching_outer_endpoints"] != 0:
                    raise ValueError("Endpoint-separated arm touches outer endpoints")
                row.update(pre_mask_strata_match_exact_per_route=True,
                           paired_common_support_sha256=object_sha(sorted(set(range(len(train)))-common_mask)),
                           paired_endpoint=endpoint_description,
                           paired_matched_random=geometry.describe_episode(train, negative, control, full_graph, common_graph, common_mask),
                           extra_positive_facts_masked_for_pairing=len(common_mask-own_mask),
                           post_mask_strata_match_claimed=False)
            else:
                rejected.append(index)
                row["rejection_reason"] = reference.get("reason", control.get("reason"))
                if not reference["feasible"]:
                    fallback = geometry.outer_only_fallback(reference)
                    fallback_mask = geometry.mask_indices(fallback)
                    _, fallback_graph = geometry.support(train, NODES, fallback_mask)
                    row["unadopted_outer_only_geometry"] = geometry.describe_episode(train, negative, fallback, full_graph, fallback_graph, fallback_mask)
                    row["fallback_adopted"] = False
            rows.append(row)
            stage("episode_geometry", episode=index, endpoint_feasible=reference["feasible"],
                  control_feasible=control["feasible"], eligible_positive=reference["eligible_positive"],
                  eligible_negative=reference["eligible_negative"])
        result["episodes"] = rows
        result["schedule"] = {"outer_episodes": len(batches), "full_outer_episodes": 60,
            "last_outer_positive_negative_rows": [30, 30], "outer_positive_rows_dropped": 0,
            "outer_redraws": 0, "query_padding": 0, "fallback_updates": 0,
            "endpoint_feasible_episodes": sum(ep["feasible"] for ep in endpoints),
            "paired_feasible_episodes": len(batches)-len(rejected), "rejected_episode_indices": rejected,
            "pre_fit_candidate_admission": "REJECT_CANDIDATE" if rejected else "FEASIBLE_GEOMETRY_ONLY; root_recipe_review_pending",
            "pre_fit_policy": "Reject entire candidate before fit if any episode infeasible; no redraw or fallback adoption",
            "eligible_positive": geometry.distribution([ep["eligible_positive"] for ep in endpoints]),
            "eligible_negative": geometry.distribution([ep["eligible_negative"] for ep in endpoints])}
        result["endpoint_exposure"] = geometry.exposure_summary(train, negative, endpoints, NODES, ROUTES)
        result["matched_random_exposure"] = geometry.exposure_summary(train, negative, controls, NODES, ROUTES)
        result["paired_mask_exposure_all_TRAIN_ids"] = geometry.distribution([masks[i] for i in range(len(train))])
        result["control_contract"] = {"same_outer_queries": True, "same_pair_support": True,
            "same_per_route_query_class_counts_and_pre_mask_degree_CN_strata": True,
            "post_mask_strata": "measured separately for both arms and every route",
            "exact_per_ID_or_node_exposure_match_claimed": False,
            "endpoint_interpretation": "conditional within-graph transfer regularization; no unbiased cross-fitting or new-node guarantee"}
        stage("complete_full_cycle_measurement", rejected_episodes=len(rejected))
        result["status"] = "PASS_TRAIN_GEOMETRY_MEASUREMENT_ONLY"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        result.update(status="FAILED_TRAIN_GEOMETRY_MEASUREMENT", failure=failure)
    finally:
        result["elapsed_seconds"] = time.monotonic()-start
        result["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        target = args.output / ("FAILURE.json" if failure else "RESULT.json")
        target.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
