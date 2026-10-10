"""Complete fresh same-runtime SAGE F reader; preparation imports stdlib only.

Two admitted seed-block shards must close all 15 banks / 24 optimizer bundles /
33 native-body records before numerical imports or selected-array reads. No
model deserialization/forward, K path, historical anchors, or TEST reader exists.
"""
import argparse
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import socket
import statistics
import time


ARMS = ("shared4_F", "joint_untied4_F", "factorized_M1_F",
        "genuine_factorized_I4_F", "ordinary_M1_F")
SPECS = {
    "shared4_F": ("shared", True, 4, 1, 1),
    "joint_untied4_F": ("joint_untied", True, 4, 4, 1),
    "factorized_M1_F": ("single", True, 1, 1, 1),
    "genuine_factorized_I4_F": ("single", True, 4, 4, 4),
    "ordinary_M1_F": ("single", False, 1, 1, 1),
}
PRIMARY_REFERENCES = ("joint_untied4_F", "genuine_factorized_I4_F")
METRICS = ("pooled_accuracy_pct", "pooled_nll", "pooled_brier",
           "mean_member_accuracy_pct", "worst_member_accuracy_pct",
           "mean_member_nll", "worst_member_nll",
           "mean_member_brier", "worst_member_brier")
PROTOCOL = {
    "schema_version": 1, "active_stage": "F", "K": None,
    "arms": list(ARMS), "banks": 15, "optimizer_bundles": 24,
    "native_body_records": 33, "paired_seeds": 3,
    "raw_authority": "archived native float32 member/pooled error flags",
    "primary_candidate": "shared4_F",
    "primary_references": list(PRIMARY_REFERENCES),
    "raw_mean_accuracy_gain_pp": 0.2,
    "raw_all_three_nonnegative": True, "raw_at_least_two_positive": True,
    "calibrated_mean_NLL_deterioration_at_most": 0.02,
    "calibrated_each_seed_NLL_deterioration_at_most": 0.05,
    "calibrated_accuracy_is_reported_not_an_extra_gate": True,
    "costs_are_diagnostic_not_a_gate": True,
    "member_quality_is_diagnostic_not_a_veto": True,
    "calibration": {"folds": 5, "fold_seed": 11709,
                    "assignment": "CPU randperm then position modulo five",
                    "scalar_fits": 75, "updates": 500,
                    "optimizer": "CPU float64 Adam", "learning_rate": 0.01,
                    "betas": [0.9, 0.999], "epsilon": 1e-8,
                    "weight_decay": 0, "initial_log_T": 0,
                    "parameters_per_fit": 1, "endpoint": "fixed final update",
                    "operator": "temperature of member log probabilities before probability mean",
                    "retries": 0, "grid": False, "all_VALID_refit": False},
    "TEST_access": False, "historical_logit_anchors": False,
    "automatic_confirmation_launch": False,
    "scope": "Complete independently fitted/stopped/selected F procedures on encountered development; one SAGE backbone suffices; no novelty or desired verdict.",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def receipt(item):
    require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
            "Exact portable path/bytes/sha256 receipt required")
    p = Path(item["path"]).resolve()
    require(p.is_file() and type(item["bytes"]) is int and p.stat().st_size == item["bytes"],
            "Receipt file or size differs: " + str(p))
    require(sha(p) == item["sha256"], "Receipt digest differs: " + str(p))
    return p


def current_receipt(path):
    p = Path(path).resolve()
    require(p.is_file(), "Required complete artifact missing: " + str(p))
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def scoped(path, parent):
    p, root = Path(path).resolve(), Path(parent).resolve()
    require(p.is_relative_to(root), "Selected artifact escaped admitted shard")
    return p


def nondevice(cfg):
    return {k: v for k, v in cfg.items() if k != "device"}


def check_native_config(cfg, seeds):
    require(cfg.get("stage") == "F" and cfg.get("K") is None
            and cfg.get("k_selection_receipt") is None, "F only; dormant K is forbidden")
    require(cfg.get("seeds") == seeds, "The entire fixed three-seed roster must be retained")
    fixed = {"depth": 2, "hidden": 128, "dropout": 0.2, "learning_rate": 0.001,
             "max_updates": 1000, "patience_updates": 300,
             "member_seed_stride": 1000003, "factor_seed_offset": 2000003,
             "dropout_seed_offset": 3000007, "block_seed_offset": 4000037}
    require(all(cfg.get(k, v) == v for k, v in fixed.items()), "The admitted native F recipe differs")
    require(isinstance(cfg.get("device"), str) and cfg["device"].startswith("cuda"),
            "Each admitted F shard must name its logical CUDA device")


def preflight(spec):
    """Only stdlib, JSON metadata and byte hashing; no arrays/models imported."""
    require(spec.get("schema_version") == 1 and isinstance(spec.get("study_id"), str)
            and spec["study_id"], "Root-supplied study identity required")
    require(receipt(spec["reader_source"]) == Path(__file__).resolve(), "Reader source pin differs")
    require(read(receipt(spec["protocol"])) == PROTOCOL, "Prospectively pinned F protocol differs")
    seeds = spec["fixed_seed_roster"]
    require(isinstance(seeds, list) and len(seeds) == 3 and len(set(seeds)) == 3
            and all(type(s) is int and s >= 0 for s in seeds), "Exactly three distinct paired seeds")
    runtime = spec["runtime"]
    require(set(runtime) == {"hostname", "fingerprint_sha256", "allowed_gpu_uuids"},
            "Exact root runtime identity required")
    require(runtime["hostname"] == socket.gethostname(), "Reader is not on the admitted 77 host")
    require(isinstance(runtime["fingerprint_sha256"], str)
            and len(runtime["fingerprint_sha256"]) == 64, "Pinned same-runtime fingerprint required")
    allowed = runtime["allowed_gpu_uuids"]
    require(isinstance(allowed, list) and len(allowed) == len(set(allowed)) == 2
            and all(isinstance(x, str) and x.startswith("GPU-") for x in allowed),
            "Root must admit exactly two distinct physical GPU UUIDs")
    role_paths = {k: receipt(v) for k, v in spec["safe_roles"].items()}
    require(set(role_paths) == {"train", "valid"}, "Only exact safe TRAIN/VALID roles")
    role_hashes = {k: v["sha256"] for k, v in spec["safe_roles"].items()}
    sources = spec["sources"]
    require(set(sources) == {"training_family", "common_routes", "factors", "models", "run_base", "run_common"},
            "All six native training sources must be pinned")
    source_paths = {k: receipt(v) for k, v in sources.items()}
    source_hashes = {str(source_paths[k]): v["sha256"] for k, v in sources.items()}
    expected_cfg = spec["expected_nondevice_config"]
    require(isinstance(expected_cfg, dict) and "device" not in expected_cfg,
            "Full root nondevice config required; only device may differ")
    shards = spec["shards"]
    require(isinstance(shards, list) and len(shards) == 2
            and len({s["name"] for s in shards}) == 2, "Exactly two named seed-block shards")
    records, archives, checkpoints, bodies, seen_seeds, seen_gpus = {}, [], {}, [], set(), set()
    shard_records = []
    for shard in shards:
        root = Path(shard["root"]).resolve()
        require(root.is_dir(), "Admitted shard root missing")
        cfg_path = receipt(shard["submitted_config"])
        cfg = read(cfg_path)
        check_native_config(cfg, seeds)
        require(nondevice(cfg) == expected_cfg, "Nondevice configs differ between shards/root pin")
        require(read(root / "CONFIG.json") == cfg, "Shard saved config differs from submitted config")
        require(Path(cfg["train_npz"]).resolve() == role_paths["train"]
                and Path(cfg["valid_npz"]).resolve() == role_paths["valid"], "Safe data paths differ")
        require(Path(cfg["common_routes"]).resolve() == source_paths["common_routes"]
                and Path(cfg["factors"]).resolve() == source_paths["factors"], "Native route/factor paths differ")
        for name in ("models", "run_base", "run_common"):
            require((Path(cfg["native_repo"]) / (name + ".py")).resolve() == source_paths[name],
                    "Native repository path differs: " + name)
        complete_path = scoped(receipt(shard["complete_family"]), root)
        complete = read(complete_path)
        block = complete.get("completed_seed_block")
        require(isinstance(block, list) and block and len(block) == len(set(block))
                and block == [s for s in seeds if s in block] and not seen_seeds.intersection(block),
                "Disjoint ordered complete seed blocks required")
        seen_seeds.update(block)
        expected_complete = {"complete": True, "stage": "F", "K": None,
                             "fixed_seed_roster": seeds, "fixed_arm_roster": list(ARMS),
                             "full_expected_groups": 15, "full_expected_optimizer_acquisition_bundles": 24,
                             "full_expected_native_body_fit_records": 33,
                             "full_native_trajectories_per_roster_forward": 42,
                             "groups": 5 * len(block), "fit_units": 8 * len(block),
                             "optimizer_acquisition_bundles": 8 * len(block),
                             "native_body_fit_records": 11 * len(block),
                             "expected_groups": 5 * len(block), "expected_fit_units": 8 * len(block),
                             "expected_native_body_fit_records": 11 * len(block),
                             "full_same_runtime_stage_complete": False,
                             "complete_F_implies_complete_K": False, "TEST_access": False,
                             "comparisons": []}
        require(all(complete.get(k) == v for k, v in expected_complete.items()),
                "Incomplete, wrong-stage, or prematurely compared seed shard")
        require(complete.get("config_sha256") == shard["submitted_config"]["sha256"]
                and complete.get("source_sha256") == source_hashes, "Shard config/source custody differs")
        require(complete.get("learning_lock") == {"stage": "F", "K": None, "pilot_provenance_read": False},
                "Active F must not read prior pilot quality")
        require(read(root / "SOURCE_HASHES.json") == {"config_sha256": shard["submitted_config"]["sha256"],
                                                      "files": source_hashes}, "Source hash receipt differs")
        execution = read(receipt(shard["execution_receipt"]))
        owner = read(receipt(shard["owner_end_receipt"]))
        gpu = execution.get("gpu_uuid")
        require(gpu in allowed and gpu not in seen_gpus, "Shard physical GPU is not uniquely admitted")
        seen_gpus.add(gpu)
        common_receipt = {"schema_version": 1, "study_id": spec["study_id"], "stage": "F", "K": None,
                          "hostname": runtime["hostname"], "runtime_fingerprint_sha256": runtime["fingerprint_sha256"],
                          "gpu_uuid": gpu, "logical_device": cfg["device"], "seed_block": block,
                          "submitted_config_sha256": shard["submitted_config"]["sha256"],
                          "training_family_sha256": sources["training_family"]["sha256"],
                          "safe_role_hashes": role_hashes, "TEST_access": False}
        require(all(execution.get(k) == v and owner.get(k) == v for k, v in common_receipt.items()),
                "Execution/owner physical GPU, runtime, config, source or safe-role receipt differs")
        require(execution.get("status") == "HOST_GPU_SOURCE_DATA_ADMITTED"
                and owner.get("status") == "COMPLETE_EXIT_0" and owner.get("exit_code") == 0,
                "Actual admitted execution and successful owner completion required")
        groups = complete["results"]
        require(len(groups) == 5 * len(block), "Every shard group must be present")
        shard_keys, unit_count, body_count, operation_sum = set(), 0, 0, {}
        for group in groups:
            arm, seed = group["arm"], group["seed"]
            key = (seed, arm)
            require(seed in block and arm in ARMS and key not in records, "Duplicate or foreign bank")
            shard_keys.add(key)
            architecture, factorized, members, body_total, bundles = SPECS[arm]
            bank_dir = root / f"{arm}_seed{seed}"
            require(read(bank_dir / "RESULT.json") == group, "Bank result differs from complete family")
            require(group.get("stage") == "F" and group.get("K") is None
                    and group.get("architecture") == architecture and group.get("serving") == "probability_mean"
                    and group.get("optimizer_acquisition_bundles") == bundles
                    and group.get("native_bodies_fitted") == body_total
                    and group.get("native_trajectories_per_forward") == members,
                    "Bank architecture/body/bundle/serving identity differs")
            selection = "independent_own_body_then_pool" if bundles == 4 else "coherent_bank" if members == 4 else "own_body"
            require(group.get("selection_scope") == selection and len(group["fits"]) == bundles,
                    "Coherent versus independent selector identity differs")
            for m, unit in enumerate(group["fits"]):
                fitted_members = 1 if bundles == 4 else members
                nbody = 4 if architecture == "joint_untied" else 1
                require(unit.get("member") == m and unit.get("architecture") == architecture
                        and unit.get("factorized") == factorized
                        and unit.get("optimizer_acquisition_bundles") == 1
                        and unit.get("native_bodies_fitted") == nbody
                        and unit.get("learning_objective") == "own_only"
                        and unit.get("within_route_supcon") is False
                        and unit.get("supcon_coefficient") == 0
                        and unit.get("supcon_temperature") is None
                        and unit.get("preclassifier_width") == 128
                        and unit.get("selection_scope") == ("coherent_bank" if fitted_members > 1 else "own_body"),
                        "Unit is not the declared fresh F acquisition")
                selected, updates = unit["selected_step"], unit["completed_updates"]
                require(type(selected) is int and type(updates) is int and 1 <= selected <= updates <= 1000,
                        "Selected/completed update range differs")
                checkpoint = scoped(unit["selected_state"], root)
                require(checkpoint == (bank_dir / f"member{m}" / "selected.pt").resolve()
                        and read(bank_dir / f"member{m}" / "RESULT.json") == unit,
                        "Native selected checkpoint/unit result path differs")
                cp = current_receipt(checkpoint)
                require(cp["bytes"] == unit["costs"]["selected_checkpoint_bytes"]
                        and str(checkpoint) not in checkpoints, "Checkpoint size or unique acquisition differs")
                checkpoints[str(checkpoint)] = dict(cp, arm=arm, seed=seed, acquisition_member=m,
                                                   selected_step=selected, native_body_records=nbody)
                body_rows = unit["body_fit_records"]
                expected_indices = list(range(4)) if architecture == "joint_untied" else [m]
                require(len(body_rows) == nbody and [r["body_index"] for r in body_rows] == expected_indices,
                        "Physical body records differ")
                for body in body_rows:
                    b = body["body_index"]
                    native_m = 0 if architecture in ("shared", "joint_untied") else b
                    native_seed = seed + 1000003 * native_m
                    routes = list(range(4)) if architecture == "shared" else [b]
                    expected_routes = [{"route": r, "factor_seed": seed + 1000003 * r + 2000003 if factorized else None,
                                        "dropout_seed": seed + 1000003 * r + 3000007} for r in routes]
                    parameter_scope = "body" if architecture == "shared" else f"bodies.{b}" if architecture == "joint_untied" else "native_body"
                    require(body.get("native_initialization_seed") == native_seed
                            and body.get("factorized") == factorized
                            and body.get("native_parameter_scope") == parameter_scope
                            and body.get("route_factor_dropout_seeds") == expected_routes
                            and scoped(body["selected_state"], root) == checkpoint
                            and body.get("selected_step") == selected
                            and body.get("selection_scope") == ("coherent_bank" if fitted_members > 1 else "own_body"),
                            "Native starts, streams, body ownership or coherent selected state differs")
                    bodies.append(dict(body, arm=arm, seed=seed, checkpoint_sha256=cp["sha256"],
                                       checkpoint_bytes=cp["bytes"]))
                require(unit.get("route_seed_metadata") == [b["route_factor_dropout_seeds"] for b in body_rows],
                        "Exported stream identity differs")
                ops = unit["operation_counts"]
                expected_ops = {"updates": updates, "backwards": updates, "adam_steps": updates,
                                "train_fullgraph_native_trajectories": fitted_members * updates,
                                "valid_selection_fullgraph_native_trajectories": fitted_members * updates,
                                "selected_valid_fullgraph_native_trajectories": fitted_members,
                                "native_graph_block_calls": 2 * fitted_members * (2 * updates + 1),
                                "train_probability_pool_loss_evaluations": updates if fitted_members == 4 else 0,
                                "train_probability_pool_backward_evaluations": 0}
                require(all(ops.get(k) == v for k, v in expected_ops.items())
                        and all(v == 0 for k, v in ops.items() if k.startswith("supcon_")),
                        "Paid native F operation records differ")
                for name, value in ops.items():
                    operation_sum[name] = operation_sum.get(name, 0) + value
                unit_count += 1
                body_count += nbody
            native_cost_keys = set(group["fits"][0]["costs"])
            require(all(set(u["costs"]) == native_cost_keys for u in group["fits"]),
                    "Acquisition cost schemas differ")
            for name in native_cost_keys:
                values = [u["costs"][name] for u in group["fits"]]
                require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values),
                        "Nonfinite/negative native cost")
                expected_cost = max(values) if "peak" in name else sum(values)
                require(group["costs"].get(name) == expected_cost, "Native group cost aggregation differs")
            gc = group["costs"]
            require(isinstance(gc.get("selected_pool_seconds"), (int, float))
                    and math.isfinite(gc["selected_pool_seconds"]) and gc["selected_pool_seconds"] >= 0
                    and gc.get("selected_serving_readout_seconds") == gc["selected_forward_and_metrics_seconds"] + gc["selected_pool_seconds"]
                    and gc.get("selected_restore_forward_and_pool_seconds") == gc["selected_restore_seconds"] + gc["selected_serving_readout_seconds"]
                    and gc.get("acquisition_restore_selected_readout_seconds") == gc["acquisition_seconds"] + gc["selected_restore_forward_and_pool_seconds"],
                    "Selected restore/serving and acquisition timing definitions differ")
            archive = current_receipt(bank_dir / "selected_VALID.npz")
            archives.append(dict(archive, arm=arm, seed=seed, members=members,
                                 physical_gpu_uuid=gpu, shard=shard["name"]))
            records[key] = group
        require(shard_keys == {(s, a) for s in block for a in ARMS}
                and unit_count == 8 * len(block) and body_count == 11 * len(block)
                and operation_sum == complete.get("operation_counts"), "Shard complete native roster/cost closure failed")
        shard_records.append({"name": shard["name"], "root": str(root), "seed_block": block,
                              "gpu_uuid": gpu, "submitted_config": shard["submitted_config"],
                              "execution_receipt": shard["execution_receipt"],
                              "owner_end_receipt": shard["owner_end_receipt"],
                              "complete_family": shard["complete_family"], "operation_counts": operation_sum})
    require(seen_seeds == set(seeds) and seen_gpus == set(allowed)
            and set(records) == {(s, a) for s in seeds for a in ARMS}
            and len(archives) == 15 and len(checkpoints) == 24 and len(bodies) == 33,
            "Entire same-runtime 15-bank/24-bundle/33-body join must close before arrays")
    return {"complete": True, "stage": "F", "K": None, "TEST_access": False,
            "seeds": seeds, "records": records, "archives": archives,
            "checkpoints": list(checkpoints.values()), "body_records": bodies,
            "shards": shard_records, "role_paths": role_paths,
            "runtime": runtime, "sources": sources,
            "nondevice_config_sha256": canonical_hash(expected_cfg),
            "current_checkpoint_hash_scope": "First-current byte identities; no deserialization, historical hash attestation or native replay."}


def payloads(np, context):
    """Called only after the complete JSON/hash/physical-runtime gate."""
    with np.load(context["role_paths"]["valid"], allow_pickle=False) as v:
        require(set(v.files) == {"ids", "y"}, "Only safe VALID ids/y allowed")
        ids, labels = v["ids"].copy(), v["y"].copy()
    require(ids.dtype == labels.dtype == np.int64 and ids.shape == labels.shape == (5274,)
            and len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701
            and labels.min() >= 0 and labels.max() < 10, "Exact WikiCS VALID role schema")
    arrays = {}
    for item in context["archives"]:
        require(current_receipt(item["path"]) == {k: item[k] for k in ("path", "bytes", "sha256")},
                "Archive changed after complete custody gate")
        with np.load(item["path"], allow_pickle=False) as f:
            require(set(f.files) == {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"},
                    "Exact selected safe VALID archive keys required")
            a = {k: f[k].copy() for k in f.files}
        members = item["members"]
        require(np.array_equal(a["ids"], ids) and np.array_equal(a["y"], labels), "Ordered VALID identity differs")
        require(a["raw_logits"].dtype == a["probability_mean"].dtype == np.float32
                and a["raw_logits"].shape == (members, 5274, 10)
                and a["probability_mean"].shape == (5274, 10)
                and a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_
                and a["member_errors"].shape == (members, 5274)
                and a["pooled_errors"].shape == (5274,), "Selected schema/dimensions/dtypes differ")
        require(np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all(),
                "Nonfinite selected raw archive")
        require(np.array_equal(a["pooled_errors"], a["probability_mean"].argmax(-1) != labels),
                "Archived probability-mean/error authority differs")
        arrays[item["seed"], item["arm"]] = a
    require(len(arrays) == 15, "All fifteen exact archive schemas before scoring/calibration")
    return arrays, ids, labels


def logsumexp(np, x, axis):
    high = x.max(axis=axis, keepdims=True)
    return (high + np.log(np.exp(x - high).sum(axis=axis, keepdims=True))).squeeze(axis)


def strict_mask(np, logits, labels):
    target = logits[:, np.arange(len(labels)), labels]
    better = (logits > target[:, :, None]).all(0)
    better[np.arange(len(labels)), labels] = False
    return better.any(-1)


def cohort(np, mask, correct, labels):
    return {"nodes": int(mask.sum()), "corrected": int((mask & correct).sum()),
            "classes": [{"class": c, "nodes": int((mask & (labels == c)).sum()),
                         "corrected": int((mask & correct & (labels == c)).sum())} for c in range(10)]}


def score(np, log_members, probability, member_errors, pooled_errors, labels):
    member_correct, pooled_correct = ~member_errors, ~pooled_errors
    coverage = member_correct.any(0)
    log_pool = logsumexp(np, log_members, 0) - math.log(len(log_members))
    truth = np.eye(10, dtype=np.float64)[labels]
    member_nll = -log_members[:, np.arange(len(labels)), labels]
    pool_nll = -log_pool[np.arange(len(labels)), labels]
    member_brier = ((np.exp(log_members) - truth[None]) ** 2).sum(-1)
    pool_brier = ((probability.astype(np.float64) - truth) ** 2).sum(-1)
    def summary(mask):
        n = int(mask.sum())
        require(n > 0, "Every reported class must occur in VALID")
        ma, mn, mb = member_correct[:, mask].mean(1) * 100, member_nll[:, mask].mean(1), member_brier[:, mask].mean(1)
        q = {"pooled_accuracy_pct": float(pooled_correct[mask].mean() * 100),
             "pooled_nll": float(pool_nll[mask].mean()), "pooled_brier": float(pool_brier[mask].mean()),
             "member_accuracy_pct": ma.tolist(), "member_nll": mn.tolist(), "member_brier": mb.tolist(),
             "mean_member_accuracy_pct": float(ma.mean()), "worst_member_accuracy_pct": float(ma.min()),
             "mean_member_nll": float(mn.mean()), "worst_member_nll": float(mn.max()),
             "mean_member_brier": float(mb.mean()), "worst_member_brier": float(mb.max())}
        counts = {"nodes": n, "pooled_correct": int(pooled_correct[mask].sum()),
                  "coverage": int(coverage[mask].sum()),
                  "lost_correct_alternatives": int((coverage & ~pooled_correct & mask).sum()),
                  "aggregation_only_correct": int((~coverage & pooled_correct & mask).sum()),
                  "unavailable_alternatives": int((~coverage & mask).sum()),
                  "served_alternatives": int((coverage & pooled_correct & mask).sum()),
                  "member_correct": member_correct[:, mask].sum(1).tolist()}
        require(counts["pooled_correct"] == counts["coverage"] - counts["lost_correct_alternatives"] + counts["aggregation_only_correct"],
                "Exact coverage/loss/aggregation-only identity")
        return q, counts
    quality, counts = summary(np.ones(len(labels), dtype=bool))
    classes = []
    for c in range(10):
        q, z = summary(labels == c)
        classes.append(dict(q, **z, **{"class": c}))
    return {"quality": quality, "counts": counts, "classes": classes,
            "P": pooled_correct, "C": coverage, "L": coverage & ~pooled_correct,
            "G": ~coverage & pooled_correct, "member_correct": member_correct}


def paired(values):
    deltas = list(values)
    require(len(deltas) == 3 and all(math.isfinite(v) for v in deltas), "Three finite paired seed differences")
    mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
    radius = 4.302652729911275 * sd / math.sqrt(3)
    flips = [abs(statistics.mean(sign * v for sign, v in zip(signs, deltas)))
             for signs in itertools.product((-1, 1), repeat=3)]
    return {"seed_deltas": deltas, "mean": mean, "sample_sd": sd,
            "min": min(deltas), "max": max(deltas),
            "descriptive_95pct_t_interval_df2": [mean - radius, mean + radius],
            "nonnegative_seed_count": sum(v >= 0 for v in deltas),
            "positive_seed_count": sum(v > 0 for v in deltas),
            "exact_sign_flip_two_sided_p": sum(v >= abs(mean) - 1e-12 for v in flips) / 8,
            "uncertainty_scope": "Three paired optimizer seeds on one encountered development graph; descriptive, not independent-node inference or significance."}


def error_diagnostic(np, candidate, reference, labels):
    p, r, c, d = candidate["P"], reference["P"], candidate["C"], reference["C"]
    repair, harm, new, removed = ~r & p, r & ~p, ~d & c, d & ~c
    def counts(mask):
        delta = {k: int(candidate["counts"][k] - reference["counts"][k]) for k in
                 ("pooled_correct", "coverage", "lost_correct_alternatives", "aggregation_only_correct")}
        if not mask.all():
            delta = {"pooled_correct": int(((p.astype(int) - r.astype(int)) * mask).sum()),
                     "coverage": int(((c.astype(int) - d.astype(int)) * mask).sum()),
                     "lost_correct_alternatives": int(((candidate["L"].astype(int) - reference["L"].astype(int)) * mask).sum()),
                     "aggregation_only_correct": int(((candidate["G"].astype(int) - reference["G"].astype(int)) * mask).sum())}
        repairs, harms = int((repair & mask).sum()), int((harm & mask).sum())
        require(repairs - harms == delta["pooled_correct"] == delta["coverage"] - delta["lost_correct_alternatives"] + delta["aggregation_only_correct"],
                "Complete paired repair/harm count identity")
        return {"repairs": repairs, "harms": harms, "count_decomposition_delta": delta,
                "new_coverage": int((new & mask).sum()), "removed_coverage": int((removed & mask).sum()),
                "new_coverage_served": int((new & p & mask).sum()),
                "new_coverage_lost": int((new & ~p & mask).sum()),
                "repair_causes": {"new_member_alternative": int((repair & c & ~d & mask).sum()),
                                  "both_banks_have_member_alternative": int((repair & c & d & mask).sum()),
                                  "aggregation_only": int((repair & ~c & mask).sum())},
                "harm_causes": {"available_alternative_lost": int((harm & c & mask).sum()),
                                "removed_member_alternative": int((harm & ~c & d & mask).sum()),
                                "lost_aggregation_only": int((harm & ~c & ~d & mask).sum())}}
    whole = counts(np.ones(len(labels), dtype=bool))
    whole["classes"] = [dict(counts(labels == c), **{"class": c}) for c in range(10)]
    return whole


def contrast(np, values, candidate, reference, labels, seeds, scope="full_F_procedures"):
    if any(values[s, candidate] is None or values[s, reference] is None for s in seeds):
        return {"candidate": candidate, "reference": reference, "status": "failed_calibration_retained",
                "quality_deltas": None, "missing_seed_readouts": [s for s in seeds if values[s, candidate] is None or values[s, reference] is None]}
    rows = [dict(error_diagnostic(np, values[s, candidate], values[s, reference], labels), seed=s) for s in seeds]
    summed = {"repairs": sum(r["repairs"] for r in rows), "harms": sum(r["harms"] for r in rows),
              "count_decomposition_delta": {k: sum(r["count_decomposition_delta"][k] for r in rows) for k in rows[0]["count_decomposition_delta"]}}
    member_count = len(values[seeds[0], candidate]["quality"]["member_accuracy_pct"])
    equal_members = member_count == len(values[seeds[0], reference]["quality"]["member_accuracy_pct"])
    members = [{"member": m, **{label: paired(values[s, candidate]["quality"][metric][m] - values[s, reference]["quality"][metric][m] for s in seeds)
                                for label, metric in (("accuracy_pp", "member_accuracy_pct"), ("nll", "member_nll"), ("brier", "member_brier"))}}
               for m in range(member_count)] if equal_members else None
    return {"candidate": candidate, "reference": reference, "status": "finite_complete",
            "quality_deltas": {k: paired(values[s, candidate]["quality"][k] - values[s, reference]["quality"][k] for s in seeds) for k in METRICS},
            "classes": [{"class": c, **{k: paired(values[s, candidate]["classes"][c][k] - values[s, reference]["classes"][c][k] for s in seeds) for k in METRICS}} for c in range(10)],
            "per_seed_error_diagnostics": rows, "summed_three_seed_readout_diagnostics": summed,
            "members": members, "member_pairing": "Route index only for equally sized banks; does not certify identical realized bodies or selected steps." if equal_members else "No route pairing for differently sized banks.",
            "comparison_scope": scope, "error_categories_are_descriptive_not_causal": True,
            "scope_explanation": "Fixed selected native bank; only equal-policy temperature changes; no member alternative acquired." if scope == "same_selected_state_calibration" else
                "Separately fitted, stopped and selected complete procedures; matched starts do not imply equal optimizer trajectories. No isolated same-state learning/averaging causal attribution.",
            "coverage_category_note": "Both-bank coverage does not certify the same route/vector or exclusive pooling causality.",
            "count_scope": "Three dependent readouts of the same 5274 development nodes."}


def calibrate(np, torch, arrays, ids, labels, seeds, output):
    start = time.perf_counter()
    permutation = torch.randperm(len(labels), generator=torch.Generator(device="cpu").manual_seed(11709))
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    y = torch.from_numpy(labels)
    records, calibrated, attempted = [], {}, 0
    for arm in ARMS:
        for seed in seeds:
            a = arrays[seed, arm]
            log_p = torch.from_numpy(a["raw_logits"]).to(torch.float64).log_softmax(-1)
            log_members = torch.full(log_p.shape, float("nan"), dtype=torch.float64)
            log_pool = torch.full((5274, 10), float("nan"), dtype=torch.float64)
            fits = []
            for fold in range(5):
                attempted += 1
                before, completed, phase = time.perf_counter(), 0, "initialization"
                trace = []
                fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                try:
                    log_T = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device="cpu"))
                    optimizer = torch.optim.Adam([log_T], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
                    fit_log, fit_y = log_p[:, fit_ids], y[fit_ids]
                    for update in range(1, 501):
                        phase = "objective"
                        optimizer.zero_grad(set_to_none=True)
                        scaled = (fit_log / log_T.exp().reshape(1, 1, 1)).log_softmax(-1)
                        pooled = torch.logsumexp(scaled, dim=0) - math.log(len(log_p))
                        loss = torch.nn.functional.nll_loss(pooled, fit_y)
                        require(torch.isfinite(loss).item(), "Nonfinite fixed temperature objective")
                        phase = "backward"
                        loss.backward()
                        require(log_T.grad is not None and torch.isfinite(log_T.grad).all().item(), "Nonfinite/missing scalar gradient")
                        phase = "optimizer_step"
                        optimizer.step()
                        completed = update
                        if update in (1, 100, 200, 300, 400, 500):
                            trace.append({"update": update, "fit_objective_before_update": loss.item()})
                    phase = "endpoint"
                    with torch.no_grad():
                        temperature = log_T.exp()
                        require(torch.isfinite(temperature).all().item() and (temperature > 0).all().item(), "One finite positive final temperature")
                        held_members = (log_p[:, held_ids] / temperature.reshape(1, 1, 1)).log_softmax(-1)
                        held_pool = torch.logsumexp(held_members, dim=0) - math.log(len(log_p))
                        require(torch.isfinite(held_members).all().item() and torch.isfinite(held_pool).all().item(), "Finite fixed calibration endpoint")
                        log_members[:, held_ids], log_pool[held_ids] = held_members, held_pool
                    fit = {"status": "finite_fixed_endpoint", "fold": fold, "updates": 500,
                           "parameters": 1, "fit_nodes": len(fit_ids), "held_nodes": len(held_ids),
                           "log_T": log_T.detach().tolist(), "temperature": temperature.item(),
                           "trace": trace, "selected_endpoint": "fixed_final_update_no_heldout_selector"}
                except (FloatingPointError, RuntimeError, ValueError, MemoryError) as error:
                    fit = {"status": "failed_retained", "fold": fold, "error": str(error),
                           "failure_phase": phase, "requested_updates": 500,
                           "actual_updates_before_failure": None if phase == "optimizer_step" else completed,
                           "known_completed_updates_lower_bound": completed, "trace": trace}
                fit["seconds"] = time.perf_counter() - before
                fits.append(fit)
            finite = all(f["status"] == "finite_fixed_endpoint" for f in fits) and torch.isfinite(log_members).all().item() and torch.isfinite(log_pool).all().item()
            record = {"arm": arm, "seed": seed, "status": "finite_complete_five_fold" if finite else "failed_retained",
                      "attempted_scalar_fits": 5, "fits": fits, "fit_seconds_sum": sum(f["seconds"] for f in fits)}
            stream = io.BytesIO()
            np.savez_compressed(stream, ids=ids, folds=folds.numpy(), member_log_probability=log_members.numpy(), pool_log_probability=log_pool.numpy())
            name = f"CALIBRATED_{arm}_seed{seed}_OOF.npz"
            data = stream.getvalue()
            (output / name).write_bytes(data)
            record["OOF_archive"] = {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            calibrated[seed, arm] = (log_members.numpy(), log_pool.numpy()) if finite else None
            records.append(record)
    require(attempted == 75 and len(records) == 15 and all(len(r["fits"]) == 5 for r in records),
            "All seventy-five scalar endpoint attempts must be terminal before contrasts")
    success = sum(f["status"] == "finite_fixed_endpoint" for r in records for f in r["fits"])
    return calibrated, records, {"attempted_scalar_fits": 75, "successful_scalar_fits": success,
           "failed_scalar_fits": 75 - success, "requested_updates_per_fit": 500,
           "requested_scalar_updates": 37500, "known_successful_scalar_updates": success * 500,
           "failed_actual_update_counts_known": all(f["actual_updates_before_failure"] is not None for r in records for f in r["fits"] if f["status"] == "failed_retained"),
           "fit_seconds_sum": sum(r["fit_seconds_sum"] for r in records), "wall_seconds": time.perf_counter() - start,
           "native_models_or_forwards_added": 0, "no_retry_or_final_refit": True,
           "scope": "Known successful scalar work; failed steps remain unknown when an optimizer call fails; fit sums are nested in wall time."}, {
           "seed": 11709, "counts": [(folds == k).sum().item() for k in range(5)],
           "assignment_sha256": hashlib.sha256(folds.numpy().tobytes()).hexdigest(),
           "ordered_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
           "scope": "Label-free assignment after native VALID selection; encountered development, not complete-pipeline unused confirmation."}


def scalar_costs(rows):
    units = [u for r in rows for u in r["fits"]]
    return {"banks": len(rows), "optimizer_acquisition_bundles": len(units),
            "native_body_records": sum(r["native_bodies_fitted"] for r in rows),
            "operation_counts": {k: sum(u["operation_counts"][k] for u in units) for k in units[0]["operation_counts"]},
            "summed_group_costs": {k: sum(r["costs"][k] for r in rows) for k in rows[0]["costs"] if "peak" not in k},
            "maximum_recorded_peaks": {k: max(r["costs"][k] for r in rows) for k in rows[0]["costs"] if "peak" in k},
            "selected_parameters_per_seed": [{"seed": r["seed"], "parameters": r["costs"]["parameters"],
                                               "inference_parameter_bytes": r["costs"]["inference_parameter_bytes"]} for r in rows],
            "scope": "Bundle/group timers and peaks retain source definitions; sums are not concurrent wall time. Process RSS is a lifetime peak; GPU overlap supplied by root. Costs diagnostic, no post hoc cost veto."}


def public_score(value):
    return None if value is None else {k: v for k, v in value.items() if k in ("quality", "counts", "classes")}


def analyze(np, raw, calibrated, arrays, labels, context):
    seeds = context["seeds"]
    raw_contrasts = {a + "_minus_" + b: contrast(np, raw, a, b, labels, seeds) for a in ARMS for b in ARMS if a != b}
    calibrated_contrasts = {a + "_minus_" + b: contrast(np, calibrated, a, b, labels, seeds) for a in ARMS for b in ARMS if a != b}
    for a in ARMS:
        for b in ARMS:
            if a == b:
                continue
            cost_deltas = {k: paired(context["records"][s, a]["costs"][k] - context["records"][s, b]["costs"][k] for s in seeds)
                           for k in ("parameters", "inference_parameter_bytes", "acquisition_seconds", "selected_restore_seconds", "selected_serving_readout_seconds", "acquisition_restore_selected_readout_seconds")}
            raw_contrasts[a + "_minus_" + b]["diagnostic_cost_deltas"] = cost_deltas
            calibrated_contrasts[a + "_minus_" + b]["diagnostic_native_cost_deltas"] = cost_deltas
    same_state = {}
    for arm in ARMS:
        values = {(s, "raw"): raw[s, arm] for s in seeds}
        values.update({(s, "calibrated"): calibrated[s, arm] for s in seeds})
        same_state[arm] = contrast(np, values, "calibrated", "raw", labels, seeds, "same_selected_state_calibration")
    gates = {}
    for ref in PRIMARY_REFERENCES:
        key = "shared4_F_minus_" + ref
        accuracy = raw_contrasts[key]["quality_deltas"]["pooled_accuracy_pct"]
        raw_gate = {"paired_raw_accuracy_pp": accuracy, "mean_at_least_0_2_pp": accuracy["mean"] >= 0.2,
                    "all_three_nonnegative": accuracy["nonnegative_seed_count"] == 3,
                    "at_least_two_positive": accuracy["positive_seed_count"] >= 2}
        raw_gate["pass"] = all(raw_gate[k] for k in ("mean_at_least_0_2_pp", "all_three_nonnegative", "at_least_two_positive"))
        cal = calibrated_contrasts[key]
        finite = cal["status"] == "finite_complete"
        confidence = {"finite_complete": finite, "paired_calibrated_NLL_deterioration": cal["quality_deltas"]["pooled_nll"] if finite else None,
                      "paired_calibrated_accuracy_pp": cal["quality_deltas"]["pooled_accuracy_pct"] if finite else None,
                      "calibrated_accuracy_is_reported_not_an_extra_gate": True}
        confidence["mean_NLL_deterioration_at_most_0_02"] = finite and confidence["paired_calibrated_NLL_deterioration"]["mean"] <= 0.02
        confidence["each_seed_NLL_deterioration_at_most_0_05"] = finite and confidence["paired_calibrated_NLL_deterioration"]["max"] <= 0.05
        confidence["pass"] = finite and confidence["mean_NLL_deterioration_at_most_0_02"] and confidence["each_seed_NLL_deterioration_at_most_0_05"]
        gates[ref] = {"raw_accuracy": raw_gate, "confidence_protection": confidence}
    policy = {"candidate": "shared4_F", "required_references": list(PRIMARY_REFERENCES), "contrasts": gates,
              "raw_accuracy_screen_pass": all(g["raw_accuracy"]["pass"] for g in gates.values()),
              "calibrated_confidence_protection_pass": all(g["confidence_protection"]["pass"] for g in gates.values()),
              "member_quality_diagnostic_not_veto": True, "costs_diagnostic_not_veto": True,
              "no_K_choice": True, "no_automatic_confirmation_launch": True,
              "favorable_findings_preserved_on_failure": True, "scope": PROTOCOL["scope"]}
    policy["joint_accuracy_confidence_development_clue"] = policy["raw_accuracy_screen_pass"] and policy["calibrated_confidence_protection_pass"]
    policy["accuracy_only_clue_with_confidence_tradeoff"] = policy["raw_accuracy_screen_pass"] and not policy["calibrated_confidence_protection_pass"]
    aggregates = []
    for readout, values in (("raw", raw), ("calibrated", calibrated)):
        for arm in ARMS:
            rs = [values[s, arm] for s in seeds]
            native = [context["records"][s, arm] for s in seeds]
            if any(r is None for r in rs):
                aggregate = {"status": "failed_calibration_retained", "missing_seed_readouts": [s for s, r in zip(seeds, rs) if r is None], "acquisition_costs": scalar_costs(native)}
            else:
                aggregate = {"status": "finite_complete", "mean_quality": {k: statistics.mean(r["quality"][k] for r in rs) for k in METRICS},
                             "summed_three_seed_readout_counts": {k: sum(r["counts"][k] for r in rs) for k in rs[0]["counts"] if k != "member_correct"},
                             "acquisition_costs": scalar_costs(native), "count_scope": "Three dependent readouts of the same 5274 development nodes."}
            aggregates.append({"readout": readout, "arm": arm, "aggregate": aggregate})
    diagnostics = []
    for seed in seeds:
        masks = {arm: strict_mask(np, arrays[seed, arm]["raw_logits"], labels) for arm in ARMS}
        for arm in ARMS:
            a, r = arrays[seed, arm], raw[seed, arm]
            raw64 = a["raw_logits"].astype(np.float64)
            log_p = raw64 - logsumexp(np, raw64, -1)[..., None]
            d = {"arm": arm, "seed": seed, "current_bank_strict_rival": cohort(np, masks[arm], r["P"], labels),
                 "fixed_F_reference_strict_rival_cohorts_raw": {ref: cohort(np, mask, r["P"], labels) for ref, mask in masks.items()},
                 "fixed_F_reference_strict_rival_cohorts_calibrated": None if calibrated[seed, arm] is None else {ref: cohort(np, mask, calibrated[seed, arm]["P"], labels) for ref, mask in masks.items()},
                 "FP64_raw_pool_argmax_disagreements_with_archived_errors": int(((np.exp(log_p).mean(0).argmax(-1) != labels) != a["pooled_errors"]).sum()),
                 "raw_logit_argmax_disagreements_with_archived_member_errors": int(((a["raw_logits"].argmax(-1) != labels) != a["member_errors"]).sum()),
                 "stable_NLL_minus_original_reported": r["quality"]["pooled_nll"] - context["records"][seed, arm]["valid"]["nll"],
                 "scope": "Fresh F selected-state cohorts; not old allocation anchors. Cross-procedure repairs include learning/stopping/selection; zero current strict-rival repairs is not zero repair of reference failures."}
            diagnostics.append(d)
    return policy, aggregates, raw_contrasts, calibrated_contrasts, same_state, diagnostics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def chunk_documents(category, rows):
    docs, chunk = [], []
    for row in rows:
        proposed = {"category": category, "offset": sum(len(d[1]["rows"]) for d in docs), "total_rows": len(rows), "rows": chunk + [row]}
        if len(json.dumps(proposed, indent=2, allow_nan=False).encode()) > 1800000:
            require(chunk, "Single compact report row exceeds transport bound")
            docs.append((f"{category}_part{len(docs) + 1}.json", dict(proposed, rows=chunk)))
            chunk = [row]
        else:
            chunk.append(row)
    if chunk:
        docs.append((f"{category}_part{len(docs) + 1}.json", {"category": category, "offset": sum(len(d[1]["rows"]) for d in docs), "total_rows": len(rows), "rows": chunk}))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input_spec).resolve()
    input_bytes = input_path.read_bytes()
    spec = json.loads(input_bytes)
    context = preflight(spec)
    # No numerical library or selected-array import before every native gate.
    import numpy as np
    import torch
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays, ids, labels = payloads(np, context)
    endpoints, calibration_records, calibration_costs, folds = calibrate(np, torch, arrays, ids, labels, context["seeds"], output)
    # All seventy-five attempts are terminal before scoring, contrasts or gates.
    raw, calibrated, arm_details = {}, {}, []
    for arm in ARMS:
        for seed in context["seeds"]:
            a = arrays[seed, arm]
            raw64 = a["raw_logits"].astype(np.float64)
            log_p = raw64 - logsumexp(np, raw64, -1)[..., None]
            raw[seed, arm] = score(np, log_p, a["probability_mean"], a["member_errors"], a["pooled_errors"], labels)
            counts, selected = raw[seed, arm]["counts"], context["records"][seed, arm]["valid"]
            require(counts["pooled_correct"] == round(selected["accuracy"] * 5274)
                    and counts["coverage"] == selected["correct_alternative_count"]
                    and counts["lost_correct_alternatives"] == selected["coverage_lost_in_pooling"]
                    and counts["member_correct"] == [round(v * 5274) for v in selected["member_accuracy"]],
                    "Raw archived integer decisions differ from complete selected records")
            endpoint = endpoints[seed, arm]
            if endpoint is None:
                calibrated[seed, arm] = None
            else:
                member_log, pool_log = endpoint
                calibrated[seed, arm] = score(np, member_log, np.exp(pool_log), a["member_errors"], pool_log.argmax(-1) != labels, labels)
            arm_details.append({"arm": arm, "seed": seed, "raw": public_score(raw[seed, arm]),
                                "calibrated": public_score(calibrated[seed, arm]),
                                "calibrated_member_argmax_disagreements_with_raw_flags": None if endpoint is None else int(((endpoint[0].argmax(-1) != labels) != a["member_errors"]).sum()),
                                "native_selected_record": context["records"][seed, arm]})
    policy, aggregates, raw_contrasts, cal_contrasts, effects, diagnostics = analyze(np, raw, calibrated, arrays, labels, context)
    for r in calibration_records:
        r["selected_readout"] = public_score(calibrated[r["seed"], r["arm"]])
    closure = {k: v for k, v in context.items() if k not in ("records", "role_paths")}
    closure["safe_roles"] = spec["safe_roles"]
    report = {"complete": True, "study_id": spec["study_id"], "stage": "F", "K": None, "TEST_access": False,
              "banks": 15, "optimizer_acquisition_bundles": 24, "native_body_fit_records": 33,
              "fixed_seed_roster": context["seeds"], "fixed_arm_roster": list(ARMS),
              "entire_same_runtime_roster_admitted_before_arrays": True,
              "all75_calibration_attempts_terminal_before_interpretation": True,
              "raw_authority_preserved": True, "protocol": PROTOCOL, "frozen_policy": policy,
              "custody": closure, "folds": folds, "calibration_costs": calibration_costs,
              "arm_aggregates": aggregates, "arm_details": arm_details, "diagnostics": diagnostics,
              "calibration_records": calibration_records, "raw_contrasts": raw_contrasts,
              "calibrated_contrasts": cal_contrasts, "calibration_minus_raw_same_selected_state": effects,
              "input_spec_sha256": hashlib.sha256(input_bytes).hexdigest(), "reader_source_sha256": sha(__file__),
              "numerical_reader_runtime": {"torch": torch.__version__, "numpy": np.__version__, "CPU_threads": torch.get_num_threads(), "calibration_device": "cpu", "calibration_dtype": "float64"},
              "limits": ["Favorable raw findings and confidence failures remain visible; no desired verdict or silent winner selection.",
                         "Costs and member quality are diagnostics, not independent vetoes on served-pool utility.",
                         "Matched starts and coherent selectors define the sharing control; untied parameter budgets and optimizer histories differ.",
                         "Native selection precedes OOF temperature fitting; encountered development, not unused whole-pipeline confirmation.",
                         "No K, old allocation logit anchors, TEST, model deserialization/forward, grid, retry or final refit.",
                         "Three seeds have descriptive paired uncertainty; no independent-node significance, primitive novelty or manuscript acceptance claim."]}
    write_json(output / "COMPLETE_REPORT.json", report)
    documents = [("SAGE_F_POLICY_SUMMARY.json", {"complete": True, "stage": "F", "K": None, "TEST_access": False,
                   "banks": 15, "optimizer_acquisition_bundles": 24, "native_body_fit_records": 33,
                   "frozen_policy": policy, "calibration_costs": calibration_costs})]
    for category, rows in (("SAGE_F_ARM_AGGREGATES", aggregates), ("SAGE_F_ARM_DETAILS", arm_details),
                           ("SAGE_F_DIAGNOSTICS", diagnostics), ("CALIBRATION_FIT_DETAILS", calibration_records),
                           ("SAGE_F_raw_full_procedure_contrasts", [{"key": k, "contrast": v} for k, v in raw_contrasts.items()]),
                           ("SAGE_F_calibrated_full_procedure_contrasts", [{"key": k, "contrast": v} for k, v in cal_contrasts.items()]),
                           ("SAGE_F_calibration_minus_raw_same_selected_state", [{"key": k, "contrast": v} for k, v in effects.items()]),
                           ("CHECKPOINT_CUSTODY", context["checkpoints"]), ("ARCHIVE_CUSTODY", context["archives"]),
                           ("NATIVE_BODY_CUSTODY", context["body_records"]), ("SHARD_CUSTODY", context["shards"])):
        documents += chunk_documents(category, rows)
    partitions = []
    for name, value in documents:
        write_json(output / name, value)
        partitions.append(current_receipt(output / name))
    summary = {"complete": True, "study_id": spec["study_id"], "stage": "F", "K": None, "TEST_access": False,
               "banks": 15, "optimizer_acquisition_bundles": 24, "native_body_fit_records": 33,
               "fixed_seed_roster": context["seeds"], "fixed_arm_roster": list(ARMS),
               "entire_same_runtime_roster_admitted_before_arrays": True,
               "all75_calibration_attempts_terminal_before_interpretation": True,
               "calibration_costs": calibration_costs, "frozen_policy": policy,
               "folds": folds, "full_report": current_receipt(output / "COMPLETE_REPORT.json"),
               "partitions": partitions, "input_spec_sha256": report["input_spec_sha256"],
               "reader_source_sha256": report["reader_source_sha256"], "raw_authority_preserved": True}
    write_json(output / "COMPLETE_ANALYSIS_SUMMARY.json", summary)
    print(json.dumps({"complete": True, "banks": 15, "bundles": 24, "body_records": 33,
                      "calibration_attempts": 75, "raw_screen": policy["raw_accuracy_screen_pass"],
                      "confidence_screen": policy["calibrated_confidence_protection_pass"],
                      "TEST_access": False}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
