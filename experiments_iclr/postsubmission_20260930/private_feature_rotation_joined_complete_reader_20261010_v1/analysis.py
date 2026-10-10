"""Fixed GAT/SAGE joined VALID reader; no training, forwards, TEST or owners."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

BACKBONES = ("GAT", "SAGE")
SEEDS = (7301, 7403, 7507)
REFERENCES = ("ordinary_M1", "ordinary_genuine_I4", "factorized_allmap_M1",
              "factorized_allmap_genuine_I4", "shared4_unchanged")
FRESH = ("shared4_coherent", "shared4_paired_graph", "shared4_rank1_lora_graph", "shared4_paired_local")
O1, O4, F1, F4, S = REFERENCES
C, P, L, F = FRESH
ARMS = REFERENCES + FRESH
OLD_ARMS = REFERENCES + ("separable_equal_size", "exchange")
DECISION_DIR = "private_feature_rotation_pilot_decision_20261010_v1"
DECISION_SHA256 = "af96dfe64ee56dce4d8ce55d78872076d99ad468e7b279b02b5dc8d5009d5288"
REFERENCE_BINDINGS_SHA256 = "da08a1f126732b379ff3f44aa689a8a920292cf51703244c6534d2a7c152685c"
PILOT_SOURCE = "common_wrapper_paired_graph_native_family_source_20261010_v1/run_family.py"
PILOT_SOURCE_SHA256 = "7af8048fe4f209922232bdf486b1b03c35f0df623842e377e2854d3ed8b111da"
CORRECTIONS_SHA256 = "58dc7ea7f15f3c0effd286e995503abb2606dab42222522985d102eceff63cc7"
ROLE_ARCHIVE_SHA256 = {"train_npz": "88c36e1983f17a9e53d3f13a473baaf9eff86f52bc4be56aba012321983fdc6d",
                       "valid_npz": "591fe3a05d4bb80b9bfaf047292c04f91163abf60c43106d13f98046258268ad"}
FRESH_CONFIG_SHA256 = {"GAT": "6607082e5244439c025f0ce50b6303300e6ed9cc3fe2b7d355cc008c0b6ea0a2",
                       "SAGE": "f8b5b48e6cf3ba5f280818eec56a29bc9979f63b87fa4972553df0858b25d7a9"}
OLD_DECISION_SHA256 = {"GAT": "75ad1b2344a21da97b1a4efde666f16f9413a8558de40936e1413b422aa4be60",
                       "SAGE": "0d4ffe35cb73b6a840282612e3d94eec87e9130920657e68267ac7e74b3a5b93"}
STATES = ("no_alternative_wrong", "aggregation_only_correct", "alternative_lost", "alternative_served")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def closed_owner(path):
    owner = read_json(path)
    require(owner.get("exit_code") == 0 and owner.get("timed_out") is False, "Successful finite owner exit required: " + str(path))
    for field in ("direct_child_wait", "child_pid_absent", "owned_cuda_pid_absent", "complete_family", "scientific_success"):
        require(owner.get(field) is True, "Owner closure missing " + field + ": " + str(path))
    require(owner.get("TEST_access") is False, "Owner must declare no TEST access")
    return owner


def completion_header(path):
    # Frozen writer places all closure metadata before results. Stop at that
    # boundary: neither result metrics nor selected archives enter this gate.
    lines = []
    with path.open() as handle:
        for line in handle:
            if line.rstrip("\n") == '  "results": [':
                return json.loads("".join(lines).rstrip().removesuffix(",") + "\n}")
            lines.append(line)
    raise ValueError("Frozen COMPLETE_FAMILY metadata/results boundary missing: " + str(path))


def check_header(header, fresh, backbone):
    groups, units = (12, 12) if fresh else (21, 39)
    require(header.get("complete") is True and header.get("TEST_access") is False, "Complete no-TEST family required")
    require(header.get("groups") == header.get("expected_groups") == groups and
            header.get("fit_units") == header.get("expected_fit_units") == units, "Exact declared family completeness required")
    if fresh:
        require(header.get("backbone") == backbone and header.get("declared_acquisition_arms") == list(FRESH), "Fixed fresh backbone/roster required")
        require(header.get("full_comparative_roster") == list(ARMS) and header.get("full_comparative_family_complete") is False,
                "Fresh family must declare reference reuse, not full comparative acquisition")


def freeze_hash(freeze, suffix):
    matches = [row["sha256"] for row in freeze["bound_files"] if row["path"].endswith(suffix)]
    require(len(matches) == 1, "Exactly one frozen binding required: " + suffix)
    return matches[0]


def source_hash(complete, suffix):
    matches = [value for path, value in complete["source_sha256"].items() if path.endswith(suffix)]
    require(len(matches) == 1, "Exactly one acquisition source binding required: " + suffix)
    return matches[0]


def sum_operations(rows):
    units = [u for r in rows for u in r["fits"]]
    keys = set(units[0]["operation_counts"])
    require(all(set(u["operation_counts"]) == keys for u in units), "Consistent operation fields within family required")
    return {key: sum(u["operation_counts"][key] for u in units) for key in sorted(keys)}


def family_records(root, backbone, fresh, owner, header, binding):
    folder = root / (DECISION_DIR + "/" + backbone if fresh else "common_wrapper_" + backbone + "_root_20261010_v1")
    output, complete_path = folder / "actual_family_v1", folder / "actual_family_v1/COMPLETE_FAMILY.json"
    # Called only after BOTH fresh header gates (and every owner closure).
    complete_sha = digest(complete_path)
    require(owner.get("complete_sha256") == complete_sha, "Owner must bind exact complete record")
    if not fresh:
        require(complete_sha == binding["complete_sha256"], "Immutable reference complete hash mismatch")
    complete, cfg, freeze = read_json(complete_path), read_json(folder / "CONFIG.json"), read_json(folder / "FREEZE.json")
    require({k: v for k, v in complete.items() if k not in ("results", "comparisons", "cost_scope")} == header,
            "Completion header changed while reading")
    config_sha = digest(folder / "CONFIG.json")
    expected_config_sha = FRESH_CONFIG_SHA256[backbone] if fresh else binding["config_sha256"]
    require(config_sha == complete["config_sha256"] == expected_config_sha, "Frozen companion config hash mismatch")
    require(read_json(output / "CONFIG.json") == cfg, "Output must preserve exact submitted config")
    require(cfg.get("backbone", "SAGE") == backbone and cfg["seeds"] == list(SEEDS), "Exact backbone and paired seed order required")
    require(freeze.get("TEST_access") is False, "Frozen no-TEST binding required")
    if fresh:
        require(cfg.get("fresh_arms_only") is True and cfg.get("correction_seed_offset") == 5000081, "Frozen fresh-arm correction config required")
        require(freeze.get("expected_groups") == freeze.get("new_fit_units") == 12, "Frozen fresh 12/12 roster required")
        require(freeze["reference_complete_sha256"] == binding["complete_sha256"] and
                freeze["reference_source_config"] == binding["config_path"] and
                freeze["original_reference_arms"] == list(REFERENCES), "Frozen reference reuse identity mismatch")
        require(freeze_hash(freeze, DECISION_DIR + "/DECISION.md") == DECISION_SHA256 and
                freeze_hash(freeze, DECISION_DIR + "/REFERENCE_BINDINGS.json") == REFERENCE_BINDINGS_SHA256, "Frozen pilot decision/reference binding mismatch")
        entry, entry_sha = PILOT_SOURCE, PILOT_SOURCE_SHA256
        require(source_hash(complete, "common_wrapper_paired_graph_native_family_source_20261010_v1/corrections.py") == CORRECTIONS_SHA256 ==
                freeze_hash(freeze, "common_wrapper_paired_graph_native_family_source_20261010_v1/corrections.py"),
                "Frozen acquired correction source required")
    else:
        entry, entry_sha = binding["source_path"], binding["source_sha256"]
        require(digest(folder / "DECISION.md") == OLD_DECISION_SHA256[backbone], "Immutable old decision identity mismatch")
    require(digest(root / entry) == source_hash(complete, entry) == entry_sha == freeze_hash(freeze, entry), "Exact acquisition entry source required")
    require(freeze_hash(freeze, folder.name + "/CONFIG.json") == config_sha, "Config must match owner freeze")
    arms = FRESH if fresh else OLD_ARMS
    rows = complete["results"]
    index = {(r["seed"], r["arm"]): r for r in rows}
    require(len(rows) == len(index) == len(arms) * 3 and set(index) == {(s, a) for s in SEEDS for a in arms}, "Complete exact arm/seed roster required")
    for (seed, arm), row in index.items():
        acquisitions = 4 if "genuine_I4" in arm else 1
        require(row["serving"] == "probability_mean" and [u["member"] for u in row["fits"]] == list(range(acquisitions)), "Frozen independent acquisition units required")
        for unit in row["fits"]:
            members = 4 if arm not in (O1, O4, F1, F4) else 1
            expected_seeds = []
            for route in range(members):
                native = seed + cfg["member_seed_stride"] * (unit["member"] + route)
                expected_seeds.append([native, native + cfg["factor_seed_offset"], native + cfg["dropout_seed_offset"]])
            require(unit["native_factor_dropout_seeds"] == expected_seeds, "Acquisition native/factor/dropout seed mismatch")
            require(1 <= unit["selected_step"] <= unit["completed_updates"] <= cfg["max_updates"], "Original selected/completed step bounds required")
    require(sum(len(r["fits"]) for r in rows) == header["fit_units"] and sum_operations(rows) == complete["operation_counts"], "Complete fit/operation aggregation mismatch")
    return dict(backbone=backbone, fresh=fresh, folder=folder, output=output, owner=owner, header=header, complete=complete,
                complete_sha256=complete_sha, cfg=cfg, freeze=freeze, index=index, owner_end_sha256=digest(folder / "OWNER_END.json"),
                freeze_sha256=digest(folder / "FREEZE.json"))


def matched_contexts(fresh, old):
    for backbone in BACKBONES:
        fc, oc = fresh[backbone], old[backbone]
        new_cfg = {k: v for k, v in fc["cfg"].items() if k not in ("fresh_arms_only", "correction_seed_offset")}
        old_cfg = dict(oc["cfg"], backbone=backbone)
        require(new_cfg == old_cfg, "Fresh/reference configs must match except declared new-arm controls")
        for suffix in ("models.py", "run_base.py", "run_common.py", "shared_fast_graph_model_interface_20261010_v1/common_routes.py",
                       "portable_internal_be_public_interface_20261007_v2/core/factors.py"):
            require(source_hash(fc["complete"], suffix) == source_hash(oc["complete"], suffix) ==
                    freeze_hash(fc["freeze"], suffix) == freeze_hash(oc["freeze"], suffix), "Shared native/wrapper source mismatch: " + suffix)
        for role in ("train_npz", "valid_npz"):
            suffix = "/".join(Path(fc["cfg"][role]).parts[-3:])
            require(freeze_hash(fc["freeze"], suffix) == freeze_hash(oc["freeze"], suffix) == ROLE_ARCHIVE_SHA256[role], "Immutable data/role hash mismatch")
    require({k: v for k, v in fresh["GAT"]["cfg"].items() if k != "backbone"} ==
            {k: v for k, v in fresh["SAGE"]["cfg"].items() if k != "backbone"}, "Companion configs must differ only by backbone")


def paired(values):
    values = [float(v) for v in values]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    flipped = [statistics.mean(a * b for a, b in zip(signs, values)) for signs in itertools.product((-1, 1), repeat=3)]
    return {"seed_deltas": values, "mean": mean, "sample_sd": sd, "min": min(values), "max": max(values),
            "descriptive_95pct_t_interval_df2": [mean - half, mean + half],
            "nonnegative_seed_count": sum(v >= 0 for v in values), "positive_seed_count": sum(v > 0 for v in values),
            "exact_sign_flip_two_sided_p": sum(abs(v) >= abs(mean) for v in flipped) / 8}


def logsumexp(a, axis):
    maximum = a.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(a - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def selected(fresh, old):
    arrays, hashes, ids, labels = {}, {}, None, None
    keys = {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"}
    for backbone in BACKBONES:
        for seed in SEEDS:
            for arm in ARMS:
                context = fresh[backbone] if arm in FRESH else old[backbone]
                path = context["output"] / f"{arm}_seed{seed}" / "selected_VALID.npz"
                with np.load(path, allow_pickle=False) as archive:
                    require(set(archive.files) == keys, "Exact selected VALID keys required")
                    a = {k: archive[k].copy() for k in keys}
                members = 1 if arm in (O1, F1) else 4
                require(a["ids"].shape == a["y"].shape == (5274,) and a["ids"].dtype == a["y"].dtype == np.int64, "Exact VALID roles required")
                require(a["raw_logits"].shape == (members, 5274, 10) and a["raw_logits"].dtype == np.float32, "Original float32 member logits required")
                require(a["probability_mean"].shape == (5274, 10) and a["probability_mean"].dtype == np.float32, "Native float32 serving probabilities required")
                require(a["member_errors"].shape == (members, 5274) and a["pooled_errors"].shape == (5274,) and
                        a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_, "Exact saved error flags required")
                require(np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all(), "Finite selected outputs required")
                if ids is None:
                    ids, labels = a["ids"], a["y"]
                    require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10)), "Ordered ten-class VALID identity required")
                require(np.array_equal(ids, a["ids"]) and np.array_equal(labels, a["y"]), "All54 archives require identical ordered IDs/labels")
                require(np.array_equal(a["probability_mean"].argmax(1) != labels, a["pooled_errors"]), "Saved serving decisions/errors disagree")
                arrays[backbone, seed, arm], hashes[str(path)] = a, digest(path)
    return arrays, hashes, ids, labels


def score(a):
    raw, y = a["raw_logits"].astype(np.float64), a["y"]
    log_probs = raw - logsumexp(raw, -1)[..., None]
    member_nll = -np.take_along_axis(log_probs, y[None, :, None], axis=2).squeeze(2)
    pool_nll = -(logsumexp(log_probs, 0) - math.log(len(raw)))[np.arange(len(y)), y]
    member_correct, pool_correct = ~a["member_errors"], ~a["pooled_errors"]
    coverage = member_correct.any(0)
    lost, aggregation_only = coverage & ~pool_correct, ~coverage & pool_correct

    def counts(mask):
        result = {"nodes": int(mask.sum()), "pooled_correct": int(pool_correct[mask].sum()), "coverage": int(coverage[mask].sum()),
                  "lost_correct_alternatives": int(lost[mask].sum()), "aggregation_only_correct": int(aggregation_only[mask].sum()),
                  "unavailable_alternatives": int((~coverage[mask]).sum()), "served_alternatives": int((coverage[mask] & pool_correct[mask]).sum()),
                  "member_correct": member_correct[:, mask].sum(1).tolist()}
        require(result["pooled_correct"] == result["coverage"] - result["lost_correct_alternatives"] + result["aggregation_only_correct"], "Coverage decomposition failed")
        return result

    classes = []
    for label in range(10):
        mask = y == label
        classes.append({"class": label, "counts": counts(mask), "pooled_accuracy_pct": float(pool_correct[mask].mean() * 100),
                        "pooled_nll": float(pool_nll[mask].mean()), "member_accuracy_pct": (member_correct[:, mask].mean(1) * 100).tolist(),
                        "member_nll": member_nll[:, mask].mean(1).tolist()})
    quality = {"pooled_accuracy_pct": float(pool_correct.mean() * 100), "pooled_nll": float(pool_nll.mean()),
               "member_accuracy_pct": (member_correct.mean(1) * 100).tolist(), "member_nll": member_nll.mean(1).tolist(),
               "mean_member_accuracy_pct": float(member_correct.mean() * 100), "mean_member_nll": float(member_nll.mean()),
               "worst_member_accuracy_pct": float(member_correct.mean(1).min() * 100)}
    return {"quality": quality, "counts": counts(np.ones(len(y), dtype=np.bool_)), "classes": classes,
            "P": pool_correct, "V": coverage, "C": member_correct, "state": 2 * coverage.astype(np.int64) + pool_correct.astype(np.int64)}


def comparison(a, b, labels):
    repair, harm = ~b["P"] & a["P"], b["P"] & ~a["P"]
    new_coverage, removed_coverage = a["V"] & ~b["V"], b["V"] & ~a["V"]
    changes = {k: a["counts"][k] - b["counts"][k] for k in ("pooled_correct", "coverage", "lost_correct_alternatives", "aggregation_only_correct")}
    require(changes["pooled_correct"] == changes["coverage"] - changes["lost_correct_alternatives"] + changes["aggregation_only_correct"], "Paired count identity failed")
    require(changes["pooled_correct"] == int(repair.sum() - harm.sum()), "Repair/harm identity failed")
    return {"repairs": int(repair.sum()), "harms": int(harm.sum()), "count_decomposition_delta": changes,
            "new_coverage": int(new_coverage.sum()), "removed_coverage": int(removed_coverage.sum()),
            "new_coverage_served": int((new_coverage & a["P"]).sum()), "new_coverage_lost": int((new_coverage & ~a["P"]).sum()),
            "repair_causes": {"new_member_alternative": int((repair & a["V"] & ~b["V"]).sum()),
                              "existing_member_alternative_served": int((repair & a["V"] & b["V"]).sum()),
                              "aggregation_only": int((repair & ~a["V"]).sum())},
            "harm_causes": {"available_alternative_lost": int((harm & a["V"]).sum()),
                            "removed_member_alternative": int((harm & ~a["V"] & b["V"]).sum()),
                            "lost_aggregation_only": int((harm & ~a["V"] & ~b["V"]).sum())},
            "transition_rows_reference_columns_candidate": np.bincount(b["state"] * 4 + a["state"], minlength=16).reshape(4, 4).tolist(),
            "classes": [{"class": k, "repairs": int((repair & (labels == k)).sum()), "harms": int((harm & (labels == k)).sum()),
                         "count_decomposition_delta": {field: a["classes"][k]["counts"][field] - b["classes"][k]["counts"][field] for field in changes}} for k in range(10)]}


def contrast(scores, candidate, reference, labels):
    a, b = [scores[s, candidate] for s in SEEDS], [scores[s, reference] for s in SEEDS]
    quality = {metric: paired(u["quality"][metric] - v["quality"][metric] for u, v in zip(a, b)) for metric in
               ("pooled_accuracy_pct", "pooled_nll", "mean_member_accuracy_pct", "mean_member_nll", "worst_member_accuracy_pct")}
    # Derive the primary accuracy delta from integer counts, avoiding float32
    # rounding in original selected/restored metric records (which are kept).
    quality["pooled_accuracy_pct"] = paired(100 * (u["counts"]["pooled_correct"] - v["counts"]["pooled_correct"]) / 5274 for u, v in zip(a, b))
    quality["mean_member_accuracy_pct"] = paired(100 * (u["C"].sum() / len(u["C"]) - v["C"].sum() / len(v["C"])) / 5274 for u, v in zip(a, b))
    classes = []
    for k in range(10):
        classes.append({"class": k, "pooled_accuracy_pp": paired(u["classes"][k]["pooled_accuracy_pct"] - v["classes"][k]["pooled_accuracy_pct"] for u, v in zip(a, b)),
                        "pooled_nll": paired(u["classes"][k]["pooled_nll"] - v["classes"][k]["pooled_nll"] for u, v in zip(a, b)),
                        "mean_member_accuracy_pp": paired(statistics.mean(u["classes"][k]["member_accuracy_pct"]) - statistics.mean(v["classes"][k]["member_accuracy_pct"]) for u, v in zip(a, b)),
                        "mean_member_nll": paired(statistics.mean(u["classes"][k]["member_nll"]) - statistics.mean(v["classes"][k]["member_nll"]) for u, v in zip(a, b))})
    same_members = len(a[0]["C"]) == len(b[0]["C"])
    members = [{"member": m, "accuracy_pp": paired(u["quality"]["member_accuracy_pct"][m] - v["quality"]["member_accuracy_pct"][m] for u, v in zip(a, b)),
                "nll": paired(u["quality"]["member_nll"][m] - v["quality"]["member_nll"][m] for u, v in zip(a, b))} for m in range(len(a[0]["C"]))] if same_members else None
    diagnostics = [{"seed": seed, **comparison(u, v, labels)} for seed, u, v in zip(SEEDS, a, b)]
    totals = {field: sum(row["count_decomposition_delta"][field] for row in diagnostics) for field in diagnostics[0]["count_decomposition_delta"]}
    return {"candidate": candidate, "reference": reference, "quality_deltas": quality, "members": members,
            "member_pairing": "same route/acquisition index" if same_members else "unequal member counts; mean-member contrast only",
            "classes": classes, "per_seed_error_diagnostics": diagnostics,
            "summed_three_seed_readout_diagnostics": {"repairs": sum(row["repairs"] for row in diagnostics), "harms": sum(row["harms"] for row in diagnostics),
                                                     "member_correct_delta": sum(int(u["C"].sum()) - int(v["C"].sum()) for u, v in zip(a, b)) if same_members else None,
                                                     "count_decomposition_delta": totals}}


def cost_summary(rows):
    return {"groups": len(rows), "acquisition_units": sum(len(r["fits"]) for r in rows),
            "acquisition_seconds_sum": sum(r["costs"]["acquisition_seconds"] for r in rows),
            "selected_serving_readout_seconds_sum": sum(r["costs"]["selected_serving_readout_seconds"] for r in rows),
            "operation_counts": sum_operations(rows),
            "all_original_group_and_fit_costs_preserved": True}


def frozen_flags(contrasts, candidate):
    stat = lambda ref, metric: contrasts[candidate + "_minus_" + ref]["quality_deltas"][metric]
    refs = (C, S, O4, F4)
    accuracy = {"positive_mean_accuracy_against_each_of_four_references": {ref: contrasts[candidate + "_minus_" + ref]["summed_three_seed_readout_diagnostics"]["count_decomposition_delta"]["pooled_correct"] > 0 for ref in refs},
                "mean_accuracy_vs_coherent_at_least_0_2pp": stat(C, "pooled_accuracy_pct")["mean"] >= .2,
                "mean_accuracy_vs_ordinary_I4_at_least_0_2pp": stat(O4, "pooled_accuracy_pct")["mean"] >= .2,
                "ordinary_I4_all_three_seed_deltas_nonnegative": stat(O4, "pooled_accuracy_pct")["nonnegative_seed_count"] == 3,
                "ordinary_I4_at_least_two_positive_seed_deltas": stat(O4, "pooled_accuracy_pct")["positive_seed_count"] >= 2}
    member, nll = stat(C, "mean_member_accuracy_pct"), stat(C, "pooled_nll")
    protection = {"mean_member_vs_coherent_nonnegative": contrasts[candidate + "_minus_" + C]["summed_three_seed_readout_diagnostics"]["member_correct_delta"] >= 0,
                  "each_seed_member_vs_coherent_at_least_minus_0_1pp": min(member["seed_deltas"]) >= -.1,
                  "mean_nll_deterioration_vs_coherent_at_most_0_02": nll["mean"] <= .02,
                  "each_seed_nll_deterioration_vs_coherent_at_most_0_05": max(nll["seed_deltas"]) <= .05}
    accuracy_pass = all(accuracy["positive_mean_accuracy_against_each_of_four_references"].values()) and all(v for k, v in accuracy.items() if k != "positive_mean_accuracy_against_each_of_four_references")
    return {"accuracy_transfer_criteria": accuracy, "accuracy_transfer_within_backbone": accuracy_pass,
            "quality_protection_criteria": protection, "quality_protection_within_backbone": all(protection.values())}


def analyze(backbone, scores, fresh, old, labels):
    groups, aggregates = [], {}
    for arm in ARMS:
        context = fresh if arm in FRESH else old
        arm_rows = []
        for seed in SEEDS:
            value, original = scores[seed, arm], context["index"][seed, arm]
            row = {"arm": arm, "seed": seed, "acquisition_origin": "new" if arm in FRESH else "immutable_reused_reference",
                   **{k: value[k] for k in ("quality", "counts", "classes")}, "original_group_record": original}
            groups.append(row)
            arm_rows.append(row)
        counts = {key: sum(r["counts"][key] for r in arm_rows) for key in arm_rows[0]["counts"] if key != "member_correct"}
        aggregates[arm] = {"summed_counts_over_three_seed_readouts": counts, "mean_quality": {key: statistics.mean(r["quality"][key] for r in arm_rows) for key in
                           ("pooled_accuracy_pct", "pooled_nll", "mean_member_accuracy_pct", "mean_member_nll", "worst_member_accuracy_pct")}, "costs": cost_summary([r["original_group_record"] for r in arm_rows])}
    pairs = [(a, b) for a in FRESH for b in REFERENCES] + [(a, C) for a in (P, L, F)] + [(P, L), (P, F)]
    contrasts = {a + "_minus_" + b: contrast(scores, a, b, labels) for a, b in pairs}
    flags = {candidate: frozen_flags(contrasts, candidate) for candidate in (P, L, F)}
    new_rows = fresh["complete"]["results"]
    reused_rows = [old["index"][s, a] for s in SEEDS for a in REFERENCES]
    return {"backbone": backbone, "groups": 27, "acquisition_units_represented": 45, "groups_detail": groups, "aggregates": aggregates,
            "paired_contrasts": contrasts, "frozen_flags": flags,
            "costs": {"new_12_acquisitions": cost_summary(new_rows), "reused_33_reference_acquisitions": cost_summary(reused_rows),
                      "entire_original_closed_39_acquisition_family": cost_summary(old["complete"]["results"]),
                      "new_owner_seconds": fresh["owner"]["seconds"], "original_reference_owner_seconds": old["owner"]["seconds"],
                      "new_overlap": fresh["owner"]["overlap"], "original_reference_overlap": old["owner"]["overlap"]},
            "source_custody": {kind: {"owner_end_sha256": ctx["owner_end_sha256"], "complete_family_sha256": ctx["complete_sha256"],
                                     "freeze_sha256": ctx["freeze_sha256"], "original_owner_end": ctx["owner"], "original_freeze": ctx["freeze"],
                                     "original_complete_family": ctx["complete"]} for kind, ctx in (("new", fresh), ("immutable_reference", old))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True, help="New JSON path; existing paths are refused")
    args = parser.parse_args()
    root, report_path = args.research_root.resolve(), args.report.resolve()
    require(not report_path.exists(), "Report must be a new file")
    decision_folder = root / DECISION_DIR
    require(digest(decision_folder / "DECISION.md") == DECISION_SHA256 and
            digest(decision_folder / "REFERENCE_BINDINGS.json") == REFERENCE_BINDINGS_SHA256, "Exact frozen joined policy/bindings required")
    bindings_document = read_json(decision_folder / "REFERENCE_BINDINGS.json")
    bindings = {r["backbone"]: r for r in bindings_document["immutable_complete_references"]}
    require(set(bindings) == set(BACKBONES) and all(r["arms"] == list(REFERENCES) and r["paired_seeds"] == list(SEEDS) and
            r["preserve_original_fit_records_costs"] is True for r in bindings.values()), "Both immutable five-arm reference bindings required")
    # Gate 1: BOTH fresh successful owner closures, before any new result read.
    fresh_owners = {b: closed_owner(decision_folder / b / "OWNER_END.json") for b in BACKBONES}
    old_owners = {b: closed_owner(root / ("common_wrapper_" + b + "_root_20261010_v1/OWNER_END.json")) for b in BACKBONES}
    # Gate 2: read only pre-results headers, then require BOTH fresh12/12.
    fresh_headers = {b: completion_header(decision_folder / b / "actual_family_v1/COMPLETE_FAMILY.json") for b in BACKBONES}
    for b in BACKBONES:
        check_header(fresh_headers[b], True, b)
    old_headers = {b: completion_header(root / ("common_wrapper_" + b + "_root_20261010_v1/actual_family_v1/COMPLETE_FAMILY.json")) for b in BACKBONES}
    for b in BACKBONES:
        check_header(old_headers[b], False, b)
    # Complete outcome records are first loaded after all closure/count gates.
    fresh = {b: family_records(root, b, True, fresh_owners[b], fresh_headers[b], bindings[b]) for b in BACKBONES}
    old = {b: family_records(root, b, False, old_owners[b], old_headers[b], bindings[b]) for b in BACKBONES}
    matched_contexts(fresh, old)
    require(digest(root / Path(PILOT_SOURCE).with_name("corrections.py")) == CORRECTIONS_SHA256, "Frozen correction helper source required")
    global np
    import numpy as np
    arrays, archives, ids, labels = selected(fresh, old)
    scores = {key: score(value) for key, value in arrays.items()}  # All54 identities checked before scoring.
    families = {b: analyze(b, {(s, a): scores[b, s, a] for s in SEEDS for a in ARMS}, fresh[b], old[b], labels) for b in BACKBONES}
    joined_flags = {candidate: {"accuracy_transfer_clue_both_backbones": all(families[b]["frozen_flags"][candidate]["accuracy_transfer_within_backbone"] for b in BACKBONES),
                                "quality_protection_both_backbones": all(families[b]["frozen_flags"][candidate]["quality_protection_within_backbone"] for b in BACKBONES)} for candidate in (P, L, F)}
    report = {"complete": True, "backbones": list(BACKBONES), "seeds": list(SEEDS), "joined_groups": 54,
              "new_acquisition_units": 24, "reused_reference_acquisition_units": 66, "total_acquisition_units_represented": 90,
              "TEST_access": False, "analysis_source_sha256": digest(Path(__file__)), "decision_sha256": DECISION_SHA256,
              "reference_bindings_sha256": REFERENCE_BINDINGS_SHA256, "original_reference_bindings": bindings_document,
              "ordered_VALID_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "ordered_VALID_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
              "selected_VALID_archive_hashes": archives, "families": families, "joined_frozen_flags": joined_flags, "transition_state_order": STATES,
              "coverage_identity": "pooled correct = any-member coverage - lost correct alternatives + aggregation-only correct",
              "gate_scope": "Accuracy flags are independent of protection flags. Protection failure blocks protected-quality claims without erasing measured accuracy.",
              "interpretation_limits": ["Fixed GAT/SAGE companions; every arm and seed retained. No outcome selects or retunes another fit.",
                  "Selected encountered development VALID; unused confirmation and capable same-operation single/I4 references remain required before paper claims.",
                  "Descriptive paired df2 t intervals measure three-seed optimization variation on one graph, not across-graph/split uncertainty or node-IID replication.",
                  "Graph-pair/LoRA/local are individual contrasts; no measured A+B or factorial interaction is inferred.",
                  "Equal private coordinate count does not match tangent freedom, Adam step geometry, FLOPs or usefulness; only the Householder product is orthogonal.",
                  "Orthogonal adapters and private LoRA ensembles are direct ancestry; neither transfer nor protection proves novelty or generality.",
                  "Original selections/restores/costs and complete records are preserved verbatim alongside recomputed stored-logit diagnostics; no checkpoint/model forward is read or run.",
                  "Fresh costs cover 24 new acquisitions. Reused costs cover 66 immutable comparator acquisitions; original unselected separable/exchange costs remain in source custody.",
                  "Observed acquisition/readout/owner costs include recorded overlap and lifetime RSS peaks; they are not isolated serving benchmarks."]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"complete": True, "report": str(report_path), "joined_frozen_flags": joined_flags}))


if __name__ == "__main__":
    main()
