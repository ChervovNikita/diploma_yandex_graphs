"""Fixed bootstrap joined reader; immutable rotation-reader math, stored VALID only."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

MATH_READER = "private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py"
MATH_READER_SHA256 = "82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d"
DECISION_DIR = "private_class_balanced_bootstrap_pilot_decision_20261010_v1"
DECISION_SHA256 = "82d24de34a7078123908b9471e609dc569fc856fdbb8f2bcfaea45c70ac0ccb2"
BINDINGS_SHA256 = {"GAT": "dafe71e21d0e973f79d9e8008c11aa8b33c23a4a6f2613fac63ec8827833d07d",
                   "SAGE": "aeefd9ad0203e1d65a4027022649bea248428e2f52f45a3194c058e94544dfb8"}
CONFIG_SHA256 = {"GAT": "5e8ca2929e8662929a95cace34030908c72e04462f41ee688dc6e94d78fcee32",
                 "SAGE": "1607828a9b1aee8fa8e0214c6eda3989d05201f3e6987d71b516918710e0a044"}
ENTRY = "common_wrapper_private_bootstrap_native_family_source_20261010_v1/run_family.py"
ENTRY_SHA256 = "eeed1e92ab79e9a9fcd5a52097ede1ff08c8cb4557af43ae5f5d200d48eaba6d"
HELPER = "common_wrapper_private_bootstrap_native_family_source_20261010_v1/bootstrap.py"
HELPER_SHA256 = "40ca31357f252f6dbea25ba1ba82c6b263b243e16f18beaed23d2cf03a6c1674"
BACKBONES, SEEDS = ("GAT", "SAGE"), (7301, 7403, 7507)
REFERENCES = ("ordinary_M1", "ordinary_genuine_I4", "factorized_allmap_M1", "factorized_allmap_genuine_I4", "shared4_unchanged")
O1, O4, F1, F4, S = REFERENCES
UNWEIGHTED = ("shared4_coherent", "shared4_paired_graph", "shared4_rank1_lora_graph")
C, A, L = UNWEIGHTED
FRESH = ("shared4_coherent_bootstrap", "shared4_paired_graph_bootstrap", "shared4_rank1_lora_graph_bootstrap")
B, AB, LB = FRESH
ARMS = REFERENCES + UNWEIGHTED + FRESH
COUNTERPART = dict(zip(FRESH, UNWEIGHTED))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_math(root):
    path = root / MATH_READER
    if digest(path) != MATH_READER_SHA256:
        raise ValueError("Immutable completed rotation-reader source required")
    spec = importlib.util.spec_from_file_location("rotation_complete_reader_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Its module scope imports stdlib only.
    return module


def folders(root, backbone):
    return {"new": root / DECISION_DIR / backbone,
            "unweighted": root / "private_feature_rotation_pilot_decision_20261010_v1" / backbone,
            "original": root / ("common_wrapper_" + backbone + "_root_20261010_v1")}


def preflight(reader, root, backbone, kind, folder, owner, header, binding):
    """Metadata/source admission only; no complete result JSON is loaded here."""
    require, freeze_hash, source_hash = reader.require, reader.freeze_hash, reader.source_hash
    groups, units = {"new": (9, 9), "unweighted": (12, 12), "original": (21, 39)}[kind]
    require(header.get("complete") is True and header.get("TEST_access") is False and
            header.get("groups") == header.get("expected_groups") == groups and
            header.get("fit_units") == header.get("expected_fit_units") == units, "Exact no-TEST complete roster required")
    complete_path = folder / "actual_family_v1/COMPLETE_FAMILY.json"
    complete_sha = digest(complete_path)
    require(owner.get("complete_sha256") == complete_sha, "Owner must bind exact closed complete file")
    cfg, freeze = reader.read_json(folder / "CONFIG.json"), reader.read_json(folder / "FREEZE.json")
    if kind == "new":
        expected_cfg, entry, entry_sha = CONFIG_SHA256[backbone], ENTRY, ENTRY_SHA256
        require(header.get("backbone") == backbone and header.get("declared_acquisition_arms") == list(FRESH) and
                header.get("full_comparative_roster") == list(ARMS) and header.get("full_comparative_family_complete") is False,
                "Fixed bootstrap9 roster and explicit reference reuse required")
        require(header.get("reused_original_reference_arms") == list(REFERENCES) and header.get("reused_unweighted_counterparts") == list(UNWEIGHTED),
                "All declared immutable comparators required")
        require(cfg.get("bootstrap_seed_offset") == 6000119 and cfg.get("correction_seed_offset") == 5000081, "Fixed bootstrap/correction seed policy required")
        require(freeze_hash(freeze, DECISION_DIR + "/DECISION.md") == DECISION_SHA256 and
                freeze_hash(freeze, DECISION_DIR + "/" + backbone + "/REFERENCE_BINDINGS.json") == BINDINGS_SHA256[backbone], "Frozen bootstrap policy/bindings mismatch")
        require(digest(root / HELPER) == source_hash(header, HELPER) == freeze_hash(freeze, HELPER) == HELPER_SHA256, "Frozen gradient/weight helper required")
    elif kind == "unweighted":
        require(binding["unweighted_parent_root"] == str(folder.relative_to(root)) and binding["unweighted_counterparts"] == list(UNWEIGHTED), "Exact unweighted parent identity required")
        require(complete_sha == binding["bootstrap_parent_closed_complete_sha256"], "Immutable unweighted complete hash mismatch")
        reader.check_header(header, True, backbone)
        expected_cfg, entry, entry_sha = binding["parent_config_sha256"], reader.PILOT_SOURCE, reader.PILOT_SOURCE_SHA256
        require(digest(folder.parent / "DECISION.md") == reader.DECISION_SHA256, "Frozen unweighted decision required")
    else:
        original = binding["original_references"]
        require(original["backbone"] == backbone and original["arms"] == list(REFERENCES) and original["paired_seeds"] == list(SEEDS) and
                original["preserve_original_fit_records_costs"] is True and complete_sha == original["complete_sha256"], "Immutable original five-reference binding mismatch")
        expected_cfg, entry, entry_sha = original["config_sha256"], original["source_path"], original["source_sha256"]
        require(str((folder / "CONFIG.json").relative_to(root)) == original["config_path"] and
                digest(folder / "DECISION.md") == reader.OLD_DECISION_SHA256[backbone], "Original config/decision identity mismatch")
    require(digest(folder / "CONFIG.json") == header["config_sha256"] == expected_cfg == freeze_hash(freeze, folder.name + "/CONFIG.json"), "Exact frozen config required")
    require(reader.read_json(folder / "actual_family_v1/CONFIG.json") == cfg and cfg.get("backbone", "SAGE") == backbone and cfg["seeds"] == list(SEEDS), "Submitted/output backbone/seed config mismatch")
    require(freeze.get("TEST_access") is False and digest(root / entry) == source_hash(header, entry) == freeze_hash(freeze, entry) == entry_sha, "Frozen no-TEST source identity required")
    if kind != "original":
        suffix = "common_wrapper_paired_graph_native_family_source_20261010_v1/corrections.py"
        require(source_hash(header, suffix) == freeze_hash(freeze, suffix) == reader.CORRECTIONS_SHA256 == digest(root / suffix), "Immutable operator helper required")
    return dict(backbone=backbone, kind=kind, folder=folder, output=folder / "actual_family_v1", owner=owner, header=header,
                complete_path=complete_path, complete_sha256=complete_sha, cfg=cfg, freeze=freeze)


def matched_metadata(reader, contexts):
    for backbone in BACKBONES:
        new, parent, original = (contexts[backbone, kind] for kind in ("new", "unweighted", "original"))
        native = {k: v for k, v in new["cfg"].items() if k != "bootstrap_seed_offset"}
        require = reader.require
        require(native == {k: v for k, v in parent["cfg"].items() if k != "fresh_arms_only"}, "Fresh/unweighted config mismatch")
        require({k: v for k, v in native.items() if k != "correction_seed_offset"} == dict(original["cfg"], backbone=backbone), "Native reference config mismatch")
        for suffix in ("models.py", "run_base.py", "run_common.py", "shared_fast_graph_model_interface_20261010_v1/common_routes.py",
                       "portable_internal_be_public_interface_20261007_v2/core/factors.py"):
            hashes = [reader.source_hash(ctx["header"], suffix) for ctx in (new, parent, original)] + [reader.freeze_hash(ctx["freeze"], suffix) for ctx in (new, parent, original)]
            require(len(set(hashes)) == 1, "Shared native source mismatch: " + suffix)
        for role in ("train_npz", "valid_npz"):
            suffix = "/".join(Path(new["cfg"][role]).parts[-3:])
            require(all(reader.freeze_hash(ctx["freeze"], suffix) == reader.ROLE_ARCHIVE_SHA256[role] for ctx in (new, parent, original)), "Immutable data/role mismatch")
    require({k: v for k, v in contexts["GAT", "new"]["cfg"].items() if k != "backbone"} ==
            {k: v for k, v in contexts["SAGE", "new"]["cfg"].items() if k != "backbone"}, "Fixed companion configs must differ only by backbone")


def records(reader, context):
    complete = reader.read_json(context["complete_path"])  # All six metadata admissions precede this.
    reader.require({k: v for k, v in complete.items() if k not in ("results", "comparisons", "cost_scope")} == context["header"], "Complete header changed")
    arms = {"new": FRESH, "unweighted": tuple(reader.FRESH), "original": tuple(reader.OLD_ARMS)}[context["kind"]]
    rows, cfg = complete["results"], context["cfg"]
    index = {(r["seed"], r["arm"]): r for r in rows}
    reader.require(len(rows) == len(index) == 3 * len(arms) and set(index) == {(s, a) for s in SEEDS for a in arms}, "Exact all-arm/seed records required")
    for (seed, arm), row in index.items():
        units = 4 if "genuine_I4" in arm else 1
        reader.require(row["serving"] == "probability_mean" and [u["member"] for u in row["fits"]] == list(range(units)), "Original acquisition/serving law required")
        for unit in row["fits"]:
            routes = 1 if arm in (O1, O4, F1, F4) else 4
            native = [seed + cfg["member_seed_stride"] * (unit["member"] + m) for m in range(routes)]
            reader.require(unit["native_factor_dropout_seeds"] == [[v, v + cfg["factor_seed_offset"], v + cfg["dropout_seed_offset"]] for v in native], "Paired acquisition seeds mismatch")
            step = unit["completed_updates"]
            reader.require(1 <= unit["selected_step"] <= step <= cfg["max_updates"], "Original step bounds required")
            if context["kind"] == "new":
                op, roles = unit["operation_counts"], unit["parameter_roles"]
                reader.require(op["updates"] == op["adam_steps"] == op["shared_own_ce_reverse_mode_calls"] == op["private_weighted_ce_reverse_mode_calls"] == step and
                               op["backwards"] == op["reverse_mode_calls"] == 2 * step, "Two block derivatives/one Adam step required")
                reader.require(roles["reverse_mode_calls_per_update"] == 2 and roles["adam_steps_per_update"] == 1 and
                               roles["private_bank_route_count"] == 4 and roles["private_bank_route_axis"] == 0 and roles["private_extra_names"] == unit["extra_private_names"], "Explicit independent private banks required")
                names = [roles[k] for k in ("shared_native_names", "private_diagonal_names", "private_extra_names")]
                reader.require(sum(len(v) for v in names) == len(set().union(*map(set, names))) and
                               set().union(*map(set, names)) == set(roles["parameter_shapes"]), "Disjoint complete parameter roles required")
                losses = unit["last_train_losses"]
                reader.require(all(math.isfinite(v) for v in [losses["own_ce"], losses["private_weighted_ce"]] + losses["members"]["own_ce"] + losses["members"]["private_weighted_ce"]), "Preserved finite own/weighted route losses required")
    reader.require(sum(len(r["fits"]) for r in rows) == complete["fit_units"] and reader.sum_operations(rows) == complete["operation_counts"], "Original fit/operation totals required")
    context.update(complete=complete, index=index)


def weight_records(reader, contexts):
    result = {}
    for backbone in BACKBONES:
        context = contexts[backbone, "new"]
        saved = context["complete"]["bootstrap_weight_records"]
        reader.require([r["family_seed"] for r in saved] == list(SEEDS), "One weight table for every paired seed required")
        for seed, record in zip(SEEDS, saved):
            folder = context["output"] / f"bootstrap_seed{seed}"
            reader.require(reader.read_json(folder / "WEIGHTS.json") == record and digest(folder / "WEIGHTS.npz") == record["archive_sha256"], "Preserved weight artifact identity mismatch")
            reader.require(record["generator_seed"] == seed + 6000119 and record["seed_offset"] == 6000119 and record["shape"] == [4, 580] and
                           record["dtype"] == "float64" and record["class_normalization"] == "mean_one_preserve_native_class_mass" and
                           record["shared_across_arms"] == list(FRESH) and record["trained_parameters_added"] == 0 and record["TEST_access"] is False, "Fixed positive bootstrap policy record required")
            reader.require(all(context["index"][seed, arm]["fits"][0]["bootstrap"] == record for arm in FRESH), "Exact table shared across all three arms required")
            result[backbone + "_seed" + str(seed)] = record
    for seed in SEEDS:
        a, b = [result[backbone + "_seed" + str(seed)] for backbone in BACKBONES]
        reader.require(all(a[k] == b[k] for k in ("raw_exponential_sha256", "weights_sha256", "ordered_TRAIN_ids_sha256", "ordered_TRAIN_labels_sha256")), "Same seeded table/ordered TRAIN roles across companions required")
    return result  # Hash custody only: no TRAIN or weight array is imported.


def selected(reader, contexts, np):
    arrays, hashes, ids, labels = {}, {}, None, None
    keys = {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"}
    for backbone in BACKBONES:
        for seed in SEEDS:
            for arm in ARMS:
                kind = "new" if arm in FRESH else "unweighted" if arm in UNWEIGHTED else "original"
                path = contexts[backbone, kind]["output"] / f"{arm}_seed{seed}/selected_VALID.npz"
                with np.load(path, allow_pickle=False) as archive:
                    reader.require(set(archive.files) == keys, "Exact saved VALID keys required")
                    a = {k: archive[k].copy() for k in keys}
                members = 1 if arm in (O1, F1) else 4
                reader.require(a["ids"].shape == a["y"].shape == (5274,) and a["ids"].dtype == a["y"].dtype == np.int64, "VALID5274 identity required")
                reader.require(a["raw_logits"].shape == (members, 5274, 10) and a["raw_logits"].dtype == np.float32 and
                               a["probability_mean"].shape == (5274, 10) and a["probability_mean"].dtype == np.float32, "Original float32 serving/logits required")
                reader.require(a["member_errors"].shape == (members, 5274) and a["pooled_errors"].shape == (5274,) and
                               a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_ and np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all(), "Finite saved outputs/error flags required")
                if ids is None:
                    ids, labels = a["ids"], a["y"]
                    reader.require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10)), "Ordered ten-class VALID required")
                reader.require(np.array_equal(ids, a["ids"]) and np.array_equal(labels, a["y"]) and
                               np.array_equal(a["probability_mean"].argmax(1) != labels, a["pooled_errors"]), "All66 archive roles/serving decisions must agree")
                arrays[backbone, seed, arm], hashes[str(path)] = a, digest(path)
    return arrays, hashes, ids, labels


def flags(reader, contrasts, candidate):
    stat = lambda ref, metric: contrasts[candidate + "_minus_" + ref]["quality_deltas"][metric]
    counterpart = COUNTERPART[candidate]
    accuracy = {"positive_mean_against_each_required_reference": {ref: contrasts[candidate + "_minus_" + ref]["summed_three_seed_readout_diagnostics"]["count_decomposition_delta"]["pooled_correct"] > 0 for ref in (C, S, O4, F4)},
                "mean_accuracy_vs_coherent_at_least_0_2pp": stat(C, "pooled_accuracy_pct")["mean"] >= .2,
                "mean_accuracy_vs_ordinary_I4_at_least_0_2pp": stat(O4, "pooled_accuracy_pct")["mean"] >= .2,
                "ordinary_I4_all_three_nonnegative": stat(O4, "pooled_accuracy_pct")["nonnegative_seed_count"] == 3,
                "ordinary_I4_at_least_two_positive": stat(O4, "pooled_accuracy_pct")["positive_seed_count"] >= 2}
    member, nll = stat(counterpart, "mean_member_accuracy_pct"), stat(counterpart, "pooled_nll")
    protection = {"mean_member_vs_own_counterpart_nonnegative": contrasts[candidate + "_minus_" + counterpart]["summed_three_seed_readout_diagnostics"]["member_correct_delta"] >= 0,
                  "each_seed_member_delta_at_least_minus_0_1pp": min(member["seed_deltas"]) >= -.1,
                  "mean_nll_deterioration_at_most_0_02": nll["mean"] <= .02, "each_seed_nll_deterioration_at_most_0_05": max(nll["seed_deltas"]) <= .05}
    return {"own_unweighted_counterpart": counterpart, "accuracy_transfer_criteria": accuracy,
            "accuracy_transfer_within_backbone": all(accuracy["positive_mean_against_each_required_reference"].values()) and all(v for k, v in accuracy.items() if k != "positive_mean_against_each_required_reference"),
            "quality_protection_criteria": protection, "quality_protection_within_backbone": all(protection.values())}


def combination(reader, np, scores, labels):
    per_seed = []
    for seed in SEEDS:
        p = {arm: scores[seed, arm]["P"] for arm in (C, A, B, AB)}
        ra, rb, rab = (~p[C] & p[arm] for arm in (A, B, AB))
        ha, hb, hab = (p[C] & ~p[arm] for arm in (A, B, AB))
        cohorts = {"A_repairs": ra, "B_repairs": rb, "AB_repairs": rab, "A_harms": ha, "B_harms": hb, "AB_harms": hab,
                   "A_only_repairs": ra & ~rb, "B_only_repairs": rb & ~ra, "shared_A_B_repairs": ra & rb,
                   "A_repairs_survive_AB": ra & p[AB], "A_repairs_become_AB_harms_versus_A": ra & ~p[AB],
                   "B_repairs_survive_AB": rb & p[AB], "B_repairs_become_AB_harms_versus_B": rb & ~p[AB],
                   "A_only_repairs_survive_AB": ra & ~rb & p[AB], "B_only_repairs_survive_AB": rb & ~ra & p[AB],
                   "shared_A_B_repairs_survive_AB": ra & rb & p[AB], "new_AB_repairs": rab & ~ra & ~rb,
                   "A_harms_survive_AB": ha & ~p[AB], "B_harms_survive_AB": hb & ~p[AB],
                   "A_harms_recovered_AB": ha & p[AB], "B_harms_recovered_AB": hb & p[AB],
                   "new_AB_harms_where_A_and_B_both_correct": hab & ~ha & ~hb}
        counts = {key: int(mask.sum()) for key, mask in cohorts.items()}
        interaction = scores[seed, AB]["counts"]["pooled_correct"] - scores[seed, A]["counts"]["pooled_correct"] - scores[seed, B]["counts"]["pooled_correct"] + scores[seed, C]["counts"]["pooled_correct"]
        reader.require(interaction == counts["AB_repairs"] - counts["AB_harms"] - counts["A_repairs"] + counts["A_harms"] - counts["B_repairs"] + counts["B_harms"], "Interaction repair/harm identity failed")
        reader.require(counts["AB_repairs"] == sum(counts[k] for k in ("A_only_repairs_survive_AB", "B_only_repairs_survive_AB", "shared_A_B_repairs_survive_AB", "new_AB_repairs")), "Repair survival partition failed")
        per_seed.append({"seed": seed, "interaction_correct_count": interaction, "cohorts": counts,
                         "classes": [{"class": k, "cohorts": {key: int((mask & (labels == k)).sum()) for key, mask in cohorts.items()}} for k in range(10)]})
    return {"roles": {"C": C, "A": A, "B": B, "A+B": AB}, "accuracy_interaction_AplusB_minus_A_minus_B_plus_C_pp": reader.paired(100 * row["interaction_correct_count"] / 5274 for row in per_seed),
            "per_seed_repair_survival": per_seed, "summed_three_seed_readout_cohorts": {key: sum(row["cohorts"][key] for row in per_seed) for key in per_seed[0]["cohorts"]},
            "scope": "A lost repair is a harm of AB versus that ingredient, not a harm versus C. Positive interaction cannot rescue worse final accuracy."}


def analyze(reader, np, backbone, scores, contexts, labels):
    local = {kind: contexts[backbone, kind] for kind in ("new", "unweighted", "original")}
    groups, aggregates = [], {}
    for arm in ARMS:
        kind = "new" if arm in FRESH else "unweighted" if arm in UNWEIGHTED else "original"
        arm_rows = []
        for seed in SEEDS:
            value, original = scores[seed, arm], local[kind]["index"][seed, arm]
            row = {"arm": arm, "seed": seed, "acquisition_origin": kind, **{k: value[k] for k in ("quality", "counts", "classes")}, "original_group_record": original}
            groups.append(row); arm_rows.append(row)
        aggregates[arm] = {"summed_counts_over_three_seed_readouts": {k: sum(r["counts"][k] for r in arm_rows) for k in arm_rows[0]["counts"] if k != "member_correct"},
                           "mean_quality": {k: statistics.mean(r["quality"][k] for r in arm_rows) for k in ("pooled_accuracy_pct", "pooled_nll", "mean_member_accuracy_pct", "mean_member_nll", "worst_member_accuracy_pct")},
                           "costs": reader.cost_summary([r["original_group_record"] for r in arm_rows])}
    pairs = [(candidate, ref) for candidate in FRESH for ref in REFERENCES + UNWEIGHTED] + [(AB, B), (LB, B), (AB, LB), (A, C), (L, C), (A, L)]
    contrasts = {a + "_minus_" + b: reader.contrast(scores, a, b, labels) for a, b in pairs}
    costs = {"new_9_acquisitions": reader.cost_summary(local["new"]["complete"]["results"]),
             "reused_9_unweighted_acquisitions": reader.cost_summary([local["unweighted"]["index"][s, a] for s in SEEDS for a in UNWEIGHTED]),
             "reused_33_original_reference_acquisitions": reader.cost_summary([local["original"]["index"][s, a] for s in SEEDS for a in REFERENCES]),
             "new_weight_setup_seconds": local["new"]["complete"]["bootstrap_setup_seconds_sum"],
             "entire_unweighted_12_acquisition_parent": reader.cost_summary(local["unweighted"]["complete"]["results"]),
             "entire_original_39_acquisition_parent": reader.cost_summary(local["original"]["complete"]["results"])}
    custody = {kind: {"owner_end_sha256": digest(ctx["folder"] / "OWNER_END.json"), "complete_family_sha256": ctx["complete_sha256"],
                      "freeze_sha256": digest(ctx["folder"] / "FREEZE.json"), "original_owner_end": ctx["owner"], "original_freeze": ctx["freeze"], "original_complete_family": ctx["complete"]} for kind, ctx in local.items()}
    return {"backbone": backbone, "groups": 33, "acquisition_units_represented": 51, "groups_detail": groups, "aggregates": aggregates,
            "paired_contrasts": contrasts, "frozen_flags": {a: flags(reader, contrasts, a) for a in FRESH},
            "C_A_B_AB_combination": combination(reader, np, scores, labels), "matched_capacity_control": AB + "_minus_" + LB,
            "costs": costs, "source_custody": custody}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True, help="New JSON path; existing paths refused")
    args = parser.parse_args()
    root, report_path = args.research_root.resolve(), args.report.resolve()
    reader = load_math(root)
    reader.require(not report_path.exists(), "Report must be a new file")
    manifest = reader.read_json(Path(__file__).with_name("SOURCE.json"))
    reader.require(manifest["files"]["analysis.py"] == digest(Path(__file__)) and manifest["roster"] == list(ARMS), "Frozen reader source/roster manifest mismatch")
    decision = root / DECISION_DIR
    reader.require(digest(decision / "DECISION.md") == DECISION_SHA256, "Exact frozen bootstrap DECISION.md required")
    bindings = {}
    for backbone in BACKBONES:
        path = decision / backbone / "REFERENCE_BINDINGS.json"
        reader.require(digest(path) == BINDINGS_SHA256[backbone], "Exact pre-fit reference binding required")
        bindings[backbone] = reader.read_json(path)
    locations = {(b, kind): folder for b in BACKBONES for kind, folder in folders(root, b).items()}
    # All six successful owners precede even metadata header reads.
    owners = {key: reader.closed_owner(folder / "OWNER_END.json") for key, folder in locations.items()}
    headers = {key: reader.completion_header(folder / "actual_family_v1/COMPLETE_FAMILY.json") for key, folder in locations.items()}
    contexts = {key: preflight(reader, root, key[0], key[1], folder, owners[key], headers[key], bindings[key[0]]) for key, folder in locations.items()}
    matched_metadata(reader, contexts)  # Both9/9 + both12/12 + both21/39 and all identities first.
    for context in contexts.values():
        records(reader, context)
    weights = weight_records(reader, contexts)
    import numpy as np
    reader.np = np
    arrays, archives, ids, labels = selected(reader, contexts, np)
    scores = {key: reader.score(value) for key, value in arrays.items()}  # All66 archive identities first.
    families = {b: analyze(reader, np, b, {(s, a): scores[b, s, a] for s in SEEDS for a in ARMS}, contexts, labels) for b in BACKBONES}
    joined_flags = {candidate: {"accuracy_transfer_clue_both_backbones": all(families[b]["frozen_flags"][candidate]["accuracy_transfer_within_backbone"] for b in BACKBONES),
                                "quality_protection_both_backbones": all(families[b]["frozen_flags"][candidate]["quality_protection_within_backbone"] for b in BACKBONES)} for candidate in FRESH}
    report = {"complete": True, "backbones": list(BACKBONES), "seeds": list(SEEDS), "roster": list(ARMS), "joined_groups": 66,
              "new_acquisition_units": 18, "reused_acquisition_units": 84, "total_acquisition_units_represented": 102, "TEST_access": False,
              "analysis_source_sha256": digest(Path(__file__)), "reader_manifest_sha256": digest(Path(__file__).with_name("SOURCE.json")),
              "immutable_math_reader_sha256": MATH_READER_SHA256, "decision_sha256": DECISION_SHA256, "reference_bindings_sha256": BINDINGS_SHA256,
              "original_reference_bindings": bindings, "weight_artifact_custody": weights,
              "ordered_VALID_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "ordered_VALID_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
              "selected_VALID_archive_hashes": archives, "families": families, "joined_frozen_flags": joined_flags, "transition_state_order": reader.STATES,
              "coverage_identity": "pooled correct = any-member coverage - lost correct alternatives + aggregation-only correct",
              "interpretation_limits": ["All fixed companions/arms/seeds retained. Accuracy transfer and own-counterpart member/NLL protection are independent flags.",
                  "Interaction cannot rescue worse final accuracy; AB versus B/A and the same-weight LoRA control are individual measured contrasts.",
                  "Encountered selected development VALID; paired df2 intervals describe three-seed optimization variation on one graph, not node-IID or graph-population uncertainty.",
                  "No bootstrap novelty, whole-model posterior calibration, protected-quality or generality claim follows automatically.",
                  "Original selected/restored metrics, losses, role records, failures, costs and complete records remain preserved. No new model/checkpoint/TEST forward or TRAIN array read occurs.",
                  "18 new versus84 reused acquisitions are separate. The unselected historical local/separable/exchange18 acquisitions remain in original parent custody/costs.",
                  "Weight setup, two reverse calls, observed overlap and process-lifetime RSS remain charged; no isolated speed claim is made.",
                  "A successful screen requires same-weight capable untiedI4, conventional all-parameter-bootstrap controls and unused graph confirmation; failure closes this exact rule without a rescue grid."]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"complete": True, "report": str(report_path), "joined_frozen_flags": joined_flags}))


if __name__ == "__main__":
    main()
