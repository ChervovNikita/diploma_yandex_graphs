"""Fixed post-closure VALID statistics. Import is stdlib-only; run is CPU NumPy."""
import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (7301, 7403, 7507)
ARMS = ("shared4_unchanged", "ordinary_genuine_I4", "factorized_allmap_genuine_I4")
REPORT_SHA = (
    "bfecd826dad5b375f41cb5a35430fa7b10a135ec69bd1ba63bab481152839711",
    "36ff1f94a4c312d4b1b2454193060e029b5b446b084cfc67f4818fd61b625133",
)
KEYS = {"ids", "y", "raw_logits", "probability_mean", "member_errors", "pooled_errors"}
EDGES = {
    "probability": tuple(i / 10 for i in range(11)),
    "probability_margin": (-1, -.5, -.2, -.1, -.05, -.01, 0, .01, .05, .1, .2, .5, 1),
    "logit_margin": (-float("inf"), -10, -5, -2, -1, -.5, 0, .5, 1, 2, 5, 10, float("inf")),
}
QUANTILES = (0, .05, .25, .5, .75, .95, .99, 1)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(np, values, kind):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(np.isfinite(a).all(), "Nonfinite diagnostic operand")
    counts = np.histogram(a, bins=EDGES[kind])[0]
    _require(int(counts.sum()) == len(a), "Histogram does not cover every value")
    return {"samples": len(a), "mean": float(a.mean()) if len(a) else None,
            "quantiles": np.quantile(a, QUANTILES, method="linear").tolist() if len(a) else None,
            "bin_counts": counts.tolist(), "bin_kind": kind}


def _frame(np, a):
    raw = a["raw_logits"].astype(np.float64)
    y, index = a["y"], np.arange(len(a["y"]))
    shifted = raw - raw.max(-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(-1, keepdims=True)
    true_raw, true_probability = raw[:, index, y], probabilities[:, index, y]
    false_raw = raw.copy()
    false_raw[:, index, y] = -np.inf
    false_probability = probabilities.copy()
    false_probability[:, index, y] = -np.inf
    pred, false_pred = raw.argmax(-1), false_raw.argmax(-1)
    correct, served = ~a["member_errors"], ~a["pooled_errors"]
    coverage = correct.any(0)
    _require(np.array_equal(pred == y[None, :], correct), "Raw argmax and saved member flags disagree")
    pool = a["probability_mean"].astype(np.float64)
    pool_false = pool.copy()
    pool_false[index, y] = -np.inf
    _require(np.array_equal(pool.argmax(-1) == y, served), "Saved pool and error flags disagree")
    rival_advantage = false_raw - true_raw[..., None]
    common = (rival_advantage > 0).all(0)
    common_count = common.sum(-1)
    common_depth = rival_advantage.min(0).max(-1)
    false_agreement = np.stack([(false_pred == k).sum(0) for k in range(10)]).max(0)
    wrong_agreement = np.stack([((pred == k) & ~correct).sum(0) for k in range(10)]).max(0)
    confidence = probabilities.max(-1)
    both = coverage & (~correct).any(0)
    max_correct = np.where(correct, confidence, -np.inf).max(0)
    max_wrong = np.where(~correct, confidence, -np.inf).max(0)
    return dict(y=y, correct=correct, served=served, coverage=coverage,
                pool_true=pool[index, y], pool_margin=pool[index, y]-pool_false.max(-1),
                member_true=true_probability, member_logit_margin=true_raw-false_raw.max(-1),
                member_probability_margin=true_probability-false_probability.max(-1),
                confidence=confidence, common_count=common_count, common_depth=common_depth,
                false_agreement=false_agreement, wrong_agreement=wrong_agreement,
                unanimous_wrong=(pred == pred[0]).all(0) & ~coverage,
                mixed_correct_wrong=both, wrong_confidence_exceeds_correct=both & (max_wrong > max_correct),
                wrong_confidence_equals_correct=both & (max_wrong == max_correct))


def _summary(np, d, mask):
    c = d["correct"][:, mask]
    n = int(mask.sum())
    return {
        "nodes": n, "correct_member_events": int(c.sum()), "wrong_member_events": int((~c).sum()),
        "correct_member_count_0_to_4": np.bincount(c.sum(0), minlength=5).tolist(),
        "best_false_rival_agreement_0_to_4": np.bincount(d["false_agreement"][mask], minlength=5).tolist(),
        "wrong_prediction_agreement_0_to_4": np.bincount(d["wrong_agreement"][mask], minlength=5).tolist(),
        "common_strict_false_rival_count_0_to_9": np.bincount(d["common_count"][mask], minlength=10).tolist(),
        "unanimous_wrong_top_prediction": int((d["unanimous_wrong"] & mask).sum()),
        "mixed_correct_wrong_nodes": int((d["mixed_correct_wrong"] & mask).sum()),
        "wrong_confidence_exceeds_correct_nodes": int((d["wrong_confidence_exceeds_correct"] & mask).sum()),
        "wrong_confidence_equals_correct_nodes": int((d["wrong_confidence_equals_correct"] & mask).sum()),
        "statistics": {
            "pool_true_class_mass": _distribution(np, d["pool_true"][mask], "probability"),
            "pool_true_minus_best_false_probability": _distribution(np, d["pool_margin"][mask], "probability_margin"),
            "member_true_class_mass": _distribution(np, d["member_true"][:, mask], "probability"),
            "member_true_minus_best_false_logit": _distribution(np, d["member_logit_margin"][:, mask], "logit_margin"),
            "member_true_minus_best_false_probability": _distribution(np, d["member_probability_margin"][:, mask], "probability_margin"),
            "correct_member_top_confidence": _distribution(np, d["confidence"][:, mask][c], "probability"),
            "wrong_member_top_confidence": _distribution(np, d["confidence"][:, mask][~c], "probability"),
            "wrong_member_true_class_mass": _distribution(np, d["member_true"][:, mask][~c], "probability"),
            "common_rival_minimum_logit_advantage": _distribution(np, d["common_depth"][mask], "logit_margin"),
        },
    }


def _stratified(np, d, mask):
    return {"all": _summary(np, d, mask),
            "by_truth_class": {str(k): _summary(np, d, mask & (d["y"] == k)) for k in range(10)}}


def _counts(d):
    p, v = d["served"], d["coverage"]
    return dict(nodes=len(p), pooled_correct=int(p.sum()), coverage=int(v.sum()),
                lost_correct_alternatives=int((v & ~p).sum()),
                aggregation_only_correct=int((~v & p).sum()),
                unavailable_alternatives=int((~v).sum()), served_alternatives=int((v & p).sum()))


def run(sage_report, native_report, report_path, archive_roots=None):
    """Read 27 bound selected archives from all CLOSED117 fits; create one new JSON."""
    report_path = Path(report_path)
    _require(not report_path.exists(), "Refuse to replace an existing report")
    paths = (Path(sage_report), Path(native_report))
    for path, expected in zip(paths, REPORT_SHA):
        _require(_sha(path) == expected, "Exact admitted complete analysis required")
    sage, native = (json.loads(p.read_text()) for p in paths)
    _require(native["complete"] is True and native["TEST_access"] is False
             and native["groups"] == 42 and native["fit_units"] == 78
             and native["backbones"] == ["GCN", "GAT"], "Exact complete companion pair required")
    families = {"SAGE": sage, **native["families"]}
    import numpy as np
    arrays, bindings, identity = {}, [], None
    for backbone, family in families.items():
        _require(family["complete"] is True and family["TEST_access"] is False
                 and family["groups"] == 21 and family["fit_units"] == 39
                 and tuple(family["seeds"]) == SEEDS, "Exact complete21/39 family required")
        hashes = family["input_bindings"]["selected_archives"]
        for seed in SEEDS:
            for arm in ARMS:
                matches = [p for p in hashes if Path(p).name == "selected_VALID.npz"
                           and Path(p).parent.name == f"{arm}_seed{seed}"]
                _require(len(matches) == 1, "Unique bound selected archive required")
                original = matches[0]
                path = (Path(archive_roots[backbone]) / f"{arm}_seed{seed}" / "selected_VALID.npz"
                        if archive_roots and backbone in archive_roots else Path(original))
                expected = hashes[original]
                _require(_sha(path) == expected, "Selected archive digest mismatch")
                with np.load(path, allow_pickle=False) as z:
                    _require(set(z.files) == KEYS, "Exact existing VALID archive schema required")
                    a = {k: z[k].copy() for k in KEYS}
                _require(_sha(path) == expected, "Selected archive changed during read")
                _require(a["ids"].shape == a["y"].shape == (5274,)
                         and a["ids"].dtype == a["y"].dtype == np.int64, "Exact VALID roles required")
                _require(a["raw_logits"].shape == (4, 5274, 10)
                         and a["raw_logits"].dtype == np.float32, "Original four-member logits required")
                _require(a["probability_mean"].shape == (5274, 10)
                         and a["probability_mean"].dtype == np.float32, "Original serving probabilities required")
                _require(a["member_errors"].shape == (4, 5274) and a["pooled_errors"].shape == (5274,)
                         and a["member_errors"].dtype == a["pooled_errors"].dtype == np.bool_, "Original flags required")
                _require(np.isfinite(a["raw_logits"]).all() and np.isfinite(a["probability_mean"]).all()
                         and (a["probability_mean"] >= 0).all()
                         and (a["probability_mean"] <= 1).all(), "Finite bounded operands required")
                _require(len(np.unique(a["ids"])) == 5274 and a["ids"].min() >= 0
                         and a["ids"].max() < 11701 and set(a["y"].tolist()) == set(range(10)), "VALID domain mismatch")
                for field in ("ids", "y"):
                    key = "ordered_VALID_ids_sha256" if field == "ids" else "ordered_VALID_labels_sha256"
                    _require(hashlib.sha256(a[field].tobytes()).hexdigest() == family["input_bindings"][key],
                             "Exact ordered role binding mismatch")
                if identity is None:
                    identity = (a["ids"], a["y"])
                _require(np.array_equal(identity[0], a["ids"]) and np.array_equal(identity[1], a["y"]),
                         "Cross-bank/backbone role mismatch")
                arrays[backbone, seed, arm] = a
                bindings.append(dict(backbone=backbone, seed=seed, arm=arm, path=str(path), sha256=expected))
    banks, pairs, signature_cells = [], [], []
    for backbone, family in families.items():
        frozen = {(r["seed"], r["arm"]): r["counts"] for r in family["groups_detail"]}
        for seed in SEEDS:
            frames = {arm: _frame(np, arrays[backbone, seed, arm]) for arm in ARMS}
            for arm, d in frames.items():
                _require(_counts(d) == frozen[seed, arm], "Frozen selected counts changed")
                v, p = d["coverage"], d["served"]
                cohorts = {"no_alternative_wrong": ~v & ~p, "aggregation_only_correct": ~v & p,
                           "alternative_lost": v & ~p, "alternative_served": v & p,
                           "all_member_wrong": ~v, "served_correct": p}
                banks.append(dict(backbone=backbone, seed=seed, arm=arm, counts=_counts(d),
                                  cohorts={k: _stratified(np, d, mask) for k, mask in cohorts.items()}))
            shared = frames[ARMS[0]]
            for ref in ARMS[1:]:
                r = frames[ref]
                q = r["served"] & ~shared["served"]
                missing = q & ~shared["coverage"]
                recoverable = q & shared["coverage"]
                certified = missing & (shared["common_count"] > 0)
                n, certificate_n = int(q.sum()), int(certified.sum())
                counts = dict(I4_correct_shared_wrong=n,
                              unrecoverable_by_member_selection=int(missing.sum()),
                              recoverable_by_member_selection=int(recoverable.sum()),
                              unrecoverable_with_common_strict_rival=certificate_n,
                              shared_correct_I4_wrong=int((shared["served"] & ~r["served"]).sum()))
                _require(counts["unrecoverable_by_member_selection"] +
                         counts["recoverable_by_member_selection"] == n, "Paired partition failed")
                masks = {"I4_correct_shared_wrong": q, "unrecoverable_by_member_selection": missing,
                         "recoverable_by_member_selection": recoverable,
                         "unrecoverable_with_common_strict_rival": certified}
                pairs.append(dict(backbone=backbone, seed=seed, reference=ref, counts=counts,
                                  cohorts={k: {"shared": _stratified(np, shared, m),
                                               "I4": _stratified(np, r, m)} for k, m in masks.items()}))
                signature_cells.append(dict(backbone=backbone, seed=seed, reference=ref,
                                            losses=n, certified_missing=certificate_n,
                                            strict_majority=n > 0 and 2 * certificate_n > n))
    edge_json = {k: [("-inf" if x < 0 else "inf") if abs(x) == float("inf") else x for x in v]
                 for k, v in EDGES.items()}
    result = dict(schema="closed117-stored-VALID-error-diagnosis-v1", complete=True,
                  source_sha256=_sha(__file__), input_analysis_sha256=list(REPORT_SHA), selected_archives=bindings,
                  backbones=list(families), paired_seeds=list(SEEDS), selected_banks=27, closed_fit_units=117,
                  TEST_access=False, new_model_forwards=0, new_training=False,
                  quantile_levels=list(QUANTILES), histogram_edges=edge_json,
                  banks=banks, pairs=pairs,
                  missing_ranking_signature=dict(cells=signature_cells,
                    consistent_across_both_references_all_nine_cells=all(c["strict_majority"] for c in signature_cells)),
                  interpretation="Descriptive frozen VALID diagnosis; no tuned aggregator, new selected score, causal or confirmation verdict.")
    with report_path.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-analysis", required=True)
    parser.add_argument("--native-analysis", required=True)
    parser.add_argument("--report", required=True)
    for backbone in ("sage", "gcn", "gat"):
        parser.add_argument("--" + backbone + "-archives")
    args = parser.parse_args()
    roots = {b: getattr(args, b.lower()+"_archives") for b in ("SAGE", "GCN", "GAT")
             if getattr(args, b.lower()+"_archives")}
    result = run(args.sage_analysis, args.native_analysis, args.report, roots)
    print(json.dumps({"complete": result["complete"], "banks": result["selected_banks"],
                      "model_forwards": 0, "TEST_access": False, "report": args.report}))
