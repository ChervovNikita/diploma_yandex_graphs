"""Reused qualified F reader math; no numerical imports at module load."""


import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics
import time


ARMS = ("FiLM_M1", "FiLM_shared4", "FiLM_I4")


METRICS = ("pooled_accuracy_pct", "pooled_nll", "pooled_brier",
           "mean_member_accuracy_pct", "worst_member_accuracy_pct",
           "mean_member_nll", "worst_member_nll",
           "mean_member_brier", "worst_member_brier")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def current_receipt(path):
    p = Path(path).resolve()
    require(p.is_file(), "Required complete artifact missing: " + str(p))
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


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
    require(len(arrays) == 9, "All nine exact archive schemas before scoring/calibration")
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
    require(attempted == 45 and len(records) == 9 and all(len(r["fits"]) == 5 for r in records),
            "All forty-five scalar endpoint attempts must be terminal before contrasts")
    success = sum(f["status"] == "finite_fixed_endpoint" for r in records for f in r["fits"])
    return calibrated, records, {"attempted_scalar_fits": 45, "successful_scalar_fits": success,
           "failed_scalar_fits": 45 - success, "requested_updates_per_fit": 500,
           "requested_scalar_updates": 22500, "known_successful_scalar_updates": success * 500,
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
