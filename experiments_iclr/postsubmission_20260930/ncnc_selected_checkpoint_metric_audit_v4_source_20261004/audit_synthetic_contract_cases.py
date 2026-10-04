"""Prospective fabricated checks of the actual successor helper functions."""
import copy
import json
from types import SimpleNamespace
from replay_gate import require
from audit_contract import LABELS
from replay_numeric import (engineering_report, valid_arrays, authenticate_pool,
                            reconstruct_i4, expected_profile, reference_metric_receipt, set_profile, private_receipt)


def rejected(function):
    try: function()
    except (RuntimeError, TypeError, ValueError): return
    raise RuntimeError("Deliberately invalid fabricated contract was accepted")


def receipt_for(positive, negative, data_api):
    return {"positive_queries": 60084, "negative_queries": 100000, "all_query_rows_complete": True,
            "encoder_calls": 1, "query_batches": [1, 1], "graph": "complete_TRAIN_only", "serving_pool": "mean_raw_logits",
            "score_digests": {"positive": data_api.tensor_sha(positive), "negative": data_api.tensor_sha(negative)}}


def run(api, context):
    import torch
    data_api, evaluate = api["pilot_data"], api["pilot_evaluate"]
    metric = evaluate.evaluator(context)
    cases = []
    def done(name): cases.append({"case": name, "status": "PASS", "work": {"planned": 0, "attempted": 0, "entered_original_scorer": 0, "returned": 0, "completed_validated": 0}, "scope": "fabricated_shared_contract_no_scorer_invocation"})
    eps = float(torch.finfo(torch.float32).eps); atol = 128 * eps
    # At zero, the absolute term alone is exact; at one, both terms contribute.
    for reference, boundary in ((0., atol), (1., 1. + 2 * atol)):
        r = torch.tensor([reference], dtype=torch.float32); equal = torch.tensor([boundary], dtype=torch.float32)
        inside = torch.nextafter(equal, torch.tensor([-float("inf")], dtype=torch.float32))
        outside = torch.nextafter(equal, torch.tensor([float("inf")], dtype=torch.float32))
        require(engineering_report(inside, r)["all_rows_within_engineering_budget"] and engineering_report(equal, r)["all_rows_within_engineering_budget"] and not engineering_report(outside, r)["all_rows_within_engineering_budget"], "Inside/equal/outside absolute/relative engineering comparator differs")
        require(engineering_report(outside, r)["acceptance_authority"] is False, "Engineering discrepancy acquired PASS authority")
    done("engineering_absolute_relative_boundaries")
    negative = torch.arange(100000, dtype=torch.float32)
    positive = torch.tensor([-1., 99950., 99951., 100001.], dtype=torch.float32).repeat(15021)
    require(evaluate.hits50(metric, positive, negative) == .5, "Official strict50th-negative ties differ")
    rejected(lambda: valid_arrays(positive.double(), negative, receipt_for(positive.double(), negative, data_api), data_api))
    rejected(lambda: valid_arrays(positive[:-1], negative, receipt_for(positive[:-1], negative, data_api), data_api))
    nonfinite = positive.clone(); nonfinite[0] = float("nan")
    rejected(lambda: valid_arrays(nonfinite, negative, receipt_for(nonfinite, negative, data_api), data_api))
    malformed = {"wall_seconds": float("nan"), "unexpected_object": object()}; malformed["cycle"] = malformed
    safe = private_receipt(malformed)
    require(safe["wall_seconds"] == {"nonfinite_float": "nan"} and safe["cycle"]["unavailable_node"] == "cyclic_receipt" and safe["unexpected_object"]["tensor_or_object_body_exported"] is False, "Malformed receipt preservation differs")
    json.dumps(safe, allow_nan=False)
    done("strict_ties_finite_shape_dtype")
    members = []
    for member in range(4):
        binding = {"arm": "FABRICATED_I4", "base_seed": 0, "member": member, "selected_state_digest": str(member) * 64,
                   "canonical_data_digests": {"positive": "FABRICATED_CANONICAL", "negative": "FABRICATED_CANONICAL"}, "original_identity": {"synthetic_only": True}}
        rows = {}
        for label in LABELS:
            # Different members and pools select different fixed labels by exact bytes.
            p = torch.full((60084,), 8. + member + LABELS.index(label), dtype=torch.float32)
            n = torch.full((100000,), 1. + member + LABELS.index(label), dtype=torch.float32)
            rows[label] = {"member": member, "label": label, "positive": p, "negative": n,
                           "receipt": receipt_for(p, n, data_api), "before_profile": expected_profile(label),
                           "after_profile": expected_profile(label), "validated": True, "binding": copy.deepcopy(binding)}
        target_p = LABELS[member % 3]; target_n = LABELS[(member + 1) % 3]
        references, provenance = {}, {"member": member}
        for pool, label in (("positive", target_p), ("negative", target_n)):
            digest = rows[label]["receipt"]["score_digests"][pool]
            references[pool], provenance[pool] = authenticate_pool(rows, expected_member=member, expected_binding=binding, pool=pool, historical_digest=digest, data_api=data_api)
            require(provenance[pool]["label"] == label, "Cross-label exact reference priority differs")
        missing, missing_receipt = authenticate_pool(rows, expected_member=member, expected_binding=binding, pool="positive", historical_digest="0" * 64, data_api=data_api)
        require(missing is None and not missing_receipt["available"], "Unavailable bytes were synthesized")
        members.append({"member": member, **references, "provenance": provenance})
        for fault in ("label", "profile", "cell", "row", "dtype", "nonfinite", "shape"):
            bad = copy.deepcopy(rows)
            if fault == "label": bad["True1"]["label"] = "True2"
            if fault == "profile": bad["True1"]["before_profile"]["deterministic_algorithms"] = False; bad["True1"]["after_profile"]["deterministic_algorithms"] = False
            if fault == "cell": bad["True1"]["binding"]["base_seed"] = 4
            if fault == "row": bad["True1"]["binding"]["canonical_data_digests"]["positive"] = "REORDERED"
            if fault == "dtype": bad["True1"]["positive"] = bad["True1"]["positive"].double()
            if fault == "nonfinite": bad["True1"]["positive"][0] = float("nan")
            if fault == "shape": bad["True1"]["positive"] = bad["True1"]["positive"][:-1]
            rejected(lambda: authenticate_pool(bad, expected_member=member, expected_binding=binding, pool="positive", historical_digest=rows[target_p]["receipt"]["score_digests"]["positive"], data_api=data_api))
    done("unavailable_partial_cross_label_references")
    done("reference_wrong_label_profile_cell_row_dtype_shape_nonfinite")
    set_profile(torch, "True2")
    partial = copy.deepcopy(members); partial[2]["negative"] = None
    rejected(lambda: reconstruct_i4(partial, evaluate, data_api))
    rp, rn, provenance = reconstruct_i4(members, evaluate, data_api)
    require(provenance["historical_pooled_digest_available"] is False and provenance["member_order"] == list(range(4)) and provenance["score_digests"] == {"positive": data_api.tensor_sha(rp), "negative": data_api.tensor_sha(rn)}, "New I4 reconstruction provenance differs")
    require(reference_metric_receipt(rp, rn, evaluate, metric, 1.)["saved_selected_hits50_equal"] and not reference_metric_receipt(rp, rn, evaluate, metric, 0.)["saved_selected_hits50_equal"], "Shared reconstructed reference metric must reject a wrong saved scalar")
    done("I4_cross_label_new_digest_reference_metric")
    rejected(lambda: reconstruct_i4(list(reversed(members)), evaluate, data_api))
    fake_evaluate = SimpleNamespace(__name__=evaluate.__name__, __file__=evaluate.__file__, mean_native_scores=lambda rows: (rows[0][0], rows[0][1]))
    rejected(lambda: reconstruct_i4(members, fake_evaluate, data_api))
    done("I4_wrong_order_wrong_reduction")
    return cases
