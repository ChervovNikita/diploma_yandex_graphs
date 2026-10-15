"""Pool four separately selected native VALID exports; no model/data construction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from common import atomic_json, atomic_torch, inside, receipt, require_allocation


def aggregate(paths, root, output):
    import torch
    paths = [inside(p, root) for p in paths]
    output = inside(output, root)
    if output.exists():
        raise ValueError("Preserve earlier bank exports; choose a fresh output")
    members = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    if len(members) != 4 or {m.get("body") for m in members} != {0, 1, 2, 3}:
        raise ValueError("I4 needs exactly the four independently trained native body authorities")
    if len({m["job_hash"] for m in members}) != 4 or len({m["seed"] for m in members}) != 4:
        raise ValueError("Independent job identities and initialization seeds required")
    members.sort(key=lambda m: m["body"])
    first = members[0]
    for member in members:
        if member.get("schema") != "own-selected-VALID-logits-v1" or member.get("role") != "VALID" \
                or member.get("TEST_accessed") is not False or member.get("procedure") != "native_single":
            raise ValueError("I4 consumes only own selected native VALID authorities")
        for field in ("cohort_id", "dataset", "recipe_id", "family", "split", "policy"):
            if member[field] != first[field]:
                raise ValueError("Misaligned selected bank contract: " + field)
        if not torch.equal(member["ids"], first["ids"]) or member["labels"].shape != first["labels"].shape \
                or not torch.allclose(member["labels"].to(torch.float64), first["labels"].to(torch.float64),
                                      atol=0, rtol=0, equal_nan=True):
            raise ValueError("Native IDs/order/labels differ between selected members")
        if member["raw_logits"].shape != first["raw_logits"].shape \
                or member["raw_logits"].dtype != first["raw_logits"].dtype \
                or not torch.isfinite(member["raw_logits"]).all().item():
            raise ValueError("Finite aligned raw-logit shapes/dtypes required")
        if first["family"] == "link":
            if (member.get("complete_VALID_positive_records"), member.get("complete_VALID_negative_records")) != (60084, 100000) \
                    or not torch.equal(member["query_endpoints"], first["query_endpoints"]):
                raise ValueError("Link I4 requires exact official VALID endpoints and complete positive/negative roles")
    pooled = torch.stack([m["raw_logits"] for m in members]).mean(0)
    labels = first["labels"]
    if first["family"] == "node":
        metrics = {"accuracy": float(pooled.argmax(-1).eq(labels).float().mean())}
        loss = float(torch.nn.functional.cross_entropy(pooled, labels))
    elif first["family"] == "ogb":
        from ogb.graphproppred import Evaluator
        metrics = {k: float(v) for k, v in Evaluator(first["dataset"]).eval(
            {"y_true": labels.numpy(), "y_pred": pooled.numpy()}).items()}
        mask = labels == labels
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(pooled.float()[mask], labels.float()[mask]))
    elif first["family"] == "link":
        from ogb.linkproppred import Evaluator
        if first["dataset"] != "ogbl-collab" or labels.ndim != 1 or len(labels) != 160084 \
                or not bool((labels[:60084] == 1).all()) or not bool((labels[60084:] == 0).all()):
            raise ValueError("Only the explicit official Collab VALID query order is supported")
        evaluator = Evaluator(name="ogbl-collab")
        evaluator.K = 50
        positive, negative = pooled[:60084], pooled[60084:]
        metrics = {"hits@50": float(evaluator.eval({"y_pred_pos": positive, "y_pred_neg": negative})["hits@50"])}
        loss = float(-torch.nn.functional.logsigmoid(positive).mean()-torch.nn.functional.logsigmoid(-negative).mean())
    else:
        raise ValueError("Explicit node, molecular or link evaluator required")
    output.mkdir(parents=True)
    bank = {"schema": "independent-own-selected-VALID-bank-v1", "role": "VALID", "TEST_accessed": False,
            "cohort_id": first["cohort_id"], "dataset": first["dataset"], "recipe_id": first["recipe_id"],
            "family": first["family"], "split": first["split"], "policy": first["policy"],
            "raw_logits": pooled, "member_raw_logits": torch.stack([m["raw_logits"] for m in members]),
            "ids": first["ids"], "labels": labels, "metrics": metrics, "loss": loss,
            "members": [{k: m[k] for k in ("job_id", "job_hash", "seed", "body", "selected_epoch", "selected_checkpoint")}
                        for m in members], "member_export_receipts": [receipt(p) for p in paths],
            "M1_exact_member0_job_hash": first["job_hash"], "aggregation": "mean_raw_logits_after_four_own_selectors",
            "extra_training_fits": 0, "pooled_epoch_reselection": False}
    if first["family"] == "link":
        bank.update(query_endpoints=first["query_endpoints"], complete_VALID_positive_records=60084,
                    complete_VALID_negative_records=100000)
    bank_receipt = atomic_torch(torch, output / "I4_selected_VALID.pt", bank)
    result = {"schema": "independent-own-selected-VALID-bank-result-v1", "status": "complete",
              "cohort_id": first["cohort_id"], "metrics": metrics, "loss": loss,
              "members": bank["members"], "M1_exact_member0_job_hash": first["job_hash"],
              "aggregation": bank["aggregation"], "bank_receipt": bank_receipt,
              "extra_training_fits": 0, "TEST_accessed": False, "full_campaign_GO": False}
    atomic_json(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--members", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_allocation()
    result = aggregate(args.members, Path(args.project_root).resolve(), args.output)
    print(json.dumps({"status": result["status"], "cohort_id": result["cohort_id"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
