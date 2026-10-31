"""Paired MolHIV readouts from one trajectory; stable float64 mixture BCE."""
import csv
import json
import math

import torch
from torch.nn import functional as F
from ogb.graphproppred import Evaluator

ALL_LAYER_ARM = "all_layer_be_gine_adaptation_prob"
READOUTS = ("raw", "prob")
READOUT_CONTRACT = {
    "training_trajectories_per_arm_seed_budget": 1,
    "reported_readouts": ["mean_member_raw_logits", "mean_member_sigmoid_probabilities"],
    "primary_selection": {"existing_arms": "raw", ALL_LAYER_ARM: "prob"},
    "selection": "one checkpoint per readout; max complete-validation ROC_AUC, min corresponding BCE, earliest epoch",
    "matched_pooling_contrasts": "raw-selected versus raw-selected and probability-selected versus probability-selected",
    "metric_precision": "CPU float64 on retained member logits; stable logsumexp mixture BCE; no probability clipping",
    "training_loss": "unchanged equal-graph mean member BCE with logits",
    "ceiling_interpretation": "No conclusive advantage when either contrasted arm selects in final10 and final10 versus preceding10 improves ROC_AUC>1e-4 or BCE falls>1e-5. Apply to both readouts/all arms equally; cap extension requires a prospective matched amendment.",
    "ceiling_practical_tolerances": {"quality_gain": 1e-4, "pooled_loss_fall": 1e-5},
}


def primary_pooling(arm):
    return "prob" if arm == ALL_LAYER_ARM else "raw"


def metrics(member_logits, targets, arm):
    z, y = member_logits.detach().cpu().double(), targets.detach().cpu().double()
    if z.ndim != 3 or y.shape != z.shape[1:] or not torch.isfinite(z).all():
        raise ValueError("MolHIV evidence shape/nonfinite logits mismatch")
    finite = torch.isfinite(y)
    if not bool(finite.any()) or not bool(((y[finite] == 0) | (y[finite] == 1)).all()):
        raise ValueError("MolHIV evidence needs finite binary labels")
    observed = finite.sum(-1) > 0
    safe = torch.where(finite, y, torch.zeros_like(y))
    def equal_graph(element):
        return float(((element * finite).sum(-1) / finite.sum(-1).clamp_min(1))[observed].mean())
    raw = z.mean(0)
    log_p = F.logsigmoid(z).logsumexp(0) - math.log(z.shape[0])
    log_q = F.logsigmoid(-z).logsumexp(0) - math.log(z.shape[0])
    member_element = F.binary_cross_entropy_with_logits(z, safe[None].expand_as(z), reduction="none").mean(0)
    evaluator = Evaluator(name="ogbg-molhiv")
    values = {"mean_member_bce": equal_graph(member_element),
              "raw_rocauc": float(evaluator.eval({"y_true": y.numpy(), "y_pred": raw.numpy()})["rocauc"]),
              "prob_rocauc": float(evaluator.eval({"y_true": y.numpy(), "y_pred": log_p.exp().numpy()})["rocauc"]),
              "raw_pooled_bce": equal_graph(F.binary_cross_entropy_with_logits(raw, safe, reduction="none")),
              "prob_pooled_bce": equal_graph(-(safe * log_p + (1 - safe) * log_q))}
    primary = primary_pooling(arm)
    values.update(rocauc=values[primary + "_rocauc"], pooled_bce=values[primary + "_pooled_bce"])
    return values


def selection_key(metrics, epoch, pooling):
    auc, loss = metrics[pooling + "_rocauc"], metrics[pooling + "_pooled_bce"]
    if not math.isfinite(auc) or not math.isfinite(loss):
        raise ValueError("Nonfinite readout selection metric")
    return auc, -loss, -epoch


def update_selected(output, best, row, checkpoint, evidence, arm):
    for pooling in READOUTS:
        key = selection_key(row, row["epoch"], pooling)
        if best.get(pooling) is not None and key <= best[pooling]:
            continue
        best[pooling] = key
        selected = {**row, "selection_pooling": pooling}
        payload = {**checkpoint, "selection_pooling": pooling}
        torch.save(payload, output / ("selected_" + pooling + ".pt"))
        torch.save(evidence, output / ("selected_validation_member_logits_" + pooling + ".pt"))
        (output / ("selected_" + pooling + ".json")).write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        if pooling == primary_pooling(arm):
            torch.save(payload, output / "selected.pt")
            torch.save(evidence, output / "selected_validation_member_logits.pt")
            (output / "selected.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")


def selections(output):
    return {pooling: json.loads((output / ("selected_" + pooling + ".json")).read_text()) for pooling in READOUTS}


def ceiling_diagnostics(output, epochs):
    rows = list(csv.DictReader((output / "validation_trace.csv").open()))
    if len(rows) != epochs or epochs < 20:
        raise ValueError("Require the complete registered trajectory for ceiling diagnostics")
    result = {}
    chosen = selections(output)
    tolerances = READOUT_CONTRACT["ceiling_practical_tolerances"]
    for pool in READOUTS:
        means = {name: [sum(float(r[pool + "_" + name]) for r in group)/10
                       for group in [rows[-20:-10], rows[-10:]]]
                 for name in ["rocauc", "pooled_bce"]}
        gain = means["rocauc"][1] - means["rocauc"][0]
        fall = means["pooled_bce"][0] - means["pooled_bce"][1]
        late = chosen[pool]["epoch"] >= epochs - 9
        improving = gain > tolerances["quality_gain"] or fall > tolerances["pooled_loss_fall"]
        result[pool] = {"selected_epoch": chosen[pool]["epoch"], "selected_in_final10": late,
                        "preceding10_mean_ROC_AUC": means["rocauc"][0], "final10_mean_ROC_AUC": means["rocauc"][1],
                        "preceding10_mean_BCE": means["pooled_bce"][0], "final10_mean_BCE": means["pooled_bce"][1],
                        "ROC_AUC_gain": gain, "BCE_fall": fall,
                        "practical_tolerances": tolerances, "nontrivial_improving_trend": improving,
                        "raw_sign_improving_trend": gain > 0 or fall > 0,
                        "still_improving_flag": late and improving,
                        "conclusive_advantage_permitted_when_either_arm_flagged": False}
    return result


def verify_retained(run, arm, epochs, targets, official_val, contract_sha256):
    rows = list(csv.DictReader((run / "validation_trace.csv").open()))
    if [int(r["epoch"]) for r in rows] != list(range(1, epochs + 1)):
        raise ValueError("Incomplete/unordered amended validation trajectory")
    def epoch_evidence(epoch):
        payload = torch.load(run / "validation_evidence" / f"epoch_{epoch:04d}.pt", map_location="cpu")
        if payload["epoch"] != epoch:
            raise ValueError("Validation epoch binding differs")
        return payload["member_logits"]
    for row in rows:
        value = metrics(epoch_evidence(int(row["epoch"])), targets, arm)
        if any(abs(v-float(row[k])) > 1e-9 for k, v in value.items()):
            raise ValueError("Retained epoch readouts do not replay")
    chosen, reports = selections(run), {}
    for pool in READOUTS:
        best = max(rows, key=lambda r: selection_key({k: float(v) for k, v in r.items()}, int(r["epoch"]), pool))
        checkpoint = torch.load(run / ("selected_" + pool + ".pt"), map_location="cpu")
        report = chosen[pool]
        if checkpoint["epoch"] != int(best["epoch"]) or report["epoch"] != checkpoint["epoch"] or checkpoint["selection_pooling"] != pool:
            raise ValueError("Readout checkpoint/epoch selection differs")
        if checkpoint["contract_sha256"] != contract_sha256:
            raise ValueError("Readout checkpoint belongs to a different contract")
        evidence = torch.load(run / ("selected_validation_member_logits_" + pool + ".pt"), map_location="cpu")
        if (not torch.equal(evidence["targets"], targets) or not torch.equal(evidence["official_graph_indices"], official_val)
                or not torch.equal(evidence["member_logits"], epoch_evidence(checkpoint["epoch"]))):
            raise ValueError("Readout target/order/selected evidence differs")
        value = metrics(evidence["member_logits"], targets, arm)
        if any(abs(v-float(best[k])) > 1e-9 or abs(v-float(report[k])) > 1e-9 or abs(v-checkpoint["validation"][k]) > 1e-9 for k,v in value.items()):
            raise ValueError("Readout reports/checkpoint metrics differ")
        reports[pool] = {"epoch": checkpoint["epoch"], "retained_metrics": value}
    primary = primary_pooling(arm)
    alias = torch.load(run / "selected.pt", map_location="cpu")
    canonical = torch.load(run / ("selected_" + primary + ".pt"), map_location="cpu")
    if json.loads((run / "selected.json").read_text()) != chosen[primary] or alias["selection_pooling"] != primary or alias["epoch"] != canonical["epoch"]:
        raise ValueError("Primary aliases differ")
    if set(alias["state_dict"]) != set(canonical["state_dict"]) or any(not torch.equal(v, alias["state_dict"][k]) for k,v in canonical["state_dict"].items()):
        raise ValueError("Primary alias state differs")
    alias_e = torch.load(run / "selected_validation_member_logits.pt", map_location="cpu")
    named_e = torch.load(run / ("selected_validation_member_logits_" + primary + ".pt"), map_location="cpu")
    if set(alias_e) != set(named_e) or any(not torch.equal(v, named_e[k]) for k,v in alias_e.items()):
        raise ValueError("Primary alias evidence differs")
    return reports
