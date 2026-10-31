"""Two declared readouts of the same retained member logits, with stable replay.

Scoring uses CPU float64. Model/loss/training state and trajectory are unchanged.
Probability NLL is evaluated by log-sum-exp without a clipping hyperparameter.
"""
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

ALL_LAYER_ARM = "all_layer_be_sage_adaptation_prob"
READOUTS = ("raw", "prob")
READOUT_CONTRACT = {
    "training_trajectories_per_arm_seed_budget": 1,
    "reported_readouts": ["mean_member_raw_logits", "mean_member_probabilities"],
    "selection_pooling": {"existing_arms": "raw", ALL_LAYER_ARM: "prob"},
    "secondary_selection": "one checkpoint per readout from the same trajectory; no duplicate fits",
    "matched_pooling_contrasts": "compare raw-selected endpoints against raw-selected endpoints and probability-selected endpoints against probability-selected endpoints",
    "within_checkpoint_pooling_contrast": "both metrics reported at the model-specific primary selected checkpoint",
    "PPI_selection": "max microF1, then min pooled BCE under the named readout, then earliest epoch",
    "MNIST_selection": "max accuracy, then min pooled CE under the named readout, then earliest epoch",
    "metric_precision": "CPU float64 on retained float32 member logits; stable mixture log probabilities; no clipping",
    "ceiling_interpretation": "No conclusive advantage when either contrasted arm has BOTH a selected epoch in final10 AND a nontrivial improving trend under the matched readout. Trend means final10 versus preceding10 quality gain>1e-4 OR pooled loss fall>1e-5. Report late and trend separately, both readouts, and raw-sign sensitivity. Thresholds are fixed equally for all arms before outcomes. Extending the cap needs a prospective matched amendment.",
    "ceiling_practical_tolerances": {"ppi": {"quality_gain": 1e-4, "pooled_loss_fall": 1e-5},
                                     "mnist": {"quality_gain": 1e-4, "pooled_loss_fall": 1e-5}},
}


def primary_pooling(arm):
    return "prob" if arm == ALL_LAYER_ARM else "raw"


def ppi_metrics(member_logits, targets, arm):
    if len(member_logits) != len(targets) or not targets:
        raise ValueError("Require one PPI member-logit tensor per complete validation graph")
    stats = {pool: [0, 0, 0, 0.0] for pool in READOUTS}
    entries = 0
    member_loss = graph_member_loss = 0.0
    for logits, target in zip(member_logits, targets):
        z, y = logits.detach().cpu().double(), target.detach().cpu().double()
        if z.ndim != 3 or z.shape[1:] != y.shape or not torch.isfinite(z).all() or not torch.isfinite(y).all():
            raise ValueError("PPI evidence shape/nonfinite mismatch")
        if not bool(((y == 0) | (y == 1)).all()):
            raise ValueError("PPI validation targets must be binary")
        raw = z.mean(0)
        log_p = F.logsigmoid(z).logsumexp(0) - math.log(z.shape[0])
        log_q = F.logsigmoid(-z).logsumexp(0) - math.log(z.shape[0])
        losses = {"raw": float(F.binary_cross_entropy_with_logits(raw, y, reduction="sum")),
                  "prob": float(-(y * log_p + (1 - y) * log_q).sum())}
        guesses = {"raw": raw >= 0, "prob": log_p >= -math.log(2)}
        truth = y.bool()
        for pool in READOUTS:
            pred = guesses[pool]
            stats[pool][0] += int((truth & pred).sum())
            stats[pool][1] += int((~truth & pred).sum())
            stats[pool][2] += int((truth & ~pred).sum())
            stats[pool][3] += losses[pool]
        risk = float(F.binary_cross_entropy_with_logits(z, y[None].expand_as(z), reduction="sum")) / z.shape[0]
        member_loss += risk
        graph_member_loss += risk / y.numel()
        entries += y.numel()
    values = {"mean_member_bce": member_loss / entries,
              "graph_mean_member_bce": graph_member_loss / len(targets)}
    for pool, (tp, fp, fn, loss) in stats.items():
        values[pool + "_micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        values[pool + "_pooled_bce"] = loss / entries
    primary = primary_pooling(arm)
    values.update(micro_f1=values[primary + "_micro_f1"], pooled_bce=values[primary + "_pooled_bce"])
    return values


def mnist_metrics(member_logits, targets, arm):
    z, y = member_logits.detach().cpu().double(), targets.detach().cpu().long()
    if z.ndim != 3 or y.shape != (z.shape[1],) or not torch.isfinite(z).all():
        raise ValueError("MNIST evidence shape/nonfinite mismatch")
    raw = z.mean(0)
    log_p = z.log_softmax(-1).logsumexp(0) - math.log(z.shape[0])
    values = {"raw_accuracy": float((raw.argmax(-1) == y).double().mean()),
              "raw_pooled_ce": float(F.cross_entropy(raw, y)),
              "prob_accuracy": float((log_p.argmax(-1) == y).double().mean()),
              "prob_pooled_ce": float(F.nll_loss(log_p, y)),
              "mean_member_ce": float(F.cross_entropy(z.reshape(-1, z.shape[-1]), y[None].expand(z.shape[0], -1).reshape(-1))),
              "mean_member_accuracy": float((z.argmax(-1) == y[None]).double().mean())}
    primary = primary_pooling(arm)
    values.update(accuracy=values[primary + "_accuracy"], pooled_ce=values[primary + "_pooled_ce"])
    return values


def selection_key(metrics, epoch, task, pooling):
    quality, loss = ("micro_f1", "pooled_bce") if task == "ppi" else ("accuracy", "pooled_ce")
    quality, loss = metrics[pooling + "_" + quality], metrics[pooling + "_" + loss]
    if not math.isfinite(quality) or not math.isfinite(loss):
        raise ValueError("Nonfinite validation selection metric")
    return quality, -loss, -epoch


def update_selected(output, best, row, checkpoint, evidence, task, arm):
    """Save both per-readout winners from this one trajectory; primary aliases explicit."""
    for pooling in READOUTS:
        key = selection_key(row, row["epoch"], task, pooling)
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


def ceiling_diagnostics(output, task, epochs):
    rows = list(csv.DictReader((output / "validation_trace.csv").open()))
    if len(rows) != epochs or epochs < 20:
        raise ValueError("Ceiling diagnostics require the complete registered trajectory")
    reports = selections(output)
    loss = "pooled_bce" if task == "ppi" else "pooled_ce"
    quality = "micro_f1" if task == "ppi" else "accuracy"
    tolerances = READOUT_CONTRACT["ceiling_practical_tolerances"][task]
    result = {}
    for pooling in READOUTS:
        previous = sum(float(row[pooling + "_" + loss]) for row in rows[-20:-10]) / 10
        final = sum(float(row[pooling + "_" + loss]) for row in rows[-10:]) / 10
        previous_quality = sum(float(row[pooling + "_" + quality]) for row in rows[-20:-10]) / 10
        final_quality = sum(float(row[pooling + "_" + quality]) for row in rows[-10:]) / 10
        quality_gain, loss_fall = final_quality - previous_quality, previous - final
        late = reports[pooling]["epoch"] >= epochs - 9
        improving = quality_gain > tolerances["quality_gain"] or loss_fall > tolerances["pooled_loss_fall"]
        result[pooling] = {"selected_epoch": reports[pooling]["epoch"], "selected_in_final10": late,
                          "preceding10_mean_pooled_loss": previous, "final10_mean_pooled_loss": final,
                          "preceding10_mean_quality": previous_quality, "final10_mean_quality": final_quality,
                          "quality_gain": quality_gain, "pooled_loss_fall": loss_fall,
                          "practical_tolerances": tolerances, "nontrivial_improving_trend": improving,
                          "raw_sign_improving_trend": quality_gain > 0 or loss_fall > 0,
                          "late_and_raw_sign_sensitivity_flag": late and (quality_gain > 0 or loss_fall > 0),
                          "still_improving_flag": late and improving,
                          "conclusive_advantage_permitted_when_either_arm_flagged": False}
    return result


def verify_retained(run, task, arm, epochs, targets, contract_sha256):
    """Numerically replay every epoch, both selection rules, and retained winners."""
    rows = list(csv.DictReader((run / "validation_trace.csv").open()))
    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError("Incomplete/unordered amended validation trajectory")
    metrics_fn = ppi_metrics if task == "ppi" else mnist_metrics
    def epoch_evidence(epoch):
        if task == "ppi":
            return torch.load(run / "validation_epochs" / f"epoch{epoch:03d}.pt", map_location="cpu")
        value = torch.load(run / "validation_evidence" / f"epoch_{epoch:04d}.pt", map_location="cpu")
        if value["epoch"] != epoch:
            raise ValueError("Validation epoch binding differs")
        return value["member_logits"]
    for row in rows:
        value = metrics_fn(epoch_evidence(int(row["epoch"])), targets, arm)
        if any(abs(v - float(row[k])) > 1e-9 for k, v in value.items()):
            raise ValueError("Raw/probability epoch metrics do not replay")
    chosen = selections(run)
    results = {}
    for pooling in READOUTS:
        best = max(rows, key=lambda row: selection_key({k: float(v) for k, v in row.items()}, int(row["epoch"]), task, pooling))
        checkpoint = torch.load(run / ("selected_" + pooling + ".pt"), map_location="cpu")
        report = chosen[pooling]
        if checkpoint["epoch"] != int(best["epoch"]) or report["epoch"] != checkpoint["epoch"] or checkpoint["selection_pooling"] != pooling:
            raise ValueError("Readout checkpoint/epoch selection differs")
        if checkpoint["contract_sha256"] != contract_sha256:
            raise ValueError("Readout checkpoint belongs to another contract")
        retained = torch.load(run / ("selected_validation_member_logits_" + pooling + ".pt"), map_location="cpu")
        if task == "mnist":
            if not torch.equal(retained["targets"], targets):
                raise ValueError("Readout target identity differs")
            retained = retained["member_logits"]
        exact = epoch_evidence(checkpoint["epoch"])
        same = len(retained) == len(exact) and all(torch.equal(left, right) for left, right in zip(retained, exact)) if task == "ppi" else torch.equal(retained, exact)
        if not same:
            raise ValueError("Readout selected logits differ from epoch evidence")
        value = metrics_fn(retained, targets, arm)
        if any(abs(v - float(best[k])) > 1e-9 or abs(v - float(report[k])) > 1e-9 or abs(v - checkpoint["validation"][k]) > 1e-9 for k, v in value.items()):
            raise ValueError("Readout selected report/checkpoint metrics differ")
        results[pooling] = {"epoch": checkpoint["epoch"], "retained_metrics": value}
    primary = primary_pooling(arm)
    primary_report = json.loads((run / "selected.json").read_text())
    primary_checkpoint = torch.load(run / "selected.pt", map_location="cpu")
    if primary_report != chosen[primary] or primary_checkpoint["selection_pooling"] != primary or primary_checkpoint["epoch"] != chosen[primary]["epoch"]:
        raise ValueError("Primary model-specific selection alias differs")
    canonical = torch.load(run / ("selected_" + primary + ".pt"), map_location="cpu")
    if set(canonical["state_dict"]) != set(primary_checkpoint["state_dict"]) or any(not torch.equal(v, primary_checkpoint["state_dict"][k]) for k, v in canonical["state_dict"].items()):
        raise ValueError("Primary state differs from its named readout checkpoint")
    primary_evidence = torch.load(run / "selected_validation_member_logits.pt", map_location="cpu")
    named_evidence = torch.load(run / ("selected_validation_member_logits_" + primary + ".pt"), map_location="cpu")
    if task == "ppi":
        identical = len(primary_evidence) == len(named_evidence) and all(torch.equal(a, b) for a, b in zip(primary_evidence, named_evidence))
    else:
        identical = set(primary_evidence) == set(named_evidence) and all(torch.equal(v, named_evidence[k]) for k, v in primary_evidence.items())
    if not identical:
        raise ValueError("Primary evidence differs from its named readout evidence")
    return results
