"""Pure epoch/selector bookkeeping; imports no model, dataset or tensor library."""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Policy:
    track: str
    maximum_epochs: int
    minimum_epochs: int
    patience: int | None
    metric: str
    direction: str = "max"
    ties: str = "first"
    evaluate_TRAIN_each_epoch: bool = False

    @classmethod
    def from_dict(cls, value, family):
        result = cls(**value)
        if result.track not in ("source_reference", "matched_extended", "matched_pilot"):
            raise ValueError("Explicit source-reference or matched protocol required")
        if type(result.maximum_epochs) is not int or type(result.minimum_epochs) is not int \
                or not 0 <= result.minimum_epochs <= result.maximum_epochs:
            raise ValueError("Finite positive cap and valid minimum required")
        if result.maximum_epochs < 1 or result.direction not in ("min", "max") or result.ties != "first":
            raise ValueError("Positive cap, metric direction and frozen first-tie selector required")
        if result.patience is not None and (type(result.patience) is not int or result.patience <= 0):
            raise ValueError("Patience must be positive or explicitly disabled")
        if result.track == "source_reference":
            expected = (2000, 0, 250, "accuracy") if family == "node" else (100, 100, None, "rocauc")
            if (result.maximum_epochs, result.minimum_epochs, result.patience, result.metric) != expected:
                raise ValueError("Source horizon/selector must retain its frozen native recipe")
            if family == "ogb" and not result.evaluate_TRAIN_each_epoch:
                raise ValueError("OGB source reference must retain native full TRAIN evaluation/RNG consumption")
        elif result.patience is None or result.minimum_epochs < 1:
            raise ValueError("Matched pilots need a frozen nonzero minimum and moderate stopping patience")
        return result


class Selector:
    def __init__(self, policy):
        self.policy = policy
        self.completed_epoch = 0
        self.best_metric = None
        self.selected_epoch = None
        self.stale = 0

    def observe(self, epoch, metrics):
        if epoch != self.completed_epoch + 1 or epoch > self.policy.maximum_epochs:
            raise ValueError("Selection requires the next completed native epoch")
        metric = float(metrics[self.policy.metric])
        if not math.isfinite(metric):
            raise ValueError("Nonfinite complete-VALID metric")
        improved = self.best_metric is None or (metric > self.best_metric if self.policy.direction == "max"
                                                else metric < self.best_metric)
        if improved:
            self.best_metric, self.selected_epoch, self.stale = metric, epoch, 0
        elif epoch >= self.policy.minimum_epochs:
            self.stale += 1
        else:
            self.stale = 0
        self.completed_epoch = epoch
        patience_stop = self.policy.patience is not None and epoch >= self.policy.minimum_epochs \
            and self.stale >= self.policy.patience
        cap_stop = epoch >= self.policy.maximum_epochs
        cap_reason = "fixed_source_horizon" if self.policy.track == "source_reference" else "cap_without_patience_plateau"
        return {"save_selected": improved, "stop": patience_stop or cap_stop,
                "stop_reason": "patience" if patience_stop else cap_reason if cap_stop else None,
                "completed_epoch": epoch, "selected_epoch": self.selected_epoch,
                "best_metric": self.best_metric, "stale_validation_checks": self.stale,
                "minimum_horizon_reached": epoch >= self.policy.minimum_epochs}

    def state(self):
        return {"completed_epoch": self.completed_epoch, "best_metric": self.best_metric,
                "selected_epoch": self.selected_epoch, "stale": self.stale,
                "policy": dict(self.policy.__dict__)}

    def load_state(self, value):
        if value["policy"] != dict(self.policy.__dict__):
            raise ValueError("Resume cannot change stopping/selection settings")
        self.completed_epoch = int(value["completed_epoch"])
        self.best_metric = value["best_metric"]
        self.selected_epoch = value["selected_epoch"]
        self.stale = int(value["stale"])


def batches_in_epoch(record_count, batch_size):
    if type(record_count) is not int or type(batch_size) is not int or record_count < 1 or batch_size < 1:
        raise ValueError("Positive complete TRAIN size/batch size required")
    return (record_count + batch_size - 1) // batch_size


def epoch_boundary_record(expected_records, visited_positions, raw_batches, optimizer_maps, skipped_batches):
    """Exact integer bookkeeping, including source-skipped singleton tails."""
    if len(visited_positions) != expected_records or set(visited_positions) != set(range(expected_records)):
        raise ValueError("A completed epoch must traverse every native TRAIN record exactly once")
    if raw_batches != optimizer_maps + skipped_batches:
        raise ValueError("Every source batch must be updated or explicitly skipped")
    return {"complete_native_TRAIN": True, "TRAIN_records_traversed": expected_records,
            "raw_batches": raw_batches, "optimizer_maps": optimizer_maps,
            "source_skipped_batches": skipped_batches}
