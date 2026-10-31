"""Sample-weighted epoch logging, separate from validation/checkpoint selection."""
import math


class EpochMeans:
    def __init__(self):
        self.samples = 0
        self.sums = {}
        self.optimizer_updates = 0

    def add(self, metrics, *, samples, optimizer_updates=1):
        if type(samples) is not int or samples < 1:
            raise ValueError("Declare the number of supervised examples in this batch")
        if type(optimizer_updates) is not int or optimizer_updates < 0:
            raise ValueError("Invalid actual optimizer-update count")
        if self.sums and set(metrics) != set(self.sums):
            raise ValueError("Epoch minibatches must log the same objective terms")
        values = {key: float(value) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite minibatch metric")
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.) + samples * value
        self.samples += samples
        self.optimizer_updates += optimizer_updates

    def finish(self):
        if not self.samples:
            raise ValueError("Empty epoch")
        return {**{key: value / self.samples for key, value in self.sums.items()},
                "sample_count": self.samples, "aggregation": "sample_weighted_epoch_mean",
                "operation_counts": {"optimizer_updates": self.optimizer_updates,
                                     "supervised_examples": self.samples}}
