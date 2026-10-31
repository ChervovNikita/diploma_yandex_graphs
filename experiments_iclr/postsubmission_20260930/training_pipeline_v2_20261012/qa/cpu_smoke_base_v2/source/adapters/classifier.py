"""Full-batch classification adapter with explicit objective and serving laws.

Torch/F, models, optimizers and role tensors are supplied by the caller. This
module neither loads labels nor chooses a backbone or learning rate.
"""
import copy
import math
import random


class ClassifierAdapter:
    def __init__(self, torch, F, model, optimizer, *, logits_fn, train_ids,
                 train_y, select_ids, select_y, regularizer, fusion, selector,
                 provenance, external_capture=None, external_restore=None,
                 custom_train=None):
        if fusion not in {"mean_probabilities", "mean_logits"}:
            raise ValueError("Bind the method's actual inference law")
        if set(selector) != {"metric", "direction", "ties", "patience_steps", "save_every"}:
            raise ValueError("Bind the complete selector")
        if selector["direction"] not in {"min", "max"} or selector["ties"] not in {"first", "last"}:
            raise ValueError("Unknown selector policy")
        if selector["save_every"] < 1 or selector["patience_steps"] < 1:
            raise ValueError("Positive checkpoint cadence and patience required")
        if (external_capture is None) != (external_restore is None):
            raise ValueError("External state capture and restoration must be paired")
        self.t, self.F, self.model, self.optimizer = torch, F, model, optimizer
        self.logits_fn, self.regularizer, self.fusion = logits_fn, regularizer, fusion
        self.train_ids, self.train_y = train_ids, train_y
        self.select_ids, self.select_y = select_ids, select_y
        self.selector, self.provenance = dict(selector), copy.deepcopy(provenance)
        self.external_capture, self.external_restore = external_capture, external_restore
        self.custom_train = custom_train
        self._best, self._best_step, self._stale = None, None, 0

    def _scalar(self, x):
        value = float(x.detach().item()) if hasattr(x, "detach") else float(x)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite objective or validation metric")
        return value

    def probabilities(self, logits):
        if logits.ndim != 3 or not self.t.isfinite(logits).all().item():
            raise ValueError("Expected finite [views,examples,classes] logits")
        return (logits.softmax(-1).mean(0) if self.fusion == "mean_probabilities"
                else logits.mean(0).softmax(-1))

    def train_step(self, step):
        self.model.train()
        if self.custom_train is not None:
            return self.custom_train(self, step)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.logits_fn(self.model, True, step)
        ce = self.t.stack([self.F.cross_entropy(z[self.train_ids], self.train_y)
                           for z in logits]).mean()
        penalty = self.regularizer(self.model)
        objective = ce + penalty
        result = {"objective": self._scalar(objective), "ce": self._scalar(ce),
                  "regularization": self._scalar(penalty), "operation_counts": {
                      "optimizer_updates": 1, "model_forwards": 1,
                      "route_predictions": int(logits.shape[0]), "backward_calls": 1,
                      "supervised_examples": int(self.train_ids.numel()),
                  }}
        objective.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not self.t.isfinite(parameter.grad).all().item():
                raise FloatingPointError("Nonfinite model gradient")
        self.optimizer.step()
        return result

    def validate(self, step):
        self.model.eval()
        with self.t.no_grad():
            logits = self.logits_fn(self.model, False, step)
            p = self.probabilities(logits)[self.select_ids]
            ce = self.t.stack([self.F.cross_entropy(z[self.select_ids], self.select_y)
                               for z in logits]).mean()
            penalty = self.regularizer(self.model)
            correct = int(p.argmax(-1).eq(self.select_y).sum().item())
            nll = -p.gather(1, self.select_y[:, None]).clamp_min(
                self.t.finfo(p.dtype).tiny).log().mean()
            value = {"objective": self._scalar(ce + penalty), "ce": self._scalar(ce),
                     "regularization": self._scalar(penalty), "correct": correct,
                     "examples": int(self.select_y.numel()),
                     "accuracy": correct / int(self.select_y.numel()),
                     "pooled_nll": self._scalar(nll), "fusion": self.fusion}
            value["selection_metric"] = value[self.selector["metric"]]
            return value

    def observe_validation(self, step, metrics):
        score = self._scalar(metrics["selection_metric"])
        if self._best is None:
            above = True
        elif self.selector["direction"] == "max":
            above = score > self._best if self.selector["ties"] == "first" else score >= self._best
        else:
            above = score < self._best if self.selector["ties"] == "first" else score <= self._best
        # For the legacy boundary model an off-cadence improvement resets
        # patience but does not change the best saved value. Preserve this law.
        previous = getattr(self, "_last_validation_step", step)
        delta = max(1, step - previous)
        self._stale = 0 if above else self._stale + delta
        eligible = step % self.selector["save_every"] == 0
        saved = above and eligible
        if saved:
            self._best, self._best_step = score, step
        self._last_validation_step = step
        self._validation_delta = delta
        return {"save_selected": saved, "stop_requested": self._stale >= self.selector["patience_steps"],
                "best_saved_metric": self._best, "best_saved_step": self._best_step,
                "stale_steps": self._stale, "checkpoint_eligible": eligible}

    def capture_state(self):
        rng = {"torch_cpu": self.t.get_rng_state().clone(), "python": random.getstate(),
               "torch_cuda": [state.clone() for state in self.t.cuda.get_rng_state_all()]
               if self.t.cuda.is_initialized() else []}
        return {"rng": rng, "selector": {"best": self._best, "best_step": self._best_step,
                "stale": self._stale, "last_validation_step": getattr(self, "_last_validation_step", None),
                "validation_delta": getattr(self, "_validation_delta", 1)},
                "provenance": copy.deepcopy(self.provenance),
                "external": self.external_capture() if self.external_capture else {}}

    def restore_state(self, state):
        if state["provenance"] != self.provenance:
            raise ValueError("Selected checkpoint has different model/data ancestry")
        rng, selector = state["rng"], state["selector"]
        self.t.set_rng_state(rng["torch_cpu"])
        random.setstate(rng["python"])
        if rng["torch_cuda"]:
            self.t.cuda.set_rng_state_all(rng["torch_cuda"])
        self._best, self._best_step, self._stale = selector["best"], selector["best_step"], selector["stale"]
        if selector["last_validation_step"] is None:
            self.__dict__.pop("_last_validation_step", None)
        else:
            self._last_validation_step = selector["last_validation_step"]
        self._validation_delta = selector["validation_delta"]
        if self.external_restore:
            self.external_restore(state["external"])
