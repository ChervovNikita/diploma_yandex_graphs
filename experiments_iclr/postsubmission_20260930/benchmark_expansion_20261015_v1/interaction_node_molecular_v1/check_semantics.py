"""Finite tensor checks of the new interaction credit; no native model/data."""
from pathlib import Path
import importlib.util
import json
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    import torch
    root = Path(__file__).resolve().parents[2]
    expansion = root / "benchmark_expansion_20261015_v1"
    rules = load(root / "reference_wrapper_redesign_20261013_v1/runtime/training_rules.py", "_interaction_credit_rules")
    tablelib = load(expansion / "structural_class_responsibility_v1/table.py", "_interaction_credit_table")
    shared = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    private = torch.nn.Parameter(torch.tensor([-.2, .1, .3, -.4], dtype=torch.float64))
    x = torch.arange(1., 9., dtype=torch.float64) / 4
    y = torch.tensor([0.] * 4 + [1.] * 4, dtype=torch.float64)
    z = shared * x[None] + private[:, None] * (x.square()[None] + .3)
    records = (z - y).square()
    table = tablelib.make_table(list(range(8)), [0] * 4 + [1] * 4, list(range(8)), role="TRAIN", permutation_seed=None)
    F, L = records.mean(), (z.mean(0) - y).square().mean()
    W = rules.weighted_native_own_loss(torch, records, table.tensor_weights(torch),
        native_reduce=lambda row: row.mean(), native_regularizer=shared.new_zeros(()))
    expected_shared = torch.autograd.grad(F, shared, retain_graph=True)[0]
    expected_private = torch.autograd.grad(.5 * W + .5 * L, private, retain_graph=True)[0]
    wrong_own_pool = torch.autograd.grad(.5 * F + .5 * L, private, retain_graph=True)[0]
    wrong_weight_only = torch.autograd.grad(W, private, retain_graph=True)[0]
    assert not torch.allclose(expected_private, wrong_own_pool) and not torch.allclose(expected_private, wrong_weight_only)

    class Core:
        _source_parameters = (("shared", shared),)
        def private_parameters(self): return (private,)

    class Recorder:
        param_groups = [{"params": [shared, private]}]
        calls = 0
        def step(self):
            self.calls += 1
            torch.testing.assert_close(shared.grad, expected_shared, atol=0, rtol=0)
            torch.testing.assert_close(private.grad, expected_private, atol=0, rtol=0)

    recorder = Recorder()
    value = rules.step_private_responsibility(torch, Core(), recorder, F=F, F_weighted=.5 * W + .5 * L)
    assert recorder.calls == 1 and value["reverse_collections"] == 2
    residual = load(expansion / "portable_private_input_v2/core.py", "_interaction_residual_semantics")
    c6 = load(expansion / "conditional_aggregation_v1/core.py", "_interaction_readout_semantics")
    reference = torch.empty(8, 8)
    bank = residual.build_residual(torch, reference, width=8, seed=101 + 901003)
    native = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 20
    assert all(torch.equal(bank(native, m), native) for m in range(4))
    logits = [torch.tensor([[.1 + m, -.2 * m], [.3 * m, -.1]], requires_grad=True) for m in range(4)]
    for mode in ("mean", "conditional"):
        gate = c6.make_aggregator(torch, mode=mode, task_kind="multiclass", seed=101, reference=reference)
        assert torch.equal(gate(logits)["served_raw_logits"], torch.stack(logits).mean(0))
    print(json.dumps({"schema": "explicit-interaction-tensor-checks-v1", "status": "passed", "torch": str(torch.__version__),
        "checks": ["I2 shared dF/private d(.5W+.5L) exact with2old-state reverse collections and1recorded transition",
            "I2 credit differs from legacy private half-pool and weight-only alternatives on finite tensors",
            "unchanged portablev2 residual exact-zero addition on all4routes",
            "unchanged frozen mean/conditional readouts have exact mean start"],
        "native_constructions": 0, "task_data_accessed": False, "native_optimizer_steps": 0,
        "recorded_nonmutating_step_calls": 1, "TEST_accessed": False}, indent=2))


if __name__ == "__main__": main()
