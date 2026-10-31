"""Runnable synthetic CPU/CUDA mechanics gate; never complete-data admission.

Run only after independent authorization for the selected runtime/device.
This author has NOT run it locally or on a remote allocation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .contract import ROOT, SCHEMA, check_pins, check_seal, confined_output
from .csr_operator import load_fixture_family
from .csr_prepare import prepare_fixture
from .dropout import DropoutKey, training_colors
from .fixture_engine import make_fixture_sources, own_member_ce, serial_own_ce_step
from .models import centered_aggregation, source_parameter_counts


def parameter_pairs(source, destination, ensemble):
    pairs = []
    if ensemble:
        for source_name, destination_prefix in (("input_be_block", "input"), ("output_be_block", "output")):
            block = getattr(source, source_name)
            for factor in ("R", "S", "B"):
                pairs.append((source_name + "." + factor, getattr(block, factor),
                              getattr(destination, destination_prefix + "_" + factor.lower()), False))
            pairs.append((source_name + ".W.weight", block.W.weight,
                          getattr(destination, destination_prefix + "_weight"), True))
    else:
        for source_name, destination_name in (("input_linear", "input"), ("output_linear", "output")):
            for field in ("weight", "bias"):
                pairs.append((source_name + "." + field, getattr(getattr(source, source_name), field),
                              getattr(getattr(destination, destination_name), field), False))
    for layer, residual in enumerate(source.residual_modules):
        parts = [(residual.normalization, destination.norm[layer], "norm"),
                 (residual.module.conv.lin_l, destination.blocks[layer].neighbor, "neighbor"),
                 (residual.module.conv.lin_r, destination.blocks[layer].root, "root"),
                 (residual.module.feed_forward_module.linear_1, destination.blocks[layer].first, "first"),
                 (residual.module.feed_forward_module.linear_2, destination.blocks[layer].second, "second")]
        for original, optimized, name in parts:
            for field in ("weight", "bias"):
                value = getattr(original, field, None)
                if value is not None:
                    pairs.append((f"layer{layer}.{name}.{field}", value,
                                  getattr(optimized, field), False))
    for field in ("weight", "bias"):
        pairs.append(("output_norm." + field, getattr(source.output_normalization, field),
                      getattr(destination.output_norm, field), False))
    if {id(s) for _, s, _, _ in pairs} != {id(p) for p in source.parameters()}:
        raise AssertionError("Source parameter map incomplete")
    if {id(d) for _, _, d, _ in pairs} != {id(p) for p in destination.parameters()}:
        raise AssertionError("Destination parameter map incomplete")
    return pairs


def run(output, device):
    output = confined_output(output)
    if output.exists():
        raise FileExistsError("Do not overwrite runtime evidence")
    source_manifest_sha256 = check_seal()
    check_pins()
    output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(991)
    source = np.array([1, 1, 2, 0, 2, 3, 3, 4, 5, 2, 4, 5], dtype=np.int64)
    receiver = np.array([0, 0, 0, 1, 2, 2, 2, 2, 4, 4, 4, 5], dtype=np.int64)
    prepare_fixture(source, receiver, 6, output / "csr", cache_half_unions=True)
    family = load_fixture_family(output / "csr/FIXTURE_CSR_MANIFEST.json", device=device,
                                 include_half_unions=True)
    checks = []

    def close(name, left, right, atol=5e-5, rtol=5e-5):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        checks.append({"name": name, "passed": True,
                       "max_absolute_error": float((left - right).detach().abs().max().cpu()),
                       "atol": atol, "rtol": rtol})

    def all_gradients(name, left, right):
        lparams, rparams = dict(left.named_parameters()), dict(right.named_parameters())
        if lparams.keys() != rparams.keys():
            raise AssertionError("Gradient parameter names changed")
        for key in lparams:
            if lparams[key].grad is None or rparams[key].grad is None:
                raise AssertionError(f"Missing gradient {key}")
            close(name + ":" + key, lparams[key].grad, rparams[key].grad)

    # Coalesced colored matrices AND explicit transposes agree with full P.
    full = family.full.forward_csr.to_dense()
    close("color partition matrix", sum(family.color(c).forward_csr.to_dense() for c in range(4)), full)
    for name, op in family.operators.items():
        close("explicit adjoint:" + name, op.adjoint_csr.to_dense(), op.forward_csr.to_dense().T)
    close("duplicate weight 2/3", full[0, 1], torch.tensor(2 / 3, device=device))
    close("isolate zero", full[3], torch.zeros(6, device=device))
    for c in range(4):
        close("cached half union:" + str(c), family.operators[f"half{c}"].forward_csr.to_dense(),
              family.color(c).forward_csr.to_dense() + family.color((c + 1) % 4).forward_csr.to_dense())

    z = torch.randn(6, 4, 3, device=device, requires_grad=True)
    upstream = torch.randn_like(z)
    fixed_mean = family.full(z.mean(dim=1))
    color_aggregates = []
    for c in range(4):
        z.grad = None
        aggregates, trace = centered_aggregation(family, z, c, trace=True)
        close("linear shared-mask mean:" + str(c), aggregates.mean(dim=1), fixed_mean)
        (aggregates * upstream).sum().backward()
        q = 4.0 * family.color(c).forward_csr.to_dense()
        expected = torch.stack([q.T @ upstream[:, member, :]
                               + (full - q).T @ upstream.mean(dim=1)
                               for member in range(4)], dim=1)
        close("differentiable mean adjoint:" + str(c), z.grad, expected)
        color_aggregates.append(aggregates.detach())
        for key in ("centered_sum_normalized_norm", "aggregation_mean_normalized_error"):
            if not torch.isfinite(trace[key]) or trace[key] > 5e-5:
                raise AssertionError("Centered aggregation diagnostic failed")
    close("all-color conditional unbiased aggregate", torch.stack(color_aggregates).mean(dim=0),
          family.full(z.detach().flatten(1)).reshape_as(z))

    features = torch.randn(6, 3, device=device)
    targets = torch.tensor([0, 1, 0, 1, 1, 0], device=device)
    train_nodes = torch.tensor([0, 1, 2, 4], device=device)
    graph = SimpleNamespace(edge_index=torch.as_tensor(np.stack((source, receiver)), device=device))
    source_ensemble, source_single, arms = make_fixture_sources()
    source_ensemble, source_single = source_ensemble.to(device), source_single.to(device)
    arms = {name: model.to(device) for name, model in arms.items()}
    source_ensemble.eval()
    source_single.eval()
    exact = arms["exact_member"].eval()
    single = arms["single_base"].eval()
    source_logits = torch.stack([source_ensemble(graph, features, member) for member in range(4)])
    packed_logits = exact(family, features)
    close("frozen source TABM logits", source_logits, packed_logits)
    close("frozen source ordinary logits", source_single(graph, features), single(family, features)[0])
    own_member_ce(source_logits, targets, train_nodes).backward()
    own_member_ce(packed_logits, targets, train_nodes).backward()
    for name, original, optimized, transposed in parameter_pairs(source_ensemble, exact, True):
        close("frozen source TABM gradients:" + name, original.grad.T if transposed else original.grad, optimized.grad)
    source_single.zero_grad(set_to_none=True)
    single.zero_grad(set_to_none=True)
    F.cross_entropy(source_single(graph, features)[train_nodes], targets[train_nodes]).backward()
    own_member_ce(single(family, features), targets, train_nodes).backward()
    for name, original, optimized, _ in parameter_pairs(source_single, single, False):
        close("frozen source ordinary gradients:" + name, original.grad, optimized.grad)
    for arm, count in source_parameter_counts(3, 2, 4).items():
        actual = sum(parameter.numel() for parameter in arms[arm].parameters())
        if actual != count:
            raise AssertionError(f"Parameter count changed for {arm}: {actual} vs {count}")
    checks.append({"name": "source parameter counts and transfer", "passed": True})

    # Training-noise equality, then independent one-step AdamW controls.
    _, _, noisy_arms = make_fixture_sources(dropout=0.2)
    noisy_arms = {name: model.to(device).train() for name, model in noisy_arms.items()}
    key, colors = DropoutKey(0, 0, 7), (1, 3)
    for arm in ("exact_member", "naive_shared_sample"):
        packed = noisy_arms[arm]
        serial = copy.deepcopy(packed)
        packed.zero_grad(set_to_none=True)
        serial.zero_grad(set_to_none=True)
        lp = packed(family, features, colors=colors, dropout_key=key)
        ls = serial(family, features, colors=colors, dropout_key=key, execution="serial")
        close("paired dropout packed/serial:" + arm, lp, ls)
        own_member_ce(lp, targets, train_nodes).backward()
        own_member_ce(ls, targets, train_nodes).backward()
        all_gradients("paired dropout packed/serial gradients:" + arm, packed, serial)
        packed.zero_grad(set_to_none=True)
        serial.zero_grad(set_to_none=True)
        op = torch.optim.AdamW(packed.parameters(), lr=0.001, weight_decay=0.0001)
        os = torch.optim.AdamW(serial.parameters(), lr=0.001, weight_decay=0.0001)
        loss = own_member_ce(packed(family, features, colors=colors, dropout_key=key), targets, train_nodes)
        loss.backward()
        op.step()
        serial_loss = serial_own_ce_step(serial, family, features, targets, train_nodes, os,
                                          colors=colors, dropout_key=key)
        close("serial one-step own CE:" + arm, loss.detach(), serial_loss)
        for name, param in packed.named_parameters():
            close("serial one-step AdamW:" + arm + ":" + name, param, dict(serial.named_parameters())[name])
        if any(float(state["step"].cpu()) != 1.0 for state in os.state.values()):
            raise AssertionError("Serial control made more than one AdamW step")

    lazy = noisy_arms["naive_shared_sample"]
    cached = copy.deepcopy(lazy)
    cached.naive_path = "cached_half"
    lazy.zero_grad(set_to_none=True)
    cached.zero_grad(set_to_none=True)
    lazy_logits = lazy(family, features, colors=colors, dropout_key=key)
    cached_logits = cached(family, features, colors=colors, dropout_key=key)
    close("lazy/cached naive logits", lazy_logits, cached_logits)
    own_member_ce(lazy_logits, targets, train_nodes).backward()
    own_member_ce(cached_logits, targets, train_nodes).backward()
    all_gradients("lazy/cached naive gradients", lazy, cached)

    for arm in ("candidate", "exact_member", "naive_shared_sample", "single_base"):
        plain = noisy_arms[arm]
        recomputed = copy.deepcopy(plain)
        recomputed.checkpoint_blocks = True
        plain.zero_grad(set_to_none=True)
        recomputed.zero_grad(set_to_none=True)
        lp = plain(family, features, colors=colors, dropout_key=key)
        lr = recomputed(family, features, colors=colors, dropout_key=key)
        close("whole-block checkpoint logits:" + arm, lp, lr)
        own_member_ce(lp, targets, train_nodes).backward()
        own_member_ce(lr, targets, train_nodes).backward()
        all_gradients("whole-block checkpoint gradients:" + arm, plain, recomputed)
    if not any(row["scope"] == "checkpoint_recompute" for row in family.ledger.sparse):
        raise AssertionError("Sparse recomputations were not counted")

    color_lists = {f"task{task}_seed{seed}": training_colors(task, seed)[:3]
                   for task in (0, 1) for seed in (0, 1, 2)}
    cuda = None
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        cuda = {"device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "unavailable")),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    result = {
        "schema": SCHEMA, "kind": "bounded_synthetic_runtime_fixture",
        "source_manifest_sha256": source_manifest_sha256,
        "status": "passed", "device": device, "cuda": cuda,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "dtype": "float32",
        "TF32_enabled": False, "AMP_enabled": False, "checks": checks,
        "color_stream_first_three_epochs": color_lists,
        "operation_summary": family.ledger.summary(),
        "graph_storage": family.storage_report(),
        "dataset_accessed": False, "predictive_training_run": False,
        "synthetic_optimizer_steps_only": True,
        "complete_graph_resource_feasibility_measured": False,
        "latency_measured": False, "promotion_admitted": False,
    }
    (output / "RUNTIME_FIXTURE_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "OPERATION_LEDGER.json").write_text(json.dumps(
        {"sparse": family.ledger.sparse, "dense": family.ledger.dense}, indent=2) + "\n")
    return {"status": "passed", "checks": len(checks), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
