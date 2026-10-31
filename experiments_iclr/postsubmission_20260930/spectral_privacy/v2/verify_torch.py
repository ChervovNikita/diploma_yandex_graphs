"""Synthetic CPU torch/PyG verifier; never loads a dataset or uses a GPU.

Run on the coordinator's existing torch/PyG server environment. Successful
execution, rather than local syntax checking, is the implementation gate.
"""

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

if __package__:
    from .adapters import SourceSAGEEnsemble
    from .collector import (GradientUnit, PreservedProbe, collect_record,
                            environment_contract, observed_bce, partition_manifest)
    from .contracts import (json_hash, pack_float64, save_new_json, seal_record,
                            state_hash, unpack_float64)
    from .replay import replay_record
    from .spectral_core import (choose_bases, output_statistics, random_matched_bases,
                                raw_gradient_matched_bases)
else:
    from adapters import SourceSAGEEnsemble
    from collector import (GradientUnit, PreservedProbe, collect_record,
                           environment_contract, observed_bce, partition_manifest)
    from contracts import (json_hash, pack_float64, save_new_json, seal_record,
                           state_hash, unpack_float64)
    from replay import replay_record
    from spectral_core import (choose_bases, output_statistics, random_matched_bases,
                               raw_gradient_matched_bases)


HERE = Path(__file__).resolve().parent


def must_fail(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Expected validation failure")


def maximum_error(left, right):
    return float((left - right).abs().max()) if left.numel() else 0.0


def load_source(path):
    module_spec = importlib.util.spec_from_file_location("frozen_source_models_for_spectral_v2", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def new_source(module):
    return module.TABMModel(model_name="SAGE", num_layers=3, input_dim=5,
                            hidden_dim=128, output_dim=3, hidden_dim_multiplier=1,
                            num_heads=1, normalization="LayerNorm", dropout=0.2,
                            tabm_inits=4, device="cpu").double()


def toy_units():
    generator = torch.Generator(device="cpu").manual_seed(902)
    units = []
    for index in range(20):
        n = 4 + index % 3
        edges = []
        for node in range(n - 1):
            edges.extend([(node, node + 1), (node + 1, node)])
        graph = Data(edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(), num_nodes=n)
        features = torch.randn(n, 5, dtype=torch.float64, generator=generator)
        targets = (torch.rand(n, 3, generator=generator) > 0.5).double()
        targets[0, 0] = float("nan")
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:max(1, n // 2)] = True
        units.append(GradientUnit(f"synthetic-train-{index:02d}", graph, features, targets, mask))
    return units


def check_covariance_and_controls():
    generator = torch.Generator(device="cpu").manual_seed(903)
    gradients = torch.randn(4, 4, 3, 2, dtype=torch.float64, generator=generator)
    stats = output_statistics(gradients, n0=1)
    centered = gradients - gradients.mean(dim=1, keepdim=True)
    explicit = torch.zeros(3, 3, dtype=torch.float64)
    for member in range(4):
        for first in range(4):
            for second in range(4):
                if first != second:
                    explicit += centered[first, member] @ centered[second, member].T
    explicit /= 4 * 4 * 3
    ustat_error = maximum_error(explicit, stats["S"])
    assert stats["T"].shape == (3, 3) and ustat_error < 1e-12

    diagonal = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    member_pair = torch.stack([diagonal, -diagonal])
    rank_example = output_statistics(torch.stack([member_pair, member_pair]))
    member_rank = int(torch.linalg.matrix_rank(member_pair.flatten(1)))
    feature_rank = int(torch.linalg.matrix_rank(rank_example["S"]))
    assert member_rank == 1 and feature_rank == 3
    assert torch.equal(torch.linalg.eigvalsh(rank_example["S"]), torch.tensor([1., 4., 9.], dtype=torch.float64))

    paired = []
    for noisy_feature in (10.0, 1.0):
        diagonal = torch.diag(torch.tensor([1.0, noisy_feature, 0.0], dtype=torch.float64))
        paired.append(torch.stack([diagonal, -diagonal]))
    noisy = output_statistics(torch.stack(paired), n0=2)
    assert torch.equal(torch.linalg.eigvalsh(noisy["S"]), torch.tensor([0., 1., 10.], dtype=torch.float64))
    assert torch.equal(torch.linalg.eigvalsh(noisy["T"]), torch.tensor([-10.25, 0., 1.], dtype=torch.float64))

    # Signal dominates a small graph-unit perturbation; at least one selected direction.
    truth = torch.randn(4, 8, 8, dtype=torch.float64, generator=generator)
    discovery = {"matrix": truth.unsqueeze(0) + 0.01 * torch.randn(4, 4, 8, 8, dtype=torch.float64, generator=generator)}
    screen = {"matrix": truth.unsqueeze(0) + 0.01 * torch.randn(4, 4, 8, 8, dtype=torch.float64, generator=generator)}
    bases, details, _ = choose_bases(discovery, screen, rank_max=3)
    ranks = {key: value.shape[1] for key, value in bases.items()}
    assert ranks["matrix"] > 0
    rng = torch.get_rng_state().clone()
    random_bases = random_matched_bases({"matrix": (8, 8)}, ranks, seed=904)
    assert torch.equal(rng, torch.get_rng_state())
    raw_bases = raw_gradient_matched_bases(discovery, ranks)
    for group in (bases, random_bases, raw_bases):
        u = group["matrix"]
        assert u.shape == (8, ranks["matrix"])
        assert torch.allclose(u.T @ u, torch.eye(ranks["matrix"], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    zero = {"matrix": torch.zeros(4, 4, 8, 8, dtype=torch.float64)}
    zero_bases, _, _ = choose_bases(zero, zero)
    assert zero_bases["matrix"].shape == (8, 0)
    assert unpack_float64(pack_float64(zero_bases["matrix"])).shape == (8, 0)
    must_fail(lambda: output_statistics(gradients[:1]))
    return {"U_statistic_vs_explicit_pairs_error": ustat_error,
            "member_contrast_rank": member_rank, "output_feature_rank": feature_rank,
            "matched_control_rank": ranks["matrix"], "zero_demand_rank": 0,
            "signed_noise_example_T_eigenvalues": [-10.25, 0., 1.]}


def check_source_identity_and_native_gradient(source, unit):
    original_hash = state_hash(source)
    keys = [f"residual_modules.{layer}.module.conv.lin_l.weight" for layer in range(3)]
    bases = random_matched_bases({key: (128, 128) for key in keys}, {key: 3 for key in keys}, seed=905)
    construction_rng = torch.get_rng_state().clone()
    model = SourceSAGEEnsemble(source, bases)
    assert torch.equal(construction_rng, torch.get_rng_state()) and state_hash(source) == original_hash
    assert model.is_tied_at_state()
    assert torch.allclose(model.C.T @ model.C, torch.eye(3, dtype=torch.float64), atol=1e-14, rtol=1e-14)
    assert torch.allclose(model.C.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    source.eval()
    model.eval()
    for member in range(4):
        expected = source(unit.graph, unit.features, tabm_seed=member)
        actual = model.forward_member(unit.graph, unit.features, member)
        assert torch.equal(expected, actual)
        assert all(adapter.current_member is None and adapter.current_contrasts is None
                   for adapter in model._neighbors.values())
    # Same dropout stream: the added path consumes no RNG.
    source.train()
    model.train()
    training_rng = torch.get_rng_state().clone()
    expected = source(unit.graph, unit.features, tabm_seed=2)
    after_source_rng = torch.get_rng_state().clone()
    torch.set_rng_state(training_rng)
    actual = model.forward_member(unit.graph, unit.features, 2)
    assert torch.equal(expected, actual) and torch.equal(after_source_rng, torch.get_rng_state())
    source.eval()
    model.eval()

    weights = model.named_neighbor_matrices()
    coefficients = model.named_private_coefficients()
    own_gradients, losses = {key: [] for key in keys}, []
    for member in range(4):
        loss, _ = observed_bce(model.forward_member(unit.graph, unit.features, member), unit)
        losses.append(loss)
        gradients = torch.autograd.grad(loss, tuple(weights[key] for key in keys), retain_graph=True)
        for key, gradient in zip(keys, gradients):
            own_gradients[key].append(gradient.detach())
    mean_loss = torch.stack(losses).mean()
    coefficient_grads = torch.autograd.grad(mean_loss, tuple(coefficients[key] for key in keys), retain_graph=True)
    shared_grads = torch.autograd.grad(mean_loss, tuple(weights[key] for key in keys))
    native_gradient_error = shared_gradient_error = correction_error = 0.0
    eta = 0.017
    for key, coefficient_gradient, shared_gradient in zip(keys, coefficient_grads, shared_grads):
        gradients = torch.stack(own_gradients[key])
        u = model.frozen_bases()[key]
        expected_coefficient = torch.einsum("mq,or,moi->qri", model.C, u, gradients) / 4
        native_gradient_error = max(native_gradient_error, maximum_error(coefficient_gradient, expected_coefficient))
        shared_gradient_error = max(shared_gradient_error, maximum_error(shared_gradient, gradients.mean(dim=0)))
        with torch.no_grad():
            coefficients[key].add_(coefficient_gradient, alpha=-eta)
        centered = gradients - gradients.mean(dim=0, keepdim=True)
        expected_delta = -eta / 4 * torch.einsum("op,mpi->moi", u @ u.T, centered)
        correction_error = max(correction_error, maximum_error(model.private_corrections()[key], expected_delta))
        assert torch.allclose(model.private_corrections()[key].sum(dim=0), torch.zeros(128, 128, dtype=torch.float64),
                              atol=1e-12, rtol=0)
    assert native_gradient_error < 1e-11 and shared_gradient_error < 1e-11 and correction_error < 1e-11
    for key, adapter in model._neighbors.items():
        layer = int(key.split(".")[1])
        original = source.residual_modules[layer].module.conv.lin_l
        assert torch.equal(adapter.base.weight, original.weight)
        assert torch.equal(adapter.base.bias, original.bias)
        assert not any("bias" in name for name, _ in adapter.named_parameters() if not name.startswith("base."))
    with torch.no_grad():
        for value in coefficients.values():
            value.zero_()
    assert model.is_tied_at_state()
    exported = model.tied_source_copy()
    assert state_hash(exported) == state_hash(source)
    for member in range(4):
        assert torch.equal(exported(unit.graph, unit.features, tabm_seed=member),
                           model.forward_member(unit.graph, unit.features, member))
    must_fail(lambda: SourceSAGEEnsemble(source, {key: torch.ones(128, 3, dtype=torch.float64) for key in keys}))
    return {"zero_function_identity": "exact_eval_and_shared_dropout_stream",
            "source_clone_preserved": True,
            "native_coefficient_gradient_error": native_gradient_error,
            "shared_gradient_error": shared_gradient_error,
            "native_eta_over_K_correction_error": correction_error,
            "added_storage_rank3": model.added_storage()}


def check_collector_and_replay(source, units, source_sha256):
    ids = [unit.unit_id for unit in units]
    manifest = partition_manifest(ids[:10], ids[10:14], ids[14:18], ids[18:])
    supplied = units[10:18]  # P/F labels never enter collector.
    probe = SourceSAGEEnsemble(source)
    probe.train()
    probe.source.residual_modules[0].normalization.eval()
    for parameter in probe.parameters():
        parameter.grad = torch.full_like(parameter, 0.123)
    before = PreservedProbe(probe)
    record = collect_record(probe, supplied, manifest, source_sha256=source_sha256,
                            dataset_identity={"study_id": "synthetic-spectral-v2-cpu", "data": "generated_only"},
                            optimizer_seed=906)
    assert before.matches()
    assert all(module.training == mode for module, mode in before.modes)
    assert set(row["unit_id"] for row in record["input_fingerprints"]) == set(ids[10:18])
    # JSON round trip must retain raw IEEE float64 bytes and canonical hashes.
    round_trip = json.loads(json.dumps(record, allow_nan=False))
    replayed = replay_record(round_trip)
    assert replayed["status"] == "EXACT_REPLAY_PASS"

    key = sorted(record["targets"])[0]
    tensor_tamper = copy.deepcopy(record)
    payload = tensor_tamper["groups"]["discovery"]["matrix_gradients"][key]
    raw = bytearray(base64.b64decode(payload["data"]))
    raw[0] ^= 1
    payload["data"] = base64.b64encode(raw).decode("ascii")
    tensor_tamper = seal_record(tensor_tamper)
    must_fail(lambda: replay_record(tensor_tamper))
    choice_tamper = copy.deepcopy(record)
    choice_tamper["selection"]["ranks"][key] += 1
    choice_tamper = seal_record(choice_tamper)
    must_fail(lambda: replay_record(choice_tamper))
    partition_tamper = copy.deepcopy(record)
    partition_tamper["partition"]["discovery"][0] = partition_tamper["partition"]["screen"][0]
    partition_payload = {k: v for k, v in partition_tamper["partition"].items() if k != "sha256"}
    partition_tamper["partition"]["sha256"] = json_hash(partition_payload)
    partition_tamper = seal_record(partition_tamper)
    must_fail(lambda: replay_record(partition_tamper))
    objective_tamper = copy.deepcopy(record)
    objective_tamper["gradient_scale"] = "member_loss_divided_by_K"
    objective_tamper = seal_record(objective_tamper)
    must_fail(lambda: replay_record(objective_tamper))
    unsealed = copy.deepcopy(record)
    unsealed["optimizer_seed"] += 1
    must_fail(lambda: replay_record(unsealed))

    # State/RNG must also be restored when a forward violates the contract.
    def consume_rng(module, inputs, output):
        torch.rand(1)
        random.random()
        return None
    handle = probe.source.input_be_block.register_forward_hook(consume_rng)
    failure_before = PreservedProbe(probe)
    try:
        must_fail(lambda: collect_record(probe, supplied, manifest, source_sha256=source_sha256,
                                         dataset_identity={"study_id": "synthetic-failure-check"},
                                         optimizer_seed=907))
    finally:
        handle.remove()
    assert failure_before.matches()
    assert all(module.training == mode for module, mode in failure_before.modes)
    bad_units = list(supplied)
    old = bad_units[0]
    bad_units[0] = GradientUnit(old.unit_id, old.graph, old.features, old.targets, old.observed_mask, split="validation")
    must_fail(lambda: collect_record(probe, bad_units, manifest, source_sha256=source_sha256,
                                     dataset_identity={"study_id": "synthetic-heldout-rejection"}, optimizer_seed=908))
    return {"record": record, "summary": {"collector_preserves_state_grad_objects_modes_rng": True,
             "failure_restoration": True, "fit_and_audit_labels_excluded": True,
             "exact_replay": replayed, "tamper_rejections": 5,
             "heldout_unit_rejected": True}}


def preserved_v1():
    manifest = json.loads((HERE / "V1_PRESERVED_MANIFEST.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        path = HERE.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError("Completed v1 evidence changed: " + name)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models", type=Path,
                        default=HERE.parents[1] / "source_snapshot" / "models.py")
    parser.add_argument("--output", type=Path, default=HERE / "CPU_VERIFICATION.json")
    parser.add_argument("--record-output", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=HERE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(901)
    random.seed(901)
    source_path = args.source_models.resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_module = load_source(source_path)
    source = new_source(source_module)
    assert all(parameter.device.type == "cpu" for parameter in source.parameters())
    assert preserved_v1()
    units = toy_units()
    checks = {"covariance_and_controls": check_covariance_and_controls(),
              "source_adapter_and_gradient": check_source_identity_and_native_gradient(source, units[0])}
    collection = check_collector_and_replay(source, units, source_sha256)
    checks["collector_and_replay"] = collection["summary"]
    assert preserved_v1()
    if args.record_output is not None:
        save_new_json(collection["record"], args.record_output, args.artifact_root)
    report = {"status": "PASS", "scope": "synthetic CPU torch/PyG only; no dataset or GPU",
              "environment": environment_contract(), "source_models_sha256": source_sha256,
              "v1_preserved": True, "checks": checks,
              "selected_projector_unbiasedness_claimed": False,
              "dataset_performance_claimed": False}
    save_new_json(report, args.output, args.artifact_root)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
