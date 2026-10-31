"""Synthetic source-compatible CPU gate; no dataset, optimizer, or remote operation."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE))

import torch
from torch_geometric.data import Data
from spectral_privacy.v2_abi_fix_01.adapters import SourceSAGEEnsemble
from spectral_privacy.v2_abi_fix_01.collector import GradientUnit, PreservedProbe, environment_contract
from spectral_privacy.v2_abi_fix_01.contracts import save_new_json, state_hash

from freeze import code_hashes, phase_path, sha
from objective_core import (GROUPS, OBJECTIVES, TARGETS, TOLERANCE, check_chain,
                            collect_one, diagnostics, losses, projected_chain)

EXPECTED_SOURCE_SHA = "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"


def source_module(path):
    path = phase_path(path)
    if sha(path) != EXPECTED_SOURCE_SHA:
        raise ValueError("Gate requires the exact original source snapshot")
    spec = importlib.util.spec_from_file_location("objective_gate_original_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_units():
    generator = torch.Generator(device="cpu").manual_seed(9401)
    units = []
    for index, nodes in ((9, 7), (18, 11)):
        edges = [(j, (j + 1) % nodes) for j in range(nodes)]
        edges += [(b, a) for a, b in edges]
        features = torch.randn(nodes, 5, dtype=torch.float64, generator=generator)
        targets = (torch.rand(nodes, 3, generator=generator) > 0.4).double()
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[:nodes // 2] = True
        targets[~mask] = float("nan")
        graph = Data(x=features, y=targets, edge_index=torch.tensor(edges).T.contiguous(), num_nodes=nodes)
        units.append(GradientUnit(f"ppi:train:{index:02d}", graph, features, targets, mask, "train"))
    return units


def synthetic_bases():
    generator = torch.Generator(device="cpu").manual_seed(9402)
    result = {}
    for group in GROUPS:
        result[group] = {}
        for key, rank in zip(TARGETS, (2, 0, 3)):
            if rank:
                draw = torch.randn(128, rank, dtype=torch.float64, generator=generator)
                result[group][key] = torch.linalg.qr(draw, mode="reduced").Q
            else:
                result[group][key] = torch.empty(128, 0, dtype=torch.float64)
    return result


def verify(source_path):
    core = PHASE / "spectral_privacy/v2_abi_fix_01"
    manifest = json.loads((core / "PROTOTYPE_MANIFEST.json").read_text())
    for name in ("adapters.py", "collector.py", "contracts.py"):
        if sha(core / name) != manifest["files_sha256"][name]:
            raise ValueError("Existing adapter/collector/contracts changed before the gate")
    module = source_module(source_path)
    torch.manual_seed(9400)
    source = module.TABMModel(model_name="SAGE", num_layers=3, input_dim=5, hidden_dim=128,
                             output_dim=3, hidden_dim_multiplier=1, num_heads=1,
                             normalization="LayerNorm", dropout=0.2, tabm_inits=4, device="cpu").double()
    source.eval()
    common = SourceSAGEEnsemble(source)
    common.eval()
    # Existing .grad references and mixed modes must survive diagnostic collection.
    parameter = next(common.parameters())
    parameter.grad = torch.ones_like(parameter) * 0.125
    old_reference = parameter.grad
    common.source.residual_modules[0].module.train()
    units, bases = synthetic_units(), synthetic_bases()
    models = {group: SourceSAGEEnsemble(source, bases[group]) for group in GROUPS}
    for model in models.values():
        model.eval()
    measurements = {group: {name: [] for name in OBJECTIVES} for group in GROUPS}
    chain_reports = {}
    source_identity = state_hash(source)
    for unit in units:
        source_gradients = collect_one(common, unit, matrices=True)
        if parameter.grad is not old_reference or not common.source.residual_modules[0].module.training:
            raise AssertionError("Existing gradient reference or mode was changed")
        chain_reports[unit.unit_id] = {}
        for group in GROUPS:
            measured = collect_one(models[group], unit)
            if not torch.equal(measured["member_logits"], source_gradients["member_logits"]):
                raise AssertionError("Zero private correction changed the original function")
            if state_hash(models[group].tied_source_copy()) != source_identity:
                raise AssertionError("Source tensor bytes changed")
            expected = projected_chain(source_gradients["source_response_matrices"],
                                       models[group].frozen_bases(), models[group].C)
            chain_reports[unit.unit_id][group] = check_chain(measured["coefficient_gradients"], expected, torch.float64)
            for name in OBJECTIVES:
                measurements[group][name].append(measured["coefficient_gradients"][name])
    # Equal whole-graph averaging is checked against one joint objective with unequal graph sizes.
    selected = models["selected"]
    coefficients = selected.named_private_coefficients()
    active = tuple(key for key in TARGETS if coefficients[key].numel())
    snapshot = PreservedProbe(selected)
    joint = {name: [] for name in OBJECTIVES}
    try:
        selected.eval()
        for unit in units:
            objectives, _, _, _ = losses([selected.forward_member(unit.graph, unit.features, m) for m in range(4)], unit)
            for name in OBJECTIVES:
                joint[name].append(objectives[name])
        actual = {}
        for name in OBJECTIVES:
            values = torch.autograd.grad((joint[name][0] + joint[name][1]) / 2,
                                         tuple(coefficients[key] for key in active), retain_graph=True)
            lookup = dict(zip(active, values))
            actual[name] = {key: (lookup[key].detach().double() if key in lookup else
                                  torch.empty(coefficients[key].shape, dtype=torch.float64)) for key in TARGETS}
        expected = {name: {key: (measurements["selected"][name][0][key] + measurements["selected"][name][1][key]) / 2
                           for key in TARGETS} for name in OBJECTIVES}
        equal_graph_report = check_chain(actual, expected, torch.float64)
        if not snapshot.matches():
            raise AssertionError("Joint objective verification changed state/grads/RNG")
    finally:
        snapshot.restore()
    null_source = copy.deepcopy(source)
    with torch.no_grad():
        for boundary in (null_source.input_be_block, null_source.output_be_block):
            for name in ("R", "S", "B"):
                value = getattr(boundary, name)
                value[1:].copy_(value[:1].expand_as(value[1:]))
    null_reports = {}
    for group in GROUPS:
        null_model = SourceSAGEEnsemble(null_source, bases[group])
        null_model.eval()
        null_reports[group] = []
        for unit in units:
            measured = collect_one(null_model, unit)
            assert all(torch.equal(measured["member_logits"][0], measured["member_logits"][m]) for m in range(1, 4))
            maxima = {}
            for name in OBJECTIVES:
                maximum = max((float(value.abs().max()) if value.numel() else 0)
                              for value in measured["coefficient_gradients"][name].values())
                assert maximum <= TOLERANCE["float64_null_max_abs"]
                maxima[name] = maximum
            null_reports[group].append({"unit_id": unit.unit_id, "max_abs": maxima, "identical_logits": True})
    # Empty rank is a valid state; zero directions have undefined cosine, represented by null.
    empty_model = SourceSAGEEnsemble(source)
    empty = {group: {name: [] for name in OBJECTIVES} for group in GROUPS}
    for unit in units:
        measured = collect_one(empty_model, unit)
        for group in GROUPS:
            for name in OBJECTIVES:
                empty[group][name].append(measured["coefficient_gradients"][name])
    empty_bases = {group: empty_model.frozen_bases() for group in GROUPS}
    empty_diagnostic = diagnostics(empty, empty_bases, empty_model.C)
    for row in empty_diagnostic["within_basis_coefficient_directions"].values():
        assert row["all_targets"]["capacity_scalars"] == 0
        assert row["all_targets"]["pool_cross_graph"]["inner_product"] == 0
        assert row["all_targets"]["pool_cross_graph"]["cosine"] is None
    bad = GradientUnit(units[0].unit_id, units[0].graph, units[0].features,
                       units[0].targets, units[0].observed_mask, "validation")
    try:
        collect_one(models["selected"], bad)
    except ValueError:
        rejection_pass = True
    else:
        raise AssertionError("Non-training graph was accepted")
    assert not torch.cuda.is_initialized()
    return {"status": "OBJECTIVE_SYNTHETIC_CPU_GATE_PASS", "code_sha256": code_hashes(),
            "source_models_sha256": sha(source_path), "environment": environment_contract(),
            "objective_chain_reports": chain_reports, "equal_graph_joint_autograd": equal_graph_report,
            "identical_members_null": null_reports, "rank_zero_direction_handling": "PASS",
            "existing_grad_and_mode_preservation": "PASS", "non_training_rejection": rejection_pass,
            "CUDA_initialized": False, "training_performed": False, "dataset_loaded": False,
            "scope": "synthetic ABI/objective/scaling verification only; real six-cell autograd remains pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models", type=Path, default=PHASE / "source_snapshot/models.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = phase_path(args.output)
    if HERE not in output.parents:
        raise ValueError("Gate output must stay inside the diagnostic folder")
    result = verify(args.source_models)
    save_new_json(result, output, HERE)
    print(json.dumps({"status": result["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
