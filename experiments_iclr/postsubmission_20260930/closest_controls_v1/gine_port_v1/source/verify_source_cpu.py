"""Exact-source chemical comparator gate; synthetic graphs, no fits or dataset access."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amendment_paths as amendment


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = amendment.output_inside(args.report)
    require(not output.exists(), "Refusing to overwrite a gate report")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require hidden CUDA devices")
    import torch
    import torch.nn.functional as F
    import torch_geometric
    import ogb
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
    from torch_geometric.nn import GINEConv
    from torch_geometric.data import Batch, Data
    import all_layer_gine_adapter as adapter
    import molhiv_readouts as readouts
    spec = importlib.util.spec_from_file_location("amended_molecular_pilot_cpu_gate", HERE / "new_tasks/molhiv_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE / "new_tasks"))
    spec.loader.exec_module(pilot)
    torch.set_num_threads(1)
    actual_runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__, "ogb": ogb.__version__,
        "runtime_class_source_sha256": {cls.__name__: hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
            for cls in [AtomEncoder, BondEncoder, GINEConv, PygGraphPropPredDataset]}}
    expected = json.loads((HERE / "expected_runtime.json").read_text())
    require(actual_runtime == expected, "Runtime differs from the preserved MolHIV engineering environment")
    provenance = json.loads((HERE / "amendment_provenance.json").read_text())
    for name in ["new_tasks/gine_adapters.py", "new_tasks/shareable_guard.py", "source_snapshot/models.py"]:
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == provenance["original_files"][name],
                "Preserved molecular dependency changed: " + name)
    require(hashlib.sha256(inspect.getsource(adapter.MemberFactorLinear).encode()).hexdigest()
            == provenance["MemberFactorLinear_origin"]["class_source_sha256"],
            "MemberFactorLinear differs from the exact validated SAGE wrapper")
    before = amendment.PACKET / "before_amendment/new_tasks/molhiv_pilot.py"
    require(hashlib.sha256(before.read_bytes()).hexdigest() == provenance["original_files"]["new_tasks/molhiv_pilot.py"],
            "Preserved original runner changed")
    old_tree, new_tree = ast.parse(before.read_text()), ast.parse((HERE / "new_tasks/molhiv_pilot.py").read_text())
    for name in ["inside", "verify_data", "load_development", "train_budgets", "graph_member_bce", "gpu_guard", "set_seed"]:
        old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        require(ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False),
                "Inherited data/loss/resource guard changed: " + name)
    checks = ["exact_original_dependencies_and_hidden_wrapper", "inherited_train_only_data_loss_and_idle_GPU_guards"]
    def same(a, b, message, tolerance=1e-10):
        require(torch.allclose(a, b, atol=tolerance, rtol=tolerance), message)
        return float((a-b).abs().max())
    def effective(layer, x, member):
        weight = layer.shared.weight * layer.S[member, :, None] * layer.R[member, None, :]
        bias = layer.B[member]
        if layer.shared.bias is not None:
            bias = bias + layer.shared.bias * layer.S[member]
        return F.linear(x, weight, bias)
    for has_bias in [False, True]:
        layer = adapter.MemberFactorLinear(torch.nn.Linear(5, 3, bias=has_bias), 4).double()
        with torch.no_grad():
            layer.R.copy_(torch.linspace(.4, 1.6, layer.R.numel()).reshape_as(layer.R))
            layer.S.copy_(torch.linspace(-.7, 1.1, layer.S.numel()).reshape_as(layer.S))
            layer.B.copy_(torch.linspace(-.2, .3, layer.B.numel()).reshape_as(layer.B))
        x = torch.linspace(-.9, 1.2, 35, dtype=torch.float64).reshape(7, 5)
        for member in range(4):
            layer.active_member = member
            actual, oracle = layer(x), effective(layer, x, member)
            same(actual, oracle, "Effective hidden affine weight/bias mismatch")
            params = tuple(layer.parameters())
            for left, right in zip(torch.autograd.grad(actual.square().mean(), params),
                                   torch.autograd.grad(oracle.square().mean(), params)):
                same(left, right, "Effective affine derivative mismatch")
    checks.append("effective_affine_weights_scaled_bias_and_gradients")
    def graph(nodes, edges, label):
        edges = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
        atoms = torch.stack([torch.arange(nodes) % d for d in get_atom_feature_dims()], dim=1)
        bonds = torch.stack([torch.arange(edges.shape[1]) % d for d in get_bond_feature_dims()], dim=1)
        return Data(x=atoms, edge_index=edges, edge_attr=bonds, y=torch.tensor([[label]], dtype=torch.float32))
    # Includes repeated directed entries, a self-edge, an isolate and an empty-edge graph.
    batch = Batch.from_data_list([
        graph(5, [[0, 0, 1, 2, 3], [1, 1, 2, 0, 3]], 1.),
        graph(3, [[0, 1], [1, 0]], 0.), graph(1, [[], []], 1.)])
    torch.manual_seed(19)
    original = adapter.GINE.MolecularProjectorEnsemble().eval()
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    model, definition = adapter.make_model()
    model.eval()
    require(torch.equal(original_rng, torch.get_rng_state()), "New constructor consumes extra RNG")
    require(len(model.factor_layers) == 6 and model.added_parameters == 13824,
            "Fixed six-affine coverage/count mismatch")
    parameters = list(model.parameters())
    require(len({p.untyped_storage().data_ptr() for p in parameters}) == len(parameters),
            "Duplicate/aliased registered parameter storage")
    mapped = {name.replace(".shared.", "."): value for name, value in model.state_dict().items()
              if not (name.startswith("blocks.") and name.endswith((".R", ".S", ".B")))}
    old_state = original.state_dict()
    require(set(mapped) == set(old_state) and all(torch.equal(v, old_state[k]) for k,v in mapped.items()),
            "Inherited atom/bond/boundary/normalization/GINE state differs")
    require(adapter.GINE.parameter_count(model) == adapter.GINE.parameter_count(original) + 13824,
            "Actual total count differs from added-factor accounting")
    with torch.no_grad():
        logits = model(batch, batch.x)
        require(logits.shape == (4, 3, 1) and torch.equal(logits, original(batch, batch.x)),
                "Identity initialization changes whole-graph member logits")
    original.train(); model.train()
    torch.manual_seed(783)
    pilot.graph_member_bce(original(batch, batch.x), batch.y).backward()
    torch.manual_seed(783)
    pilot.graph_member_bce(model(batch, batch.x), batch.y).backward()
    old_params = dict(original.named_parameters())
    shared_diff = 0.0
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), "Missing/nonfinite gradient")
        if name.startswith("blocks.") and name.endswith((".R", ".S", ".B")):
            continue
        shared_diff = max(shared_diff, same(parameter.grad, old_params[name.replace(".shared.", ".")].grad,
                                           "Matched-dropout shared first-step gradient differs", 1e-5))
    checks.append("original_GNNM_initial_state_RNG_logits_and_matched_dropout_shared_gradients")
    model.eval(); model.zero_grad(set_to_none=True)
    model.forward_member(batch, batch.x, 2).square().mean().backward()
    for layer in model.factor_layers:
        for parameter in [layer.R, layer.S, layer.B]:
            require(parameter.grad is not None and torch.count_nonzero(parameter.grad[[0,1,3]]) == 0,
                    "Cross-member hidden-factor gradient leakage")
        require(layer.shared.weight.grad is not None, "Missing shared-map derivative")
    checks.append("member_specific_gradient_rows_and_shared_trainability")
    # An independent scatter/readout oracle verifies the entire bond-aware path.
    model = model.double().eval()
    with torch.no_grad():
        for index, layer in enumerate(model.factor_layers):
            layer.R.copy_(torch.linspace(.8,1.2,layer.R.numel()).reshape_as(layer.R))
            layer.S.copy_(torch.linspace(.7,1.3,layer.S.numel()).reshape_as(layer.S))
            layer.B.copy_(torch.linspace(-.03,.04,layer.B.numel()).reshape_as(layer.B))
        for block in model.blocks:
            block.conv.eps.fill_(.17)
    def boundary(layer, x, member):
        return F.linear(x * layer.R[member], layer.W.weight) * layer.S[member] + layer.B[member]
    def oracle_member(member):
        x = F.gelu(boundary(model.input_projector, model.encoder(batch.x), member))
        src, dst = batch.edge_index
        for block in model.blocks:
            norm = block.normalization
            h = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
            bonds = block.bond_encoder(batch.edge_attr)
            message = F.relu(h[src] + bonds)
            incoming = torch.zeros_like(h).index_add(0, dst, message)
            inner = incoming + (1 + block.conv.eps) * h
            mlp = block.conv.nn
            inner = effective(mlp[3], F.gelu(effective(mlp[0], inner, member)), member)
            x = x + inner
        norm = model.output_normalization
        x = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
        pooled = torch.stack([x[batch.batch == graph_index].mean(0) for graph_index in range(batch.num_graphs)])
        return boundary(model.output_projector, pooled, member)
    actual, oracle = model(batch, batch.x), torch.stack([oracle_member(m) for m in range(4)])
    whole_diff = same(actual, oracle, "Bond-aware whole-graph scatter/affines/residual/readout differs", 1e-8)
    parameters = tuple(model.parameters())
    grads_actual = torch.autograd.grad(actual.square().mean(), parameters)
    grads_oracle = torch.autograd.grad(oracle.square().mean(), parameters)
    whole_grad_diff = max(same(a,b,"Whole-graph independent-oracle gradient differs",1e-8)
                          for a,b in zip(grads_actual,grads_oracle))
    checks.append("nonidentity_full_graph_bond_scatter_residual_readout_and_gradients")
    # Distinct pooling rankings and extreme logits catch accidental raw-only scoring.
    fixture = torch.tensor([[[-1.],[.3]],[[-1.],[.3]],[[-1.],[.3]],[[6.],[.3]]], dtype=torch.float64)
    target = torch.tensor([[1.],[0.]], dtype=torch.float64)
    value = readouts.metrics(fixture,target,readouts.ALL_LAYER_ARM)
    require(value["raw_rocauc"] == 1 and value["prob_rocauc"] == 0,
            "Synthetic readout ranking reversal not detected")
    probs = fixture.sigmoid().mean(0)
    same(torch.tensor(value["prob_pooled_bce"],dtype=torch.float64), F.binary_cross_entropy(probs,target),
         "Stable mixture BCE differs from moderate-logit direct arithmetic")
    extreme = torch.full((4,2,1),1000.,dtype=torch.float64)
    extreme_value = readouts.metrics(extreme,target,readouts.ALL_LAYER_ARM)
    require(all(torch.isfinite(torch.tensor(v)) for v in extreme_value.values())
            and abs(extreme_value["prob_pooled_bce"]-500.) < 1e-9,
            "Mixture BCE is unstable or clipped for extreme logits")
    checks.append("stable_float64_mixture_BCE_and_distinct_raw_probability_rankings")
    fixture_root = output.parent / (output.stem + "_synthetic_fixtures")
    require(not fixture_root.exists(), "Refusing to overwrite gate fixtures")
    fixture_root.mkdir(parents=True)
    chosen_epochs = {}
    for arm in ["gnnm", readouts.ALL_LAYER_ARM]:
        run = fixture_root / arm; run.mkdir(); (run / "validation_evidence").mkdir()
        best, rows = {}, []
        for epoch, z in enumerate([fixture, torch.zeros_like(fixture), fixture*.5],1):
            metrics = readouts.metrics(z,target,arm)
            row = {"epoch":epoch,**metrics};rows.append(row)
            torch.save({"epoch":epoch,"member_logits":z},run/"validation_evidence"/f"epoch_{epoch:04d}.pt")
            readouts.update_selected(run,best,row,
                {"state_dict":{"synthetic":torch.tensor([epoch])},"epoch":epoch,"validation":metrics,"contract_sha256":"synthetic"},
                {"member_logits":z,"targets":target,"official_graph_indices":torch.tensor([0,1])},arm)
        import csv
        with (run/"validation_trace.csv").open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        checked=readouts.verify_retained(run,arm,3,target,torch.tensor([0,1]),"synthetic")
        require(checked["raw"]["epoch"] == 1 and checked["prob"]["epoch"] == 2,
                "Independent readout selections do not retain their own trajectory winners")
        chosen_epochs[arm]={p:checked[p]["epoch"] for p in readouts.READOUTS}
        name=run/"selected_prob.json";unchanged=name.read_text()
        tampered=json.loads(unchanged);tampered["epoch"]=99;name.write_text(json.dumps(tampered))
        try:
            readouts.verify_retained(run,arm,3,target,torch.tensor([0,1]),"synthetic")
        except ValueError:
            pass
        else:
            raise RuntimeError("Tampered readout selection was not rejected")
        name.write_text(unchanged)
    checks.append("same_trajectory_two_readout_selection_full_retained_replay_primary_alias_and_tamper_rejection")
    rng = torch.get_rng_state().clone()
    plan = pilot.comparison_plan(["middle_private","gnnm","untied","base","multihead","ens",
                                  "base_parameters","ens_parameters",readouts.ALL_LAYER_ARM])
    require(torch.equal(rng,torch.get_rng_state()), "CPU parameter planning changes initialization RNG")
    require(plan[readouts.ALL_LAYER_ARM]["parameters"] == definition["parameters"], "Factory count differs")
    checks.append("actual_CPU_parameter_counts_and_RNG_preserving_comparison_plan")
    require(not torch.cuda.is_initialized(), "Synthetic source gate created a CUDA context")
    report={"status":"PASS","scope":"synthetic chemical-comparator source/affine/member/full-graph/readout gate",
            "source_sha256":amendment.source_hashes(),"runtime":actual_runtime,"checks":checks,
            "fixed_comparator_definition":definition,"actual_CPU_parameter_counts":plan,
            "max_shared_first_step_gradient_difference":shared_diff,
            "whole_graph_logit_max_abs_difference":whole_diff,"whole_graph_gradient_max_abs_difference":whole_grad_diff,
            "synthetic_selection_epochs":chosen_epochs,"fixture_root":str(fixture_root),
            "dataset_loads":0,"optimizer_steps":0,"performance_evidence":False,
            "validation_or_test_scored":False,"CUDA_context_created":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
