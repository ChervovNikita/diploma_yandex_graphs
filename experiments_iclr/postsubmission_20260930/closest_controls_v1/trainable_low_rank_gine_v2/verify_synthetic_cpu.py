"""Synthetic initialization/autograd/learning/count gates; no dataset or checkpoint."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for name, digest in bindings["phase_files_sha256"].items():
        require(sha(PHASE / name) == digest, "Source binding changed: " + name)
    manifest = json.loads((HERE / "artifact_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        require(sha(HERE / name) == digest, "Prepared packet file changed: " + name)
    old = ast.parse((PHASE / "new_tasks/molhiv_pilot.py").read_text())
    new = ast.parse((HERE / "loss.py").read_text())
    old_loss = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "graph_member_bce")
    new_loss = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "graph_member_bce")
    require(ast.dump(old_loss, include_attributes=False) == ast.dump(new_loss, include_attributes=False),
            "Inherited graph/member BCE changed")
    return bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and output != HERE and not output.exists(), "Use new engineering folder inside packet")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            and os.environ.get("GNNM_PHASE_ROOT") == str(PHASE), "Require explicit confined CPU wrapper")
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS",
                 "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                 "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR"):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(PHASE), "Unconfined cache: " + name)
    bindings = check_sources()
    output.mkdir(parents=True)
    try:
        import torch
        import torch.nn.functional as F
        import torch_geometric
        import ogb
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        from torch_geometric.data import Batch, Data
        from torch_geometric.nn import GINEConv
        import adapter
        from loss import graph_member_bce
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.get_default_dtype() == torch.float32, "Require original float32 production default")
        require(not torch.cuda.is_initialized(), "CUDA context forbidden")
        runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
                   "torch_geometric": torch_geometric.__version__, "ogb": ogb.__version__,
                   "classes_sha256": {cls.__name__: hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
                                      for cls in (AtomEncoder, BondEncoder, GINEConv)}}
        expected = bindings["expected_runtime"]
        require(all(runtime[k] == expected[k] for k in ("python", "torch", "torch_geometric", "ogb"))
                and runtime["classes_sha256"] == expected["classes_sha256"], "Production molecular runtime changed")
        checks = []
        # Exact legacy initialization oracle, without its unsupported generator
        # keyword. fork_rng restores global/dropout RNG around the oracle.
        for dtype in (torch.float32, torch.float64):
            shared = torch.nn.Linear(5, 3).to(dtype=dtype)
            rng_before = torch.get_rng_state().clone()
            layer = adapter.MemberLowRankLinear(shared, 4, 2, 901)
            require(torch.equal(rng_before, torch.get_rng_state()), "Private factor initializer changed common RNG")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(901)
                expected_A = torch.empty_like(layer.A)
                for member in range(4):
                    torch.nn.init.kaiming_uniform_(expected_A[member], a=math.sqrt(5))
            require(torch.equal(rng_before, torch.get_rng_state()), "Legacy initializer oracle failed to restore RNG")
            require(torch.equal(layer.A, expected_A), "Explicit private uniform differs from legacy Kaiming draws")
            require(not torch.equal(layer.A[0], layer.A[1]), "Member A draws accidentally tied")
            bound = math.sqrt(3.0) * (torch.nn.init.calculate_gain("leaky_relu", math.sqrt(5)) / math.sqrt(5))
            require(bool((layer.A.abs() <= bound).all()) and not bool(torch.count_nonzero(layer.B)),
                    "Kaiming range or zero-B initialization violated")
        checks.append("legacy_kaiming_exact_factor_draws_float32_float64_private_generator_and_global_RNG_isolation")
        max_effective_error = max_gradient_error = 0.0
        # Independent affine oracle covers trainable factors, shared W and shared bias.
        for has_bias in (False, True):
            torch.manual_seed(111)
            layer = adapter.MemberLowRankLinear(torch.nn.Linear(5, 3, bias=has_bias).double(), 4, 2, 901)
            with torch.no_grad():
                layer.B.copy_(torch.linspace(-.3, .2, layer.B.numel(), dtype=torch.float64).reshape_as(layer.B))
            x = torch.linspace(-1.2, .9, 35, dtype=torch.float64).reshape(7, 5)
            for member in range(4):
                layer.active_member = member
                got = layer(x)
                expected_output = F.linear(x, layer.shared.weight + layer.B[member] @ layer.A[member], layer.shared.bias)
                max_effective_error = max(max_effective_error, float((got - expected_output).abs().max()))
                require(torch.allclose(got, expected_output, atol=1e-10, rtol=1e-10), "Dense low-rank affine oracle mismatch")
                parameters = tuple(layer.parameters())
                left = torch.autograd.grad(got.square().mean(), parameters)
                right = torch.autograd.grad(expected_output.square().mean(), parameters)
                for a, b in zip(left, right):
                    max_gradient_error = max(max_gradient_error, float((a - b).abs().max()))
                    require(torch.allclose(a, b, atol=1e-10, rtol=1e-10), "Affine autograd oracle mismatch")
        checks.append("effective_BA_affine_and_all_parameter_gradients")

        def graph(nodes, edges, label):
            edge_index = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
            atoms = torch.stack([(torch.arange(nodes) + i) % d for i, d in enumerate(get_atom_feature_dims())], 1)
            bonds = torch.stack([(torch.arange(edge_index.shape[1]) + i) % d for i, d in enumerate(get_bond_feature_dims())], 1)
            return Data(x=atoms, edge_index=edge_index, edge_attr=bonds,
                        y=torch.tensor([[label]], dtype=torch.float32))
        # Directed/repeated edges, self-edge, isolate, empty-edge graph, both labels.
        batch = Batch.from_data_list([
            graph(5, [[0, 0, 1, 2, 3], [1, 1, 2, 0, 3]], 1.),
            graph(3, [[0, 1], [1, 0]], 0.), graph(1, [[], []], 1.)])
        torch.manual_seed(19)
        source = adapter.GINE.MolecularProjectorEnsemble(adapter.FIXED_OPTIONS, members=4)
        rng_before = torch.get_rng_state().clone()
        model, definition = adapter.make_model(source)
        require(torch.equal(rng_before, torch.get_rng_state()), "Count choice or factor construction consumes common RNG")
        require(len(model.low_rank_layers) == 6, "Wrong hidden-map coverage")
        actual = adapter.counts(model)
        expected_added = 4 * model.rank * 6 * (128 + 256)
        require(model.added_parameters == expected_added
                and actual["parameter_scalars"] == adapter.counts(source)["parameter_scalars"] + expected_added,
                "Actual parameter count differs from all six member A/B tensors")
        require(actual["buffer_scalars"] == adapter.counts(source)["buffer_scalars"]
                and actual["buffer_bytes"] == adapter.counts(source)["buffer_bytes"], "Unexpected low-rank buffer storage")
        require(actual["trainable_parameter_scalars"] == actual["parameter_scalars"], "Shared or private factors accidentally frozen")
        parameters = list(model.parameters())
        require(len({p.untyped_storage().data_ptr() for p in parameters}) == len(parameters), "Aliased registered parameter storage")
        mapped = {name.removeprefix("base.").replace(".shared.", "."): value
                  for name, value in model.state_dict().items() if not name.endswith((".A", ".B"))
                  or not name.startswith("base.blocks.")}
        require(set(mapped) == set(source.state_dict())
                and all(torch.equal(t, source.state_dict()[name]) for name, t in mapped.items()), "Inherited source tensor changed")
        require(all(bool(torch.count_nonzero(layer.A)) and not bool(torch.count_nonzero(layer.B))
                    for layer in model.low_rank_layers), "Require random-A/zero-B initialization")
        source.eval(); model.eval()
        with torch.no_grad():
            original_logits, initial_logits = source(batch), model(batch)
            require(initial_logits.shape == (4, 3, 1) and torch.equal(initial_logits, original_logits), "Zero delta changes complete molecular logits")
            require(torch.equal(model.pooled_forward(batch), initial_logits.mean(0)), "Wrong mean raw-logit pooling")
        checks.append("actual_production_count_rank_choice_zero_delta_inherited_state_and_eval_logits")

        source.train(); model.train()
        torch.manual_seed(783)
        original_dropout_logits = source(batch)
        original_after = torch.get_rng_state().clone()
        graph_member_bce(original_dropout_logits, batch.y).backward()
        torch.manual_seed(783)
        new_dropout_logits = model(batch)
        require(torch.equal(original_dropout_logits, new_dropout_logits)
                and torch.equal(original_after, torch.get_rng_state()), "Changed dropout streams or initial stochastic function")
        graph_member_bce(new_dropout_logits, batch.y).backward()
        old_params = dict(source.named_parameters())
        max_shared_gradient_error = 0.0
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing/nonfinite parameter gradient")
            if name.startswith("base.blocks.") and name.endswith((".A", ".B")):
                continue
            old_gradient = old_params[name.removeprefix("base.").replace(".shared.", ".")].grad
            require(old_gradient is not None and torch.allclose(parameter.grad, old_gradient, atol=1e-5, rtol=1e-5),
                    "Shared initial gradient differs under matched dropout")
            max_shared_gradient_error = max(max_shared_gradient_error, float((parameter.grad - old_gradient).abs().max()))
        first_B_norms = [float(layer.B.grad.norm()) for layer in model.low_rank_layers]
        require(all(n > 0 for n in first_B_norms), "Initial trainable correction branch is dead")
        require(all(not bool(torch.count_nonzero(layer.A.grad)) for layer in model.low_rank_layers), "Zero-B first-step A derivative should be zero")
        checks.append("matched_dropout_logits_streams_shared_gradients_live_B_and_expected_zero_A_derivative")

        # Only member2 can receive private gradients from a member2 loss.
        model.eval(); model.zero_grad(set_to_none=True)
        graph_member_bce(model.forward_member(batch, member=2)[None], batch.y).backward()
        for layer in model.low_rank_layers:
            require(layer.B.grad is not None and bool(torch.count_nonzero(layer.B.grad[2]))
                    and not bool(torch.count_nonzero(layer.B.grad[[0, 1, 3]])), "Private member gradient leaks across rows")
            require(layer.A.grad is not None and not bool(torch.count_nonzero(layer.A.grad)), "Initial A derivative inconsistent")
        checks.append("private_member_gradient_isolation")

        # Fresh synthetic optimization: after B updates, A must become trainable.
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., eps=1e-8, betas=(.9, .999))
        require(not optimizer.state, "Synthetic optimizer must start fresh")
        A_before = [layer.A.detach().clone() for layer in model.low_rank_layers]
        learning_losses = []
        second_A_norms = None
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = graph_member_bce(model(batch), batch.y)
            require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
            loss.backward()
            if step == 1:
                second_A_norms = [float(layer.A.grad.norm()) for layer in model.low_rank_layers]
                require(all(n > 0 for n in second_A_norms), "A does not receive a live gradient after the first B update")
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Nonfinite optimizer gradient")
            optimizer.step()
            learning_losses.append(float(loss.detach()))
        require(all(bool(torch.count_nonzero(layer.B @ layer.A)) for layer in model.low_rank_layers), "No learned nonzero dense correction")
        require(all(not torch.equal(before, layer.A) for before, layer in zip(A_before, model.low_rank_layers)), "Trainable A never updated")
        with torch.no_grad():
            require(not torch.equal(model(batch), initial_logits), "Synthetic training failed to change the model function")
        checks.append("fresh_three_step_nonzero_BA_and_A_learning")

        # A single member intervention cannot alter another member's hidden path.
        separate = adapter.TrainableLowRankGINE(source, model.rank).eval()
        with torch.no_grad():
            before = separate(batch)
            for layer in separate.low_rank_layers:
                layer.B[2].fill_(.03)
            after = separate(batch)
            require(torch.equal(before[[0, 1, 3]], after[[0, 1, 3]])
                    and not torch.equal(before[2], after[2]), "Member hidden paths were pooled or coupled")
        checks.append("single_member_nonzero_intervention_changes_only_its_full_graph_path")

        # Independent nonzero whole-graph oracle: directed bond messages,
        # residuals, both low-rank maps, per-node norm and final graph readout.
        oracle_model = copy.deepcopy(model).double().eval()
        def boundary(layer, x, member):
            return F.linear(x * layer.R[member], layer.W.weight) * layer.S[member] + layer.B[member]
        def effective(layer, x, member):
            weight = layer.shared.weight + layer.B[member] @ layer.A[member]
            return F.linear(x, weight, layer.shared.bias)
        def oracle_member(member):
            base = oracle_model.base
            x = F.gelu(boundary(base.input_projector, base.encoder(batch.x), member))
            src, dst = batch.edge_index
            for block in base.blocks:
                norm = block.normalization
                h = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
                bonds = block.bond_encoder(batch.edge_attr)
                message = F.relu(h[src] + bonds)
                incoming = torch.zeros_like(h).index_add(0, dst, message)
                inner = incoming + (1 + block.conv.eps) * h
                mlp = block.conv.nn
                inner = effective(mlp[3], F.gelu(effective(mlp[0], inner, member)), member)
                x = x + inner
            norm = base.output_normalization
            x = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
            pooled = torch.stack([x[batch.batch == i].mean(0) for i in range(batch.num_graphs)])
            return boundary(base.output_projector, pooled, member)
        actual_graph = oracle_model(batch)
        oracle_graph = torch.stack([oracle_member(member) for member in range(4)])
        full_graph_error = float((actual_graph - oracle_graph).abs().max())
        require(torch.allclose(actual_graph, oracle_graph, atol=1e-8, rtol=1e-8), "Nonzero whole-graph scatter/readout oracle mismatch")
        parameters = tuple(oracle_model.parameters())
        actual_gradients = torch.autograd.grad(actual_graph.square().mean(), parameters)
        oracle_gradients = torch.autograd.grad(oracle_graph.square().mean(), parameters)
        full_graph_gradient_error = 0.0
        for a, b in zip(actual_gradients, oracle_gradients):
            full_graph_gradient_error = max(full_graph_gradient_error, float((a - b).abs().max()))
            require(torch.allclose(a, b, atol=1e-8, rtol=1e-8), "Nonzero whole-graph autograd oracle mismatch")
        require(torch.allclose(oracle_model.pooled_forward(batch), oracle_graph.mean(0), atol=1e-8, rtol=1e-8),
                "Nonzero final raw-logit readout mismatch")
        checks.append("nonzero_whole_graph_bond_scatter_residual_norm_lowrank_readout_and_autograd_oracle")

        # The autograd gate must reject the common zero-A/zero-B dead branch.
        dead = adapter.MemberLowRankLinear(torch.nn.Linear(5, 3).double(), 4, 2, 901)
        with torch.no_grad():
            dead.A.zero_(); dead.B.zero_()
        dead(torch.ones(7, 5, dtype=torch.float64)).square().sum().backward()
        require(not bool(torch.count_nonzero(dead.A.grad)) and not bool(torch.count_nonzero(dead.B.grad)), "Dead-branch fixture inconsistent")
        rejected = False
        try:
            require(float(dead.B.grad.norm()) > 0, "Dead low-rank branch rejected")
        except ValueError:
            rejected = True
        require(rejected, "Autograd admission failed to reject zero-zero initialization")
        checks.append("zerozero_dead_branch_rejection")

        # Rank zero: no new storage, same dropout, same function after a real update.
        rankzero_source = copy.deepcopy(source)
        rankzero_source.zero_grad(set_to_none=True)
        rankzero = adapter.TrainableLowRankGINE(rankzero_source, 0)
        require(adapter.counts(rankzero) == adapter.counts(rankzero_source) and rankzero.added_parameters == 0,
                "Rank zero must have exactly original parameter/buffer storage")
        rankzero_source.train(); rankzero.train()
        opts = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0.) for m in (rankzero_source, rankzero)]
        before_outputs = []
        for m, opt in zip((rankzero_source, rankzero), opts):
            torch.manual_seed(997)
            opt.zero_grad(set_to_none=True)
            logits = m(batch)
            before_outputs.append(logits.detach())
            graph_member_bce(logits, batch.y).backward(); opt.step()
        require(torch.equal(*before_outputs), "Rank-zero dropout function differs")
        rankzero_source.eval(); rankzero.eval()
        with torch.no_grad():
            require(torch.equal(rankzero_source(batch), rankzero(batch)), "Rank-zero function differs after fresh optimizer update")
        for name, value in rankzero.state_dict().items():
            require(torch.equal(value, rankzero_source.state_dict()[name.removeprefix("base.").replace(".shared.", ".")]),
                    "Rank-zero updated common state differs")
        checks.append("rankzero_exact_function_storage_dropout_and_real_update_identity")

        report = {"status": "PASS", "scope": "engineering synthetic graphs only; not predictive evidence",
                  "checks": checks, "runtime": runtime, "source_bindings_sha256": sha(HERE / "SOURCE_BINDINGS.json"),
                  "initializer_compatibility_repair_provenance_sha256": sha(HERE / "REPAIR_PROVENANCE.json"),
                  "prepared_manifest_sha256": sha(HERE / "artifact_manifest.json"), "definition": definition,
                  "actual_production_counts": actual, "rankzero_counts": adapter.counts(rankzero),
                  "actual_three_step_optimizer_tensor_bytes": sum(t.numel() * t.element_size()
                      for state in optimizer.state.values() for t in state.values() if isinstance(t, torch.Tensor)),
                  "maximum_effective_affine_error": max_effective_error, "maximum_affine_gradient_error": max_gradient_error,
                  "maximum_shared_initial_gradient_error": max_shared_gradient_error,
                  "nonzero_whole_graph_logit_max_abs_error": full_graph_error,
                  "nonzero_whole_graph_gradient_max_abs_error": full_graph_gradient_error,
                  "first_step_B_gradient_norms": first_B_norms, "second_step_A_gradient_norms": second_A_norms,
                  "synthetic_three_step_losses": learning_losses, "nonzero_synthetic_learning_claim_only": True,
                  "dataset_or_model_checkpoint_access": False, "test_data_accessed": False,
                  "predictive_fit_performed": False, "novelty_claim": False, "author_TabLoRA_reproduction": False,
                  "CUDA_context_created": False, "synthetic_graphs": 3, "synthetic_nodes": 9,
                  "source_inventory_rule": "explicit immutable manifest only; logs/outputs excluded"}
        (output / "SYNTHETIC_CPU_GATE.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "PASS", "rank": model.rank, "parameters": actual["parameter_scalars"],
                          "parameter_mismatch": definition["rank_plan"]["absolute_parameter_mismatch"]}), flush=True)
    except Exception as error:
        (output / "FAILURE.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
