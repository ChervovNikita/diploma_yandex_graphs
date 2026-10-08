"""Small fabricated CPU semantics fixture, never scientific quality evidence.

Uses hash-pinned native Polynormer/BE sources. No dataset, checkpoint, target,
optimizer step, GPU call or training. One fresh Adam is constructed for exact
parameter-membership checks. Run with the existing allocation CPU runtime.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PINS = {
    "polynormer": ("wikics_native_polynormer_r_donor_preparation_20261007_v1/vendor/native_polynormer.py",
                   "9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8"),
    "factors": ("portable_internal_be_public_interface_20261007_v2/core/factors.py",
                "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3")}


def load(path, name, digest=None):
    if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Pinned fixture source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(phase):
    import torch
    import torch_geometric
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(9311)
    native = load(phase/PINS["polynormer"][0], "_private_att_native", PINS["polynormer"][1])
    factors = load(phase/PINS["factors"][0], "_private_att_factors", PINS["factors"][1])
    hook = load(Path(__file__).with_name("private_local_attention.py"), "_private_att_hook")
    options = dict(in_channels=5, hidden_channels=8, out_channels=3, local_layers=2,
                   global_layers=1, heads=1, in_dropout=.2, dropout=.2,
                   global_dropout=.2, beta=-1, pre_ln=False)

    def build():
        model = native.Polynormer(**options)
        model.reset_parameters()
        factors.install_factors(model, 4)
        return model.double()

    original = build()
    # Different known factor rows make an incorrectly selected BE route visible.
    with torch.no_grad():
        for m in range(4):
            original.lin_in.r[m].copy_(1 + .05*m*torch.arange(5, dtype=torch.float64))
    candidate = copy.deepcopy(original)  # Copy BEFORE parametrization installation.
    original_scorers = {f"local_convs.{i}.{n}": getattr(c,n).detach().clone()
                        for i,c in enumerate(candidate.local_convs)
                        for n in ("att_src", "att_dst")}
    nonscorer_ids = {n:id(p) for n,p in candidate.named_parameters()
                    if n not in original_scorers}
    nonscorer_values = {n:p.detach().clone() for n,p in candidate.named_parameters()
                       if n not in original_scorers}
    rng_before = torch.get_rng_state().clone()
    controller = hook.install_private_local_attention(candidate, 4)
    assert torch.equal(rng_before, torch.get_rng_state()), "Installer consumed RNG"
    params = dict(candidate.named_parameters())
    for n, ident in nonscorer_ids.items():
        assert id(params[n]) == ident and torch.equal(params[n], nonscorer_values[n])
    for i,c in enumerate(candidate.local_convs):
        for name in ("att_src", "att_dst"):
            bank = getattr(c.parametrizations,name).original
            assert tuple(bank.shape) == (4,1,1,8)
            for m in range(4):
                assert torch.equal(bank[m], original_scorers[f"local_convs.{i}.{name}"])
    assert controller.metadata()["additional_prediction_parameters"] == 96
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in original.parameters()) + 96
    bank_ids = {id(getattr(c.parametrizations,n).original)
                for c in candidate.local_convs for n in ("att_src","att_dst")}
    assert set(map(id,candidate.parameters())) == set(nonscorer_ids.values()) | bank_ids
    optimizer = torch.optim.Adam(candidate.parameters(),lr=.001,weight_decay=0.,eps=1e-8)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    assert {id(p) for p in optimized} == {id(p) for p in candidate.parameters()}
    assert bank_ids <= {id(p) for p in optimized} and len(optimizer.state) == 0
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == .001
    assert optimizer.param_groups[0]["weight_decay"] == 0.
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["betas"] == (.9,.999)

    x = torch.arange(30, dtype=torch.float64).reshape(6,5)/17 - .4
    # Includes duplicates, an explicit self edge and an isolated receiver.
    edges = torch.tensor([[0,1,1,2,3,3,4,4], [1,0,0,3,2,3,0,2]], dtype=torch.long)
    labels = torch.tensor([0,1,2,0,1,2])
    checks = {"copied_rows_no_RNG_and_preserved_nonscorer_aliases": True,
              "exact_small_fixture_parameter_delta": True,
              "fresh_Adam_exact_membership_no_unused_original_scorers":True}

    def route(model, member, attention=None):
        with factors.member_context(model, member):
            if attention is None:
                return model(x,edges)
            with attention.member_context(member):
                return model(x,edges)

    for global_stage in (False,True):
        original._global = candidate._global = global_stage
        for train_mode in (False,True):
            original.train(train_mode); candidate.train(train_mode)
            for m in range(4):
                torch.manual_seed(1331+m)
                reference = route(original,m)
                torch.manual_seed(1331+m)
                copied = route(candidate,m,controller)
                assert torch.equal(reference,copied), "Initial native function differs"
    checks["exact_initial_local_global_and_matched_dropout_functions"] = True

    original.eval(); candidate.eval()
    original._global = candidate._global = True
    reference_loss = torch.stack([F.cross_entropy(route(original,m), labels)
                                  for m in range(4)]).mean()
    candidate_loss = torch.stack([F.cross_entropy(route(candidate,m,controller), labels)
                                  for m in range(4)]).mean()
    reference_loss.backward(); candidate_loss.backward()
    reference_parameters = dict(original.named_parameters())
    for name in nonscorer_ids:
        a,b = reference_parameters[name].grad, params[name].grad
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
    for i,c in enumerate(candidate.local_convs):
        for name in ("att_src", "att_dst"):
            p = getattr(c.parametrizations,name).original
            ref = getattr(original.local_convs[i],name).grad
            assert p.grad is not None and ref is not None
            torch.testing.assert_close(p.grad.sum(0),ref,rtol=1e-8,atol=1e-10)
    checks["initial_joint_loss_shared_gradient_and_summed_scorer_gradient"] = True

    candidate.zero_grad(set_to_none=True)
    F.cross_entropy(route(candidate,2,controller), labels).backward()
    active_total = 0.
    for c in candidate.local_convs:
        for name in ("att_src", "att_dst"):
            g = getattr(c.parametrizations,name).original.grad
            assert g is not None
            assert torch.count_nonzero(g[[0,1,3]]) == 0
            active_total += float(g[2].abs().sum())
    assert active_total > 0, "Native loss never reached active scorer rows"
    checks["only_active_scorer_row_receives_native_gradient"] = True

    try:
        candidate(x,edges)
    except RuntimeError as error:
        assert "member context" in str(error)
    else:
        raise AssertionError("Unscoped forward did not fail")
    with controller.member_context(1):
        try:
            with controller.member_context(3):
                raise ValueError("fabricated exception")
        except ValueError:
            pass
        assert all(s.member == 1 for _,_,s in controller.sites)
    assert all(s.member is None for _,_,s in controller.sites)
    checks["unscoped_failure_and_nested_exception_restoration"] = True

    # Engineering score intervention: fixed values, one private scorer row only.
    conv = candidate.local_convs[0]
    h = torch.arange(48, dtype=torch.float64).reshape(6,8)/13 - 1.
    snapshots = []
    for m in range(4):
        with controller.member_context(m), factors.member_context(candidate,m):
            values = conv.lin(h).detach().clone()
            _,(_,alpha) = conv(h,edges,return_attention_weights=True)
            snapshots.append((values,alpha.detach().clone()))
    with torch.no_grad():
        conv.parametrizations.att_src.original[1,0,0,0].add_(2.)
    changed = False
    for m in range(4):
        with controller.member_context(m), factors.member_context(candidate,m):
            assert torch.equal(conv.lin(h),snapshots[m][0])
            _,(returned_edges,alpha) = conv(h,edges,return_attention_weights=True)
            assert torch.equal(returned_edges,edges), "Support/duplicates/self edges changed"
            if m == 1:
                changed = not torch.equal(alpha,snapshots[m][1])
            else:
                assert torch.equal(alpha,snapshots[m][1])
    assert changed, "Known scorer perturbation did not change attention"
    checks["scorer_only_changes_active_attention_not_values_or_support"] = True

    restored = build()
    restored_controller = hook.install_private_local_attention(restored,4)
    restored.load_state_dict(candidate.state_dict(),strict=True)
    restored.eval(); restored._global = candidate._global
    for m in range(4):
        assert torch.equal(route(restored,m,restored_controller),route(candidate,m,controller))
    checks["strict_state_dict_roundtrip_after_rebuild"] = True
    try:
        hook.install_private_local_attention(candidate,4)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated installation accepted")
    checks["repeat_installation_rejected"] = True

    # Shape/count constructor check of the actual recipe; no graph forward.
    full = native.Polynormer(in_channels=300,hidden_channels=512,out_channels=10,
                            local_layers=7,global_layers=2,heads=1,beta=-1,pre_ln=False)
    full.reset_parameters(); factors.install_factors(full,4)
    before_counts = factors.factor_counts(full)
    full_controller = hook.install_private_local_attention(full,4)
    metadata = full_controller.metadata()
    after_counts = factors.factor_counts(full)
    assert after_counts["private_factors"] == before_counts["private_factors"]
    assert metadata["original_shared_scorer_scalars"] == 7168
    assert metadata["private_scorer_scalars"] == 28672
    assert metadata["additional_prediction_parameters"] == 21504
    metadata.update(original_BE_private_factor_scalars=before_counts["private_factors"],
                    candidate_BE_private_factor_scalars=after_counts["private_factors"],
                    total_original_private_scalars=before_counts["private_factors"],
                    total_candidate_private_scalars=after_counts["private_factors"]+28672,
                    total_original_shared_scalars=before_counts["non_factor_parameters"],
                    total_candidate_shared_scalars=after_counts["total"]-
                        after_counts["private_factors"]-28672)
    checks["actual_recipe_scorer_shapes_and_21504_parameter_delta"] = True
    return {"schema":"private-local-attention-CPU-engineering-fixture-v1",
            "torch":torch.__version__,"torch_geometric":torch_geometric.__version__,
            "device":"cpu","fixture_dtype":"float64","fabricated_nodes":6,
            "checks":checks,"passed_checks":len(checks),
            "actual_recipe_parameter_metadata":metadata,
            "source_pins":{k:{"path":v[0],"sha256":v[1]} for k,v in PINS.items()},
            "scientific_model_or_score_access":False,"training_updates":0,
            "optimizer_constructions":1,"optimizer_steps":0,"datasets_or_checkpoints_loaded":0,
            "GPU_calls":0,"performance_evidence":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",type=Path,default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    print(json.dumps(run(args.phase),indent=2))
