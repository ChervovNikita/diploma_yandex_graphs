"""Synthetic-only CPU engineering gate; never opens a dataset or checkpoint."""
import argparse
from dataclasses import replace
import math
import platform
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch_geometric.data import Data
import torch_geometric
import ogb

from lookahead_contract import (SOURCE_ONLY, RECIPE, RULE, PLACEMENTS, LOSSES,
                                sources, write_new, refuse_real_input_admission)
from lookahead_core import (Unit, source_module, objective_values, checked_units,
                            initial_models, paired_lookahead, mechanism_rule,
                            state_hash, tensor_hash, update_schedule)


def rejects(function, expected=(ValueError,)):
    try:
        function()
    except expected as exc:
        return str(exc)
    raise AssertionError("Required refusal did not happen")


def gradient_gate():
    """Check graph-equal reduction and both analytic logit derivatives."""
    z = torch.linspace(-1.7, 2.3, 36, dtype=torch.float64).reshape(4, 3, 3)
    y = torch.tensor([[1., float("nan"), float("nan")],
                      [0., 1., float("nan")], [1., 0., 1.]], dtype=torch.float64)
    observed = torch.isfinite(y)
    safe = torch.where(observed, y, torch.zeros_like(y))
    counts = observed.sum(-1)
    results = {}
    for family in LOSSES:
        logits = z.clone().requires_grad_(True)
        values, rows = objective_values(logits, y)
        actual = torch.autograd.grad(values[family], logits)[0]
        if family == "own":
            expected = (torch.sigmoid(z) - safe[None]) * observed[None]
            expected = expected / (4 * 3 * counts[None, :, None])
            explicit_rows = torch.stack([
                torch.stack([(torch.nn.functional.softplus(z[m, g, observed[g]])
                              - y[g, observed[g]] * z[m, g, observed[g]]).mean()
                             for m in range(4)]).mean() for g in range(3)])
        else:
            pooled = z.mean(0)
            expected = ((torch.sigmoid(pooled) - safe) * observed
                        / (4 * 3 * counts[:, None]))[None].expand_as(z)
            explicit_rows = torch.stack([
                (torch.nn.functional.softplus(pooled[g, observed[g]])
                 - y[g, observed[g]] * pooled[g, observed[g]]).mean()
                for g in range(3)])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(rows[family], explicit_rows, rtol=1e-12, atol=1e-12)
        assert torch.equal(actual[:, ~observed], torch.zeros_like(actual[:, ~observed]))
        results[family] = {"maximum_logit_gradient_error":float((actual-expected).abs().max()),
                           "maximum_graph_loss_error":float((rows[family]-explicit_rows).abs().max()),
                           "observed_targets_per_graph":counts.tolist()}
    bad_logits=z.clone();bad_logits[0,0,0]=float("nan")
    hidden=y.clone();hidden[0]=float("nan")
    infinite=y.clone();infinite[0,0]=float("inf")
    nonbinary=y.clone();nonbinary[0,0]=.3
    for logits, labels in ((bad_logits,y),(z,hidden),(z,infinite),(z,nonbinary)):
        rejects(lambda logits=logits,labels=labels:objective_values(logits,labels))
    return {"analytic_checks":results,"invalid_cases_refused":4,
            "scope":"analytic logit fixture; not molecular evidence"}


def graph_fixture(index):
    n = 4 + index % 2
    x = torch.zeros((n,9), dtype=torch.long)
    x[:,0] = (torch.arange(n) + index) % 5
    x[:,1] = index % 2
    edges=[]
    for i in range(n):
        j=(i+1)%n;edges.extend([(i,j),(j,i)])
    edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()
    edge_attr=torch.zeros((len(edges),3),dtype=torch.long)
    edge_attr[:,0]=(torch.arange(len(edges))+index)%2
    return Data(x=x,edge_index=edge_index,edge_attr=edge_attr,
                y=torch.tensor([[float(index%2)]],dtype=torch.float32))


def units(prefix, count, offset):
    return [Unit(f"{prefix}:unit:{i:02d}",f"{prefix}:group:{i:02d}",
                 graph_fixture(offset+i),synthetic=True) for i in range(count)]


def paired_gate():
    torch.manual_seed(6019)
    module=source_module()
    common=module.MolecularProjectorEnsemble(
        module.gine_options(output_dim=1,hidden_dim=128,num_layers=3,
                            dropout=.1,hidden_dim_multiplier=2),members=4)
    common.train()
    # Sentinel gradients catch mutations that a state_dict hash would miss.
    gradients=[]
    for parameter in common.parameters():
        parameter.grad=torch.full_like(parameter,.03125)
        gradients.append((parameter,parameter.grad,parameter.grad.detach().clone()))
    modes=[(m,m.training) for m in common.modules()]
    update=units("Q-update",8,0);screen=units("Q-screen",6,20)
    source_groups={"P:synthetic-group"}
    start=state_hash(common)
    state=torch.random.get_rng_state().clone()
    result=paired_lookahead(common,update,screen,source_groups,seed=0,
                            fixture_mode=True,include_null=True)
    assert torch.equal(state,torch.random.get_rng_state())
    assert state_hash(common)==start
    for parameter,reference,values in gradients:
        assert parameter.grad is reference and torch.equal(parameter.grad,values)
    assert all(m.training is before for m,before in modes)
    rows=result["trajectories"]
    assert len(rows)==10
    for family in LOSSES:
        tied=rows[family+"_tied"];null=rows[family+"_null_tied_copy"]
        assert tied["final_state_sha256"]==null["final_state_sha256"]
        assert tied["steps"]==null["steps"]
        assert tied["final_screen"]==null["final_screen"]
        assert null["paired_pooled_bce_gain_vs_actual_tied_after_K"]==0.
        assert all(g==0. for g in null["paired_graph_gains_vs_actual_tied_after_K"])
        assert tied["final_state_sha256"]!=start
        assert tied["initial_screen"]["pooled_bce"]!=tied["final_screen"]["pooled_bce"]
        for placement in PLACEMENTS:
            row=rows[family+"_"+placement]
            assert row["initial_screen"]==tied["initial_screen"]
            assert row["optimizer_step_counts"]==[8.]
            assert [r["unit_ids"] for r in row["steps"]]==[r["unit_ids"] for r in tied["steps"]]
            assert [r["dropout_seed"] for r in row["steps"]]==[r["dropout_seed"] for r in tied["steps"]]
            expected=tied["final_screen"]["pooled_bce"]-row["final_screen"]["pooled_bce"]
            assert row["paired_pooled_bce_gain_vs_actual_tied_after_K"]==expected
            assert row["final_screen"]["unit_ids"]==sorted(u.unit_id for u in screen)
            per_graph=row["paired_graph_gains_vs_actual_tied_after_K"]
            assert math.isclose(math.fsum(per_graph)/len(per_graph),expected,rel_tol=0.,abs_tol=1e-12)
        private_counts=[rows[family+"_"+p]["parameter_count"] for p in PLACEMENTS[1:]]
        assert len(set(private_counts))==1
        assert private_counts[0]-tied["parameter_count"]==3*tied["private_block_parameter_count"]
    bad_screen=[replace(screen[0],group_id=update[0].group_id),*screen[1:]]
    reused=[replace(screen[0],unit_id=update[0].unit_id),*screen[1:]]
    nontrain=[replace(screen[0],split="valid"),*screen[1:]]
    unmarked=[replace(screen[0],synthetic=False),*screen[1:]]
    for bad in (bad_screen,reused,nontrain,unmarked):
        rejects(lambda bad=bad:checked_units(update,bad,source_groups,True))
    rejects(lambda:checked_units(update,screen,{update[0].group_id},True))
    rejects(lambda:paired_lookahead(common,update,screen,source_groups,0,fixture_mode=False))
    # A partial update minibatch remains partial; coverage is never invented.
    wider=units("Q-large-synthetic",130,50)
    schedule=update_schedule(wider,seed=1)
    assert len(schedule)==8 and [len(s) for s in schedule]==[128,2]*4
    assert {u.unit_id for s in schedule for u in s}=={u.unit_id for u in wider}
    result["gate_assertions"]={"exact_null_for_both_training_losses":True,
        "actual_tied_after_K_differs_from_unchanged_source":True,
        "initial_function_and_capacity_matching":True,
        "source_gradient_references_and_values_preserved":True,
        "source_modes_and_external_CPU_RNG_preserved":True,
        "all_screening_graphs_scored":True,"invalid_scope_cases_refused":6,
        "partial_minibatch_schedule_checked":True}
    return result


def rule_gate():
    blank={(s,f,l,p):0. for s in range(3) for f in (.1,1.)
           for l in LOSSES for p in PLACEMENTS}
    result=mechanism_rule(blank)
    assert all(v["abstain"] for v in result["loss_families"].values())
    good=dict(blank)
    for s in range(3):
        good[s,.1,"pooled","first"]=.002
        good[s,.1,"pooled","last"]=.003
    result=mechanism_rule(good)
    assert result["loss_families"]["pooled"]["candidate_for_separate_full_predictive_study"]=="last"
    assert result["loss_families"]["own"]["abstain"]
    tied_candidates=dict(blank)
    for s in range(3):
        for p in ("first","middle","last"):
            tied_candidates[s,.1,"own",p]=.002
    result=mechanism_rule(tied_candidates)
    assert result["loss_families"]["own"]["candidate_for_separate_full_predictive_study"]=="first"
    too_small=dict(blank)
    for s in range(3):too_small[s,.1,"own","first"]=RULE["primary_each_seed_minimum_gain"]
    assert mechanism_rule(too_small)["loss_families"]["own"]["abstain"]
    one_seed_fails=dict(tied_candidates)
    for p in PLACEMENTS[1:]:one_seed_fails[2,.1,"own",p]=0.
    assert mechanism_rule(one_seed_fails)["loss_families"]["own"]["abstain"]
    adverse=dict(tied_candidates)
    for p in PLACEMENTS[1:]:adverse[1,1.,"own",p]=-RULE["full_label_each_seed_maximum_harm"]
    assert mechanism_rule(adverse)["loss_families"]["own"]["abstain"]
    missing=dict(blank);missing.pop(next(iter(missing)))
    false_tied=dict(blank);false_tied[0,.1,"own","tied"]=1e-12
    nonfinite=dict(blank);nonfinite[0,.1,"own","first"]=float("nan")
    for invalid in (missing,false_tied,nonfinite):rejects(lambda invalid=invalid:mechanism_rule(invalid))
    # Ensure admission refuses before inspecting even the binding path.
    with patch.object(Path,"open",side_effect=AssertionError("Source-only refusal opened an input")):
        refusal=rejects(lambda:refuse_real_input_admission("nonexistent_future_binding.json"))
    return {"scope":"synthetic arithmetic; not MolHIV evidence",
        "complete_48_arm_requirement":True,"tied_gain_exact_zero_requirement":True,
        "practical_and_numerical_abstention":True,"all_seed_requirement":True,
        "full_label_harm_guard":True,"loss_families_separate":True,
        "deterministic_placement_ties":True,"no_input_opened_by_admission":True,
        "admission_refusal":refusal}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    # No dataset provider is imported or constructed. Fresh process required.
    if torch.cuda.is_initialized():raise ValueError("CPU gate started with CUDA initialized")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not SOURCE_ONLY:raise ValueError("This gate is for source-only v1")
    frozen=sources()
    result={"schema":"gnnm-molhiv-lookahead-synthetic-CPU-gate-v1",
        "scope":"synthetic engineering and analytic checks only","source_only":True,
        "environment":{"python":sys.version,"platform":platform.platform(),
            "torch":torch.__version__,"torch_geometric":torch_geometric.__version__,
            "ogb":ogb.__version__,"threads":torch.get_num_threads(),
            "interop_threads":torch.get_num_interop_threads()},
        "sources":frozen,"recipe":RECIPE,"rule":RULE,
        "logit_autograd":gradient_gate(),"paired_actual_AdamW":paired_gate(),
        "fixed_rule":rule_gate(),"dataset_loaded":False,"checkpoint_loaded":False,
        "accuracy_or_superiority_evidence":False,"passed":True}
    if frozen!=sources():raise ValueError("Source changed during CPU gate")
    if torch.cuda.is_initialized():raise ValueError("CPU gate created a CUDA context")
    write_new(args.output,result)
    print("Synthetic CPU gate passed; real-input admission remains closed.")


if __name__=="__main__":main()
