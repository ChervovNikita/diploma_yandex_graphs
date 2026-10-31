"""Synthetic GPU paired-null gate; call only through authorized one-GPU wrapper."""
import argparse
import os
from pathlib import Path
import sys

# The wrapper must set this before importing torch or creating a CUDA context.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
    raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the approved child command environment")

import torch
from lookahead_contract import sources,sha,write_new,HERE
from lookahead_core import source_module,state_hash,initial_models
from gpu_lookahead_core_v2 import paired_gpu_lookahead,SOURCE_ONLY
from verify_cpu import gradient_gate,units,rule_gate,rejects


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if not SOURCE_ONLY:raise ValueError("Synthetic v2 source-only gate required")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(6019)
    module=source_module()
    common=module.MolecularProjectorEnsemble(module.gine_options(
        output_dim=1,hidden_dim=128,num_layers=3,dropout=.1,hidden_dim_multiplier=2),members=4).to("cuda:0")
    common.train();gradients=[]
    for parameter in common.parameters():
        parameter.grad=torch.full_like(parameter,.03125)
        gradients.append((parameter,parameter.grad,parameter.grad.detach().clone()))
    modes=[(m,m.training) for m in common.modules()]
    initial,block_count=initial_models(common)
    initial_tied_count=sum(p.numel() for p in common.parameters())
    for family in ("own","pooled"):
        for placement,index in (("first",0),("middle",1),("last",2)):
            model,optimizer=initial[family+"_"+placement]
            copies=model.private_blocks[str(index)]
            original=common.blocks[index].state_dict()
            assert optimizer.state=={}
            assert sum(p.numel() for p in model.parameters())==initial_tied_count+3*block_count
            for name,source_tensor in original.items():
                tensors=[c.state_dict()[name] for c in copies]
                assert all(torch.equal(t,source_tensor) for t in tensors)
                assert len({source_tensor.data_ptr(),*(t.data_ptr() for t in tensors)})==5
    del initial
    update=units("Q-update",8,0);screen=units("Q-screen",6,20)
    frozen={**sources(),**{str((HERE/n).relative_to(HERE.parent)):sha(HERE/n)
        for n in ("gpu_lookahead_core_v2.py","verify_gpu_v2.py")}}
    state=state_hash(common);cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(0).clone()
    report,artifacts=paired_gpu_lookahead(common,update,screen,{"P:synthetic-group"},0,
                                          fixture_mode=True,include_null=True)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(0))
    assert state_hash(common)==state
    assert all(m.training is before for m,before in modes)
    for parameter,reference,values in gradients:
        assert parameter.grad is reference and torch.equal(parameter.grad,values)
    rows=report["trajectories"];assert len(rows)==10
    for family in ("own","pooled"):
        tied=rows[family+"_tied"];null=rows[family+"_null_tied_copy"]
        assert tied["final_state_sha256"]==null["final_state_sha256"]
        assert tied["steps"]==null["steps"] and tied["final_screen"]==null["final_screen"]
        assert null["paired_pooled_bce_gain_vs_actual_tied_after_K"]==0.
        assert tied["final_state_sha256"]!=state
        assert tied["initial_screen"]["pooled_bce"]!=tied["final_screen"]["pooled_bce"]
        for placement in ("tied","first","middle","last"):
            row=rows[family+"_"+placement]
            assert row["initial_screen"]==tied["initial_screen"]
            assert row["optimizer_step_counts"]==[8.]
            assert [r["unit_ids"] for r in row["steps"]]==[r["unit_ids"] for r in tied["steps"]]
            assert [r["dropout_seed"] for r in row["steps"]]==[r["dropout_seed"] for r in tied["steps"]]
            assert row["final_screen"]["unit_ids"]==sorted(u.unit_id for u in screen)
        # Registered state and optimizer identity must cover complete private copies.
        for placement,index in (("first",0),("middle",1),("last",2)):
            model,optimizer,_=artifacts[family+"_"+placement]
            copies=model.private_blocks[str(index)]
            source_block=common.blocks[index]
            original_names=set(source_block.state_dict())
            assert len(copies)==4 and isinstance(model.blocks[index],torch.nn.Identity)
            assert all(set(c.state_dict())==original_names for c in copies)
            for name in original_names:
                # Updates may differ, but copy shapes, dtypes and storage remain separate.
                values=[c.state_dict()[name] for c in copies]
                assert all(v.shape==source_block.state_dict()[name].shape and
                    v.dtype==source_block.state_dict()[name].dtype for v in values)
                assert len({v.data_ptr() for v in values})==4
            parameters=list(model.parameters())
            opt_parameters=[p for g in optimizer.param_groups for p in g["params"]]
            assert len(opt_parameters)==len(parameters)
            assert {id(p) for p in parameters}=={id(p) for p in opt_parameters}
            private_row=rows[family+"_"+placement]
            assert private_row["private_block_parameter_count"]==block_count
            assert private_row["parameter_count"]==initial_tied_count+3*block_count
    rejects(lambda:paired_gpu_lookahead(common,update,screen,{"P:synthetic-group"},0,fixture_mode=False))
    result={"schema":"gnnm-GPU-lookahead-source-only-synthetic-gate-v2",
        "source_only":True,"scope":"synthetic GPU engineering and analytic fixtures only",
        "sources":frozen,"python":sys.version,"logit_autograd":gradient_gate(),
        "fixed_rule":rule_gate(),"actual_GPU_AdamW":report,
        "exact_GPU_paired_null":True,"full_block_storage_optimizer_identity_checked":True,
        "source_gradient_modes_CPU_and_GPU_RNG_preserved":True,
        "dataset_loaded":False,"checkpoint_loaded":False,"CPU_molecular_sibling_required":False,
        "passed":True}
    now={**sources(),**{str((HERE/n).relative_to(HERE.parent)):sha(HERE/n)
        for n in ("gpu_lookahead_core_v2.py","verify_gpu_v2.py")}}
    if now!=frozen:raise ValueError("Sources changed during GPU gate")
    write_new(args.output,result)
    print("Synthetic GPU paired-null gate passed; real admission remains closed.")


if __name__=="__main__":main()
