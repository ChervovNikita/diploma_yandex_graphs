"""Relocated clean-tensor checkpoint replay gate, tied/rank-zero and nonzero bases.

Synthetic engineering only: no dataset, optimizer step or predictive claim.
"""
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/"bundle_template/code")]
import export_verified_fit as exporter
import replay as portable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    portable.require(output.is_relative_to(HERE) and not output.exists(),"Use a new synthetic-gate folder inside the packet")
    portable.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Require hidden CUDA devices")
    output.mkdir(parents=True)
    portable.configure_environment(output)
    import torch
    import torch_geometric
    import numpy
    from torch_geometric.nn import SAGEConv
    from types import SimpleNamespace
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    code=HERE/"bundle_template/code"
    models=portable.load_module("portable_synthetic_original_models",code/"models.py")
    adapters=portable.load_module("portable_synthetic_original_adapters",code/"adapters.py")
    metrics=portable.load_module("portable_synthetic_exact_metrics",code/"metrics.py")
    runtime={"python":sys.version.split()[0],"torch":torch.__version__,"torch_geometric":torch_geometric.__version__,
             "numpy":numpy.__version__,"SAGEConv_sha256":__import__("hashlib").sha256(inspect.getsource(SAGEConv).encode()).hexdigest(),
             "deterministic_algorithms":torch.are_deterministic_algorithms_enabled()}
    source_paths={name:code/name for name in ("models.py","adapters.py","replay.py","metrics.py")}
    graphs=[]
    for n in (5,7):
        x=torch.linspace(-.9,1.1,n*50).reshape(n,50)
        edges=torch.tensor([[0,0,1,2,3],[1,1,2,0,3]],dtype=torch.long)
        y=((torch.arange(n)[:,None]+torch.arange(121)[None,:])%3==0).float()
        graphs.append({"x":x,"edge_index":edges,"y":y})
    reports={}
    for name,ranks in [("tied_rank_zero",[0,0,0]),("nonzero_basis",[1,2,3])]:
        folder=output/name;folder.mkdir()
        torch.manual_seed(901)
        source=models.TABMModel(model_name="SAGE",num_layers=3,input_dim=50,hidden_dim=128,output_dim=121,
            hidden_dim_multiplier=1,num_heads=4,normalization="LayerNorm",dropout=.2,tabm_inits=4,device="cpu")
        bases={f"residual_modules.{i}.module.conv.lin_l.weight":torch.eye(128)[:,:rank] for i,rank in enumerate(ranks)}
        model=adapters.SourceSAGEEnsemble(source,bases=bases).eval()
        with torch.no_grad():
            for p in model.named_private_coefficients().values():
                if p.numel():p.copy_(torch.linspace(-.003,.003,p.numel()).reshape_as(p))
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rebuilt=portable.rebuild(state,code)
        portable.require(all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in state.items()),"Reconstruction changed synthetic state")
        with torch.no_grad():logits=[model(SimpleNamespace(**g,num_nodes=len(g["x"])),g["x"]).detach() for g in graphs]
        values=metrics.metrics(logits,[g["y"] for g in graphs])
        trace=folder/"synthetic_trace.csv"
        with trace.open("x",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["epoch",*values]);writer.writeheader()
            for epoch in range(101):writer.writerow({"epoch":epoch,**values})
        metadata={"schema":"gnnm-portable-spectral-checkpoint-v1","validation_scope":"synthetic_engineering",
            "test_inputs_or_targets_included":False,"fresh_training_reproduction":False,"runtime":runtime,
            "selected":{"epoch":0,"metrics":values},
            "frozen_buffer_sha256":portable.canonical_hash({k:portable.tensor_identity(v) for k,v in model.named_buffers()}),
            "synthetic_fixture":"untrained complete M4/h128/L3 state with fixed rank and optional nonzero private coefficients"}
        bundle=folder/"bundle"
        exporter.populate(bundle,state,graphs,lambda epoch:[v.clone() for v in logits],trace,metadata,source_paths)
        relocated=folder/"relocated"/"arbitrary_directory_name";relocated.parent.mkdir()
        shutil.move(str(bundle),relocated)
        command=[sys.executable,str(relocated/"code/replay.py"),"--output","results/synthetic_replay_run01.json"]
        with (folder/"portable_stdout_stderr.log").open("x") as stream:
            result=subprocess.run(command,cwd=relocated,stdout=stream,stderr=subprocess.STDOUT,check=False)
        portable.require(result.returncode==0,"Relocated synthetic checkpoint replay failed; retain its log")
        receipt=json.loads((relocated/"results/synthetic_replay_run01.json").read_text())
        portable.require(receipt["status"]=="PASS" and receipt["validation_scope"]=="synthetic_engineering"
            and receipt["selected_epoch"]==0 and receipt["weights_only_tensor_loading"] is True,
            "Unexpected synthetic portable receipt")
        # Corrupt the saved metric while re-sealing file hashes; numerical replay must reject it.
        trace_path=relocated/"evidence/validation_trace.csv";old=trace_path.read_text()
        rows=list(csv.DictReader(old.splitlines()));rows[0]["pooled_bce"]=str(float(rows[0]["pooled_bce"])+.1)
        with trace_path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        exporter.seal_bundle(relocated)
        reject=subprocess.run([sys.executable,str(relocated/"code/replay.py"),"--output","results/tamper_must_fail.json"],
            cwd=relocated,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
        (folder/"tamper_rejection.log").write_text(reject.stdout)
        portable.require(reject.returncode!=0 and not (relocated/"results/tamper_must_fail.json").exists(),"Metric tamper was accepted")
        trace_path.write_text(old);exporter.seal_bundle(relocated)
        reports[name]={"ranks":ranks,"exact_state_reconstruction":True,"relocated_portable_replay":"PASS",
            "maximum_fresh_logit_error":receipt["maximum_fresh_logit_error"],
            "threshold_decisions_changed":receipt["threshold_decisions_changed"],"resealed_metric_tamper_rejected":True}
    portable.require(not torch.cuda.is_initialized(),"Synthetic gate initialized CUDA")
    report={"status":"PASS","scope":"synthetic tied/rank-zero and nonzero-state relocation plus101-epoch selection replay",
        "cases":reports,"runtime":runtime,"dataset_loads":0,"optimizer_steps":0,"predictive_fit":False,
        "test_inputs_or_targets_loaded":False,"fresh_training_reproduction":False,"performance_evidence":False,
        "packet_source_sha256":{str(p.relative_to(HERE)):portable.sha(p) for p in sorted(HERE.rglob("*")) if p.is_file()
            and not p.is_relative_to(output) and p.name!="artifact_manifest.json"}}
    (output/"SYNTHETIC_CPU_GATE.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","cases":list(reports),"dataset_loads":0,"optimizer_steps":0}))


if __name__=="__main__":main()
