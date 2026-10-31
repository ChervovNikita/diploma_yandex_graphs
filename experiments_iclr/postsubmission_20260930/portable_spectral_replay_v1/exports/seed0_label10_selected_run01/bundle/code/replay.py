"""Portable CPU checkpoint replay. All evidence/output paths are bundle-contained.

This verifies an already fitted checkpoint and its saved selection trajectory.
It does not reproduce training or establish a new predictive improvement.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parents[1]
CACHE_NAMES = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS",
               "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
               "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            value.update(part)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def contained(relative):
    require(type(relative) is str and not Path(relative).is_absolute(), "Require a bundle-relative path")
    path = (BUNDLE / relative).resolve()
    require(path.is_relative_to(BUNDLE) and path != BUNDLE, "Path escapes the bundle")
    return path


def configure_environment(bundle):
    bundle=bundle.resolve()
    for name in CACHE_NAMES:
        path = (bundle / "runtime" / name.lower()).resolve()
        require(path.is_relative_to(bundle),"Cache/temp symlink escapes the bundle")
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_identity(tensor):
    value=tensor.detach().cpu().contiguous()
    metadata={"dtype":str(value.dtype),"shape":list(value.shape),"byteorder":sys.byteorder}
    raw=value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
    metadata["sha256"]=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(",",":")).encode()+raw).hexdigest()
    return metadata


def rebuild(state, code):
    import torch
    require(type(state) is dict and state and all(type(k) is str and isinstance(v,torch.Tensor) for k,v in state.items()),
            "Selected state must be a clean string-to-tensor mapping")
    require(all(v.device.type=="cpu" and v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()),
            "Require finite float32 CPU state tensors")
    models=load_module("portable_original_gnnm_models",code/"models.py")
    adapters=load_module("portable_original_spectral_adapters",code/"adapters.py")
    source=models.TABMModel(model_name="SAGE",num_layers=3,input_dim=50,hidden_dim=128,output_dim=121,
        hidden_dim_multiplier=1,num_heads=4,normalization="LayerNorm",dropout=.2,tabm_inits=4,device="cpu")
    bases={f"residual_modules.{i}.module.conv.lin_l.weight":
           state[f"source.residual_modules.{i}.module.conv.lin_l.U"] for i in range(3)}
    model=adapters.SourceSAGEEnsemble(source,bases=bases)
    require(torch.equal(model.C,state["C"]),"Saved member contrasts differ from exact Helmert construction")
    model.load_state_dict(state,strict=True)
    require(all(torch.equal(v,state[k]) for k,v in model.state_dict().items()),"Clean state did not reconstruct exactly")
    return model.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/replay_run01.json")
    args=parser.parse_args()
    output=contained(args.output)
    require(output.is_relative_to(BUNDLE/"results") and not output.exists(), "Use a new bundle/results output")
    # Confinement is installed before Torch/PyG imports; credentials are never recorded.
    configure_environment(BUNDLE)
    import torch
    import torch_geometric
    from torch_geometric.nn import SAGEConv
    from types import SimpleNamespace
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32)
    manifest=json.loads(contained("manifest.json").read_text())
    immutable={str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob("*") if p.is_file()
               and not p.is_relative_to(BUNDLE/"runtime") and not p.is_relative_to(BUNDLE/"results")
               and p!=BUNDLE/"manifest.json"}
    require(set(manifest)==immutable,"Immutable bundle file coverage differs")
    for name,digest in manifest.items():
        require(sha(contained(name))==digest,"Bundle artifact changed: "+name)
    metadata=json.loads(contained("metadata.json").read_text())
    require(metadata["schema"]=="gnnm-portable-spectral-checkpoint-v1" and metadata["test_inputs_or_targets_included"] is False,
            "Unrecognized portable validation-only schema")
    require(metadata["validation_scope"] in ("official_PPI_validation", "synthetic_engineering"),"Wrong graph scope")
    expected=metadata["runtime"]
    runtime={"python":sys.version.split()[0],"torch":torch.__version__,"torch_geometric":torch_geometric.__version__,
             "numpy":importlib.metadata.version("numpy"),
             "SAGEConv_sha256":hashlib.sha256(inspect.getsource(SAGEConv).encode()).hexdigest()}
    require(runtime["torch"].split("+")[0]==expected["torch"].split("+")[0]
            and runtime["torch_geometric"]==expected["torch_geometric"]
            and runtime["numpy"]==expected["numpy"]
            and runtime["SAGEConv_sha256"]==expected["SAGEConv_sha256"],
            "Require recorded Torch core/PyG/SAGEConv versions; CPU build suffix may differ")
    torch.use_deterministic_algorithms(expected["deterministic_algorithms"])
    graphs=torch.load(contained("data/validation_graphs.pt"),map_location="cpu",weights_only=True)
    require(type(graphs) is list and len(graphs)==2,"Require the two validation graph dictionaries")
    inputs=[]
    for index,g in enumerate(graphs):
        require(type(g) is dict and set(g)=={"x","edge_index","y"},"Graph dictionary contains undeclared data")
        require(g["x"].dtype==torch.float32 and g["y"].dtype==torch.float32 and g["edge_index"].dtype==torch.long,
                "Wrong graph tensor dtype")
        n=g["x"].shape[0]
        require(g["x"].shape==(n,50) and g["y"].shape==(n,121) and n>0
                and g["edge_index"].ndim==2 and g["edge_index"].shape[0]==2,"Wrong graph tensor shape")
        require(torch.isfinite(g["x"]).all() and ((g["y"]==0)|(g["y"]==1)).all(),"Nonfinite/nonbinary graph data")
        require(not g["edge_index"].numel() or int(g["edge_index"].min())>=0 and int(g["edge_index"].max())<n,
                "Edge index outside graph")
        require({k:tensor_identity(v) for k,v in g.items()}==metadata["validation_graph_tensor_identities"][index],
                "Validation graph identity differs")
        inputs.append(SimpleNamespace(**g,num_nodes=n))
    code=contained("code/replay.py").parent
    metric_module=load_module("portable_exact_spectral_metrics",code/"metrics.py")
    targets=[g["y"] for g in graphs]
    rows=list(csv.DictReader(contained("evidence/validation_trace.csv").open()))
    require([int(r["epoch"]) for r in rows]==list(range(101)),"Require all101 registered validation epochs")
    actual_rows=[];maximum_metric_error=0.
    for row in rows:
        epoch=int(row["epoch"])
        logits=torch.load(contained(f"evidence/epochs/epoch{epoch:03d}.pt"),map_location="cpu",weights_only=True)
        values=metric_module.metrics(logits,targets)
        maximum_metric_error=max(maximum_metric_error,max(abs(v-float(row[k])) for k,v in values.items()))
        actual_rows.append({"epoch":epoch,**values})
    require(maximum_metric_error<=2e-5,"Retained logits do not reproduce saved metrics")
    key=lambda r:(float(r["micro_f1"]),-float(r["pooled_bce"]),-int(r["epoch"]))
    trace_best=max(rows,key=key);recomputed_best=max(actual_rows,key=key)
    selected=metadata["selected"]
    require(int(trace_best["epoch"])==recomputed_best["epoch"]==selected["epoch"],
            "Recomputed/trace checkpoint selection differs")
    for k,v in selected["metrics"].items():
        require(abs(v-float(trace_best[k]))<=1e-12,"Selected metadata differs from saved trace")
    kept=torch.load(contained("evidence/selected_member_logits.pt"),map_location="cpu",weights_only=True)
    named=torch.load(contained(f"evidence/epochs/epoch{selected['epoch']:03d}.pt"),map_location="cpu",weights_only=True)
    require(len(kept)==2 and all(torch.equal(a,b) for a,b in zip(kept,named)),"Selected logits differ from selected epoch")
    state=torch.load(contained("state/selected_state.pt"),map_location="cpu",weights_only=True)
    require({k:tensor_identity(v) for k,v in state.items()}==metadata["selected_state_tensor_identities"],"State tensor identities differ")
    model=rebuild(state,code)
    buffers=canonical_hash({k:tensor_identity(v) for k,v in model.named_buffers()})
    require(buffers==metadata["frozen_buffer_sha256"],"Frozen bases/member contrasts changed")
    with torch.no_grad():fresh=[model(g,g.x).detach().cpu() for g in inputs]
    require(all(torch.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(fresh,kept)),"Fresh selected predictions differ")
    fresh_values=metric_module.metrics(fresh,targets)
    changed=sum(int(((a.mean(0)>=0)!=(b.mean(0)>=0)).sum()) for a,b in zip(fresh,kept))
    require(not torch.cuda.is_initialized(),"Replay unexpectedly initialized CUDA")
    report={"status":"PASS","scope":"selected-checkpoint and complete validation-selection replay; not fresh training reproduction",
            "validation_scope":metadata["validation_scope"],"selected_epoch":selected["epoch"],"runtime":runtime,
            "immutable_manifest_sha256":sha(contained("manifest.json")),"maximum_stored_metric_error":maximum_metric_error,
            "maximum_fresh_logit_error":max(float((a-b).abs().max()) for a,b in zip(fresh,kept)),
            "threshold_decisions_changed":changed,"fresh_cpu_metrics":fresh_values,
            "reported_selected_metrics":selected["metrics"],"cross_runtime_atol_rtol":[2e-4,2e-4],
            "weights_only_tensor_loading":True,"optimizer_steps":0,"fresh_training_reproduction":False,
            "author_calibration_or_gradient_artifacts_loaded":False,"test_inputs_or_targets_loaded":False,
            "CUDA_context_created":False,"cache_and_temp_paths":{k:str(Path(os.environ[k]).resolve().relative_to(BUNDLE)) for k in CACHE_NAMES}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps({"status":"PASS","selected_epoch":selected["epoch"],
                      "maximum_fresh_logit_error":report["maximum_fresh_logit_error"],"fresh_training_reproduction":False}))


if __name__=="__main__":
    main()
