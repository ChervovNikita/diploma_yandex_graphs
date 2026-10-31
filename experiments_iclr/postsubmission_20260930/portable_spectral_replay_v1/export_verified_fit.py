"""Export one complete, author-replayed CPU-v3 fit; never export a running fit.

Only explicitly whitelisted tensor evidence and unchanged model code enter
the public bundle. No processed dataset pickle, test split, calibration
archive, optimizer state, credentials or author path is copied.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
sys.path.insert(0,str(HERE/"bundle_template/code"))
import replay as portable


def require(ok,message):
    if not ok:raise ValueError(message)


def inside(path):
    path=Path(path).resolve()
    require(path.is_relative_to(PHASE) and path!=PHASE,"Author input must stay in the authorized phase")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    require(type(ref["path"]) is str and not Path(ref["path"]).is_absolute(),"Require a phase-relative reference")
    path=inside(PHASE/ref["path"])
    require(portable.sha(path)==ref["sha256"],"Referenced author artifact changed")
    return path


def tensor_list(path):
    import torch
    value=torch.load(path,map_location="cpu",weights_only=True)
    require(type(value) is list and len(value)==2 and all(isinstance(t,torch.Tensor) for t in value),
            "Require a clean two-validation-graph tensor list")
    return [t.detach().cpu().contiguous().clone() for t in value]


def legacy_identity(tensor):
    import torch
    value=tensor.detach().cpu().contiguous()
    metadata={"dtype":str(value.dtype),"shape":list(value.shape)}
    metadata["sha256"]=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()
        +value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return metadata


def validation_graphs(protocol,manifest):
    """Reconstruct valid arrays only; frozen tensor identities must match exactly."""
    import numpy as np
    import networkx as nx
    import torch
    import torch_geometric
    from torch_geometric.datasets import PPI
    provider=next(r for r in manifest["provider_sources"] if r["class"]=="PPI")
    actual=hashlib.sha256(inspect.getsource(PPI).encode()).hexdigest()
    require(actual==protocol["environment"]["classes"]["PPI"]==provider["sha256"],"PPI provider source differs")
    cache=inside(manifest["cache"])
    paths={}
    for name in ("valid_feats.npy","valid_labels.npy","valid_graph_id.npy","valid_graph.json"):
        relative="provider/raw/"+name
        row=next(r for r in manifest["raw_and_processed_files"] if r["relative_path"]==relative)
        path=inside(cache/relative)
        require(path.stat().st_size==row["bytes"] and portable.sha(path)==row["sha256"],"Frozen validation raw file changed")
        paths[name]=path
    x=torch.from_numpy(np.load(paths["valid_feats.npy"],allow_pickle=False)).float()
    y=torch.from_numpy(np.load(paths["valid_labels.npy"],allow_pickle=False)).float()
    ids=torch.from_numpy(np.load(paths["valid_graph_id.npy"],allow_pickle=False)).long()
    ids=ids-ids.min()
    document=read_json(paths["valid_graph.json"])
    # Explicit links/edges choice accommodates NetworkX's API default change.
    field="links" if "links" in document else "edges"
    signature=inspect.signature(nx.node_link_graph)
    kwargs={"edges":field} if "edges" in signature.parameters else {"link":field}
    graph=nx.DiGraph(nx.node_link_graph(document,**kwargs))
    require(sorted(ids.unique().tolist())==[0,1],"Require exactly the official two validation graphs")
    identities=manifest["split_summaries"]["val"]["graph_tensor_fingerprints"]
    require(len(identities)==2,"Missing official validation tensor identities")
    result=[]
    for index in range(2):
        mask=ids==index
        induced=graph.subgraph(mask.nonzero(as_tuple=False).reshape(-1).tolist())
        edges=torch.tensor(list(induced.edges),dtype=torch.long).t().contiguous()
        require(edges.numel()>0,"Unexpected empty official validation topology")
        edges=edges-edges.min()
        edges=edges[:,edges[0]!=edges[1]]
        value={"x":x[mask].contiguous(),"edge_index":edges.contiguous(),"y":y[mask].contiguous()}
        require(all(legacy_identity(t)==identities[index][k] for k,t in value.items()),
                "Raw reconstruction differs from the exact frozen validation graph; export refused")
        result.append(value)
    return result,{name:portable.sha(path) for name,path in paths.items()}


def seal_bundle(bundle):
    files={str(p.relative_to(bundle)):portable.sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()
           and not p.is_relative_to(bundle/"runtime") and not p.is_relative_to(bundle/"results") and p!=bundle/"manifest.json"}
    (bundle/"manifest.json").write_text(json.dumps(files,indent=2,sort_keys=True)+"\n")


def populate(bundle,state,graphs,logit_loader,trace,metadata,source_paths):
    import torch
    bundle.mkdir(parents=True)
    for name in ("code","data","state","evidence/epochs"):(bundle/name).mkdir(parents=True,exist_ok=True)
    for name in ("models.py","adapters.py","replay.py","metrics.py"):
        shutil.copyfile(source_paths[name],bundle/"code"/name)
    torch.save(state,bundle/"state/selected_state.pt")
    torch.save(graphs,bundle/"data/validation_graphs.pt")
    shutil.copyfile(trace,bundle/"evidence/validation_trace.csv")
    for epoch in range(101):
        logits=logit_loader(epoch)
        require(len(logits)==2,"Missing validation graph evidence")
        torch.save(logits,bundle/"evidence/epochs"/f"epoch{epoch:03d}.pt")
        if epoch==metadata["selected"]["epoch"]:torch.save(logits,bundle/"evidence/selected_member_logits.pt")
    metadata.update(selected_state_tensor_identities={k:portable.tensor_identity(v) for k,v in state.items()},
                    validation_graph_tensor_identities=[{k:portable.tensor_identity(v) for k,v in g.items()} for g in graphs])
    (bundle/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (bundle/"requirements.txt").write_text("torch=="+metadata["runtime"]["torch"].split("+")[0]+"\n"
        +"torch-geometric=="+metadata["runtime"]["torch_geometric"]+"\n"
        +"numpy=="+metadata["runtime"]["numpy"]+"\n")
    (bundle/"README.txt").write_text("Portable selected-checkpoint replay\n\n"
        "Use the recorded Python3.11/Torch/PyG CPU environment. No downloading occurs.\n"
        "From any directory: python /path/to/bundle/code/replay.py --output results/replay_run02.json\n"
        "The script checks every immutable file, all101 validation metric epochs and checkpoint selection,\n"
        "then reconstructs the exact frozen bases from the state_dict and executes fresh selected predictions.\n"
        "All tensor loads use weights_only=True. Temporary/cache/output paths remain inside this bundle.\n"
        "This is checkpoint/selection replay, NOT independent fresh training reproduction.\n"
        "The data contains only the two validation graphs; no test split or author calibration records.\n"
        "Float32 cross-runtime tolerance is atol=rtol=2e-4; threshold changes and fresh metrics are reported.\n"
        "Both trace-selected and independently recomputed winners must match.\n"
        "CPU build suffix may differ; Torch core/PyG version and SAGEConv source must match.\n"
        "Recorded state construction, basis provenance and training validity remain author audit obligations.\n")
    seal_bundle(bundle)


def runtime_record(environment):
    return {"python":environment["python"].split()[0],"torch":environment["torch"],
            "torch_geometric":environment["packages"]["torch-geometric"],
            "numpy":environment["packages"]["numpy"],"SAGEConv_sha256":environment["classes"]["SAGEConv"],
            "deterministic_algorithms":environment["deterministic"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--author-replay",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(os.environ.get("PYTHONDONTWRITEBYTECODE")=="1" and os.environ.get("CUDA_VISIBLE_DEVICES")=="",
            "Use the confined CPU-only repository wrapper")
    for name in portable.CACHE_NAMES:
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(PHASE),"Unconfined author cache/temp path")
    run=inside(args.run);replay_path=inside(args.author_replay);output=inside(args.output)
    require(output.is_relative_to(HERE/"exports") and not output.exists(),"Use a new exporter-owned exports folder")
    # All admission checks precede tensor loading or creation of a public bundle.
    artifact=read_json(run/"artifact_manifest.json")
    actual={str(p.relative_to(run)):portable.sha(p) for p in run.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"}
    require(actual==artifact,"Complete fit artifact inventory/hashes changed")
    contract=read_json(run/"contract.json");result=read_json(run/"result.json");selected=read_json(run/"selected.json")
    protocol=read_json(run/"protocol.json");receipt=read_json(replay_path);reviewed=read_json(HERE/"REVIEWED_SOURCE.json")
    require(protocol["protocol_sha256"]==portable.canonical_hash({k:v for k,v in protocol.items() if k!="protocol_sha256"}),"Runtime protocol seal differs")
    require(protocol["schema"]==reviewed["expected_schema"] and protocol["recipe"]==reviewed["expected_recipe"]
        and protocol["status"]=="runtime_frozen_no_predictive_fit" and protocol["test_access_planned"] is False,"Wrong runtime protocol")
    require(contract["source_sha256"]==protocol["source_sha256"] and contract["environment"]==protocol["environment"]
        and contract["protocol_sha256"]==protocol["protocol_sha256"] and contract["recipe"]==protocol["recipe"],"Fit source/runtime/protocol mismatch")
    for name,digest in protocol["source_sha256"].items():
        require(portable.sha(inside(PHASE/name))==digest and portable.sha(inside(run/"source"/name))==digest,"Current/archived fitted source changed")
    require(all(protocol["source_sha256"].get(k)==v for k,v in reviewed["source_sha256"].items()),"Reviewed exporter model/runner source differs")
    gate=read_json(checked(protocol["preflight"]))
    require(gate["status"]=="PASS" and gate["source_sha256"]==protocol["source_sha256"]
        and gate["environment"]==protocol["environment"] and gate["prospective_protocol_sha256"]==protocol["prospective_protocol_sha256"]
        and gate["validation_loader_accessed"] is False and gate["test_loader_accessed"] is False,"Incomplete source/runtime preflight admission")
    require(result["status"]=="completed_predictive_development_fit" and result["fit_graph_steps"]==2000
        and result["validation_epochs"]==101 and result["selected"]==selected and result["execution_device"]=="cpu"
        and result["CUDA_context_created"] is False and result["test_loader_accessed"] is False
        and contract["test_loader_accessed"] is False and contract["cpu_execution_gate"]["CUDA_context_created"] is False,
        "Require complete validation-only CPU fit")
    checkpoint=inside(run/selected["checkpoint"])
    bindings={"contract_sha256":portable.sha(run/"contract.json"),"selected_checkpoint_sha256":portable.sha(checkpoint),
              "fit_artifact_manifest_sha256":portable.sha(run/"artifact_manifest.json"),"protocol_sha256":protocol["protocol_sha256"],
              "source_sha256":protocol["source_sha256"],"cell":contract["cell"],"arm":contract["arm"],"selected_epoch":selected["epoch"]}
    require(receipt["status"]=="PASS" and receipt["test_loader_accessed"] is False
            and all(receipt.get(k)==v for k,v in bindings.items()),"Passed numerical author replay is missing or mismatched")
    import torch
    import torch_geometric
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    runtime=runtime_record(contract["environment"])
    require(torch.__version__==runtime["torch"] and torch_geometric.__version__==runtime["torch_geometric"]
            and sys.version.split()[0]==runtime["python"] and not torch.cuda.is_initialized(),"Exporter runtime differs from the admitted CPU fit")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    require(saved["epoch"]==selected["epoch"] and saved["contract_sha256"]==bindings["contract_sha256"],"Checkpoint identity differs")
    state={k:t.detach().cpu().contiguous().clone() for k,t in saved["state_dict"].items()}
    require(all(isinstance(t,torch.Tensor) for t in state.values()),"Checkpoint state contains non-tensors")
    metric_names=("micro_f1","pooled_bce","mean_member_bce","graph_mean_member_bce","graph_mean_pooled_bce","graph_mean_bce_jensen_gap")
    require(all(saved["validation"][k]==selected[k] for k in metric_names),"Selected checkpoint metrics differ")
    manifest=read_json(checked(protocol["data_manifest"]))
    graphs,raw_hashes=validation_graphs(protocol,manifest)
    targets=tensor_list(run/"validation_targets.pt")
    require(all(torch.equal(g["y"],y) for g,y in zip(graphs,targets)),"Official validation targets differ from the fit")
    source_paths={"models.py":run/"source/source_snapshot/models.py",
                  "adapters.py":run/"source/spectral_privacy/v2_abi_fix_01/adapters.py",
                  "replay.py":HERE/"bundle_template/code/replay.py","metrics.py":HERE/"bundle_template/code/metrics.py"}
    model=portable.rebuild(state,HERE/"bundle_template/code")
    frozen=portable.canonical_hash({k:portable.tensor_identity(v) for k,v in model.named_buffers()})
    require(frozen==contract["basis_buffers_initial_sha256"]==result["basis_buffers_final_sha256"],"Exported frozen basis/member contrasts differ")
    metadata={"schema":"gnnm-portable-spectral-checkpoint-v1","validation_scope":"official_PPI_validation",
        "test_inputs_or_targets_included":False,"fresh_training_reproduction":False,"runtime":runtime,
        "selected":{"epoch":selected["epoch"],"metrics":{k:selected[k] for k in metric_names}},"frozen_buffer_sha256":frozen,
        "fit":{"arm":contract["arm"],"seed":contract["seed"],"label_fraction":contract["label_fraction"],"recipe":contract["recipe"]},
        "provenance":{"source_contract_sha256":bindings["contract_sha256"],"source_selected_checkpoint_sha256":bindings["selected_checkpoint_sha256"],
            "source_fit_artifact_manifest_sha256":bindings["fit_artifact_manifest_sha256"],"source_runtime_protocol_sha256":protocol["protocol_sha256"],
            "source_numerical_replay_sha256":portable.sha(replay_path),"source_data_manifest_sha256":protocol["data_manifest"]["sha256"],
            "original_models_sha256":portable.sha(source_paths["models.py"]),"original_adapters_sha256":portable.sha(source_paths["adapters.py"]),
            "original_metric_body_sha256":reviewed["metric_body_sha256"],"validation_raw_file_sha256":raw_hashes}}
    output.mkdir(parents=True);bundle=output/"bundle"
    populate(bundle,state,graphs,lambda epoch:tensor_list(run/"validation_epochs"/f"epoch{epoch:03d}.pt"),
             run/"validation_trace.csv",metadata,source_paths)
    command=[sys.executable,str(bundle/"code/replay.py"),"--output","results/export_replay_run01.json"]
    with (output/"portable_replay_stdout_stderr.log").open("x") as stream:
        completed=subprocess.run(command,cwd=bundle,stdout=stream,stderr=subprocess.STDOUT,check=False)
    require(completed.returncode==0,"Portable fresh replay failed; preserve attempt, export not admitted")
    public_receipt=read_json(bundle/"results/export_replay_run01.json")
    require(public_receipt["status"]=="PASS" and public_receipt["validation_scope"]=="official_PPI_validation","Portable official-data gate failed")
    admission={"status":"PASS","scope":"complete source/runtime/artifact-verified fit plus author and portable numerical checkpoint replays",
        "source_bindings":bindings,"public_manifest_sha256":portable.sha(bundle/"manifest.json"),
        "public_replay_sha256":portable.sha(bundle/"results/export_replay_run01.json"),
        "exported_graphs":2,"exported_validation_epochs":101,"fresh_training_reproduction":False,
        "test_inputs_or_targets_exported":False,"credentials_or_calibration_artifacts_exported":False}
    (output/"EXPORT_ADMISSION_PASS.json").write_text(json.dumps(admission,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","bundle":str(bundle),"public_manifest_sha256":admission["public_manifest_sha256"],
                      "fresh_training_reproduction":False}))


if __name__=="__main__":main()
