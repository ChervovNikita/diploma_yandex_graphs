"""Four-arm frozen-basis PPI warm-start fits; no test or downloader path."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
AUTHORIZED_PHASES = (Path("/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930"),
                    Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930"))
if PHASE not in AUTHORIZED_PHASES: raise ValueError("Unrecognized authorized phase")
sys.path[:0] = [str(HERE), str(PHASE), str(PHASE / "protocols"),
               str(PHASE / "spectral_privacy/ppi_integration_v1")]
from prepare import (ARMS, RECIPE, SCHEMA, checked, inside, json_hash, ref, seal, sha,
                     source_hashes, verify_seal, write_new)
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from torch_geometric.datasets import PPI
from torch_geometric.nn import SAGEConv
import baseline_adapters as baseline
from collect_ppi import (CachedTrainPPI, checked_masks, legacy_tensor_fingerprint,
                         load_train, validate_lock as validate_integration)
from spectral_privacy.ppi_f2_audit_v1.freeze_audit import fixed_bases
from spectral_privacy.v2_abi_fix_01.adapters import SourceSAGEEnsemble
from spectral_privacy.v2_abi_fix_01.contracts import state_hash, tensor_fingerprint, unpack_float64
from spectral_privacy.v2_abi_fix_01.replay import replay_record
from new_tasks.shareable_guard import require_shareable_manifest

TARGETS = tuple(f"residual_modules.{i}.module.conv.lin_l.weight" for i in range(3))
CACHE_VARIABLES = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS",
                   "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                   "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR")


def confinement():
    if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE) or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("Source the explicit authorized repository wrapper")
    values = {k: os.environ.get(k) for k in CACHE_VARIABLES}
    if any(not v or not Path(v).resolve().is_relative_to(PHASE) for v in values.values()):
        raise ValueError("All cache/temp paths must stay inside the authorized phase")
    return values


def environment():
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda, "packages": dict(sorted(
                (d.metadata["Name"],d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
            "classes": {c.__name__:hashlib.sha256(inspect.getsource(c).encode()).hexdigest() for c in (PPI,SAGEConv)},
            "dtype": str(torch.get_default_dtype()), "deterministic":torch.are_deterministic_algorithms_enabled(),
            "cuda_tf32":torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32":torch.backends.cudnn.allow_tf32}


def setup():
    confinement(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if torch.get_default_dtype() != torch.float32: raise ValueError("Require float32 source runtime")


def protocol(path, runtime=False):
    p = inside(path); v = json.loads(p.read_text()); verify_seal(v)
    if (v["schema"] != SCHEMA or v["recipe"] != RECIPE or v["arms"] != list(ARMS)
            or v["source_sha256"] != source_hashes() or v["test_access_planned"] is not False):
        raise ValueError("Prospective source/recipe changed")
    for cell in v["cells"]:
        for k in ("checkpoint","label_masks","contract","DE_matrix_record","DE_basis_lock","DE_replay_report"): checked(cell[k])
    for value in v["prior_evidence"].values(): checked(value)
    checked(v["audit_input_lock"]); checked(v["data_manifest"])
    if runtime:
        if v.get("status") != "runtime_frozen_no_predictive_fit" or v["environment"] != environment():
            raise ValueError("Exact runtime freeze required")
        gate = json.loads(checked(v["preflight"]).read_text())
        if gate["status"] != "PASS" or gate["prospective_protocol_sha256"] != v["prospective_protocol_sha256"]:
            raise ValueError("Passed real training graph preflight required")
    return v


def data_inputs(v):
    inputs = validate_integration(checked(v["integration_input_lock"]))
    train, fingerprints = load_train(inputs)
    manifest = require_shareable_manifest(json.loads(checked(v["data_manifest"]).read_text()), "ppi")
    if manifest["manifest_amendment"]["parent_manifest_sha256"] != inputs["data_manifest"]["sha256"]:
        raise ValueError("Shareable parent does not bind the common source dataset")
    rows = manifest["split_summaries"]["train"]["graph_tensor_fingerprints"]
    graphs = [train[i] for i in range(20)]
    for i,g in enumerate(graphs):
        if (g.y.shape != (g.num_nodes,121) or legacy_tensor_fingerprint(g.y) != rows[i]["y"]
                or not bool(torch.isfinite(g.y).all()) or not bool(((g.y==0)|(g.y==1)).all())):
            raise ValueError("Training targets differ from frozen official inputs")
    return inputs, graphs, manifest, fingerprints


def load_validation(v, inputs, manifest):
    cache = inside(PHASE / inputs["data_cache"])
    path = inside(cache / "provider/processed/val.pt")
    identity = next(r for r in manifest["raw_and_processed_files"] if r["relative_path"] == "provider/processed/val.pt")
    if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
        raise ValueError("Frozen validation provider differs")
    dataset = CachedTrainPPI(root=str(cache / "provider"),split="val")
    rows = manifest["split_summaries"]["val"]["graph_tensor_fingerprints"]
    graphs = list(dataset)
    if len(graphs) != 2 or len(rows) != 2: raise ValueError("Require complete official two-graph validation split")
    for i,g in enumerate(graphs):
        if any(legacy_tensor_fingerprint(getattr(g,k)) != rows[i][k] for k in ("x","edge_index","y")):
            raise ValueError("Validation input/target fingerprints differ")
    return graphs


def common_source(cell):
    saved = torch.load(checked(cell["checkpoint"]),map_location="cpu",weights_only=False)
    contract = json.loads(checked(cell["contract"]).read_text())
    if (saved["epoch"] != 25 or saved["seed"] != cell["seed"] or saved["label_fraction"] != cell["label_fraction"]
            or saved["protocol_sha256"] != contract["protocol_sha256"]
            or any(float(s["step"]) != 250 for s in saved["optimizer"]["state"].values())):
        raise ValueError("Require exact recorded P-only epoch25 source")
    if any(not k.startswith("model.") for k in saved["state_dict"]): raise ValueError("Only tied source tensors allowed")
    state = {k[6:]:t for k,t in saved["state_dict"].items()}
    # Constructor RNG is discarded; warm-start and arm construction consume no fit RNG.
    with torch.random.fork_rng(devices=[]):
        source = baseline._MODELS.TABMModel(**baseline.model_options(50,121,128,3,0.2),tabm_inits=4,device="cpu")
    source.load_state_dict(state,strict=True)
    if (state_hash(source) != cell["original_source_state_sha256"]
            or any(not torch.equal(source.state_dict()[k],t) for k,t in state.items())):
        raise ValueError("Common P25 tensor bytes differ")
    return source


def models_for_cell(cell):
    record = json.loads(checked(cell["DE_matrix_record"]).read_text())
    replay_record(record,expected_probe_state_sha256=cell["adapter_state_sha256"])
    packets = fixed_bases(record["selection"])
    if json_hash(packets) != cell["fixed_bases_sha256"] or record["selection"]["ranks"] != cell["ranks"]:
        raise ValueError("Existing basis/rank rule changed")
    source = common_source(cell); rng = torch.get_rng_state().clone(); models = {}
    for arm in ARMS:
        group = "raw_gradient" if arm == "raw" else arm
        bases = None if arm == "tied" else {k:unpack_float64(x).float() for k,x in packets[group].items()}
        model = SourceSAGEEnsemble(source,bases=bases)
        if not model.is_tied_at_state() or state_hash(model.tied_source_copy()) != state_hash(source):
            raise ValueError("Nonzero initial coefficients or changed source")
        if not torch.equal(torch.get_rng_state(),rng): raise ValueError("Arm construction changed RNG")
        models[arm] = model
    return models


def optimizer(model):
    parameters = [p for p in model.parameters() if p.requires_grad and p.numel()]
    return torch.optim.AdamW(parameters,lr=RECIPE["lr"],weight_decay=RECIPE["weight_decay"],
                            eps=RECIPE["eps"],betas=tuple(RECIPE["betas"]))


def cost(model,opt=None):
    private = model.added_storage(); parameter_scalars = sum(p.numel() for p in model.parameters())
    private_scalars = private["private_trainable_scalars"]
    optimizer_bytes = 0 if opt is None else sum(t.numel()*t.element_size()
        for s in opt.state.values() for t in s.values() if isinstance(t,torch.Tensor))
    return {**private, "parameter_scalars":parameter_scalars, "source_parameter_scalars":parameter_scalars-private_scalars,
            "parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "buffer_bytes":sum(p.numel()*p.element_size() for p in model.buffers()),
            "optimizer_tensor_bytes":optimizer_bytes, "member_hidden_width":128,
            "member_paths":4, "graph_propagations_per_full_forward":12,
            "activation_or_propagation_reduction_claimed":False}


def basis_hash(model):
    return json_hash({k:tensor_fingerprint(t) for k,t in model.named_buffers()})


def step_rng(seed,epoch,index,device):
    value = 67001+10000*seed+20*epoch+index
    torch.manual_seed(value)
    if device.type == "cuda": torch.cuda.manual_seed(value)
    return value


def observed_loss(logits,graph,mask):
    y = graph.y[mask]
    if not bool(torch.isfinite(y).all()): raise ValueError("Observed training labels must be finite")
    return bce(logits[:,mask],y[None].expand(4,-1,-1))


def finite_step(model,opt,graph,mask):
    model.train(); opt.zero_grad(set_to_none=True); logits = model(graph,graph.x)
    if logits.shape != (4,graph.num_nodes,121) or not bool(torch.isfinite(logits).all()): raise ValueError("Bad member output")
    loss = observed_loss(logits,graph,mask)
    if not bool(torch.isfinite(loss)): raise ValueError("Nonfinite observed training loss")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.numel()):
        raise ValueError("Inactive/nonfinite trainable parameter")
    opt.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()): raise ValueError("Nonfinite optimizer update")
    return float(loss.detach()),logits.detach()


def metrics(logits,targets):
    if len(logits) != len(targets) or not targets: raise ValueError("Incomplete validation evidence")
    tp=fp=fn=entries=0; pooled_loss=member_loss=graph_member_loss=graph_pooled_loss=0.0
    for pred,y in zip(logits,targets):
        if pred.shape != (4,y.shape[0],121) or not bool(torch.isfinite(pred).all()): raise ValueError("Invalid validation member logits")
        pooled=pred.mean(0); truth=y.bool(); guessed=pooled>=0
        tp+=int((truth&guessed).sum()); fp+=int((~truth&guessed).sum()); fn+=int((truth&~guessed).sum())
        this_pooled=float(bce(pooled,y,reduction="sum")); pooled_loss+=this_pooled
        risk=float(bce(pred,y[None].expand_as(pred),reduction="sum"))/4
        graph_pooled_loss+=this_pooled/y.numel()
        member_loss+=risk; graph_member_loss+=risk/y.numel(); entries+=y.numel()
    gap=(graph_member_loss-graph_pooled_loss)/len(targets)
    if gap < -2e-7: raise ValueError("Convex-BCE member/pooled Jensen gap is negative")
    return {"micro_f1":2*tp/max(1,2*tp+fp+fn),"pooled_bce":pooled_loss/entries,
            "mean_member_bce":member_loss/entries,"graph_mean_member_bce":graph_member_loss/len(targets),
            "graph_mean_pooled_bce":graph_pooled_loss/len(targets),"graph_mean_bce_jensen_gap":gap}


def evaluate(model,graphs):
    model.eval()
    with torch.no_grad(): logits=[model(g,g.x).detach().cpu() for g in graphs]
    return metrics(logits,[g.y.detach().cpu() for g in graphs]),logits


def artifact_manifest(folder):
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "artifact_manifest.json"}


def preflight(args):
    v=protocol(args.protocol); output=inside(args.output)
    if HERE not in output.parents or output.exists(): raise ValueError("Use a new preflight folder inside this study")
    output.mkdir(parents=True); start=time.perf_counter(); reports=[]
    try:
        inputs,graphs,manifest,fingerprints=data_inputs(v); graph=graphs[0]
        for cell in v["cells"]:
            original=next(c for c in inputs["arms"] if c["arm_id"]==cell["arm_id"])
            masks,_=checked_masks(original,json.loads(checked(cell["contract"]).read_text()),graphs)
            safe=graph.clone(); safe.y[~masks[0]]=float("nan")
            torch.manual_seed(72001+cell["seed"]); before=torch.get_rng_state().clone(); models=models_for_cell(cell)
            if not torch.equal(before,torch.get_rng_state()): raise ValueError("Common construction changed RNG")
            evals={}; stochastic={}; initial_sources={}; costs={}; updates={}
            for arm,m in models.items():
                initial_sources[arm]=state_hash(m.tied_source_copy()); m.eval()
                with torch.no_grad(): evals[arm]=m(safe,safe.x)
                step_rng(cell["seed"],1,0,torch.device("cpu")); m.train()
                with torch.no_grad(): stochastic[arm]=m(safe,safe.x)
            if any(not torch.equal(evals[a],evals["tied"]) or not torch.equal(stochastic[a],stochastic["tied"]) for a in ARMS):
                raise ValueError("Initial eval/dropout function identity failed")
            states={}
            for arm,m in models.items():
                opt=optimizer(m)
                if opt.state: raise ValueError("Optimizer moments are not empty")
                frozen=basis_hash(m); initial=state_hash(m)
                step_rng(cell["seed"],1,0,torch.device("cpu")); loss,_=finite_step(m,opt,safe,masks[0])
                if basis_hash(m)!=frozen or any(t.requires_grad for t in m.frozen_bases().values()): raise ValueError("Basis changed or requires gradients")
                if state_hash(m)==initial: raise ValueError("One-step source update was inactive")
                maximum_center_error=max(float(d.sum(0).abs().max()) for d in m.private_corrections().values())
                if maximum_center_error>1e-7: raise ValueError("Private corrections ceased to be centered")
                expected=0 if arm=="tied" else 3*128*sum(cell["ranks"].values())
                costs[arm]=cost(m,opt)
                if costs[arm]["private_trainable_scalars"]!=expected: raise ValueError("Actual private capacity differs from fixed ranks")
                updates[arm]={"finite_loss":loss,"center_sum_max_abs":maximum_center_error,
                              "frozen_buffers_sha256":frozen,"fresh_adamw_steps":[float(s["step"]) for s in opt.state.values()]}
                if any(s!=1 for s in updates[arm]["fresh_adamw_steps"]): raise ValueError("Optimizer did not start fresh")
                states[arm]=m.state_dict()
            if sum(cell["ranks"].values())==0 and any(
                    any(not torch.equal(t,states["tied"][k]) for k,t in states[a].items()) for a in ARMS):
                raise ValueError("Rank-zero arm identity failed after update")
            reports.append({"cell":cell["arm_id"],"source_initial_sha256":initial_sources,
                            "eval_and_dropout_initial_outputs_identical":True,"constructor_rng_preserved":True,
                            "official_graph_index":0,"nodes":graph.num_nodes,"observed_nodes":int(masks[0].sum()),
                            "logits_shape":[4,graph.num_nodes,121],"costs":costs,"one_step_updates":updates})
        report={"status":"PASS","prospective_protocol_sha256":v["protocol_sha256"],"source_sha256":source_hashes(),
                "environment":environment(),"confinement":confinement(),"cells":reports,
                "train_input_identity_sha256":json_hash(fingerprints),"engineering_updates":24,
                "validation_loader_accessed":False,"test_loader_accessed":False,
                "predictive_fit_performed":False,"seconds":time.perf_counter()-start}
        write_new(output/"PREFLIGHT.json",report); write_new(output/"artifact_manifest.json",artifact_manifest(output))
        print(json.dumps({"status":"PASS","engineering_updates":24,"seconds":report["seconds"]}),flush=True)
    except Exception as e:
        write_new(output/"FAILURE.json",{"type":type(e).__name__,"message":str(e),"seconds":time.perf_counter()-start}); raise


def freeze_runtime(args):
    v=protocol(args.protocol); gate=json.loads(inside(args.preflight).read_text())
    if (gate["status"]!="PASS" or gate["prospective_protocol_sha256"]!=v["protocol_sha256"]
            or gate["source_sha256"]!=source_hashes() or gate["environment"]!=environment()
            or gate["validation_loader_accessed"] or gate["test_loader_accessed"]): raise ValueError("Passed matching preflight required")
    new={k:x for k,x in v.items() if k!="protocol_sha256"}
    new.update(status="runtime_frozen_no_predictive_fit",prospective_protocol_sha256=v["protocol_sha256"],
               preflight=ref(args.preflight),environment=environment(),confinement=confinement())
    write_new(args.output,seal(new)); print(json.dumps({"status":new["status"]}),flush=True)


def guard_gpu(args):
    if args.device!="cuda:0" or args.physical_gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.physical_gpu):
        raise ValueError("Fitting requires one explicit authorized physical GPU as logical cuda:0")
    q=subprocess.run(["nvidia-smi","-i",str(args.physical_gpu),"--query-gpu=memory.used,utilization.gpu",
                      "--format=csv,noheader,nounits"],check=True,text=True,capture_output=True)
    memory,utilization=map(int,q.stdout.strip().split(","))
    if memory>100 or utilization!=0: raise RuntimeError("GPU occupied; no CUDA context/model created")
    return {"physical_gpu":args.physical_gpu,"idle_memory_mib":memory,"idle_utilization_percent":utilization}


def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)


def snapshot(model,opt,epoch,values,contract):
    return {"state_dict":{k:t.detach().cpu() for k,t in model.state_dict().items()},
            "optimizer":copy.deepcopy(opt.state_dict()),"epoch":epoch,"validation":values,
            "contract_sha256":contract,"rng_cpu":torch.get_rng_state(),"rng_python":random.getstate(),
            "rng_cuda":torch.cuda.get_rng_state().cpu() if next(model.parameters()).device.type=="cuda" else None}


def fit(args):
    v=protocol(args.protocol,runtime=True); output=inside(args.output)
    if HERE not in output.parents or output.exists(): raise ValueError("Use a new fit folder inside this study")
    # GPU gate precedes any CUDA call, including torch.manual_seed.
    gpu=guard_gpu(args); inputs,graphs,manifest,fingerprints=data_inputs(v)
    cell=next(c for c in v["cells"] if c["seed"]==args.seed and c["label_fraction"]==args.label_fraction)
    original=next(c for c in inputs["arms"] if c["arm_id"]==cell["arm_id"])
    masks,mask_manifest=checked_masks(original,json.loads(checked(cell["contract"]).read_text()),graphs)
    model=models_for_cell(cell)[args.arm]; frozen=basis_hash(model)
    device=torch.device(args.device); model.to(device); opt=optimizer(model)
    if opt.state: raise ValueError("Fresh optimizer required")
    train=[]
    for g,m in zip(graphs,masks):
        safe=g.clone(); safe.y[~m]=float("nan"); train.append(safe.to(device))
    masks_device=[m.to(device) for m in masks]
    val=[g.to(device) for g in load_validation(v,inputs,manifest)]
    output.mkdir(parents=True); (output/"validation_epochs").mkdir(); (output/"checkpoints").mkdir()
    for name in v["source_sha256"]:
        dst=output/"source"/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(PHASE/name,dst)
    shutil.copyfile(args.protocol,output/"protocol.json")
    torch.save([g.y.cpu() for g in val],output/"validation_targets.pt")
    torch.save(masks,output/"label_masks.pt")
    torch.save({"state_dict":model.state_dict(),"optimizer":opt.state_dict()},output/"initial.pt")
    contract={"schema":SCHEMA,"arm":args.arm,"cell":cell["arm_id"],"seed":args.seed,"label_fraction":args.label_fraction,
              "protocol_sha256":v["protocol_sha256"],"source_sha256":source_hashes(),"environment":environment(),
              "recipe":RECIPE,"initial_sha256":sha(output/"initial.pt"),"mask_manifest":mask_manifest,
              "train_input_identity_sha256":json_hash(fingerprints),"basis_buffers_initial_sha256":frozen,
              "cost_initial":cost(model,opt),"warm_start_checkpoint":cell["checkpoint"],"fixed_basis_cell":cell,
              "gpu_idle_gate":gpu,"test_loader_accessed":False,"hidden_train_labels_are_NaN":True,
              "data_manifest":v["data_manifest"],"confinement":confinement()}
    write_new(output/"contract.json",contract); contract_sha=sha(output/"contract.json")
    start=time.perf_counter(); fit_seconds=eval_seconds=0.; best=None; selected=None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        with (output/"validation_trace.csv").open("x",newline="") as stream:
            fields=["epoch","micro_f1","pooled_bce","mean_member_bce","graph_mean_member_bce",
                    "graph_mean_pooled_bce","graph_mean_bce_jensen_gap",
                    "train_graph_mean_member_bce","fit_epoch_seconds","validation_seconds","elapsed_seconds"]
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for epoch in range(101):
                loss_sum=0.; seconds=0.
                if epoch:
                    order=list(range(20)); random.Random(19709+1000*args.seed+epoch).shuffle(order)
                    sync(device); t=time.perf_counter()
                    for index in order:
                        step_rng(args.seed,epoch,index,device)
                        loss,_=finite_step(model,opt,train[index],masks_device[index]); loss_sum+=loss
                    sync(device); seconds=time.perf_counter()-t; fit_seconds+=seconds
                    if basis_hash(model)!=frozen: raise ValueError("Frozen basis/contrast buffer changed")
                sync(device); t=time.perf_counter(); values,logits=evaluate(model,val); sync(device)
                validation_seconds=time.perf_counter()-t; eval_seconds+=validation_seconds
                torch.save(logits,output/"validation_epochs"/f"epoch{epoch:03d}.pt")
                row={"epoch":epoch,**values,"train_graph_mean_member_bce":loss_sum/20 if epoch else 0.,
                     "fit_epoch_seconds":seconds,"validation_seconds":validation_seconds,"elapsed_seconds":time.perf_counter()-start}
                writer.writerow(row); stream.flush(); key=(values["micro_f1"],-values["pooled_bce"],-epoch)
                if best is None or key>best:
                    best=key; checkpoint=f"checkpoints/epoch{epoch:03d}.pt"
                    torch.save(snapshot(model,opt,epoch,values,contract_sha),output/checkpoint)
                    selected={"epoch":epoch,**values,"checkpoint":checkpoint}
                if epoch==100 and not (output/"checkpoints/epoch100.pt").exists():
                    torch.save(snapshot(model,opt,epoch,values,contract_sha),output/"checkpoints/epoch100.pt")
                if epoch==0 or epoch%10==0: print(json.dumps({"cell":cell["arm_id"],"arm":args.arm,"epoch":epoch,**values}),flush=True)
        peak={"allocated":torch.cuda.max_memory_allocated(device),"reserved":torch.cuda.max_memory_reserved(device)}
        final_cost=cost(model,opt); final_values=values
        selected_ckpt=torch.load(output/selected["checkpoint"],map_location=device,weights_only=False)
        model.load_state_dict(selected_ckpt["state_dict"],strict=True); model.eval()
        with torch.no_grad():
            for _ in range(2): model(train[0],train[0].x)
            sync(device); torch.cuda.reset_peak_memory_stats(device); times=[]
            for _ in range(10):
                sync(device); t=time.perf_counter(); model(train[0],train[0].x); sync(device); times.append(time.perf_counter()-t)
        write_new(output/"selected.json",selected)
        result={"status":"completed_predictive_development_fit","cell":cell["arm_id"],"arm":args.arm,
                "selected":selected,"epoch100":final_values,"fit_graph_steps":2000,"validation_epochs":101,
                "fit_seconds":fit_seconds,"validation_seconds":eval_seconds,"total_run_seconds":time.perf_counter()-start,
                "peak_fit_memory_bytes":peak,"cost_final":final_cost,"prior_shared_costs":v["prior_probe_costs"],
                "prior_shared_cost_evidence":v["prior_evidence"],"selected_latency_seconds":times,
                "selected_latency_median_seconds":statistics.median(times),"latency_workload":"official training graph0 full four-member forward",
                "selected_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"test_loader_accessed":False,
                "confirmation_claimed":False,"basis_buffers_final_sha256":basis_hash(model)}
        write_new(output/"result.json",result); write_new(output/"artifact_manifest.json",artifact_manifest(output))
    except Exception as e:
        write_new(output/"FAILURE.json",{"type":type(e).__name__,"message":str(e),"seconds":time.perf_counter()-start}); raise


def replay(args):
    v=protocol(args.protocol,runtime=True); folder=inside(args.run)
    if artifact_manifest(folder)!=json.loads((folder/"artifact_manifest.json").read_text()): raise ValueError("Fit artifact coverage/hashes changed")
    contract=json.loads((folder/"contract.json").read_text()); result=json.loads((folder/"result.json").read_text())
    if (contract["protocol_sha256"]!=v["protocol_sha256"] or contract["source_sha256"]!=source_hashes()
            or result["status"]!="completed_predictive_development_fit" or result["fit_graph_steps"]!=2000):
        raise ValueError("Require complete matching predictive fit")
    inputs,_,manifest,_=data_inputs(v); val=load_validation(v,inputs,manifest)
    targets=torch.load(folder/"validation_targets.pt",map_location="cpu",weights_only=False)
    if len(targets)!=2 or any(not torch.equal(g.y,y) for g,y in zip(val,targets)): raise ValueError("Validation targets changed")
    with (folder/"validation_trace.csv").open() as f: rows=list(csv.DictReader(f))
    if [int(r["epoch"]) for r in rows]!=list(range(101)): raise ValueError("Incomplete validation trajectory")
    maximum_metric_error=0.
    for row in rows:
        kept=torch.load(folder/"validation_epochs"/f"epoch{int(row['epoch']):03d}.pt",map_location="cpu",weights_only=False)
        values=metrics(kept,targets)
        maximum_metric_error=max(maximum_metric_error,max(abs(values[k]-float(row[k])) for k in values))
    if maximum_metric_error>2e-5: raise ValueError("Stored validation logits/metrics do not replay")
    best=max(rows,key=lambda r:(float(r["micro_f1"]),-float(r["pooled_bce"]),-int(r["epoch"])))
    selected=json.loads((folder/"selected.json").read_text())
    if result["selected"]!=selected or selected["epoch"]!=int(best["epoch"]): raise ValueError("Wrong selected epoch")
    saved=torch.load(folder/selected["checkpoint"],map_location="cpu",weights_only=False)
    if saved["epoch"]!=selected["epoch"] or saved["contract_sha256"]!=sha(folder/"contract.json"): raise ValueError("Checkpoint contract/epoch changed")
    for k,x in saved["validation"].items():
        if abs(x-float(best[k]))>1e-12 or abs(x-selected[k])>1e-12: raise ValueError("Selected metrics differ")
    cell=next(c for c in v["cells"] if c["arm_id"]==contract["cell"])
    model=models_for_cell(cell)[contract["arm"]]; frozen=basis_hash(model)
    model.load_state_dict(saved["state_dict"],strict=True)
    if frozen!=basis_hash(model) or frozen!=contract["basis_buffers_initial_sha256"]: raise ValueError("Basis buffers changed at selected checkpoint")
    values,fresh=evaluate(model,val)
    kept=torch.load(folder/"validation_epochs"/f"epoch{selected['epoch']:03d}.pt",map_location="cpu",weights_only=False)
    maximum=max(float((a-b).abs().max()) for a,b in zip(fresh,kept))
    if any(not torch.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(fresh,kept)): raise ValueError("Fresh selected forward drift")
    changed=sum(int(((a.mean(0)>=0)!=(b.mean(0)>=0)).sum()) for a,b in zip(fresh,kept))
    write_new(args.output,{"status":"PASS","scope":"numerical selected-checkpoint replay; not independent training reproduction",
                          "cell":contract["cell"],"arm":contract["arm"],"selected_epoch":selected["epoch"],
                          "maximum_stored_metric_error":maximum_metric_error,"maximum_fresh_logit_error":maximum,
                          "threshold_decisions_changed":changed,"fresh_cpu_metrics":values,"reported_selected_metrics":saved["validation"],
                          "contract_sha256":sha(folder/"contract.json"),"selected_checkpoint_sha256":sha(folder/selected["checkpoint"]),
                          "fit_artifact_manifest_sha256":sha(folder/"artifact_manifest.json"),"protocol_sha256":v["protocol_sha256"],
                          "source_sha256":source_hashes(),"test_loader_accessed":False})
    print(json.dumps({"status":"PASS","maximum_fresh_logit_error":maximum,"threshold_decisions_changed":changed}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("preflight","freeze-runtime","fit","replay")); p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--preflight",type=Path); p.add_argument("--run",type=Path)
    p.add_argument("--arm",choices=ARMS); p.add_argument("--seed",type=int,choices=(0,1,2)); p.add_argument("--label-fraction",type=float,choices=(0.1,1.0))
    p.add_argument("--device",choices=("cpu","cuda:0"),default="cpu"); p.add_argument("--physical-gpu",type=int)
    args=p.parse_args(); setup()
    if args.command=="fit" and any(x is None for x in (args.arm,args.seed,args.label_fraction)): p.error("fit needs arm/seed/label-fraction")
    if args.command=="freeze-runtime" and args.preflight is None: p.error("freeze-runtime needs preflight")
    if args.command=="replay" and args.run is None: p.error("replay needs run")
    {"preflight":preflight,"freeze-runtime":freeze_runtime,"fit":fit,"replay":replay}[args.command](args)
