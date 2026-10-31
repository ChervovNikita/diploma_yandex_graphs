"""Selected boundary link export; source TEST target custody/protocol retained."""
from functools import wraps
import gc
import json
from pathlib import Path
try:
    from ..engine import restore_checkpoint, AdapterStepError
    from ..run_study import load_factory, resolve_selected_config
    from ..spec import fingerprint
    from .link import bound_file, file_sha
    from .link_boundary import validate_roster
except ImportError:
    from engine import restore_checkpoint, AdapterStepError
    from run_study import load_factory, resolve_selected_config
    from spec import fingerprint
    from adapters.link import bound_file, file_sha
    from adapters.link_boundary import validate_roster


def receipt(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():raise ValueError("Confine endpoint artifact")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def load_bound(torch, root, row):
    return torch.load(bound_file(root, row), map_location="cpu", weights_only=True)


def retain_costs(function):
    @wraps(function)
    def wrapped(**kwargs):
        counts = {"optimizer_updates": 0, "scientific_backbone_fits": 0, "model_constructions": 0,
            "checkpoint_restore_attempts": 0, "checkpoint_restores": 0, "prediction_pack_attempts": 0,
            "prediction_packs_completed": 0, "metric_evaluator_attempts": 0, "metric_evaluator_calls": 0}
        try:
            result = function(operation_counts=counts, **kwargs)
            result["operation_counts"] = dict(counts);return result
        except BaseException as error:
            failure = {"status": "failed", "failure_type": type(error).__name__, "failure": str(error),
                "operation_counts": counts, "partial_inference_counts_are_completed_lower_bounds": True}
            output = Path(kwargs["output_dir"]);output.mkdir(parents=True, exist_ok=True)
            (output/"failure_costs.json").write_text(json.dumps(failure, indent=2)+"\n")
            error.endpoint_costs = failure;raise
    return wrapped


def query_pack(torch, pack, nodes):
    if not isinstance(pack, dict) or set(pack) != {"ids", "pairs"}:raise ValueError("Export reads only label-free {ids,pairs}")
    ids, pairs = pack["ids"], pack["pairs"]
    if any(not torch.is_tensor(v) or v.device.type != "cpu" or v.dtype != torch.long or v.requires_grad for v in (ids,pairs)):
        raise ValueError("Immutable CPU int64 query authority required")
    if ids.ndim != 1 or not len(ids) or len(torch.unique(ids)) != len(ids) or bool((ids < 0).any()) or pairs.shape != (len(ids),2):
        raise ValueError("Complete unique opaque query IDs required")
    if int(pairs.min()) < 0 or int(pairs.max()) >= nodes:raise ValueError("Query nodes exceed frozen graph")
    return pack


def pool_scores(torch, scores):
    if not torch.is_tensor(scores) or scores.ndim != 2 or scores.shape[0] not in (1,4) or not scores.shape[1] or not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("One or four complete finite raw selected member scores required")
    return scores.mean(0)


def targets(torch, pack, ids):
    if not isinstance(pack,dict) or set(pack) != {"ids","is_positive","original_index"}:raise ValueError("Separate sealed target/index authority required")
    if any(not torch.is_tensor(v) or v.device.type!="cpu" or v.requires_grad for v in pack.values()):raise ValueError("Immutable CPU target/index tensors required")
    if not torch.equal(pack["ids"],ids) or pack["ids"].dtype != torch.long:raise ValueError("Target IDs/order differ from raw bank")
    mask, original = pack["is_positive"], pack["original_index"]
    if mask.dtype != torch.bool or original.dtype != torch.long or mask.shape != ids.shape or original.shape != ids.shape:
        raise ValueError("Target/index fields differ")
    for role in (mask,~mask):
        index=original[role]
        if not len(index) or not torch.equal(index.sort().values,torch.arange(len(index))):raise ValueError("Official within-pool index coverage changed")
    return mask,original


def whole_fit_roster_closed(study, registry):
    if study["release"].get("global_plan_gates_required") is not True:raise ValueError("Global search/comparative closures are mandatory")
    for row in study["runs"]:
        if row["config"].get("stage") in {"search","search_selection","comparative"}:
            actual=registry.run(study["study_id"],row["run_id"])
            if actual["state"] not in {"complete","failed"}:raise ValueError("Whole declared fit/search roster has not closed")


def _parent(study,run,registry,identifier,kind):
    if identifier not in run["parents"]:raise ValueError("Parent is outside endpoint dependency roster")
    row=registry.run(study["study_id"],identifier)
    if row["state"] != "complete" or row["contract"]["kind"] != kind:raise ValueError("Endpoint requires complete own parent")
    return row


def _inputs(study,config):
    if config["support_policy"] != "full_TRAIN_only" or config["fusion"] != "mean_raw_scores":raise ValueError("Freeze native TRAIN-only raw mean serving")
    inputs=study["data"]["evidence"]["endpoint_inputs"]
    if config["query_input"] != inputs["evaluate_queries"]:raise ValueError("Query authority differs from frozen study")
    return inputs


def _selected_authority(project_root,output_root,study,run,registry,identifier):
    row=_parent(study,run,registry,identifier,"model_fit")
    actual=resolve_selected_config(study,row["contract"],registry)
    if row["result"]["completed_steps"] != actual["budget"]["maximum"]:raise ValueError("Partial fit cannot supply selected bank")
    selected=row["result"]["selected"];relative=Path(selected["path"])
    if relative.is_absolute() or ".." in relative.parts:raise ValueError("Confine each own selected checkpoint")
    if not 1<=selected["step"]<=actual["budget"]["maximum"]:raise ValueError("Own selector epoch is outside full paid budget")
    checkpoint=Path(output_root)/study["study_id"]/identifier/relative
    ancestry={"run_id":identifier,"body":actual["body"],"checkpoint":receipt(project_root,checkpoint),
        "selected_epoch":selected["step"],"config_hash":fingerprint(actual["config"]),"initialization_seed":actual["config"]["initialization_seed"]}
    return actual,selected,checkpoint,ancestry


def predict_unlabelled(adapter,pairs):
    """Exactly one native encoder and all source query tails, no role labels."""
    torch=adapter.t
    counts={"encoder_call_attempts":0,"encoder_calls":0,"decoder_query_passes":0,"query_records":0,
        "native_decoder_call_attempts":0,"native_decoder_completed_calls":0,"native_decoder_query_rows":0}
    adapter.model.eval();adapter._active_counts=counts
    try:
        with adapter._scope(training=False),torch.no_grad():
            counts["encoder_call_attempts"]+=1
            h=adapter.model.encoder(adapter.graph["x"],adapter.full_graph);counts["encoder_calls"]+=1
            output=[];pairs=pairs.to(adapter.device)
            for ids in adapter.utils.PermIterator(adapter.device,len(pairs),adapter.config["training"]["eval_batch_size"],False):
                score=adapter.model.decoder(h,adapter.full_graph,pairs[ids].T).flatten()
                counts["decoder_query_passes"]+=1
                if score.shape != (len(ids),) or not bool(torch.isfinite(score).all()):raise ValueError("Incomplete/nonfinite native raw query scoring")
                output.append(score.cpu());counts["query_records"]+=len(ids)
            result=torch.cat(output)
            if len(result) != len(pairs):raise ValueError("Query tail omitted")
        return result,counts
    except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
    finally:adapter._active_counts=None


@retain_costs
def export_bank(*,project_root,study,run,output_root,output_dir,registry,device,operation_counts):
    import torch
    config=run["config"];_inputs(study,config);whole_fit_roster_closed(study,registry)
    pack=query_pack(torch,load_bound(torch,project_root,config["query_input"]),study["data"]["evidence"]["endpoint_nodes"])
    if "reuse_export_parent" in config:
        source=_parent(study,run,registry,config["reuse_export_parent"],"ensemble_assembly")["result"]
        if config.get("single_reference") != "fixed_I4_body0" or source["query_input"] != config["query_input"] or source["support_policy"] != config["support_policy"]:
            raise ValueError("Singleton reuse must bind the exact prospective I4 body0/query/support")
        original=load_bound(torch,project_root,source["bank"]);pool_scores(torch,original["scores"])
        if original["scores"].shape[0] != 4 or not torch.equal(original["ids"],pack["ids"]):raise ValueError("Reuse requires complete exact four-body bank")
        ancestry=source["ancestry"][:1]
        if ancestry[0]["body"] != 0 or [ancestry[0]["run_id"]] != config["fit_parents"]:raise ValueError("Wrong singleton selected member ancestry")
        actual,_,_,expected=_selected_authority(project_root,output_root,study,run,registry,ancestry[0]["run_id"])
        validate_roster(study,actual)
        if actual["config"]["arm"]!="native_independent4_body" or ancestry[0]!=expected:
            raise ValueError("Reuse differs from the exact current own-selected I4 body0 authority")
        scores=original["scores"][:1].clone()
        operation_counts["reused_body0_prediction_records"]=len(pack["ids"])
    else:
        fits=config["fit_parents"]
        if len(fits) not in (1,4) or len(set(fits)) != len(fits):raise ValueError("One/four distinct own-selected fits required")
        outputs,ancestry,contracts=[],[],[]
        for identifier in fits:
            actual,selected,checkpoint,member_ancestry=_selected_authority(project_root,output_root,study,run,registry,identifier)
            if len(fits)==4:
                if actual["config"]["arm"] != "native_independent4_body":raise ValueError("Typed genuine I4 bodies required")
                validate_roster(study,actual)
            elif actual["config"]["arm"] != "native_single":raise ValueError("Typed I4 singleton must reuse its exact bank body0")
            adapter=load_factory(project_root,actual["adapter"])(torch=torch,project_root=project_root,study=study,run=actual,device=device)
            operation_counts["model_constructions"]+=1
            for key,value in adapter.provenance["adapter_setup_operations"].items():operation_counts[key]=operation_counts.get(key,0)+value
            operation_counts["checkpoint_restore_attempts"]+=1
            restored=restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=identifier)
            operation_counts["checkpoint_restores"]+=1
            if restored["step"] != selected["step"]:raise ValueError("Selected checkpoint epoch differs")
            operation_counts["prediction_pack_attempts"]+=1
            try:score,counts=predict_unlabelled(adapter,pack["pairs"])
            except BaseException as error:
                for key,value in getattr(error,"operation_counts",{}).items():operation_counts[key]=operation_counts.get(key,0)+value
                raise
            for key,value in counts.items():operation_counts[key]=operation_counts.get(key,0)+value
            operation_counts["prediction_packs_completed"]+=1;outputs.append(score)
            ancestry.append(member_ancestry)
            contracts.append(actual);del adapter;gc.collect()
        if len(fits)==4:
            group=contracts[0]["config"]["independence_group"]
            expected={row["run_id"] for row in study["runs"] if row["kind"]=="model_fit" and row["config"].get("independence_group")==group}
            if set(fits)!=expected or {row["body"] for row in contracts}!={0,1,2,3}:raise ValueError("Bind complete own I4 group")
            recipes=[{k:v for k,v in row["config"].items() if k!="initialization_seed"} for row in contracts]
            if any(value!=recipes[0] for value in recipes[1:]):raise ValueError("Resolved I4 member recipes differ")
            order=sorted(range(4),key=lambda index:ancestry[index]["body"])
            outputs=[outputs[index] for index in order];ancestry=[ancestry[index] for index in order]
        scores=torch.stack(outputs)
    pool_scores(torch,scores)
    path=Path(output_dir)/"raw_query_bank.pt";torch.save({"ids":pack["ids"],"scores":scores},path)
    return {"status":"complete","bank":receipt(project_root,path),"ancestry":ancestry,"query_input":config["query_input"],
        "fusion":"mean_raw_scores","support_policy":"full_TRAIN_only","evaluation_targets_read":False,
        "scientific_backbone_fits":0,"singleton_reference_is_fixed_body0":"reuse_export_parent" in config}


@retain_costs
def evaluate_bank(*,project_root,study,run,output_root,output_dir,registry,device,operation_counts):
    import inspect
    import torch
    from ogb.linkproppred import Evaluator
    config=run["config"];inputs=_inputs(study,config);whole_fit_roster_closed(study,registry)
    if config["target_input"] != inputs["evaluate_targets"]:raise ValueError("Sealed target authority changed")
    exported=_parent(study,run,registry,config["export_parent"],"ensemble_assembly")["result"]
    if exported["query_input"]!=config["query_input"] or exported["support_policy"]!=config["support_policy"]:raise ValueError("Bank serving authority changed")
    bank=load_bound(torch,project_root,exported["bank"]);scores=pool_scores(torch,bank["scores"])
    if set(bank)!={"ids","scores"} or len(scores)!=len(bank["ids"]):raise ValueError("Raw bank shape changed")
    mask,original=targets(torch,load_bound(torch,project_root,config["target_input"]),bank["ids"])
    positive=scores[mask][original[mask].argsort()];negative=scores[~mask][original[~mask].argsort()]
    if len(negative)<50 or config["metric"]!={"hits_k":50,"evaluator_source_sha256":file_sha(Path(inspect.getfile(Evaluator)))}:
        raise ValueError("Qualified native shared-pool metric changed")
    evaluator=Evaluator.__new__(Evaluator);evaluator.name="ogbl-collab";evaluator.K=50;evaluator.eval_metric="hits@50"
    operation_counts["metric_evaluator_attempts"]+=1
    hits=float(evaluator.eval({"y_pred_pos":positive,"y_pred_neg":negative})["hits@50"])
    operation_counts["metric_evaluator_calls"]+=1
    if hits != int((positive>negative.topk(50).values[-1]).sum())/len(positive):raise ValueError("Native strict Hits tie law changed")
    return {"status":"complete","hits@50":hits,"positive_records":len(positive),"negative_records":len(negative),
        "export_parent":config["export_parent"],"target_receipt":config["target_input"],"fusion":"mean_raw_scores",
        "support_policy":"full_TRAIN_only","CAL_used":False,"confidence_claims_available":False,"author_protocol_replication":False,
        "scientific_backbone_fits":0}
