"""Selected-bank source-pattern and member/served VALID diagnostics."""
from pathlib import Path
from time import perf_counter
import torch
from torch.nn import functional as F
from graph_ops import Graph
from pilot_common import require, read_bound_json, file_sha, atomic_json, DIAGNOSTIC_SEED, utc
from pilot_model import seed_all, train_flag
from pilot_state import restore_snapshot, rng_state, rng_digest
from pilot_data import epoch_stream, tensor_sha
from pilot_evaluate import evaluator, hits50
from pattern_model import make_pattern
from pattern_teacher import ObservationTeacher
from pattern_train import batch_forward
from pattern_objective import diagnostic_sums


def load_fit(context,arm):
    pin=context["release"]["fit_receipts"][arm]
    receipt=read_bound_json(pin["path"],pin["sha256"])
    require(receipt["schema"]=="ncnc-pattern-complete-fit-v1" and receipt["identity"]==context["identity"] and receipt["arm"]==arm and receipt["seed"]==0 and receipt["epochs"]==100 and receipt["optimizer_steps"]==1700 and receipt["selection_candidates"]==100 and receipt["selected_roundtrip_and_full_served_replay"] is True,"Incomplete/different fit receipt")
    selected=receipt["selected_checkpoint"]
    require(selected["path"]=="SELECTED_"+arm+".pt", "Selected file is not this arm's own output")
    path=Path(pin["path"]).parent/selected["path"]
    require(path.stat().st_size==selected["bytes"] and file_sha(path)==selected["sha256"],"Selected checkpoint custody differs")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    require(payload["identity"]==context["identity"] and payload["arm"]==arm and payload["seed"]==0 and payload["serving_pool"]=="mean_raw_logits" and payload["test_stage_supported"] is False,"Selected source/pool differs")
    return receipt,payload


def score_valid_routes(model,data,mods):
    train_flag(model,False)
    graph=Graph.from_pairs(data["pairs"],len(data["x"]))
    before=rng_digest(rng_state())
    torch.cuda.synchronize(0);start=perf_counter()
    with torch.no_grad():
        h=model.encoder(data["x"],graph)
        banks=[]
        for query in (data["valid_positive"],data["valid_negative"]):
            rows={}
            for ids in mods["native_utils"].PermIterator(query.device,len(query),131072,False):
                routes=model.decoder.diagnostic_routes(h,graph,query[ids])
                for name,score in routes.items():
                    require(score.shape==(len(ids),4) and bool(torch.isfinite(score).all()),"Fixed-bank VALID coverage differs")
                    rows.setdefault(name,[]).append(score.cpu())
            banks.append({name:torch.cat(parts) for name,parts in rows.items()})
    torch.cuda.synchronize(0)
    require(rng_digest(rng_state())==before,"Selected diagnostic VALID consumes RNG")
    coverage={"positive_queries":60084,"negative_queries":100000,"route_count":5,
              "one_encoder_and_one_completion_scorer_bank":True,"recipient_features_and_decoder_held_fixed":True,
              "crossed_member_rule":"recipient_m_receives_donor_(m+shift)_mod4_clamped_weights",
              "pooled_rule":"mean_clamped_completion_weights_then_each_recipient_native_decoder_then_mean_raw_logits",
              "wall_seconds":perf_counter()-start,"scope":"fixed_bank_counterfactual_diagnostic_not_selected_prediction"}
    return banks,coverage


def selected_valid(model,data,mods,metric):
    banks,coverage=score_valid_routes(model,data,mods)
    pos,neg=(b["own"] for b in banks)
    require(pos.shape==(60084,4) and neg.shape==(100000,4),"Official diagnostic pools incomplete")
    served=hits50(metric,pos.mean(1),neg.mean(1))
    members=[hits50(metric,pos[:,m],neg[:,m]) for m in range(4)]
    thresholds=torch.topk(neg,50,dim=0).values[-1]
    correct=pos>thresholds
    pair_error=torch.stack([((~correct[:,m]) & (~correct[:,k])).float().mean() for m in range(4) for k in range(m+1,4)]).tolist()
    route_results={name:{"served_hits50":hits50(metric,banks[0][name].mean(1),banks[1][name].mean(1)),
                         "served_balanced_BCE":float((F.softplus(-banks[0][name].mean(1)).mean()+F.softplus(banks[1][name].mean(1)).mean())/2)}
                   for name in banks[0]}
    return {"served_hits50":served,"member_hits50":members,"positive_coincident_error_fractions":pair_error,
            "positive_oracle_union":float(correct.any(1).float().mean()),
            "member_balanced_BCE":((F.softplus(-pos).mean(0)+F.softplus(neg).mean(0))/2).tolist(),
            "served_balanced_BCE":float((F.softplus(-pos.mean(1)).mean()+F.softplus(neg.mean(1)).mean())/2),
            "positive_queries":60084,"negative_queries":100000,"resource_coverage":coverage,
            "fixed_bank_counterfactual_routes":route_results,
            "score_digests":{"positive_bank":tensor_sha(pos),"negative_bank":tensor_sha(neg)},
            "not_a_selector":True,"graph":"complete_TRAIN_only","serving_pool":"mean_raw_logits"}


def run(context,mods,data,sampler,device,output,attempts):
    teacher=ObservationTeacher.from_train(data["pairs"],len(data["x"]))
    results={};supports={};streams={}
    metric=evaluator(context)
    for arm in ("J","F"):
        receipt,payload=load_fit(context,arm)
        model,opt=make_pattern(mods,0,device);restore_snapshot(model,opt,payload["states"][0])
        attempts.phase("selected_complete_member_and_served_VALID_diagnostics",arm=arm)
        valid=selected_valid(model,data,mods,metric)
        require(valid["served_hits50"]==payload["selection"]["hits50"],"Selected diagnostic does not reproduce selected metric")
        # Held-out mask events, not held-out edge truth; label teacher was used
        # in training. Fixed seed and native stream are identical across arms.
        seed_all(DIAGNOSTIC_SEED);train_flag(model,False)
        negative_pairs,iterator,stream=epoch_stream(data,mods,sampler)
        batches=[];support=[]
        attempts.phase("17_fresh_TRAIN_mask_events_source_pattern_diagnostics",arm=arm,completed_batches=0)
        torch.cuda.synchronize(0);started=perf_counter()
        with torch.no_grad():
            for index,ids in enumerate(iterator,start=1):
                main,records=batch_forward(model,data,teacher,negative_pairs,ids,auxiliary_grad=False)
                record={"native_member_main_loss":(-F.logsigmoid(records[0]["logits"]).mean(0)-F.logsigmoid(-records[1]["logits"]).mean(0)).cpu().tolist()}
                for name,row in zip(("positive","negative"),records):
                    record[name]=diagnostic_sums(row["detail"]["t"],row["detail"]["native_q"],row["rows"],row["labels"],len(ids),row["values"],common_rows=row["detail"]["neighbors"].common[0])
                    support.append(teacher.support_digest(row["query"],row["detail"]["neighbors"],row["labels"]))
                batches.append(record);attempts.progress({"completed_batches":index})
        torch.cuda.synchronize(0)
        require(len(batches)==17,"Incomplete diagnostic mask epoch")
        results[arm]={"selected_epoch":payload["selection"]["order"],"VALID":valid,"mask_event_batches":batches,
                      "source_pattern_mask_wall_seconds":perf_counter()-started,"source_teacher":teacher.receipt(),
                      "mask_seed":DIAGNOSTIC_SEED,"TRAIN_stream":stream,"support_digests":support,
                      "auxiliary_scorer_backward":False,"new_optimization_or_selection":False,
                      "scope":"new_mask_events_not_independent_graphs_or_unseen_source_edge_labels"}
        streams[arm]=stream;supports[arm]=support
        del model,opt,payload
    require(streams["J"]==streams["F"] and supports["J"]==supports["F"],"Diagnostic mask/support events differ")
    atomic_json(output/"PRIVATE_DIAGNOSTICS.json",{"identity":context["identity"],"arms":results})
    return {"schema":"ncnc-pattern-diagnostics-complete-v1","identity":context["identity"],"arms":["J","F"],"status":"COMPLETE",
            "private_diagnostics":{"path":"PRIVATE_DIAGNOSTICS.json","bytes":(output/"PRIVATE_DIAGNOSTICS.json").stat().st_size,"sha256":file_sha(output/"PRIVATE_DIAGNOSTICS.json")},
            "full_selected_VALID_evaluations":2,"full_TRAIN_mask_epochs":2,"matched_mask_supports":True,
            "predictive_values_exposed":False,"test_file_opened":False,"UTC":utc()}
