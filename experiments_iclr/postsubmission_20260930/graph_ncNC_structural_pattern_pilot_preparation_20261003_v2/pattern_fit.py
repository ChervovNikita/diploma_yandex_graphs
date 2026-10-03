"""Exactly two independently served seed0 fits, 100 epochs each."""
from pathlib import Path
import torch
from pilot_common import require, utc, file_sha
from pilot_state import (snapshot, restore_snapshot, rng_state, rng_digest, state_digest,
                         write_journal, read_journal, write_selected)
from pilot_evaluate import score_valid, hits50
from pattern_model import make_pattern
from pattern_teacher import ObservationTeacher
from pattern_train import train_epoch
from pattern_checks import close


def fit(context,mods,data,sampler,metric,device,output,arm,resume,attempts):
    teacher=ObservationTeacher.from_train(data["pairs"],len(data["x"]))
    model,opt=make_pattern(mods,0,device)
    if resume:
        state=read_journal(output,context,arm,0)
        require(state["arm"]==arm and state["fresh_scientific_initialization"] is True and state["resource_state_donor"] is False,"Wrong own fit provenance")
        restore_snapshot(model,opt,state["current"])
    else:
        initial=snapshot(model,opt)
        state={"arm":arm,"epoch":0,"phase":"epochs","current":initial,"best":None,"best_state":None,"epochs":[],
               "initial_state_sha256":state_digest(initial),"initial_rng_sha256":rng_digest(initial["rng"]),
               "fresh_scientific_initialization":True,"resource_state_donor":False,"teacher":teacher.receipt()}
        write_journal(output,context,arm,0,state)
    require(0<=state["epoch"]<=100 and len(state["epochs"])==state["epoch"],"Own journal epoch coverage differs")
    for epoch in range(state["epoch"]+1,101):
        restore_snapshot(model,opt,state["current"])
        start_rng=rng_digest(rng_state())
        attempts.phase("native_TRAIN_with_pattern_auxiliary",arm=arm,epoch=epoch,completed_batches=0,attempted_batch=0)
        train=train_epoch(model,opt,data,mods,sampler,teacher,arm,progress=attempts.progress)
        train_rng=rng_state()
        attempts.phase("complete_official_VALID_selection",epoch=epoch)
        before=rng_digest(train_rng)
        pos,neg,valid=score_valid(model,data,mods,mode="private")
        require(rng_digest(rng_state())==before,"VALID consumed training RNG")
        quality=hits50(metric,pos,neg)
        current=snapshot(model,opt,rng=train_rng)
        best,replace=mods["design"].select_validation_candidate(state["best"],candidate_id="epoch_"+str(epoch),hits50=quality,order=epoch)
        state["best"]=best
        if replace:state["best_state"]=current
        state["current"]=current
        state["epochs"].append({"epoch":epoch,"start_rng_sha256":start_rng,"end_rng_sha256":before,"TRAIN":train,"VALID":valid,"private_hits50":quality})
        state["epoch"]=epoch
        attempts.phase("durable_complete_epoch_commit",epoch=epoch)
        write_journal(output,context,arm,0,state)
        print("EPOCH_COMPLETE arm="+arm+" seed=0 epoch="+str(epoch)+"/100",flush=True)
    require(state["epoch"]==100 and len(state["epochs"])==100,"Incomplete fixed fit")
    state["phase"]="complete"
    write_journal(output,context,arm,0,state)
    attempts.phase("own_selected_checkpoint_serialization_and_complete_serving_replay")
    restore_snapshot(model,opt,state["best_state"])
    ref_rng=rng_digest(rng_state())
    pos,neg,reference=score_valid(model,data,mods,mode="private")
    ref_quality=hits50(metric,pos,neg)
    require(ref_quality==state["best"]["hits50"] and rng_digest(rng_state())==ref_rng,"Selected serving does not reproduce chosen score/RNG")
    pin=write_selected(output,context,arm,arm,0,state["best"],[state["best_state"]])
    path=Path(output)/pin["path"]
    require(path.stat().st_size==pin["bytes"] and file_sha(path)==pin["sha256"],"Own selected file differs")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    require(payload["identity"]==context["identity"] and payload["arm"]==arm and payload["seed"]==0 and payload["test_stage_supported"] is False,"Selected identity differs")
    restored,ropt=make_pattern(mods,0,device)
    restore_snapshot(restored,ropt,payload["states"][0])
    require(state_digest(snapshot(restored,ropt))==state_digest(state["best_state"]),"Selected typed model/Adam/RNG/flags roundtrip differs")
    replay_rng=rng_digest(rng_state())
    rpos,rneg,replay=score_valid(restored,data,mods,mode="private")
    close(rpos,pos,"selected complete VALID positive replay"); close(rneg,neg,"selected complete VALID negative replay")
    require(hits50(metric,rpos,rneg)==ref_quality and rng_digest(rng_state())==replay_rng,"Selected replay metric/RNG differs")
    def json_receipt(name):
        path=Path(output)/name
        return {"path":name,"bytes":path.stat().st_size,"sha256":file_sha(path)}
    return {"schema":"ncnc-pattern-complete-fit-v1","identity":context["identity"],"arm":arm,"seed":0,"epochs":100,
            "optimizer_steps":1700,"selection_candidates":100,"selected_checkpoint":pin,
            "initial_state_sha256":state["initial_state_sha256"],"initial_rng_sha256":state["initial_rng_sha256"],
            "private_selection":json_receipt("PRIVATE_SELECTION_"+arm+".json"),"journal":json_receipt("JOURNAL.json"),
            "teacher":teacher.receipt(),"epoch_streams":[{"epoch":r["epoch"],"start_rng_sha256":r["start_rng_sha256"],"end_rng_sha256":r["end_rng_sha256"],"stream":r["TRAIN"]["stream"],
                "support_digests":[b["support_digest"] for b in r["TRAIN"]["private_training_diagnostics"]]} for r in state["epochs"]],
            "selected_roundtrip_and_full_served_replay":True,"extra_complete_VALID_replay_evaluations":2,
            "full_VALID_evaluations":102,"reference_replay":reference,"serialized_replay":replay,
            "predictive_values_exposed":False,"resource_state_donor":False,"test_file_opened":False,"UTC":utc()}
