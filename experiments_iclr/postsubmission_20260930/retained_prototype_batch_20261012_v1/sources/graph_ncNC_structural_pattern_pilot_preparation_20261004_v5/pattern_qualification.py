"""Root-released numerical and complete-graph resource stages only."""
import torch
from pilot_common import (require, utc, atomic_json, ENGINEERING_SEED, ENGINEERING_SIGN_SEED)
from pilot_model import train_flag
from pilot_state import snapshot, state_digest, rng_state, rng_digest, restore_rng
from pattern_diagnostics import score_valid_routes
from pattern_model import make_pattern
from pattern_teacher import ObservationTeacher
from checkpoint_parity import checkpoint_parity
from pattern_train import train_epoch
from pattern_checks import (mathematical_checks, fixture, teacher_checks,
                            target_correspondence, serialized_replay)


def run(context, mods, data, sampler, device, output, stage, attempts):
    checks=[]
    if stage == "numerical":
        attempts.phase("fabricated_direct_products_analytical_gradients")
        checks.extend(mathematical_checks(device))
        data=fixture(device)
        teacher=ObservationTeacher.from_train(data["pairs"],len(data["x"]))
        checks.extend(teacher_checks(data,teacher))
        attempts.phase("fabricated_saved_V3_checkpoint_forward_gradient_RNG_two_update_parity")
        parity=checkpoint_parity(mods,data,teacher,device,full_graph=False)
        attempts.phase("fabricated_native_forward_and_gradient_boundary")
        checks.extend(target_correspondence(mods,data,teacher,device)["checks"])
        attempts.phase("fabricated_serialized_model_Adam_RNG_replay")
        replay=serialized_replay(mods,data,teacher,device,output,sign_seed=ENGINEERING_SIGN_SEED)
        checks.append("actual_serialized_next_step_and_serving_replay_J_F")
        return {"schema":"ncnc-pattern-qualification-v1","identity":context["identity"],"stage":stage,"status":"PASS",
                "checks":checks,"replay":replay,"activation_checkpoint_parity":parity,"fabricated_inputs_only":True,"project_metric_computed":False,
                "state_donor":False,"test_file_opened":False,"UTC":utc()}
    require(stage == "full_graph", "Wrong qualification stage")
    attempts.phase("complete_TRAIN_label_only_teacher_index")
    teacher=ObservationTeacher.from_train(data["pairs"],len(data["x"]))
    require(teacher.nodes==235868 and len(data["pairs"])==1179052,"Full official graph required")
    attempts.phase("bounded_four_record_saved_V3_checkpoint_parity_on_complete_real_graph")
    parity=checkpoint_parity(mods,data,teacher,device,full_graph=True)
    attempts.phase("full_graph_native_target_correspondence_and_scorer_boundary")
    correspondence=target_correspondence(mods,data,teacher,device,full_graph=True)
    attempts.phase("full_graph_actual_serialized_next_update_and_served_replay_J_F")
    replay=serialized_replay(mods,data,teacher,device,output,sign_seed=ENGINEERING_SIGN_SEED)
    torch.cuda.synchronize(0)
    probe_peak={"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(0),
                "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(0)}
    attempts.row["observed_CUDA_peak_before_arm_resets"]=probe_peak
    attempts.flush()
    # Fresh after all numerical/replay probes; no probe state or RNG is reused.
    records=[]
    for arm in ("J","F"):
        torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
        attempts.phase("fresh_full_graph_engineering_initialization", arm=arm)
        model,opt=make_pattern(mods,ENGINEERING_SEED,device,engineering_sign_seed=ENGINEERING_SIGN_SEED)
        initial=snapshot(model,opt)
        attempts.phase("complete_17_batch_main_and_auxiliary_epoch",arm=arm)
        train=train_epoch(model,opt,data,mods,sampler,teacher,arm,progress=attempts.progress)
        private=train.pop("private_training_diagnostics")
        atomic_json(output/("PRIVATE_RESOURCE_DIAGNOSTICS_"+arm+".json"),private)
        attempts.phase("complete_official_VALID_resource_no_metric",arm=arm)
        before=rng_digest(rng_state())
        banks,valid=score_valid_routes(model,data,mods)
        require(rng_digest(rng_state())==before,"Resource VALID changes RNG")
        require(all(value.shape==(60084,4) for value in banks[0].values()) and all(value.shape==(100000,4) for value in banks[1].values()),"Full VALID route coverage required")
        torch.cuda.synchronize(0)
        records.append({"arm":arm,"initial_state_sha256":state_digest(initial),"initial_rng_sha256":rng_digest(initial["rng"]),
                        "final_rng_sha256":before,"TRAIN":train,"VALID":valid,
                        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(0),"cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(0)})
        attempts.row["observed_CUDA_peak_before_arm_resets"]={key:max(probe_peak[key],*(r[key] for r in records)) for key in probe_peak}
        attempts.flush()
        del model,opt,banks,initial
    a,b=records
    require(a["initial_state_sha256"]==b["initial_state_sha256"] and a["initial_rng_sha256"]==b["initial_rng_sha256"] and a["TRAIN"]["stream"]==b["TRAIN"]["stream"] and a["final_rng_sha256"]==b["final_rng_sha256"],"J/F constructor/native sampling/dropout work differs")
    require(a["TRAIN"]["residual_slots"]==b["TRAIN"]["residual_slots"] and a["TRAIN"]["synthetic_removed_observed_slots"]==b["TRAIN"]["synthetic_removed_observed_slots"],"J/F candidate/teacher coverage differs")
    require(a["TRAIN"]["support_digests"]==b["TRAIN"]["support_digests"],"J/F actual candidate/link support differs")
    return {"schema":"ncnc-pattern-qualification-v1","identity":context["identity"],"stage":stage,"status":"PASS",
            "teacher":teacher.receipt(),"correspondence":correspondence,"serialized_replay":replay,"records":records,
            "activation_checkpoint_parity":parity,
            "full_TRAIN_epochs":2,"full_VALID_evaluations":2,"full_graph_numerical_and_serialized_replay":True,
            "numerical_probe_peaks":probe_peak,
            **{key:max(probe_peak[key],*(r[key] for r in records)) for key in probe_peak},
            "project_metric_computed":False,"raw_scores_saved":False,"state_donor":False,"test_file_opened":False,"UTC":utc()}
