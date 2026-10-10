"""Disabled independent fits/own selectors/assembled references. No CLI or owner."""
import gc
import math
import resource
import sys
import time
from typed_label_context_factor_source_prototype_20261010_v3.caps import CLOSED
from typed_label_context_factor_source_prototype_20261010_v3 import driver as v3_driver
from typed_label_context_factor_source_prototype_20261010_v3.provider import prepare_static, prepare_seed
from typed_label_context_factor_source_prototype_20261010_v3.runtime_gate import runtime, require, inside, read, binding
from .model import FAMILIES
from .runtime_gate import release_gate
from .session import ReferenceSession, body_spec


def replay(rt, engine, inputs, costs, family, spec, identity, checkpoint, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    torch = rt["torch"]
    caller = engine.capture_rng(rt["numpy"], torch)
    session = saved = None
    try:
        with costs.measure("reference_owned_selected_checkpoint_read"):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        scalar = torch.cuda.amp.GradScaler()  # Never the original lane's master.
        session = ReferenceSession(rt, engine, inputs, scalar, costs, family, spec, identity, caps)
        session.restore(saved)  # Exact V3 full state/buffers/Adam/scaler/RNG checks.
        scores, outputs = session.evaluate("reference_fresh_selected_own_serving")
        diagnostics = {}
        for role in ("TRAIN", "VALID"):
            old, new = saved["outputs"][role], outputs[role]
            require(old["ids"] == new["ids"], "Exact own-selected reference row order")
            expected_contexts = 1 if role == "VALID" else 0
            require(len(old["context_member_logits"]) == len(new["context_member_logits"]) == expected_contexts
                    and all(len(pair) == 2 for pair in (*old["context_member_logits"], *new["context_member_logits"])),
                    "Complete two-context selected replay evidence; no zipped subset")
            row = {"pool_logit_drift": float((old["pool_logits"]-new["pool_logits"]).abs().max().item()),
                   "member_logit_drift": float((old["member_logits"]-new["member_logits"]).abs().max().item()),
                   "context_member_logit_drift": max((float((a-b).abs().max().item())
                       for pa, pb in zip(old["context_member_logits"], new["context_member_logits"])
                       for a, b in zip(pa, pb)), default=0.0),
                   "probability_drift": float((old["probabilities"]-new["probabilities"]).abs().max().item()),
                   "prediction_changes": int(((old["probabilities"]>.5)!=(new["probabilities"]>.5)).sum().item())}
            require(max(row["pool_logit_drift"], row["member_logit_drift"], row["context_member_logit_drift"], row["probability_drift"]) <= .001
                    and row["prediction_changes"] == 0, "Own selected reference numerical replay gate")
            diagnostics[role] = row
        return {"selected_epoch": saved["epoch"], "scores": scores,
                "exact_selected_state_restore": True, "diagnostics": diagnostics,
                "counters": dict(session.counters), "output_bitwise_equality_required": False}, outputs
    finally:
        session = saved = None
        gc.collect()
        torch.cuda.empty_cache()
        engine.restore_rng(rt["numpy"], torch, caller)
        require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), caller), "Reference replay preserves original lane/master stream")


def fit_one(rt, engine, inputs, scalar, family, spec, identity, folder, qualify, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    folder.mkdir(exist_ok=False)
    costs = engine.Costs(folder, rt["torch"], rt["device"])
    torch, started = rt["torch"], time.perf_counter()
    caller = engine.capture_rng(rt["numpy"], torch)
    session = outputs = None
    record = {"complete": False, "status": "started", "family": family, "body_spec": spec,
              "individually_selected": True, "no_shared_loss_or_optimizer": True,
              "qualification_mode": qualify, "TEST_access": False}
    try:
        torch.cuda.reset_peak_memory_stats(rt["device"])
        session = ReferenceSession(rt, engine, inputs, scalar, costs, family, spec, identity, caps)
        record["ownership"] = session.bank.verify_ownership()
        fitted = v3_driver.fit(session, folder, qualify=qualify)
        record.update(fitted)
        require(fitted["counters"]["TRAIN_query_rows"] == fitted["counters"]["epochs"]*len(inputs.train.ids)
                and fitted["counters"]["TRAIN_native_calls"] == 2*fitted["counters"]["epochs"], "Complete once-per-target own supervision")
        master_scalar = engine.cpu_tree(torch, scalar.state_dict())
        session = None
        gc.collect()
        torch.cuda.empty_cache()
        record["fresh_selected"], outputs = replay(rt, engine, inputs, costs, family, spec, identity,
                                                  folder/"SELECTED_STATE.pt", caps)
        require(engine.exact(torch, scalar.state_dict(), master_scalar), "Reference own lane scaler survives fresh replay")
        record.update(complete=True, status="complete", qualification_passed=qualify,
                      own_selector="strict complete own VALID two-context raw-logit BCE; earliest ties;200/50",
                      independent_initialization_seed=spec["native_initialization_seed"])
        return record, outputs
    except BaseException as error:
        record.update(status="failed", complete=False, error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        active_error, cleanup_error = sys.exc_info()[0] is not None, None
        try:
            session = None
            gc.collect()
            torch.cuda.empty_cache()
            engine.restore_rng(rt["numpy"], torch, caller)
            require(engine.exact(torch, engine.capture_rng(rt["numpy"], torch), caller), "Fit cleanup preserves caller/master stream")
            record.update(cleanup_completed=True,
                          peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(rt["device"]),
                          peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(rt["device"]))
        except BaseException as error:
            cleanup_error = error
            record.update(status="failed", complete=False, cleanup_completed=False,
                          cleanup_error={"type": type(error).__name__, "message": str(error)})
        finally:
            record.update(seconds=time.perf_counter()-started,
                          cumulative_RSS_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)),
                          RSS_is_cumulative=True)
            v3_driver.write(folder/"RESULT.json", record)
        if cleanup_error is not None and not active_error:
            raise cleanup_error


def assemble(rt, inputs, selected, records, family, folder, costs, caps=CLOSED):
    """No joint selector: exactly four original independently selected states."""
    caps.require("source_bound", "model", "data", "runtime")
    torch = rt["torch"]
    require(family in FAMILIES and len(selected) == len(records) == 4
            and [r["body_spec"]["body_index"] for r in records] == list(range(4))
            and all(r["complete"] and r["family"] == family for r in records), "All four ordered successful own selectors/replays before assembly")
    report, payload = {}, {}
    for role, truth in (("TRAIN", inputs.train.targets), ("VALID", inputs.validation.targets)):
        ids = selected[0][role]["ids"]
        require(all(s[role]["ids"] == ids and s[role]["member_logits"].shape == (1, len(ids), 5) for s in selected), "Same complete selected reference cohort")
        with costs.measure("selected_four_body_"+role+"_CPU_FP32_assembly"):
            bank = torch.cat([s[role]["member_logits"] for s in selected], dim=0)
            require(bank.dtype == torch.float32 and bank.device.type == "cpu", "Native selected reference CPU FP32 pool")
            pool = bank.mean(dim=0)  # CPU FP32, exactly the V3 post-context mean rule.
            probabilities = torch.sigmoid(pool)
            micro, macro = rt["native"].evaluator(truth, (probabilities>.5).int())
            bce = float(torch.nn.functional.binary_cross_entropy_with_logits(pool, truth.float()).item())
            require(all(math.isfinite(x) for x in (bce, float(micro), float(macro)))
                    and torch.isfinite(bank).all().item(), "Complete finite independently selected committee")
        report[role] = {"BCE": bce, "micro_F1": float(micro), "macro_F1": float(macro),
                        "member_scores": [r["fresh_selected"]["scores"][role] for r in records]}
        payload[role] = {"ids": ids, "member_logits": bank, "pool_logits": pool, "probabilities": probabilities}
    state = {"family": family, "body_selected_epochs": [r["selected_epoch"] for r in records],
             "body_checkpoint_bindings": [r["checkpoint"] for r in records], "outputs": payload,
             "source": "individually selected states; no fitted voting rule or common epoch"}
    target = folder/"INDIVIDUALLY_SELECTED_ENSEMBLE.pt"
    with costs.measure("selected_four_body_payload_serialization") as event:
        torch.save(state, target)
        event["serialized_bytes"] = target.stat().st_size
    return {"complete": True, "family": family, "scores": report,
            "selected_body_epochs": state["body_selected_epochs"], "selected_payload": binding(target),
            "four_body_training_paths_and_four_selectors_and_replays_charged": True,
            "four_full_fits_charged": all(not r["qualification_mode"] for r in records),
            "qualification_is_one_update_per_body": all(r["qualification_mode"] for r in records),
            "selected_replay_body_calls": 8, "no_joint_or_posthoc_selector": True,
            "ordinary_native_lacks_extra_mass_information": family == "ordinary_native_independent4"}


def execute(release_path, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    release, roles, sources, release_binding = release_gate(release_path, caps)
    output = inside(release["output_directory"], exists=False)
    output.mkdir(parents=True, exist_ok=False)
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    qualify = release["action"] == "qualify_independently_selected_references"
    report = {"complete": False, "status": "started", "qualification_passed": False,
              "mode": "qualification" if qualify else "scientific_reference_acquisition",
              "reference_source_seal_sha256": release["reference_source_seal_sha256"],
              "reference_protocol_sha256": release["reference_protocol_sha256"],
              "roles": release["roles"], "input_files": release["development_files"],
              "seed_specs": release["seed_specs"],
              "release": release_binding, "families": list(FAMILIES), "runs": [], "ensembles": [],
              "TEST_access": False, "automatic_retry": False, "superiority_or_novelty_admitted": False}
    rt = static = operators = None
    v3_driver.write(output/"COHORT_REPORT.json", report)
    try:
        rt, engine, loader, report["runtime"] = runtime(release, sources, caps)
        costs = engine.Costs(output, rt["torch"], rt["device"])
        static, operators = prepare_static(rt, engine, loader, release, sources, roles, costs, caps)
        initial, contexts = {}, {}
        for family in FAMILIES:
            family_folder = output/family
            family_folder.mkdir()
            # Separate private AMP lane per body; each persists over ordered roles.
            scalars = [rt["torch"].cuda.amp.GradScaler() for _ in range(4)]
            for role_spec in release["seed_specs"]:
                role_seed = role_spec["role_seed"]
                role_folder = family_folder/("role"+str(role_seed))
                role_folder.mkdir()
                inputs = prepare_seed(rt, static, operators, roles[role_seed], release["roles"][str(role_seed)], role_spec, costs, caps)
                require(contexts.setdefault(role_seed, inputs.paired.plan.identity_sha256) == inputs.paired.plan.identity_sha256, "Identical contexts across both references")
                selected, members = [], []
                for member in range(4):
                    spec = body_spec(role_spec, member)
                    identity = {"reference_source_seal_sha256": release["reference_source_seal_sha256"],
                                "reference_protocol_sha256": release["reference_protocol_sha256"],
                                "candidate_source_seal_sha256": release["source_seal_sha256"],
                                "candidate_protocol_sha256": release["protocol_sha256"], "family": family,
                                "body_spec": spec, "role_binding": release["roles"][str(role_seed)], "release": release_binding}
                    folder = role_folder/("body"+str(member))
                    try:
                        row, served = fit_one(rt, engine, inputs, scalars[member], family, spec, identity, folder, qualify, caps)
                    finally:
                        if (folder/"RESULT.json").exists():
                            report["runs"].append(read(folder/"RESULT.json"))
                            v3_driver.write(output/"COHORT_REPORT.json", report)
                    require(row["complete"] and (not qualify or row["qualification_passed"]), "No successful-body subset promotion")
                    key = (role_seed, member)
                    require(initial.setdefault(key, row["native_initial_digest"]) == row["native_initial_digest"], "Matched native starts across ordinary/contextual references")
                    members.append(row)
                    selected.append(served)
                require(len({r["native_initial_digest"] for r in members}) == 4, "Four genuinely different complete native initial states")
                report["ensembles"].append(assemble(rt, inputs, selected, members, family, role_folder, costs, caps))
                selected = members = inputs = None
                gc.collect()
        require(len(report["runs"]) == 8*len(release["seed_specs"]) and len(report["ensembles"]) == 2*len(release["seed_specs"]), "Complete fixed reference roster")
        require(loader.bindings(loader.permitted_files(release["input_root"])) == release["development_files"], "Inputs unchanged throughout reference acquisition")
        authority = [release_binding, *release["roles"].values(), release["root_review"],
                     release["schema_receipt"], release["native_qualification_receipt"], release["candidate_qualification_receipt"]]
        require(all(binding(row["path"]) == row for row in authority), "Reference authority/input qualification descriptors unchanged throughout acquisition")
        report.update(complete=True, status="complete", qualification_passed=qualify,
                      matched_native_initial_digests={str(seed): [initial[seed, m] for m in range(4)] for seed in contexts},
                      matched_context_identity_by_role=contexts,
                      observed_results_do_not_admit_candidate_science=True)
    except BaseException as error:
        report.update(complete=False, status="failed", qualification_passed=False,
                      error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if rt is not None:
            static = operators = None
            gc.collect()
            rt["torch"].cuda.empty_cache()
        final = resource.getrusage(resource.RUSAGE_SELF)
        report.update(seconds_including_shared_full_graph_preparation=time.perf_counter()-started,
                      CPU_user_seconds=final.ru_utime-usage.ru_utime,
                      CPU_system_seconds=final.ru_stime-usage.ru_stime,
                      cumulative_process_RSS_peak_bytes=int(final.ru_maxrss*(1 if sys.platform == "darwin" else 1024)),
                      nested_cost_event_seconds_must_not_be_summed=True)
        v3_driver.write(output/"COHORT_REPORT.json", report)
    return report
