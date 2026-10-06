#!/usr/bin/env python3
"""Root-released detached owner: assigned qualification, then fixed b0 waves.

Uses the reviewed ownership helper and existing one-child supervision loop.
No numerical imports, payloads, score selection, other-process signals or retry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from types import SimpleNamespace

REPO = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git")
PHASE = REPO/"experiments_iclr/postsubmission_20260930"
SOURCE = PHASE/"citeseer_known_ranking_control_gpu77_b0_preparation_20261006_v1"
SOURCE_SHA = "dd09a2d21fc18032ed87f69b224a8ed4f15212ee6bb0c933bce80f5a14c5d52d"
COMMAND_SHA = "cf4ee9057eb664ce8d64c2aba67091b1128ee4fbd4a86b4ae86795f6457e3056"
PLAN_SHA = "7f873be9f5c4a7ae276632ba986c9b6662a4a988825052835e838f6950a0ab7b"
GPU_UUIDS = ("GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998", "GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced")
HELPER = PHASE/"shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py"
HELPER_SHA = "e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd"
FIT_ROOT = PHASE/"citeseer_known_ranking_control_gpu77_b0_execution_root_20261006_v1"
QUAL_ROOT = PHASE/"citeseer_known_ranking_control_gpu77_b0_qualification_root_20261006_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value): raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def phase_file(relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts: raise ValueError("Require phase-relative path")
    path = (PHASE/value).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve(strict=True)) or not path.is_file(): raise ValueError("File leaves authorized phase")
    return path


def physical_host():
    if platform.system() != "Linux" or socket.gethostname() != "peptide" or Path.cwd().resolve() != REPO.resolve():
        raise ValueError("Require exact authorized peptide repository")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10)
    if tuple(rows.split()) != GPU_UUIDS: raise ValueError("Physical GPU inventory changed")


def verify_source():
    if sha(SOURCE/"SOURCE_MANIFEST.json") != SOURCE_SHA or sha(SOURCE/"COMMANDS_DISABLED.json") != COMMAND_SHA or sha(SOURCE/"PLAN.json") != PLAN_SHA:
        raise ValueError("Reviewed source/commands/plan bytes changed")
    for row in read(SOURCE/"SOURCE_MANIFEST.json")["files"]:
        path = SOURCE/row["path"]
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]: raise ValueError("Reviewed source file changed")
    if sha(HELPER) != HELPER_SHA: raise ValueError("Reviewed ownership helper changed")


def lane(uuid):
    # Separate module globals per physical GPU; concurrent lanes cannot rebind
    # one another's helper.GPU value.
    spec = importlib.util.spec_from_file_location("rank77_owned_"+uuid[-4:], HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helper.GPU = uuid
    context = SimpleNamespace(REPO=REPO, SOURCE=SOURCE, SOURCE_SHA=SOURCE_SHA, GPU_UUID=uuid,
        GPU_UUIDS=GPU_UUIDS, phase_file=phase_file, physical_host=physical_host, sha=sha, write=write)
    return helper, context


def output_bytes(path):
    """Only this freshly created fit output, without following any symlink."""
    root=Path(path)
    if not root.exists():return 0
    total=0;pending=[root]
    while pending:
        with os.scandir(pending.pop()) as items:
            for item in items:
                if item.is_symlink():raise ValueError('Own fit output contains a symlink; refuse traversal')
                if item.is_dir(follow_symlinks=False):pending.append(Path(item.path))
                elif item.is_file(follow_symlinks=False):total+=item.stat(follow_symlinks=False).st_size
                else:raise ValueError('Unexpected nonregular entry in own fit output')
    return total


def await_resources(s,limits,receipt_path,c):
    """Fixed finite pre-child resource window; no fit starts or method is skipped."""
    started=time.monotonic();attempts=0
    while True:
        c.physical_host();attempts+=1
        rows=s.query(['--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],limits['telemetry_timeout_seconds'])
        if tuple(row.split(',')[0].strip() for row in rows)!=c.GPU_UUIDS:
            raise ValueError('Physical two-GPU inventory changed')
        selected=next(row for row in rows if row.split(',')[0].strip()==c.GPU_UUID)
        uuid,free=[part.strip() for part in selected.split(',')]
        if uuid!=c.GPU_UUID or not free.isdigit():raise ValueError('Authorized GPU/free-memory telemetry differs')
        status={'UTC':s.now(),'GPU_UUID':uuid,'GPU_free_bytes':int(free)*1024**2,'attempts':attempts,
                'elapsed_seconds':time.monotonic()-started,'scientific_child_started':False}
        c.write(receipt_path,status)
        if status['GPU_free_bytes']>=limits['minimum_fresh_GPU_free_bytes']:return status
        remaining=limits['resource_wait_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Fixed fresh-GPU resource window exhausted before child launch')
        time.sleep(min(limits['poll_interval_seconds'],remaining))


def run_fit(s,root,entry,queue,environment,output,c):
    cell=entry['cell_id'];job=c.phase_file(entry['job_relative'])
    limits=dict(queue['resource_limits'],external_hard_seconds_per_cell=entry['hard_seconds'])
    await_resources(s,limits,root/'logs'/(cell+'.PREFLIGHT.json'),c)
    argv=entry['argv']
    stdout_path=root/'logs'/(cell+'.stdout.log');stderr_path=root/'logs'/(cell+'.stderr.log')
    started=time.monotonic();reason=None;owner=None;raw_owner=None;cwd=None;signals=[];refusal=None;observed=False
    max_rss=max_gpu=max_logs=max_output=0;incomplete_count=0;last_incomplete=None
    with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,cwd=c.REPO,env=environment,
                                 stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            raw_owner=s.identity(process.pid)
            cwd=(Path('/proc')/str(process.pid)/'cwd').resolve(strict=True)
            if raw_owner is None or raw_owner['argv']!=argv or raw_owner['pgid']!=process.pid or raw_owner['sid']!=process.pid or cwd!=c.REPO.resolve():
                reason='Initial fresh owned child/session/argv/cwd could not be verified; no signal authorized'
            else:owner=raw_owner
        except Exception as error:
            reason='Initial identity observation: '+type(error).__name__+': '+str(error)
        c.write(root/'logs'/(cell+'.CURRENT_PROCESS.json'),{'UTC':s.now(),'cell_id':cell,'identity':owner,'argv':argv,
                'raw_identity_observation':raw_owner,'identity_admitted_for_signals':owner is not None,
                'cwd':str(c.REPO),'observed_cwd':str(cwd) if cwd is not None else None,'limits':limits,'source_manifest_sha256':c.SOURCE_SHA})
        c.write(root/'logs'/(cell+'.CHILD_STARTED.json'),{'UTC':s.now(),'identity':owner,'argv':argv,
                'raw_identity_observation':raw_owner,'identity_admitted_for_signals':owner is not None,
                'cwd':str(c.REPO),'observed_cwd':str(cwd) if cwd is not None else None,'limits':limits})
        try:
            while process.poll() is None:
                if reason is not None:break
                remaining=entry['hard_seconds']-(time.monotonic()-started)
                if remaining<=0:reason='external_hard_wall_bound'
                else:
                    rows=s.owned_tree(owner)
                    incomplete=[row for row in rows if not row.get('observation_complete',True)]
                    if incomplete:
                        incomplete_count+=1;last_incomplete={'UTC':s.now(),'rows':incomplete}
                        if process.poll() is not None:break
                    rss,gpu=s.resources(rows,limits,remaining)
                    max_rss=max(max_rss,rss);max_gpu=max(max_gpu,gpu)
                    max_logs=max(max_logs,stdout_path.stat().st_size+stderr_path.stat().st_size)
                    max_output=max(max_output,output_bytes(output))
                    if max_rss>limits['owned_tree_RSS_cap_bytes']:reason='owned_tree_RSS_cap'
                    elif max_gpu>limits['owned_tree_GPU_memory_cap_bytes']:reason='owned_tree_GPU_memory_cap'
                    elif max_logs>limits['combined_child_log_cap_bytes']:reason='combined_child_log_cap'
                    elif max_output>limits['own_fit_output_cap_bytes']:reason='own_fit_output_cap'
                    elif time.monotonic()-started>=entry['hard_seconds']:reason='external_hard_wall_bound'
                    c.write(root/'logs'/(cell+'.CURRENT_RESOURCES.json'),{'UTC':s.now(),'cell_id':cell,'identity':owner,
                            'elapsed_seconds':time.monotonic()-started,'owned_RSS_bytes':rss,'owned_GPU_bytes':gpu,
                            'own_output_bytes':max_output,'own_log_bytes':max_logs,'scores_read':False})
                if reason:
                    try:
                        sent=s.kill_owned(process,owner,reason)
                        if sent:signals.append(sent)
                    except s.IncompleteExitObservation as error:refusal=type(error).__name__+': '+str(error)
                    break
                time.sleep(min(limits['poll_interval_seconds'],max(.01,remaining)))
        except (Exception,KeyboardInterrupt) as error:
            reason='required_bounds_telemetry_failure: '+type(error).__name__+': '+str(error)
            try:
                sent=s.kill_owned(process,owner,reason) if owner is not None else None
                if sent:signals.append(sent)
            except Exception as error:refusal=type(error).__name__+': '+str(error)
        finally:
            reason,refusal,observed=s.observe_terminal(process,owner,started,limits,signals,reason,refusal)
    max_logs=max(max_logs,stdout_path.stat().st_size+stderr_path.stat().st_size)
    max_output=max(max_output,output_bytes(output))
    if reason is None and max_logs>limits['combined_child_log_cap_bytes']:reason='combined_child_log_cap_at_exit'
    if reason is None and max_output>limits['own_fit_output_cap_bytes']:reason='own_fit_output_cap_at_exit'
    receipt={'UTC':s.now(),'cell_id':cell,'job_sha256':entry['job_sha256'],
        'exit_code':process.returncode if observed else None,'exit_code_authority':'subprocess.Popen.wait/poll' if observed else None,
        'terminal_wait_observed':observed,'child_identity':owner,'cwd':str(c.REPO),'observed_cwd':str(cwd) if cwd is not None else None,'reason':reason,'signals_sent':signals,
        'raw_identity_observation':raw_owner,'identity_admitted_for_signals':owner is not None,
        'signal_refusal':refusal,'incomplete_exit_observation_samples':incomplete_count,'last_incomplete_exit_observation':last_incomplete,
        'elapsed_seconds':time.monotonic()-started,'max_sampled_owned_RSS_bytes':max_rss,'max_sampled_owned_GPU_bytes':max_gpu,
        'max_sampled_own_output_bytes':max_output,'max_sampled_own_log_bytes':max_logs,
        'stdout_sha256':c.sha(stdout_path),'stderr_sha256':c.sha(stderr_path),'scores_read':False,'attempts':1,'retry':False}
    c.write(root/'logs'/(cell+'.EXIT.json'),receipt)
    return receipt


def run_child(index, commands, jobs, owner_root, limits, phase_started, phase_seconds):
    command = commands[index]; argv = command["argv"]; job_path = Path(argv[4]); output = Path(argv[6])
    job = jobs[index]
    verify_source(); physical_host()
    if sha(job_path) != job["sha256"]: raise ValueError("Root-bound activated job changed")
    if output.exists() or not output.parent.is_dir(): raise ValueError("Fresh child output required; no retry")
    if phase_seconds-(time.monotonic()-phase_started) < command["external_hard_seconds"]+limits["resource_wait_seconds"]:
        raise TimeoutError("Fixed phase budget cannot cover next whole child/wait")
    s, c = lane(command["env"]["CUDA_VISIBLE_DEVICES"])
    environment = dict(os.environ, **command["env"]); environment.pop("PYTHONHOME", None)
    cell = ("qualification_"+job_path.parent.name) if index < 2 else job_path.stem
    entry = {"cell_id": cell, "job_relative": str(job_path.relative_to(PHASE)), "job_sha256": job["sha256"],
             "hard_seconds": command["external_hard_seconds"], "argv": argv}
    receipt = run_fit(s, owner_root, entry, {"resource_limits": limits}, environment, output, c)
    child = receipt["raw_identity_observation"]
    if child is None: raise ValueError("Owned child identity was not observed")
    absent = s.identity(child["PID"]) is None
    rows = s.query(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], limits["telemetry_timeout_seconds"])
    no_cuda = not any(len(parts) >= 2 and parts[1].strip() == str(child["PID"]) for parts in (r.split(",") for r in rows))
    physical = {"UTC": s.now(), "child_PID": child["PID"], "child_start_ticks": child["start_ticks"],
        "physical_gpu_uuid": c.GPU_UUID, "owned_PID_absent": absent, "owned_PID_no_CUDA_rows": no_cuda,
        "terminal_wait_observed": receipt["terminal_wait_observed"], "compute_rows": rows}
    write(owner_root/"logs"/(cell+".PHYSICAL_TERMINAL.json"), physical)
    if receipt["exit_code"] != 0 or receipt["reason"] is not None or receipt["signals_sent"] or not receipt["terminal_wait_observed"] or not absent or not no_cuda:
        raise RuntimeError(cell+" failed owned terminal closure; preserve artifacts, no retry")
    if (output/"FAILURE.json").exists(): raise ValueError("Child failure record remains")
    if index >= 2:
        freeze_path = output/"FREEZE.json"; freeze = read(freeze_path)
        # Inspect only completion/source/job custody; never selection or scores.
        if freeze["last_cycle"] != 60 or freeze["source_manifest_sha256"] != SOURCE_SHA or freeze["job_sha256"] != job["sha256"]:
            raise ValueError("Fit lacks exact complete sixty-cycle source/job freeze")
        receipt.update(freeze_path=str(freeze_path.relative_to(PHASE)), freeze_sha256=sha(freeze_path))
    return receipt


def parallel(indices, *args):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_child, index, *args) for index in indices]
        return [future.result() for future in futures]


def admit_actual_gates(commands, jobs, owner_root, release):
    combined = {"scope": "discarded_FP32_training_step_qualification", "passed": True,
        "source_manifest_sha256": SOURCE_SHA, "ordinary_three_pass_qualification": True,
        "architectures": {}, "source_receipts": [], "fits": 0, "VALID_TEST_values_access": False, "states_discarded": True}
    cost_evidence = []; cost_decisions = []
    geometry = None
    arm_fit = {jobs[i]["value"]["arm"]: jobs[i]["value"] for i in (2, 3, 4)}
    for index in (0, 1):
        output = Path(commands[index]["argv"][6]); path = output/"RESULT.json"; result = read(path)
        job = jobs[index]["value"]
        if result.get("passed") is not True or result.get("ordinary_three_pass_qualification") is not True or result["source_manifest_sha256"] != SOURCE_SHA or result["job_sha256"] != jobs[index]["sha256"]:
            raise ValueError("Exact actual ordinary qualification did not pass")
        if result["physical_gpu_uuid"] != job["physical_gpu_uuid"] or set(result["architectures"]) != set(job["architectures"]):
            raise ValueError("Assigned architecture/physical GPU qualification differs")
        if result.get("states_discarded") is not True or result.get("VALID_TEST_values_access") is not False or result.get("fits") != 0 or result.get("native_alias_restored") is not True or result["tolerances"] != job["tolerances"]:
            raise ValueError("TRAIN-only discarded qualification custody differs")
        if geometry is None: geometry = result["geometry"]
        elif geometry != result["geometry"]: raise ValueError("Assigned first-episode geometry differs")
        combined["source_receipts"].append({"path": str(path.relative_to(PHASE)), "sha256": sha(path)})
        for arm, architecture in result["architectures"].items():
            if arm in combined["architectures"] or architecture.get("passed") is not True: raise ValueError("Architecture gate failed/duplicated")
            cost_path = output/(arm+"_COST_RESULT.json"); cost = read(cost_path)
            if cost != architecture["complete_cycle_cost"] or cost["source_manifest_sha256"] != SOURCE_SHA or cost["physical_gpu_uuid"] != job["physical_gpu_uuid"]:
                raise ValueError("Raw measured new-loss cost custody differs")
            if cost["last_cycle"] != 1 or cost["counters"]["episodes"] != 61 or cost["counters"]["ordinary_joint_updates"] != 183 or cost["exposure"]["outer_positive_instances"] != 3870 or cost["exposure"]["outer_positive_ids_unseen"] != 0:
                raise ValueError("Measured discarded TRAIN cycle incomplete")
            if cost["states_discarded"] is not True or cost["VALID_TEST_values_access"] is not False: raise ValueError("Cost gate accessed forbidden values")
            measured = cost["inclusive_seconds"]*60
            ceiling = release["TRAIN_cost_fraction_of_fit_soft_cap"]*arm_fit[arm]["soft_seconds"]
            if measured > ceiling: raise ValueError("New measured TRAIN cost exceeds fixed prospective fit budget allowance: "+arm)
            cost_decisions.append({"arm": arm, "measured_one_cycle_inclusive_seconds": cost["inclusive_seconds"],
                "sixty_cycle_TRAIN_projection_seconds": measured, "fixed_TRAIN_allowance_seconds": ceiling, "passed": True})
            cost_evidence.append({"path": str(cost_path.relative_to(PHASE)), "sha256": sha(cost_path)})
            combined["architectures"][arm] = architecture
    if set(combined["architectures"]) != {"shared_f4", "capable_single", "ordinary_native4"}: raise ValueError("All three new ordinary architecture gates required")
    combined["geometry"] = geometry
    gate_path = QUAL_ROOT/"ALL_THREE_GATE.json"
    if gate_path.exists(): raise ValueError("No prior gate overwrite")
    write(gate_path, combined)
    plan = read(SOURCE/"PLAN.json")
    plan.update(root_adopted_after_TRAIN_cost=True, complete_cycle_cost_evidence=cost_evidence,
        selection_budget_fairness_approved=release["selection_budget_fairness_approved"], fits_authorized=True, launch_authorized=True)
    plan["actual_new_loss_cost_admission"] = {"root_release_sha256": sha(FIT_ROOT/"ROOT_RELEASE.json"),
        "TRAIN_cost_fraction_of_fit_soft_cap": release["TRAIN_cost_fraction_of_fit_soft_cap"], "decisions": cost_decisions,
        "no_cap_expansion_or_retry": True, "qualification_terminal_receipts": [str((owner_root/"logs"/("qualification_"+label+".EXIT.json")).relative_to(PHASE)) for label in ("a998", "8ced")]}
    plan_path = FIT_ROOT/"PLAN.json"
    if plan_path.exists(): raise ValueError("No prior adopted plan overwrite")
    write(plan_path, plan)
    for index in (2, 3, 4):
        job = copy.deepcopy(jobs[index]["value"])
        # The root already approves fit scope/source/budgets. Only evidence that
        # did not exist before measured qualification is filled by this owner.
        job["training_step_gate"] = {"approved": True, "path": str(gate_path.relative_to(PHASE)), "sha256": sha(gate_path)}
        job["cohort_plan_relative"] = str(plan_path.relative_to(PHASE)); job["cohort_plan_sha256"] = sha(plan_path)
        path = Path(commands[index]["argv"][4]); write(path, job)
        jobs[index] = {"value": job, "sha256": sha(path)}
    write(owner_root/"GATE_ADMISSION.json", {"passed": True, "gate_sha256": sha(gate_path), "plan_sha256": sha(plan_path),
        "source_receipts": combined["source_receipts"], "cost_evidence": cost_evidence, "cost_decisions": cost_decisions,
        "activated_fit_jobs": [{"path": commands[i]["argv"][4], "sha256": jobs[i]["sha256"]} for i in (2, 3, 4)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args(); release = read(args.release)
    if args.release.resolve(strict=True) != FIT_ROOT/"ROOT_RELEASE.json": raise ValueError("Exact root release path required")
    if release.get("execution_enabled") is not True or release.get("qualification_authorized") is not True or release.get("fits_conditionally_authorized") is not True or release.get("selection_budget_fairness_approved") is not True:
        raise ValueError("Root conditional qualification plus first-b0 fit approval required")
    if release.get("retry") is not False or release.get("TEST_access") is not False or release.get("TRAIN_cost_fraction_of_fit_soft_cap") != .8:
        raise ValueError("Require fixed cost margin, closed TEST and no retry")
    if sha(__file__) != release["owner_program_sha256"]: raise ValueError("Reviewed detached owner changed")
    verify_source(); physical_host()
    contract = read(SOURCE/"COMMANDS_DISABLED.json"); commands = contract["commands"]; limits = contract["resource_limits"]
    if len(release["jobs"]) != 5: raise ValueError("Require exact two qualification and three b0 job bindings")
    jobs = []
    for index, binding in enumerate(release["jobs"]):
        path = phase_file(binding["path"])
        if path != Path(commands[index]["argv"][4]) or sha(path) != binding["sha256"]: raise ValueError("Root job/order binding differs")
        job = read(path)
        preview_path = phase_file(commands[index]["job_preview"]["path"])
        if sha(preview_path) != commands[index]["job_preview"]["sha256"]: raise ValueError("Reviewed disabled job preview changed")
        preview = read(preview_path)
        scientific = ("seed", "factor_seed", "outer_size", "inner_size", "soft_seconds", "runtime_versions", "cpu_threads", "cpu_interop_threads", "expected_hostname")
        scientific += ("architectures", "reachable_history_steps", "full_TRAIN_cycles_per_architecture", "episode_index", "tolerances") if index < 2 else ("arm", "rule", "geometry", "schedule", "paired_seed_block", "cell_id")
        if any(job[key] != preview[key] for key in scientific): raise ValueError("Root job changed fixed scientific/runtime/soft-cap contract")
        if job["source_review_approved"] is not True or job["source_review"]["approved"] is not True or job["external_hard_bound_confirmed"] is not True or job["TEST_access"] is not False or job["retry"] is not False:
            raise ValueError("Root source/scope/hard-bound job approval unresolved")
        if job["source_manifest_sha256"] != SOURCE_SHA or job["program_sha256"] != sha(Path(commands[index]["argv"][2])) or job["output_directory"] != commands[index]["argv"][6] or job["physical_gpu_uuid"] != commands[index]["env"]["CUDA_VISIBLE_DEVICES"]:
            raise ValueError("Exact job source/program/output/physical GPU differs")
        if job["fits_authorized"] is not (index >= 2) or job["VALID_values_access"] is not (index >= 2): raise ValueError("Exact root job qualification/fit scope differs")
        jobs.append({"value": job, "sha256": binding["sha256"]})
    owner_root = FIT_ROOT/"owner"
    if owner_root.exists(): raise ValueError("Fresh detached owner only; no retry/resume")
    owner_root.mkdir(); (owner_root/"logs").mkdir()
    owner_helper, _ = lane(GPU_UUIDS[0]); owner_identity = owner_helper.identity(os.getpid())
    if owner_identity is None or owner_identity["pgid"] != os.getpid() or owner_identity["sid"] != os.getpid(): raise ValueError("Owner must be root-launched in a detached fresh session")
    started = time.monotonic(); completed = []
    write(owner_root/"OWNER_STARTED.json", {"UTC": owner_helper.now(), "owner_identity": owner_identity,
        "root_release_sha256": sha(args.release), "initial_job_bindings": release["jobs"], "combined_fixed_phase_allowance_seconds": 52200,
        "training_started": False, "retry": False, "TEST_access": False})
    try:
        qualification_started = time.monotonic()
        completed += parallel((0, 1), commands, jobs, owner_root, limits, qualification_started, 5100)
        admit_actual_gates(commands, jobs, owner_root, release)
        fit_started = time.monotonic()
        write(owner_root/"TRAINING_PHASE_ADMITTED.json", {"UTC": owner_helper.now(), "full_batch_fits": 3, "wave1": [commands[i]["argv"][4] for i in (2, 3)],
            "wave2": commands[4]["argv"][4], "fixed_phase_hard_seconds": 47100, "actual_child_start_authority": "logs/*.CHILD_STARTED.json"})
        completed += parallel((2, 3), commands, jobs, owner_root, limits, fit_started, 47100)
        write(owner_root/"WAVE1_COMPLETE.json", {"UTC": owner_helper.now(), "completed": completed[-2:], "scores_read": False})
        completed.append(run_child(4, commands, jobs, owner_root, limits, fit_started, 47100))
        write(owner_root/"BLOCK_COMPLETE.json", {"UTC": owner_helper.now(), "complete": True, "new_fits": 3, "completed": completed,
            "inclusive_seconds": time.monotonic()-started, "TEST_access": False, "scores_read": False, "retry": False})
    except (Exception, KeyboardInterrupt) as error:
        write(owner_root/"OWNER_FAILURE.json", {"UTC": owner_helper.now(), "error": type(error).__name__+": "+str(error),
            "completed": completed, "inclusive_seconds": time.monotonic()-started, "partial_outputs_preserved": True, "retry": False, "TEST_access": False})
        raise


if __name__ == "__main__": main()
