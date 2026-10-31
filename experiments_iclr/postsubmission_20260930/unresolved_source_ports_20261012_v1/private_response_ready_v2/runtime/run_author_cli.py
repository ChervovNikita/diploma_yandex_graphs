"""Run one explicitly approved unchanged native author CLI group.

Raw source stdout/stderr are quarantined. The public registry/result/progress
schema contains TRAIN/SELECT values and process/cost metadata only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import subprocess
import time

from author_cli_spec import (load_author_spec, register_author_spec, validate_author_spec,
                             verify_receipt)
from registry import Registry
from run_study import owner_identity, require_approval
from spec import canonical, confined, fingerprint, validate

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
NUMBER = rb"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
# TEST tokens are skipped, never captured or converted into values.
EPOCH = re.compile(rb"^(\d+) epochs trained\. Current acc: " + NUMBER + rb" " + NUMBER
                   + rb" \S+ epochs trained\. Current loss: " + NUMBER + rb" " + NUMBER + rb" \S+$")
REPEAT = re.compile(rb"^RP (\d+) Launched\.\.\.$")
RETURNED = re.compile(rb"^run_(\d+)\s+test_acc:")


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_receipt(root, path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(Path(root).resolve())), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(), "content_policy": "quarantined_TEST_bearing_bytes_not_interpreted"}


def verify_allocation():
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Author CLI is currently allocation-only")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10)
    if output.splitlines() != [ALLOCATION_UUID]:
        raise ValueError("Author CLI allocation UUID check failed")


def plan_cohort(root, plan):
    records, assigned = {}, {}
    for entry in plan["studies"]:
        path = confined(root, entry["path"])
        import gzip
        raw = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        value = json.loads(raw)
        if all(r.get("kind") == "author_cli_group" for r in value["runs"]):
            validate_author_spec(value)
        else:
            validate(value)
        if fingerprint(value) != entry["spec_hash"]:
            raise ValueError("Master source study changed")
        for run in value["runs"]:
            key = value["study_id"] + "/" + run["run_id"]
            if key in records:
                raise ValueError("Duplicate master run identity")
            records[key] = run
    for group in plan["groups"]:
        for key in group["run_keys"]:
            if key not in records or key in assigned:
                raise ValueError("Missing or duplicate master placement")
            assigned[key] = group["gpu_assignment"]
    if set(assigned) != set(records):
        raise ValueError("Every master run needs exactly one placement")
    return records, assigned


def require_global_author_gate(root, plan, plan_hash, name, records, assigned):
    """Recheck exact whole-master cohort; never read result payloads or scores."""
    gate = plan.get("global_gates", {}).get(name)
    stages = {"search", "search_selection"} if name == "whole_search" else {"author_cli"}
    if not gate or set(gate.get("source_stages", [])) != stages:
        raise ValueError("Missing explicit whole-master author gate: " + name)
    expected = sorted(key for key, row in records.items() if row["config"].get("stage") in stages)
    if not expected or gate.get("all_packet_source_run_keys") != expected:
        raise ValueError("Author gate does not cover every materialized master source run")
    hosts = sorted({assigned[key].split("/")[0] for key in expected})
    if gate.get("required_hosts") != hosts:
        raise ValueError("Author gate host roster differs from its complete source cohort")
    if name == "whole_search" and "author_cli" not in gate.get("required_for_stages", []):
        raise ValueError("Master HPO closure must explicitly gate author CLI groups")
    path = confined(root, gate["path"])
    if not path.is_file():
        raise ValueError("The whole master phase has not closed: " + name)
    value = json.loads(path.read_text())
    if (value.get("status") != "complete" or value.get("plan_hash") != plan_hash
            or value.get("closed_phase") != name or set(value.get("host_receipts", {})) != set(hosts)):
        raise ValueError("Author global closure receipt does not belong to the approved plan")
    for host, bound in value["host_receipts"].items():
        proof = json.loads(verify_receipt(root, bound).read_text())
        keys = [key for key in expected if assigned[key].split("/")[0] == host]
        if (proof.get("status") != "complete" or proof.get("plan_hash") != plan_hash
                or proof.get("closed_phase") != name or proof.get("host") != host
                or proof.get("run_keys") != keys or set(proof.get("results", {})) != set(keys)):
            raise ValueError("Copied author gate host proof changed or omitted source jobs")
        for key in keys:
            row, result = records[key], proof["results"][key]
            terminal_allowed = name == "whole_author" and row["kind"] == "author_cli_group"
            failed_search_allowed = name == "whole_search" and row["kind"] == "model_fit" and row["config"].get("stage") == "search"
            if result["state"] != "complete" and not ((terminal_allowed or failed_search_allowed) and result["state"] == "failed"):
                raise ValueError("Incomplete required author/HPO source job")
            if result.get("contract_hash") != fingerprint(row) or not result.get("result_hash"):
                raise ValueError("Author gate host proof has different run contracts")
    return {"gate": name, "plan_hash": plan_hash, "source_run_keys": expected,
            "receipt": file_receipt(root, path)}


def require_master_approval(root, value, digest, run, approval):
    relative = approval.get("approved_plan_path")
    if not relative:
        raise ValueError("Author CLI requires a source-bound master plan")
    plan = json.loads(confined(root, relative).read_text())
    plan_hash = fingerprint(plan)
    if plan_hash != approval.get("approved_plan_hash"):
        raise ValueError("Master plan changed after GO")
    records, assigned = plan_cohort(root, plan)
    key = value["study_id"] + "/" + run["run_id"]
    if (key not in records or records[key] != run
            or not any(row["spec_hash"] == digest for row in plan["studies"])
            or assigned[key] != "allocation/gpu0"):
        raise ValueError("Author CLI group is outside the approved allocation assignment")
    return require_global_author_gate(root, plan, plan_hash, "whole_search", records, assigned)


class Progress:
    """Source-specific TEST-free observation and conservative actual costs."""
    def __init__(self):
        self.current = None
        self.repeats = {}
        self.anomalies = []
        self.last_train_select = None

    def feed(self, line):
        line = line.rstrip(b"\r\n")
        start = REPEAT.match(line)
        if start:
            repeat = int(start[1])
            if repeat != len(self.repeats) or not 0 <= repeat < 10 or (self.current is not None and not self.repeats[self.current]["returned"]):
                self.anomalies.append("unexpected_repeat_order")
                return None
            self.current = repeat
            self.repeats[repeat] = {"observed_epoch_updates": 0, "returned": False, "best_SELECT_accuracy": 0.0}
            return {"event": "repeat_launched", "repeat": repeat}
        returned = RETURNED.match(line)
        if returned:
            repeat = int(returned[1])-1
            if repeat != self.current or self.repeats[repeat]["returned"]:
                self.anomalies.append("unexpected_repeat_return")
                return None
            self.repeats[repeat]["returned"] = True
            return {"event": "repeat_returned", "repeat": repeat}
        epoch = EPOCH.match(line)
        if not epoch:
            return None
        if self.current is None:
            self.anomalies.append("epoch_without_repeat")
            return None
        index = int(epoch[1])
        row = self.repeats[self.current]
        if index != row["observed_epoch_updates"] or index >= 2000 or row["returned"]:
            self.anomalies.append("noncontiguous_or_excess_epoch")
            return None
        values = list(map(float, epoch.group(2,3,4,5)))
        if not all(math.isfinite(v) for v in values) or not all(0 <= v <= 1 for v in values[:2]) or any(v < 0 for v in values[2:]):
            self.anomalies.append("invalid_TRAIN_SELECT_diagnostic")
            return None
        row["observed_epoch_updates"] += 1
        row["best_SELECT_accuracy"] = max(row["best_SELECT_accuracy"], values[1])
        self.last_train_select = {"event": "TRAIN_SELECT_epoch", "repeat": self.current, "epoch": index,
            "TRAIN_accuracy": values[0], "SELECT_accuracy": values[1], "TRAIN_loss": values[2], "SELECT_loss": values[3]}
        return self.last_train_select

    def costs(self, *, child_started, exit_code):
        complete = sum(r["returned"] for r in self.repeats.values())
        known = sum(r["observed_epoch_updates"] for r in self.repeats.values())
        all_observed = (len(self.repeats) == complete == 10 and not self.anomalies
                        and all(r["observed_epoch_updates"] >= 250 for r in self.repeats.values()))
        exact = (not child_started) or all_observed
        return {"author_cli_processes_started": int(child_started), "declared_repeat_trajectories": 10,
                "repeat_launch_events_observed": len(self.repeats), "repeat_returns_observed": complete,
                "known_completed_epoch_optimizer_updates": known,
                "actual_optimizer_updates": known if exact else None,
                "actual_optimizer_updates_exact": exact,
                "unobserved_update_count_is_unknown": not exact,
                "registered_maximum_optimizer_updates": 20000,
                "known_TRAIN_model_forwards_lower_bound": known,
                "known_validation_TEST_model_forwards_lower_bound": known*4,
                "cuda_peak_bytes": None, "cuda_peak_measured": False,
                "complete_native_repeat_observation": all_observed,
                "protocol_observation_anomalies": sorted(set(self.anomalies)),
                "exit_code": exit_code}


def preflight_group(root, value, run):
    cli = run["cli"]
    repository = Path(cli["execution_repository"]).resolve()
    if not Path(root).resolve().is_relative_to(repository):
        raise ValueError("Author execution repository does not contain the project")
    python = Path(cli["python_executable"])
    if not python.is_file() or not python.resolve().is_relative_to(repository):
        raise ValueError("Author Python must remain in the declared repository runtime")
    for row in cli["source_files"] + cli["raw_files"]:
        verify_receipt(root, row)
    work = confined(root, cli["cwd"])
    for key in ["basis_dir", "processed_dir"]:
        path = confined(root, cli[key])
        if not path.is_dir() or list(path.iterdir()):
            raise ValueError("Author initial workdirs must have fresh empty basis/processed directories")
    for path in cli["quarantine"].values():
        if confined(root, path).exists():
            raise ValueError("Retain existing author quarantine; never overwrite or rerun a group")
    env = dict(os.environ, **cli["environment"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        raise ValueError("The authorized single-GPU allocation uses visible device0")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if set(cli["environment"]) != {"PYTHONPATH", "LD_LIBRARY_PATH", "DGLBACKEND"} or env["DGLBACKEND"] != "pytorch":
        raise ValueError("Unexpected author private environment override")
    if Path(env["PYTHONPATH"]).resolve() != confined(root, cli["private_dependency_dir"]).resolve():
        raise ValueError("Author dependency overlay differs from its frozen path")
    for path in env["LD_LIBRARY_PATH"].split(":"):
        if not Path(path).is_dir() or not Path(path).resolve().is_relative_to(repository):
            raise ValueError("Author library search paths must be existing repository directories")
    # Metadata-only child check does not import models or consume the CLI's RNG.
    check_env=dict(env,CUDA_VISIBLE_DEVICES="")
    check="import importlib.metadata as m,json,sys;expected=json.loads(sys.argv[1]);observed={n:m.version(n) for n in expected};assert observed==expected, 'Declared author runtime versions changed'"
    checked=subprocess.run([str(python),"-B","-c",check,canonical(cli["runtime_versions"])],cwd=work,env=check_env,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    if checked.returncode:
        raise ValueError("Declared author runtime version metadata changed")
    return [str(python), *cli["argv"]], work, env


def execute_claimed_group(root, value, run, registry, destination, owner, admission):
    """Process custody for an already claimed group; tests use synthetic CLI bytes."""
    command, work, env = admission
    row = registry.run(value["study_id"], run["run_id"])
    if row["state"] != "running" or row["owner"] != owner:
        raise ValueError("Native CLI subprocess requires this exact real registry claim")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    deadline = tick + run["config"]["wall_seconds"]
    deadline_UTC = datetime.fromtimestamp(time.time()+run["config"]["wall_seconds"], timezone.utc).isoformat()
    observer, process, child_owner, reason = Progress(), None, None, None
    stop = {"signal": None}
    old_handlers = {}
    for signum in [signal.SIGTERM, signal.SIGINT]:
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda received, frame: stop.update(signal=received))
    stdout_path = confined(root, run["cli"]["quarantine"]["stdout"])
    stderr_path = confined(root, run["cli"]["quarantine"]["stderr"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    def stop_child():
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    try:
        with stdout_path.open("xb") as raw_stdout, stderr_path.open("xb") as raw_stderr, (destination/"TRAIN_SELECT_PROGRESS.jsonl").open("x") as progress:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=raw_stderr,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            child_owner = {"pid": process.pid, "hostname": socket.gethostname(), "pgid": process.pid}
            stat = Path("/proc")/str(process.pid)/"stat"
            if stat.exists():
                child_owner["start_ticks"] = int(stat.read_text().rsplit(")",1)[1].split()[19])
            started = {"owner": owner, "child_owner": child_owner, "command": command,
                       "cwd": str(work), "declared_environment": run["cli"]["environment"],
                       "deadline_UTC": deadline_UTC, "study_hash": fingerprint(value), "run_hash": fingerprint(run)}
            atomic(destination/"START.json", started)
            registry.event(value["study_id"], run["run_id"], "author_cli_child_started", started)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            pending, eof = b"", False
            try:
                while not eof or process.poll() is None:
                    if stop["signal"] is not None or time.monotonic() >= deadline:
                        reason = "interrupted_signal" if stop["signal"] is not None else "declared_wall_deadline"
                        stop_child()
                    for key, _ in selector.select(timeout=.2):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj);eof=True;continue
                        raw_stdout.write(chunk);raw_stdout.flush()
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n",1)
                            observed = observer.feed(line)
                            if observed is not None:
                                progress.write(canonical(observed)+"\n")
                                if observed["event"] != "TRAIN_SELECT_epoch" or (observed["epoch"]+1)%100 == 0:
                                    progress.flush()
                                    safe = {"last_TRAIN_SELECT": observer.last_train_select, "cost": observer.costs(child_started=True, exit_code=None)}
                                    atomic(destination/"PROGRESS.json",safe)
                                    registry.event(value["study_id"], run["run_id"], "author_TRAIN_SELECT_progress",safe)
                        if len(pending)>1024*1024:
                            pending=b"";observer.anomalies.append("oversize_unparsed_source_line")
                    if reason and process.poll() is not None:
                        # Drain promptly available stdout, then retain terminal custody.
                        if not selector.select(timeout=.2):
                            break
                process.wait(timeout=5)
            finally:
                selector.close();process.stdout.close()
        costs = observer.costs(child_started=True, exit_code=process.returncode)
        success = process.returncode == 0 and costs["complete_native_repeat_observation"] and reason is None
        result = {"status":"complete" if success else "failed", "owner":owner,"child_owner":child_owner,
                  "deadline_UTC":deadline_UTC,"exit_code":process.returncode,"wall_seconds":time.monotonic()-tick,
                  "failure_type":None if success else (reason or "author_process_or_protocol_failure"),
                  "cost":costs,"quarantine":{"stdout":file_receipt(root,stdout_path),"stderr":file_receipt(root,stderr_path)},
                  "author_TEST_outcomes_interpreted":False,"author_aggregate_interpreted":False,
                  "whole_author_release_required":True,"resume_supported":False}
    except BaseException as error:
        stop_child()
        result = {"status":"failed","failure_type":type(error).__name__,"owner":owner,"child_owner":child_owner,
                  "deadline_UTC":deadline_UTC,"wall_seconds":time.monotonic()-tick,
                  "cost":observer.costs(child_started=process is not None,exit_code=process.returncode if process else None),
                  "author_TEST_outcomes_interpreted":False,"whole_author_release_required":True,"resume_supported":False}
        # Exception text/traceback can contain TEST-bearing source output; do not publish it.
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum,handler)
    atomic(destination/"RESULT.json",result)
    registry.finish(value["study_id"],run["run_id"],result,success=result["status"]=="complete")
    return result


def execute(root, study_path, run_id, registry_path, output, approval_path, device="cuda:0", *, preflight_only=False):
    verify_allocation()  # Must precede any project metadata read or write.
    root = Path(root).resolve()
    for path in [study_path,registry_path,output,approval_path]:
        if path and not Path(path).resolve().is_relative_to(root):
            raise ValueError("Author study, registry, outputs and approval must remain inside the project")
    value,digest = load_author_spec(study_path,root)
    if value.get("source", {}).get("author_family") == "snap_stanford_ogb_molecular_batchnorm_v1":
        from run_ogb_author_cli import execute as execute_ogb
        return execute_ogb(root, study_path, run_id, registry_path, output, approval_path,
                           device=device, preflight_only=preflight_only)
    rows = [r for r in value["runs"] if r["run_id"]==run_id]
    if len(rows)!=1 or device!="cuda:0":
        raise ValueError("Declare one author CLI group on visible cuda0")
    run=rows[0];admission=preflight_group(root,value,run)
    if preflight_only:
        return {"preflight":"passed","kind":"author_cli_group","spec_hash":digest,"run_id":run_id,
                "scientific_training_runs":0,"GO_created":False}
    approval=require_approval(approval_path,digest)
    gate=require_master_approval(root,value,digest,run,approval)
    registry=Registry(registry_path)
    try:
        register_author_spec(registry,value)
        owner=owner_identity();owner.update(kind="author_cli_group",study_hash=digest,run_hash=fingerprint(run),gate=gate)
        registry.claim(value["study_id"],run_id,owner)
        return execute_claimed_group(root,value,run,registry,Path(output)/value["study_id"]/run_id,owner,admission)
    finally:
        registry.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["project-root","study","run-id","registry","output"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--approval");parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    if not args.preflight_only and not args.approval:
        parser.error("explicit GO and whole-master HPO closure are required")
    result=execute(args.project_root,args.study,args.run_id,args.registry,args.output,args.approval,args.device,preflight_only=args.preflight_only)
    print(canonical(result))
    raise SystemExit(0 if result.get("status","complete")=="complete" else 1)


if __name__=="__main__":
    main()
