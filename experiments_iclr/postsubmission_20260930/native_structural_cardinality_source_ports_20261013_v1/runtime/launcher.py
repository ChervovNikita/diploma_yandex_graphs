"""Two supervised experiment slots per GPU, with fixed three-GPU assignments.

The launcher contains no learning code. Children own registry claims, logs and
checkpoints. A lost supervisor never grants permission to repeat a fit.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time

from registry import Registry
from run_study import owner_identity, require_approval
from spec import canonical, confined, fingerprint, load
from phase_closure import write_host_receipts

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"


def hardware():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = []
    for line in output.splitlines():
        index, uuid, total, free = [part.strip() for part in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "total_mib": int(total), "free_mib": int(free)})
    return rows


def admission_capacities(host, observed, active, plan, owned_usage_mib):
    """Reserve future fit memory without subtracting owned live memory twice.

    Recovered owned usage is actual process memory, never the reservation.
    Unidentified usage remains external and therefore conservatively excluded.
    """
    result = {}
    for row in observed:
        worker = host + '/gpu' + str(row['index'])
        live = max(0, row['free_mib'] + owned_usage_mib.get(worker, 0) - 1024)
        declared = plan.get('worker_capacities_cuda_bytes', {}).get(worker)
        result[worker] = min(live, declared // 1024**2) if declared is not None else live
    return result


def owned_gpu_usage(host, observed, active):
    """Only exact active process groups contribute recoverable live VRAM."""
    if not active:
        return {}
    known_groups = {item['process'].pid for item in active.values()}
    uuid_workers = {row['uuid']: host + '/gpu' + str(row['index']) for row in observed}
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    used = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            continue
        pid, uuid, memory = int(fields[0]), fields[1], int(fields[2])
        stat = Path('/proc') / str(pid) / 'stat'
        try:
            state = stat.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        if int(state[2]) in known_groups and state[0] != 'Z' and uuid in uuid_workers:
            worker = uuid_workers[uuid]
            used[worker] = used.get(worker, 0) + memory
    return used


def host_check(host, plan):
    rows = hardware()
    if host == "allocation":
        if socket.gethostname() != "anogena-2-0" or [r["uuid"] for r in rows] != [ALLOCATION_UUID]:
            raise ValueError("The scientific one-GPU allocation route failed verification")
    elif host == "server77":
        binding = plan.get("host_bindings", {}).get(host)
        if not binding or binding.get("status") != "verified":
            raise ValueError("18.77 needs a verified hostname/UUID binding after synchronization")
        if socket.gethostname() != binding["hostname"] or [r["uuid"] for r in rows] != binding["gpu_uuids"]:
            raise ValueError("18.77 hardware differs from the frozen host binding")
    else:
        raise ValueError("Use allocation or server77")
    return rows


def runtime_preflight(plan, python):
    """Check the actual worker interpreter before starting scientific workers."""
    if not plan.get('runtime_compatibility_required'):
        return {'status': 'legacy_plan_without_runtime_preflight', 'scientific_fits': 0}
    expected = plan.get('runtime_requirements')
    if not isinstance(expected, dict):
        raise ValueError('Final master requires captured numerical runtime compatibility pins')
    package = Path(__file__).resolve().parent
    for name in ('prepare_sync_environment.py', 'prepare_sync_bundle.py'):
        if not (package / name).is_file():
            raise ValueError('Missing frozen runtime compatibility source')
    code = ('import json,sys;sys.path.insert(0,sys.argv[1]);'
            'from prepare_sync_environment import verify_runtime_compatibility;'
            'print(json.dumps(verify_runtime_compatibility(json.loads(sys.argv[2]))))')
    raw = subprocess.check_output([str(python), '-B', '-c', code, str(package), canonical(expected)],
                                  text=True, timeout=120)
    value = json.loads(raw)
    if (value.get('status') != 'required_runtime_compatibility_passed'
            or value.get('source_environment_hash') != fingerprint(expected)
            or value.get('mandatory_numerical_pins_verified') is not True
            or value.get('native_binary_compatibility_verified') is not True
            or value.get('CUDA_initialized') is not False
            or value.get('scientific_fits') != 0):
        raise ValueError('Runtime compatibility probe did not verify the approved numerical environment')
    return value


def validate_plan(root, plan):
    if plan.get("schema_version") != 1 or plan.get("slots_per_gpu") != 2:
        raise ValueError("Expected the fixed two-slot execution plan")
    studies, runs, placement = {}, {}, {}
    for row in plan["studies"]:
        path = confined(root, row["path"])
        spec, digest = load(path, root)
        if digest != row["spec_hash"] or spec["study_id"] in studies:
            raise ValueError("Changed or repeated study contract")
        studies[spec["study_id"]] = {"spec": spec, "path": path, "hash": digest}
        for run in spec["runs"]:
            key = spec["study_id"] + "/" + run["run_id"]
            runs[key] = {"study": spec["study_id"], "run": run, "path": path}
    for group in plan["groups"]:
        worker = group["gpu_assignment"]
        if worker not in {"allocation/gpu0", "server77/gpu0", "server77/gpu1"}:
            raise ValueError("Unexpected hardware assignment")
        for key in group["run_keys"]:
            if key not in runs or key in placement:
                raise ValueError("Every job must have exactly one assignment")
            placement[key] = {"worker": worker, "group": group["id"]}
    if set(placement) != set(runs):
        raise ValueError("Unassigned jobs remain")
    # Dependencies are keyed within a study; global barriers are declared by key.
    dependencies = {}
    for key, record in runs.items():
        dependencies[key] = [record["study"] + "/" + p for p in record["run"]["parents"]]
        dependencies[key] += plan.get("barriers", {}).get(key, [])
        if any(parent not in runs for parent in dependencies[key]):
            raise ValueError("Missing parent for " + key)
        if any(placement[p]["worker"].split("/")[0] != placement[key]["worker"].split("/")[0]
               for p in dependencies[key]):
            raise ValueError("Dependencies must be staged on the same host")
    # A cycle must be found before starting any processes.
    done, visiting = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("Cyclic run dependencies")
        if key in done:
            return
        visiting.add(key)
        for parent in dependencies[key]:
            visit(parent)
        visiting.remove(key)
        done.add(key)
    for key in runs:
        visit(key)
    from retention_lifecycle import validate_plan_policy
    validate_plan_policy(root,plan,runs,studies)
    return studies, runs, placement, dependencies


def load_plan(root, path, approval_path=None):
    plan = json.loads(Path(path).read_text())
    studies, runs, placement, dependencies = validate_plan(root, plan)
    digest = fingerprint(plan)
    if approval_path is not None:
        approval = json.loads(Path(approval_path).read_text())
        if approval.get("approved_plan_hash") != digest:
            raise ValueError("This assignment plan has not received explicit GO")
        for row in studies.values():
            require_approval(approval_path, row["hash"])
    return plan, digest, studies, runs, placement, dependencies



def enabled_workers(output, host, digest, observed):
    """Read atomic GPU requests; an absent file retains legacy host startup."""
    available = {host + '/gpu' + str(row['index']) for row in observed}
    path = output / (host + '.gpu-admission.json')
    if not path.exists():
        return available
    request = json.loads(path.read_text())
    if request.get('host') != host or request.get('plan_hash') != digest:
        raise ValueError('GPU admission belongs to a different host or plan')
    enabled = request.get('enabled_workers')
    if (not isinstance(enabled, list) or not enabled or len(enabled) != len(set(enabled))
            or any(not isinstance(worker, str) or worker not in available for worker in enabled)):
        raise ValueError('GPU admission requests unavailable hardware')
    return set(enabled)


def request_gpu_admission(output, host, digest, observed, gpu_index):
    """Enable requested GPUs under the caller's exclusive startup lock."""
    available = {host + '/gpu' + str(row['index']) for row in observed}
    requested = available if gpu_index is None else {host + '/gpu' + str(gpu_index)}
    if not requested <= available:
        raise ValueError('Requested GPU index is absent from verified hardware')
    path = output / (host + '.gpu-admission.json')
    # A fresh single-GPU request starts only that GPU. Existing requests accumulate.
    enabled = enabled_workers(output, host, digest, observed) if path.exists() else set()
    enabled |= requested
    request = {'schema': 'go-bound-host-gpu-admission-v1', 'host': host,
               'plan_hash': digest, 'enabled_workers': sorted(enabled), 'slots_per_gpu': 2}
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        stream.write(canonical(request) + '\n')
    temporary.replace(path)
    return sorted(enabled)


def schedule_ready(runs, states, placement, dependencies, active, capacities, host, *, slots=2):
    """Pure admission policy used by real launcher and engineering checks."""
    result, occupied, reserved = [], {}, {}
    active_keys = {item["key"] for item in active}
    for item in active:
        worker = item["worker"]
        occupied[worker] = occupied.get(worker, 0) + 1
        reserved[worker] = reserved.get(worker, 0) + item["reserve_mib"]
    keys = sorted(runs, key=lambda key: (-runs[key]["run"]["config"].get("estimated_seconds", 0), key))
    for key in keys:
        worker = placement[key]["worker"]
        if worker.split("/")[0] != host or worker not in capacities or states[key] != "registered" or key in active_keys:
            continue
        terminal_parents = runs[key]["run"].get("kind") == "selection"
        allowed_states = {"complete", "failed"} if terminal_parents else {"complete"}
        if any(states[p] not in allowed_states for p in dependencies[key]):
            continue
        if occupied.get(worker, 0) >= slots:
            continue
        reserve = runs[key]["run"]["config"].get("reservation_mib")
        if type(reserve) is not int or reserve < 1:
            raise ValueError("Each run requires a positive GPU-memory reservation")
        # free_mib is a conservative admission margin. Current external use is
        # respected; reservations prevent simultaneously starting two large fits.
        if reserve + reserved.get(worker, 0) > capacities[worker]:
            continue
        result.append(key)
        occupied[worker] = occupied.get(worker, 0) + 1
        reserved[worker] = reserved.get(worker, 0) + reserve
    return result


def run_supervisor(root, plan_path, approval_path, host, python, *, preflight_only=False):
    if host == 'allocation':
        host_check(host, {})  # Route identity before any project metadata read.
    plan, digest, studies, runs, placement, dependencies = load_plan(
        root, plan_path, None if preflight_only else approval_path)
    observed = host_check(host, plan)
    runtime = runtime_preflight(plan, python)
    if preflight_only:
        print(canonical({"preflight": "passed", "host": host, "hardware": observed,
                         "plan_hash": digest, "jobs": len(runs), "scientific_training_runs": 0,
                         "runtime_compatibility": runtime}))
        return
    output = confined(root, plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / (host + ".supervisor.lock")).open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = Registry(output / (host + ".sqlite"))
    for study in studies.values():
        registry.register(study["spec"])
    active, stop, closed_local_gates = {}, {"requested": False}, set()
    def request_stop(signum, frame):
        stop["requested"] = True  # Current children finish and retain their state.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    journal = (output / (host + ".supervisor.jsonl")).open("a", buffering=1)
    def event(kind, payload):
        journal.write(canonical({"time": time.time(), "kind": kind, **payload}) + "\n")
    event("start", {"owner": owner_identity(), "plan_hash": digest, "hardware": observed})
    try:
        while True:
            for key in list(active):
                item = active[key]
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                event("child_terminal", {"key": key, "exit_code": code, "pid": item["process"].pid})
                del active[key]
                row = runs[key]
                from retention_lifecycle import sweep_closed_owner
                releases=sweep_closed_owner(root,studies[row["study"]]["spec"],registry,row["run"]["run_id"])
                if releases:event("prospective_retention_release",{"key":key,"receipts":releases})
                if registry.run(row["study"], row["run"]["run_id"])["state"] == "registered":
                    raise RuntimeError("Child terminated before its registry claim; inspect its log before retrying")
            all_states = registry.states()
            states = {key: all_states[key] for key in runs}
            write_host_receipts(root, plan, digest, runs, placement, registry, host, closed_local_gates)
            local = [key for key in runs if placement[key]["worker"].startswith(host + "/")]
            if not active and all(states[key] in {"complete", "failed"} for key in local):
                event("roster_terminal", {"all_successful": all(states[k] == "complete" for k in local)})
                return
            if stop["requested"] and not active:
                event("stopped_after_current_children", {})
                return
            if not stop["requested"]:
                current = hardware()
                capacities = admission_capacities(host, current, active, plan,
                                                 owned_gpu_usage(host, current, active))
                enabled = enabled_workers(output, host, digest, current)
                capacities = {worker: value for worker, value in capacities.items() if worker in enabled}
                stages = {row["run"]["config"].get("stage", "search") for row in runs.values()}
                phase_allowed = {stage: gate_ready(root, plan, digest, stage) for stage in stages}
                admissible_states = {key: state if key in plan.get("label_free_export_run_keys",[]) or phase_allowed[runs[key]["run"]["config"].get("stage", "search")]
                                     else "phase_wait" for key, state in states.items()}
                ready = schedule_ready(runs, admissible_states, placement, dependencies,
                                       list(active.values()), capacities, host)
                for key in ready:
                    row = runs[key]
                    run = row["run"]
                    worker = placement[key]["worker"]
                    index = int(worker.rsplit("gpu", 1)[1])
                    kind = run["kind"]
                    script = Path(__file__).with_name(worker_script(kind))
                    if not script.is_file():
                        raise ValueError("Missing executable endpoint runner")
                    log_path = output / "logs" / row["study"] / (run["run_id"] + ".log")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open("a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index),
                               GNNM_TORCH_THREADS=str(plan.get("torch_threads_per_fit", 2)),
                               OMP_NUM_THREADS=str(plan.get("torch_threads_per_fit", 2)))
                    command = [str(python), "-B", "-u", str(script), "--project-root", str(root),
                               "--study", str(row["path"]), "--run-id", run["run_id"],
                               "--registry", str(output / (host + ".sqlite")), "--output", str(output / "runs"),
                               "--approval", str(approval_path), "--device", "cuda:0"]
                    process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    active[key] = {"key": key, "process": process, "log": log, "worker": worker,
                                   "reserve_mib": run["config"]["reservation_mib"]}
                    event("spawn", {"key": key, "pid": process.pid, "worker": worker, "command": command})
            # Failed parents and pre-existing unowned running records need action,
            # not an automatic restart or an eternal unchanged loop.
            if not active:
                pending = [k for k in local if states[k] == "registered"]
                blocked = [k for k in pending if runs[k]["run"]["kind"] != "selection"
                           and any(states[p] == "failed" for p in dependencies[k])]
                unowned = [k for k in local if states[k] == "running"]
                if blocked or unowned:
                    event("attention_required", {"failed_parent_jobs": blocked, "existing_running_records": unowned})
                    raise RuntimeError("Inspect failed parents or existing process custody before continuing")
            time.sleep(5)
    finally:
        registry.close()
        journal.close()
        lock.close()


def gate_ready(root, plan, digest, stage):
    for name, gate in plan.get("global_gates", {}).items():
        if stage not in gate["required_for_stages"]:
            continue
        path = confined(root, gate["path"])
        if not path.is_file():
            return False
        value = json.loads(path.read_text())
        if (value.get("status") != "complete" or value.get("plan_hash") != digest
                or value.get("closed_phase") != name):
            raise ValueError("Global phase receipt does not belong to this frozen plan")
        if set(value.get("host_receipts", {})) != set(gate["required_hosts"]):
            raise ValueError("Global phase receipt omits an assigned host")
        for host, receipt in value["host_receipts"].items():
            raw = confined(root, receipt["path"]).read_bytes()
            import hashlib
            if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("Global phase proof changed after collection")
    return True


def worker_script(kind):
    if kind == 'model_fit':
        return 'run_study.py'
    if kind == 'author_cli_group':
        return 'run_author_cli.py'
    if kind in {'selection', 'ensemble_assembly', 'calibration', 'evaluation'}:
        return 'run_endpoint.py'
    raise ValueError('Unsupported execution kind: ' + str(kind))


def process_identity(pid):
    """Linux process identity; a recycled PID never establishes custody."""
    try:
        value = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    if value[0] == 'Z':
        return None
    return {'pid': pid, 'start_ticks': int(value[19]), 'pgid': int(value[2])}


def live_owned(value):
    if not isinstance(value, dict) or set(value) != {'pid', 'start_ticks', 'pgid'}:
        raise ValueError('Incomplete startup process custody')
    return process_identity(value['pid']) == value


def receipt_authentication(root, plan):
    """Check the approved existing remote before starting scientific children."""
    from phase_rendezvous import transport_policy
    policy = transport_policy(plan, plan.get('phase_rendezvous', {}).get('remote_name'))
    if plan.get('phase_rendezvous_capacity', {}).get('all_hosts_fit_fixed_limits') is not True:
        raise ValueError('The approved plan must qualify every host phase proof against transport limits')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    if 'GIT_SSH_COMMAND' not in env:
        env['GIT_SSH_COMMAND'] = 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes'
    def command(args):
        result = subprocess.run(['git', '-C', str(root), *args], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=60, check=False)
        if result.returncode:
            # Git diagnostics may echo private URLs. Never print them.
            raise RuntimeError('Existing Git authentication is unavailable for phase synchronization')
        return result.stdout.strip()
    repository = Path(command(['rev-parse', '--show-toplevel']).decode()).resolve()
    if not Path(root).resolve().is_relative_to(repository):
        raise ValueError('Receipt source must be the project Git repository')
    remote = command(['remote', 'get-url', policy['remote_name']])
    if hashlib.sha256(remote).hexdigest() != policy['remote_url_sha256']:
        raise ValueError('Existing Git remote differs from the approved receipt destination')
    command(['ls-remote', '--heads', policy['remote_name'],
             'refs/heads/codex/phase-receipts/' + fingerprint(plan) + '/*'])
    return repository, policy


def install_startup_owner(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def local_terminal_roster(output, host, studies, runs, placement):
    """Read existing terminal custody without registering, resuming or refitting."""
    database = output / (host + '.sqlite')
    expected = {key for key in runs if placement[key]['worker'].startswith(host + '/')}
    if not database.is_file() or not expected:
        return None
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        actual_specs = dict(connection.execute('SELECT id,spec_hash FROM studies'))
        for name in {runs[key]['study'] for key in expected}:
            if actual_specs.get(name) != studies[name]['hash']:
                raise ValueError('Existing host registry belongs to changed study contracts')
        states = {study + '/' + identifier: state
                  for study, identifier, state in connection.execute('SELECT study,id,state FROM runs')}
    finally:
        connection.close()
    if not all(states.get(key) in {'complete', 'failed'} for key in expected):
        return None
    return {'terminal_records': len(expected), 'all_successful': all(states[key] == 'complete' for key in expected)}


def all_global_gates_closed(root, plan, digest):
    from phase_rendezvous import verify_global
    gates = plan.get('global_gates', {})
    return bool(gates) and all(verify_global(root, plan, digest, name) for name in gates)


def detach_host(root, plan_path, approval_path, host, python, *, gpu_index=None):
    """Start one detached supervisor and one GO-bound receipt sidecar per host."""
    plan, digest, studies, runs, placement, _ = load_plan(root, plan_path, approval_path)
    observed = host_check(host, plan)
    if gpu_index is not None and (type(gpu_index) is not int or gpu_index not in {r["index"] for r in observed}):
        raise ValueError("Requested GPU index is absent from verified hardware")
    repository, policy = receipt_authentication(root, plan)
    output = confined(root, plan['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    owner_path = output / (host + '.startup-owner.json')
    with (output / (host + '.startup.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(owner_path.read_text()) if owner_path.exists() else None
        alive = {}
        if previous is not None:
            if previous.get('host') != host or previous.get('plan_hash') != digest:
                raise ValueError('Startup custody belongs to a different plan or host')
            if any(name not in previous for name in ('supervisor', 'rendezvous')):
                raise RuntimeError('Partial startup custody remains; inspect it before continuing')
            alive = {name: live_owned(previous[name]) for name in ('supervisor', 'rendezvous')}
            if alive['supervisor'] and not alive['rendezvous'] and all_global_gates_closed(root, plan, digest):
                request_gpu_admission(output, host, digest, observed, gpu_index)
                return {'status': 'already_running', **previous, 'phase_sync_finished': True,
                        'scientific_children_restarted': False, 'slots_per_gpu': 2}
            terminal = local_terminal_roster(output, host, studies, runs, placement)
            if terminal is not None:
                return {'status': 'host_roster_terminal', **previous, **terminal,
                        'phase_sync_alive': alive['rendezvous'], 'scientific_children_restarted': False}
            if any(alive.values()) and not all(alive.values()):
                raise RuntimeError('Only one startup process remains alive; inspect custody before continuing')
            if all(alive.values()):
                request_gpu_admission(output, host, digest, observed, gpu_index)
                return {'status': 'already_running', **previous,
                        'scientific_children_restarted': False, 'slots_per_gpu': 2}
        request_gpu_admission(output, host, digest, observed, gpu_index)
        # Check the actual runtime scripts before either detached process starts.
        runtime = runtime_preflight(plan, python)
        package = Path(__file__).resolve().parent
        sidecar = package / 'phase_rendezvous.py'
        if not sidecar.is_file():
            raise ValueError('Missing frozen phase synchronization runtime')
        commands = {
            'rendezvous': [str(python), '-B', '-u', str(sidecar), '--project-root', str(root),
                '--plan', str(Path(plan_path).resolve()), '--approval', str(Path(approval_path).resolve()),
                '--host', host, '--source-repository', str(repository),
                '--remote', policy['remote_name'], '--watch'],
            'supervisor': [str(python), '-B', '-u', str(package / 'launcher.py'),
                '--project-root', str(root), '--plan', str(Path(plan_path).resolve()),
                '--approval', str(Path(approval_path).resolve()), '--host', host, '--python', str(python)]}
        owner = {'host': host, 'plan_hash': digest, 'runtime_compatibility': runtime}
        for name in ('rendezvous', 'supervisor'):
            log_path = output / (host + '.' + name + '.startup.log')
            with log_path.open('a') as log:
                child = subprocess.Popen(commands[name], cwd=root, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(child.pid)
            if identity is None:
                raise RuntimeError(name + ' exited before startup custody could be recorded')
            owner[name] = identity
            # Preserve partial startup custody even if the second process fails.
            partial = output / (host + '.startup-attempt-' + str(child.pid) + '.json')
            with partial.open('x') as stream:
                stream.write(canonical(owner) + '\n')
            install_startup_owner(owner_path, owner)
            time.sleep(1)
            if child.poll() is not None:
                raise RuntimeError(name + ' exited during startup; inspect ' + str(log_path))
        return {'status': 'started', **owner, 'survives_SSH_or_MacLink_disconnect': True,
                'slots_per_gpu': 2, 'phase_sync_requires_MacLink': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--host", choices=("allocation", "server77"), required=True)
    parser.add_argument("--python")
    parser.add_argument("--gpu-index", type=int, help="Enable one GPU in the single host supervisor")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Start a session-independent supervisor after GO verification")
    args = parser.parse_args()
    if args.gpu_index is not None and not args.detach:
        parser.error("--gpu-index requires --detach")
    if args.host == 'allocation':
        host_check(args.host, {})
    root = Path(args.project_root).resolve()
    if not args.preflight_only and not args.approval:
        raise ValueError("Explicit GO receipt is required")
    for path in (args.plan, args.approval):
        if path and not Path(path).resolve().is_relative_to(root):
            raise ValueError("Plan and approval must stay in the project")
    python = Path(args.python).resolve() if args.python else root / ".venv/bin/python"
    if args.detach:
        if args.preflight_only:
            raise ValueError("A preflight does not start a supervisor")
        print(canonical(detach_host(root, args.plan, args.approval, args.host, python, gpu_index=args.gpu_index)))
    else:
        run_supervisor(root, Path(args.plan), Path(args.approval) if args.approval else None,
                       args.host, python, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
