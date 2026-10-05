#!/usr/bin/env python3
"""Directly own one full native PENCIL zero-update resource probe.

This supervisor does not import numerical libraries or open scientific outcomes.
It signals only identities in the new child's dedicated process session.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PLAN = PHASE / 'pencil_citeseer_bounded_zero_update_resource_plan_20261005_v1'
PROBE = PHASE / 'pencil_citeseer_heart_resource_plan_audit_20261005_v1/probe.py'
COHORT = PHASE / 'citeseer_endpoint_frame_paired_development_20261005_v1'
RUN = HERE / 'run01'
GIB = 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def proc(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    pgid=int(fields[2]), sid=int(fields[3]), start_ticks=int(fields[19]),
                    RSS_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def owned_session(child_pid, identities):
    current = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        row = proc(int(entry.name))
        if row and row['sid'] == child_pid and row['pgid'] == child_pid:
            prior = identities.get(str(row['pid']))
            require(prior is None or prior['start_ticks'] == row['start_ticks'],
                    'Owned PID identity changed inside dedicated session')
            identities[str(row['pid'])] = row
            if row['state'] != 'Z':
                current.append(row)
    return current


def stop_owned(child, identities, reason):
    events = []
    # The child starts its own session. Never use process names or broad pkill.
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        rows = owned_session(child.pid, identities)
        for row in sorted(rows, key=lambda r: r['pid'] == child.pid):
            current = proc(row['pid'])
            if (current and current['start_ticks'] == row['start_ticks']
                    and current['sid'] == current['pgid'] == child.pid):
                try:
                    os.kill(row['pid'], sig)
                    events.append(dict(UTC=now(), pid=row['pid'], start_ticks=row['start_ticks'],
                                       signal=int(sig), reason=reason))
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()
            if not owned_session(child.pid, identities):
                break
            time.sleep(0.2)
        if not owned_session(child.pid, identities):
            break
    return events


def main():
    receipt = dict(schema='pencil_owned_zero_update_supervisor_v1', UTC=now(),
                   hostname=socket.gethostname(), cwd=str(Path.cwd()),
                   supervisor_identity=proc(os.getpid()), status='GATING',
                   source_changes=False, scientific_fits=0, optimizer_updates=0,
                   TEST_access=False, comparative_outcomes_opened=False,
                   retries=0, other_jobs_signalled=False,
                   timing_profile='co_resident_not_uncontended_efficiency',
                   output_path_adjustment='Fresh child run01 under owned execution directory')
    child = None
    identities = {}
    signals_sent = []
    started = time.monotonic()
    try:
        require(Path.cwd().resolve() == REPO and socket.gethostname() == 'anogena-2-0',
                'Authorized repository/hostname differs')
        require(HERE.resolve().is_relative_to(PHASE) and not RUN.exists(),
                'Execution output must be fresh inside authorized phase')
        require(sha(PROBE) == '36a4ccef3f5973ce0583ecbb060672371cb171513b99b48cebb3ef50177ecf51',
                'Unchanged approved probe source differs')
        require(sha(PROBE.parent / 'MANIFEST.json') ==
                'f1a5cae24d746f32360bafe7dcb1eb7557356a05cbb53b28d41a28dcd93894a3',
                'Approved probe manifest differs')
        freeze = json.loads((COHORT / 'COHORT_FREEZE.json').read_text())
        require(freeze.get('complete') is True and len(freeze['completed_physical_fits']) == 36
                and freeze['plan_sha256'] ==
                'fd9fec451a81d0512cd8431ba2a990c580592ff6a6e8e12d784386a99c5e9daa'
                and freeze['source_manifest_sha256'] ==
                'efa95806d86e3cc261a8506042204d8d32e39506386d6faf90a625d52be12ff9',
                'Complete exact 36-fit cohort metadata required')
        receipt['cohort_gate'] = dict(complete=True, completed_physical_fits=36,
                                     freeze_sha256=sha(COHORT / 'COHORT_FREEZE.json'),
                                     outcome_files_opened=False)
        release_path = PLAN / 'ROOT_RESOURCE_RELEASE.json'
        release = json.loads(release_path.read_text())
        receipt['release_sha256'] = sha(release_path)
        require(release['caps'] == dict(wall_seconds=3600,
                max_parent_plus_live_descendant_RSS_bytes=128 * GIB,
                max_cuda_allocated_bytes=24 * GIB, max_cuda_reserved_bytes=28 * GIB),
                'Authorized caps differ')
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(PHASE / 'pencil_one_gpu_dependency_overlay_20261005_v1')
                + ':' + str(PHASE / 'native_ncn_dependency_overlay_20261005_v1')
                + ':' + str(REPO / '.venv/lib/python3.11/site-packages'),
                RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', CUDA_VISIBLE_DEVICES='0',
                PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                WANDB_MODE='disabled', OUTDATED_IGNORE='1')
        command = [str(PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'),
                   str(PROBE), '--release', str(release_path),
                   '--release-sha256', receipt['release_sha256'], '--output', str(RUN)]
        receipt['command'] = command
        receipt['environment'] = {key: environment[key] for key in (
                'PYTHONPATH', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES',
                'PYTHONDONTWRITEBYTECODE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'WANDB_MODE', 'OUTDATED_IGNORE')}
        # Last gate immediately precedes the one child launch. No numeric imports.
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], check=True, capture_output=True,
                text=True, timeout=30)
        rows = [line.split(',') for line in query.stdout.strip().splitlines()]
        require(len(rows) == 1 and rows[0][0].strip() == GPU_UUID,
                'Authorized singleton GPU UUID differs')
        gpu_free = int(rows[0][1].strip()) * 1024 ** 2
        receipt['immediate_gpu_gate'] = dict(UTC=now(), UUID=GPU_UUID,
                free_bytes=gpu_free, minimum_free_bytes=34 * GIB,
                used_bytes=int(rows[0][2].strip()) * 1024 ** 2,
                utilization_percent=int(rows[0][3].strip()))
        require(gpu_free >= 34 * GIB, 'Fresh free GPU memory is below 34 GiB; no launch or retry')
        log = (HERE / 'probe_stdout_stderr.log').open('xb')
        child_started = time.monotonic()
        child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        identity = proc(child.pid)
        require(identity is not None and identity['sid'] == identity['pgid'] == child.pid,
                'Dedicated directly owned child session identity absent')
        identities[str(child.pid)] = identity
        receipt.update(status='RUNNING', child_identity=identity, child_started_UTC=now())
        write('SUPERVISOR_RECEIPT.json', receipt)
        peak_rss = 0
        peak_supervisor_rss = 0
        while True:
            live = owned_session(child.pid, identities)
            total_rss = sum(row['RSS_bytes'] for row in live)
            peak_rss = max(peak_rss, total_rss)
            own = proc(os.getpid())
            peak_supervisor_rss = max(peak_supervisor_rss, own['RSS_bytes'] if own else 0)
            elapsed = time.monotonic() - child_started
            receipt.update(elapsed_child_seconds=elapsed,
                    max_parent_plus_loader_descendants_RSS_bytes=peak_rss,
                    max_supervisor_RSS_bytes=peak_supervisor_rss,
                    live_owned_process_count=len(live), observed_owned_identities=identities)
            failure = None
            if total_rss > release['caps']['max_parent_plus_live_descendant_RSS_bytes']:
                failure = 'OWNED_PROCESS_TREE_RSS_CAP'
            if elapsed > release['caps']['wall_seconds']:
                failure = 'OWNED_CHILD_WALL_CAP'
            progress_path = RUN / 'PROGRESS.json'
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                receipt['latest_probe_progress'] = progress
                for key in ('max_cuda_allocated_bytes', 'max_cuda_reserved_bytes'):
                    if progress.get(key, 0) > release['caps'][key]:
                        failure = key
            if failure:
                signals_sent = stop_owned(child, identities, failure)
                receipt.update(status='FAIL_RESOURCE_CAP_PRESERVED', resource_failure=failure)
                break
            exit_code = child.poll()
            if exit_code is not None:
                receipt['child_exit_code'] = exit_code
                if owned_session(child.pid, identities):
                    signals_sent = stop_owned(child, identities, 'OWNED_CHILD_ENDED_WITH_DESCENDANTS')
                if exit_code == 0 and (RUN / 'RESULT.json').exists():
                    result = json.loads((RUN / 'RESULT.json').read_text())
                    require(result['status'] == 'PASS_ZERO_UPDATE_RESOURCE_ONLY',
                            'Child success without expected complete resource result')
                    receipt.update(status='PASS_ZERO_UPDATE_RESOURCE_ONLY',
                                   probe_RESULT_sha256=sha(RUN / 'RESULT.json'))
                else:
                    receipt['status'] = 'FAIL_CHILD_PRESERVED'
                break
            write('SUPERVISOR_RECEIPT.json', receipt)
            time.sleep(1)
    except BaseException as error:
        receipt.update(status='FAIL_GATE_OR_SUPERVISOR_PRESERVED',
                       error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        if child is not None:
            signals_sent = stop_owned(child, identities, 'SUPERVISOR_EXCEPTION')
    finally:
        if child is not None:
            try:
                receipt['child_exit_code'] = child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                receipt['child_exit_code'] = None
        receipt.update(terminal_UTC=now(), inclusive_supervisor_seconds=time.monotonic() - started,
                       signals_sent=signals_sent)
        for name in ('RESULT.json', 'FAILURE.json', 'PROGRESS.json'):
            path = RUN / name
            if path.exists():
                receipt['probe_' + name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        write('SUPERVISOR_RECEIPT.json', receipt)
        write('TERMINAL_RECEIPT.json', receipt)
        print(json.dumps(dict(status=receipt['status'], UTC=now()), sort_keys=True))


if __name__ == '__main__':
    main()
