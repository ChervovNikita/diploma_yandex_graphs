#!/usr/bin/env python3
"""Four remaining discarded TRAIN control cost cycles; stdlib supervisor only.

Requires a separate root release. No fit, VALID or TEST execution is admitted.
Only the freshly spawned and identity-verified scientific child group can be
terminated when a declared bound is exceeded or required bound telemetry fails.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HOST = 'anogena-2-0'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CELLS = [('capable_single', 'live_transfer'), ('untied4', 'live_transfer'), ('ordinary_native4', 'ordinary_joint'),
         ('shared_f4', 'ordinary_joint')]


class IncompleteExitObservation(RuntimeError):
    """Same recorded process/group, but empty cmdline cannot authorize signals."""


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1024 * 1024):
            value.update(part)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    directory = Path('/proc') / str(pid)
    def stat():
        raw = (directory / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'PID': pid, 'start_ticks': int(fields[19]), 'state': fields[0],
                'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'sid': int(fields[3])}
    # cmdline becomes empty during exit. A second stat observation identifies
    # confirmed zombies, while start/group changes remain identity failures.
    for attempt in range(3):
        try:
            before = stat()
            argv = [s.decode() for s in (directory / 'cmdline').read_bytes().split(b'\0') if s]
            after = stat()
        except FileNotFoundError:
            return None
        if any(before[key] != after[key] for key in ('start_ticks', 'pgid', 'sid')):
            raise RuntimeError('Process identity changed during observation; no signal authorized')
        after['argv'] = argv
        after['observation_complete'] = bool(argv or after['state'] == 'Z')
        if after['state'] == 'Z' or argv or attempt == 2:
            return after
        time.sleep(.01)


def owned_tree(owner):
    leader = identity(owner['PID'])
    if leader is None or leader['state'] == 'Z':
        return []
    if any(leader[key] != owner[key] for key in ('start_ticks', 'pgid', 'sid')):
        raise RuntimeError('Owned child identity changed; no signal authorized')
    if not leader['observation_complete']:
        return [leader]  # Explicit incomplete exit observation, never signal identity.
    if leader['argv'] != owner['argv']:
        raise RuntimeError('Owned child argv changed; no signal authorized')
    rows, pending = [], [leader]
    while pending:
        row = pending.pop()
        rows.append(row)
        children = Path('/proc') / str(row['PID']) / 'task' / str(row['PID']) / 'children'
        try:
            pids = children.read_text().split()
        except FileNotFoundError:
            continue
        for token in pids:
            child = identity(int(token))
            if child is None:
                continue
            if child['ppid'] != row['PID'] or child['pgid'] != owner['PID'] or child['sid'] != owner['PID']:
                raise RuntimeError('Owned descendant escaped declared child group')
            pending.append(child)
    return rows


def kill_owned(process, owner, reason):
    live = identity(owner['PID'])
    if live is None or live['state'] == 'Z' or process.poll() is not None:
        return None
    if any(live[key] != owner[key] for key in ('start_ticks', 'pgid', 'sid')):
        raise RuntimeError('Identity changed; refusing group termination')
    if not live['observation_complete'] or not live['argv']:
        raise IncompleteExitObservation('Empty cmdline exit observation; no signal authorized')
    if live['argv'] != owner['argv']:
        raise RuntimeError('Argv changed; refusing group termination')
    if live['pgid'] != live['sid'] or live['pgid'] != process.pid:
        raise RuntimeError('Fresh scientific child session no longer matches')
    os.killpg(process.pid, signal.SIGKILL)
    return {'UTC': now(), 'PID': process.pid, 'start_ticks': owner['start_ticks'],
            'signal': 'SIGKILL', 'reason': reason, 'owned_group_only': True}


def observe_terminal(process, owner, started, limits, signals, reason, signal_refusal):
    """Only the direct Popen wait/poll can provide this child's exit status."""
    observed = False
    remaining = max(.01, limits['external_hard_seconds_per_cell'] - (time.monotonic() - started))
    try:
        process.wait(timeout=remaining)
        observed = True
    except subprocess.TimeoutExpired:
        reason = 'external_hard_wall_bound'
        try:
            sent = kill_owned(process, owner, reason) if owner is not None else None
            if sent: signals.append(sent)
        except Exception as refusal:
            signal_refusal = type(refusal).__name__ + ': ' + str(refusal)
        if process.poll() is not None:
            observed = True
        elif signals:
            try:
                process.wait(timeout=5)
                observed = True
            except subprocess.TimeoutExpired:
                pass
    return reason, signal_refusal, observed


def query(arguments, timeout):
    return subprocess.check_output(['nvidia-smi', *arguments], text=True,
                                   timeout=timeout).strip().splitlines()


def physical_preflight(limits):
    if socket.gethostname() != HOST:
        raise RuntimeError('Authorized one-GPU host differs')
    rows = query(['--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                 limits['telemetry_timeout_seconds'])
    if len(rows) != 1:
        raise RuntimeError('Require the physical singleton authorized GPU')
    uuid, free = [part.strip() for part in rows[0].split(',')]
    if uuid != GPU or int(free) * 1024 ** 2 < limits['minimum_fresh_GPU_free_bytes']:
        raise RuntimeError('Authorized singleton GPU or fresh free-memory bound failed')
    return {'UTC': now(), 'hostname': HOST, 'GPU_UUID': uuid,
            'GPU_free_bytes': int(free) * 1024 ** 2}


def resources(rows, limits, remaining):
    rss = 0
    for row in rows:
        try:
            status = (Path('/proc') / str(row['PID']) / 'status').read_text()
        except FileNotFoundError:
            continue
        rss += sum(int(line.split()[1]) * 1024 for line in status.splitlines()
                   if line.startswith('VmRSS:'))
    pids = {row['PID'] for row in rows}
    gpu = {}
    timeout = max(.05, min(limits['telemetry_timeout_seconds'], remaining))
    for line in query(['--query-compute-apps=gpu_uuid,pid,used_memory',
                       '--format=csv,noheader,nounits'], timeout):
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 3 and parts[0] == GPU and parts[1].isdigit() and int(parts[1]) in pids:
            if not parts[2].isdigit():
                raise RuntimeError('Owned GPU memory telemetry unavailable')
            gpu[int(parts[1])] = int(parts[2]) * 1024 ** 2
    return rss, sum(gpu.values())


def verify_packet():
    manifest = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        path = (HERE / row['path']).resolve(strict=True)
        if not path.is_relative_to(HERE) or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise RuntimeError('Reviewed queue source bytes changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or Path.cwd().resolve() != REPO or not HERE.is_relative_to(PHASE.resolve(strict=True)):
        raise RuntimeError('Require normal execution in the authorized repository')
    release_path = args.release.resolve(strict=True)
    if not release_path.is_relative_to(PHASE.resolve(strict=True)):
        raise RuntimeError('Root release leaves authorized project phase')
    release = json.loads(release_path.read_text())
    plan = json.loads((HERE / 'PLAN.json').read_text())
    limits = json.loads((HERE / 'LIMITS.json').read_text())
    verify_packet()
    if release.get('approved') is not True or release.get('scope') != 'four_remaining_discarded_complete_TRAIN_cost_cycles':
        raise RuntimeError('Separate root approval required; template is disabled')
    for field, actual in [('queue_source_manifest_sha256', sha(HERE / 'SOURCE_MANIFEST.json')),
                          ('queue_program_sha256', sha(__file__)), ('plan_sha256', sha(HERE / 'PLAN.json'))]:
        if release.get(field) != actual:
            raise RuntimeError('Root release does not bind exact queue bytes')
    if release.get('limits') != limits or sha(HERE / 'LIMITS.json') != plan['limits_sha256']:
        raise RuntimeError('Prospective resource bounds changed')
    if any(release.get(key) is not False for key in ('fits_authorized', 'VALID_values_access', 'TEST_access', 'retry')):
        raise RuntimeError('Cost queue admits no fits, VALID, TEST or retry')
    if not release.get('root_review_evidence'):
        raise RuntimeError('Root queue source review evidence required')
    for evidence in release['root_review_evidence']:
        path = (PHASE / evidence['path']).resolve(strict=True)
        if not path.is_relative_to(PHASE.resolve(strict=True)) or sha(path) != evidence['sha256']:
            raise RuntimeError('Root source-review evidence changed')
    output = Path(plan['queue_output_directory'])
    if release.get('queue_output_directory') != str(output) or output.exists() or not output.parent.is_dir() or not output.resolve().is_relative_to(PHASE.resolve(strict=True)):
        raise RuntimeError('Require exact fresh queue output under repository')
    source = PHASE / plan['source_name']
    if sha(source / 'SOURCE_MANIFEST.json') != plan['source_manifest_sha256'] or sha(source / 'run.py') != plan['run_sha256']:
        raise RuntimeError('Qualified method source differs')
    if [(row['arm'], row['rule']) for row in plan['cells']] != CELLS:
        raise RuntimeError('Four remaining prospectively requested control cells differ')
    precursor_path = HERE / plan['precursor_binding_file']
    if sha(precursor_path) != plan['precursor_binding_sha256']:
        raise RuntimeError('Completed candidate precursor binding changed')
    precursor = json.loads(precursor_path.read_text())
    for key in ('completed_candidate', 'candidate_child_started', 'v1_queue_failure'):
        evidence = precursor[key]
        path = (PHASE / evidence['remote_phase_relative_path']).resolve(strict=True)
        if not path.is_relative_to(PHASE.resolve(strict=True)) or sha(path) != evidence['sha256']:
            raise RuntimeError('Completed candidate or preserved v1 failure changed')
    gate = PHASE / plan['training_step_gate']['path']
    if sha(gate) != plan['training_step_gate']['sha256'] or json.loads(gate.read_text()).get('passed') is not True:
        raise RuntimeError('Exact passed FP32 gate changed')
    physical_preflight(limits)
    output.mkdir()
    summary = {'UTC_started': now(), 'scope': 'four_remaining_discarded_TRAIN_control_cost_cycles',
               'source_manifest_sha256': plan['source_manifest_sha256'],
               'queue_source_manifest_sha256': sha(HERE / 'SOURCE_MANIFEST.json'),
               'root_release_sha256': sha(release_path), 'supervisor_identity': identity(os.getpid()),
               'precursor_binding_sha256': plan['precursor_binding_sha256'],
               'fits': 0, 'VALID_TEST_values_access': False, 'retry': False, 'cells': [], 'passed': False}
    save(output / 'QUEUE_STARTED.json', summary)
    try:
        for cell in plan['cells']:
            verify_packet()
            job_path = HERE / cell['job_file']
            if sha(job_path) != cell['job_sha256']:
                raise RuntimeError('Cost cell job bytes changed')
            job = json.loads(job_path.read_text())
            if job['purpose'] != 'TRAIN_only_discarded_complete_cycle_cost' or any(job[key] is not False for key in ('fits_authorized', 'VALID_values_access', 'TEST_access', 'retry')):
                raise RuntimeError('Cell is not one discarded TRAIN cost measurement')
            if (job['seed'], job['factor_seed'], job['outer_size'], job['inner_size'], job['cycles'], job['soft_seconds'], job['geometry']) != (20261005, 20261005, 64, 256, 1, 600, 'endpoint'):
                raise RuntimeError('Prospectively requested episode recipe changed')
            cell_root = output / cell['cell_id']
            if job['output_directory'] != str(cell_root / 'result'):
                raise RuntimeError('Fresh cell output binding differs')
            preflight = physical_preflight(limits)
            cell_root.mkdir()
            save(cell_root / 'PREFLIGHT.json', preflight)
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                       PYTHONPATH=str(PHASE / 'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO / '.venv/lib/python3.11/site-packages'))
            env.pop('PYTHONHOME', None)
            argv = [str(PHASE / plan['interpreter_relative']), str(source / 'run.py'),
                    '--job', str(job_path), '--output', job['output_directory']]
            started = time.monotonic()
            reason, signals, max_rss, max_gpu = None, [], 0, 0
            signal_refusal, terminal_wait_observed = None, False
            incomplete_exit_samples, last_incomplete_exit = 0, None
            owner = None
            with (cell_root / 'child.stdout.log').open('xb') as stdout, (cell_root / 'child.stderr.log').open('xb') as stderr:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           cwd=REPO, env=env, start_new_session=True)
                try:
                    owner = identity(process.pid)
                except Exception as error:
                    reason = 'Initial child observation failed: ' + type(error).__name__ + ': ' + str(error)
                if owner is None or owner['argv'] != argv or owner['pgid'] != process.pid or owner['sid'] != process.pid:
                    reason = 'Initial owned child identity could not be verified; no signal sent'
                save(cell_root / 'CHILD_STARTED.json', {'UTC': now(), 'identity': owner, 'argv': argv, 'limits': limits})
                try:
                    while process.poll() is None:
                        if reason is not None:
                            break
                        remaining = limits['external_hard_seconds_per_cell'] - (time.monotonic() - started)
                        if remaining <= 0:
                            reason = 'external_hard_wall_bound'
                        else:
                            rows = owned_tree(owner)
                            incomplete = [row for row in rows if not row.get('observation_complete', True)]
                            if incomplete:
                                incomplete_exit_samples += 1
                                last_incomplete_exit = {'UTC': now(), 'rows': incomplete}
                                if process.poll() is not None:
                                    break  # Authoritative termination, never inferred from /proc.
                            rss, gpu = resources(rows, limits, remaining)
                            max_rss, max_gpu = max(max_rss, rss), max(max_gpu, gpu)
                            log_bytes = sum((cell_root / name).stat().st_size for name in ('child.stdout.log', 'child.stderr.log'))
                            if max_rss > limits['owned_tree_RSS_cap_bytes']: reason = 'owned_tree_RSS_cap'
                            elif max_gpu > limits['owned_tree_GPU_memory_cap_bytes']: reason = 'owned_tree_GPU_memory_cap'
                            elif log_bytes > limits['combined_child_log_cap_bytes']: reason = 'combined_child_log_cap'
                            elif time.monotonic() - started >= limits['external_hard_seconds_per_cell']: reason = 'external_hard_wall_bound'
                        if reason:
                            try:
                                sent = kill_owned(process, owner, reason)
                                if sent: signals.append(sent)
                            except IncompleteExitObservation as refusal:
                                signal_refusal = type(refusal).__name__ + ': ' + str(refusal)
                            break
                        time.sleep(min(limits['poll_interval_seconds'], max(.01, remaining)))
                except (Exception, KeyboardInterrupt) as error:
                    reason = 'required_bounds_telemetry_failure: ' + type(error).__name__ + ': ' + str(error)
                    try:
                        sent = kill_owned(process, owner, reason) if owner is not None else None
                        if sent: signals.append(sent)
                    except Exception as refusal:
                        signal_refusal = type(refusal).__name__ + ': ' + str(refusal)
                finally:
                    reason, signal_refusal, terminal_wait_observed = observe_terminal(
                        process, owner, started, limits, signals, reason, signal_refusal)
            receipt = {'UTC': now(), 'cell_id': cell['cell_id'], 'job_sha256': cell['job_sha256'],
                       'exit_code': process.returncode if terminal_wait_observed else None,
                       'exit_code_authority': 'subprocess.Popen.wait/poll' if terminal_wait_observed else None,
                       'terminal_wait_observed': terminal_wait_observed, 'signal_refusal': signal_refusal,
                       'incomplete_exit_observation_samples': incomplete_exit_samples,
                       'last_incomplete_exit_observation': last_incomplete_exit,
                       'reason': reason, 'child_identity': owner,
                       'signals_sent': signals, 'inclusive_seconds': time.monotonic() - started,
                       'max_sampled_owned_tree_RSS_bytes': max_rss, 'max_sampled_owned_tree_GPU_memory_bytes': max_gpu,
                       'attempts': 1, 'fits': 0, 'VALID_TEST_values_access': False, 'retry': False,
                       'logs': {name: {'sha256': sha(cell_root / name), 'bytes': (cell_root / name).stat().st_size}
                                for name in ('child.stdout.log', 'child.stderr.log')}}
            result_path = cell_root / 'result' / 'COST_RESULT.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                receipt['cost_result_sha256'] = sha(result_path)
                receipt['complete_cycle_costs'] = result['cycle_costs']
                receipt['passed'] = (terminal_wait_observed and process.returncode == 0 and reason is None and
                    result.get('scope') == 'discarded_TRAIN_complete_cycle_cost' and result.get('states_discarded') is True and
                    result.get('complete_VALID_scoring') is False and result.get('TEST_access') is False and result.get('last_cycle') == 1 and
                    result.get('source_manifest_sha256') == plan['source_manifest_sha256'] and result.get('job_sha256') == cell['job_sha256'])
            else:
                receipt['passed'] = False
            save(cell_root / 'EXECUTION_RECEIPT.json', receipt)
            summary['cells'].append(receipt)
            save(output / 'QUEUE_PROGRESS.json', summary)
            if not receipt['passed']:
                raise RuntimeError('Cost cell failed; stop queue without retry')
        summary.update(passed=True, UTC_finished=now())
        save(output / 'QUEUE_RESULT.json', summary)
    except (Exception, KeyboardInterrupt) as error:
        summary.update(passed=False, UTC_finished=now(), error=type(error).__name__ + ': ' + str(error))
        save(output / 'QUEUE_FAILURE.json', summary)
        raise


if __name__ == '__main__':
    main()
