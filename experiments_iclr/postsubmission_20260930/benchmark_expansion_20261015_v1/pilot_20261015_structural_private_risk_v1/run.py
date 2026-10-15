"""Frozen six-fit preflight and one-worker execution through the original runner.

Stdlib only. Root publication/resource admission is separate from this source.
No TRAIN/VALID score interpretation or selected-state collection occurs here.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

HOST = 'anogena-2-0'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
EXPANSION = Path('experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1')
SEEDS = [101, 202, 303]


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid())) if replace else path
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    if replace: temporary.replace(path)


def load(path, name):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition); sys.modules[name] = module
    definition.loader.exec_module(module)
    return module


def guard():
    require(socket.gethostname() == HOST and subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() == [GPU],
        'Exact dedicated allocation required before project reads')


def safe(root, path):
    root, path = Path(root).resolve(), Path(path)
    value = (path if path.is_absolute() else root / EXPANSION / path).resolve()
    require(value.is_relative_to(root), 'Project-contained path required')
    return value


def verified(root, receipt):
    path = safe(root, receipt['path']); actual = pin(path)
    require(actual['sha256'] == receipt['sha256'] and ('bytes' not in receipt or actual['bytes'] == receipt['bytes']),
            'Frozen input changed: ' + str(path))
    return path


def validate_structure(spec, jobs):
    require(spec['schema'] == 'molbace-structural-private-risk-six-fit-pilot-v1' and spec['new_fits'] == 6 and
            spec['seeds'] == SEEDS and spec['TEST_access'] is False and spec['full_campaign_GO'] is False and
            spec['campaign_roster_additions'] == spec['new_control_fits'] == 0, 'Exact six-fit scope required')
    order = [(seed, variant) for seed in SEEDS for variant in ('structured', 'permuted')]
    require([(r['seed'], r['variant']) for r in spec['jobs']] == order and len(jobs) == 6 and
            len({j['id'] for j in jobs}) == 6 and [digest(j) for j in jobs] == [r['job_hash'] for r in spec['jobs']],
            'Frozen six-job order/identity differs')
    require(all(j['TEST_access'] is False and j['policy'] == jobs[0]['policy'] and j['device'] == 'cuda:0' for j in jobs),
            'Unchanged native controller and closed TEST required')
    require(spec['cost_bounds']['maximum_wall_seconds_per_fit'] == 14400 and
            spec['cost_bounds']['maximum_concurrent_new_GPU_workers'] == 1, 'One new worker/four-hour fit bound required')


def terminal_gate(end, outcomes, rows):
    require(len(rows) == 6 and len({r['id'] for r in rows}) == 6 and end.get('status') == 'complete' and
            end.get('TEST_accessed') is False and end.get('outcomes') == outcomes and len(outcomes) == 6 and
            [r.get('id') for r in outcomes] == [r['id'] for r in rows] and all(
                r.get('status') == 'complete' and r.get('returncode') == 0 and not r.get('timed_out') and
                r.get('RESULT', {}).get('sha256') for r in outcomes),
            'All six successful terminal outcomes required; no partial analysis/collection')


def preflight(root, here):
    spec = json.loads((here / 'SPEC.json').read_text())
    require(Path(spec['project_root']).resolve() == root, 'Frozen repository root differs')
    manifest = json.loads((here / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        path = (here / row['path']).resolve()
        require(path.is_relative_to(here), 'Pilot source escapes package')
        verified(root, dict(row, path=str(path)))
    for receipt in spec['source_pins'] + spec['data_role_pins']:
        verified(root, receipt)
    require(all(not safe(root, path).exists() for path in spec['absent_role_overrides']), 'Native role override appeared')
    jobs = []
    for row in spec['jobs']:
        path = here / row['file']
        require(pin(path)['sha256'] == row['file_sha256'] and path.stat().st_size == row['file_bytes'], 'Frozen JOB file changed')
        jobs.append(json.loads(path.read_text()))
    validate_structure(spec, jobs)
    extension = load(safe(root, 'research/structural_residual_composition_v2/extension.py'), '_frozen_structural_v2_source')
    extension.source_bindings(root, spec['producer_manifest']['sha256'])
    for row, job in zip(spec['jobs'], jobs):
        parent = json.loads((extension.PARENT / ('JOB_seed' + str(row['seed']) + '.json')).read_text())
        require(job == extension.descriptor(parent, row['procedure'], spec['producer_manifest']['sha256']),
                'JOB differs from exact genuine v2 producer and parent')
    for qualification in spec['actual_v2_qualifications']:
        value = json.loads(verified(root, qualification['authority']).read_text()); result = value['result']
        require(value['status'] == 'complete' and result['status'] == 'passed' and value['exit_code'] == 0 and
                result['variant'] == qualification['variant'] and result['TEST_accessed'] is False and
                result['optimizer_calls'] == {'attempts': [2], 'completed': [2]} and
                result['native_forwards_completed'] == 73 and result['reverse_completed_including_checks'] == 8 and
                result['VALID_completed'] == 3 and result['fresh_restore_VALID_bit_exact'] is True and
                result['native_constructions_completed'] == 2 and result['initial_native_eval_bit_exact'] is True and
                result['changed_state_rejected'] is True and result['correct_shared_F_private_W_same_old_state'] is True and
                all(result['fresh_optimizer_restore_contract'][key] is True for key in (
                    'actual_parameter_IDs_and_group_boundaries_unchanged', 'every_group_option_unchanged', 'optimizer_defaults_unchanged')),
                'Actual exact v2 CPU qualification differs')
        require(any(r['sha256'] == spec['producer_manifest']['sha256'] and r['path'].endswith('/structural_residual_composition_v2/MANIFEST.json')
                    for r in result['source_receipts']), 'Qualification used another producer')
    verified(root, spec['retained_failed_v1_qualification'])
    prior = json.loads(verified(root, spec['completed_uniform_and_33_bank_authority']).read_text())
    require(prior['status'] in ('complete', 'complete_with_warnings') and prior['TEST_accessed'] is False and
            len(prior['rows']) == 33 and len(prior['training_fit_outcomes']) == 39 and
            prior['training_fit_outcomes'] == spec['reused_training_outcomes'], 'Completed 33-bank/39-fit authority differs')
    for row in spec['reused_training_outcomes']:
        value = json.loads(verified(root, row['receipt']).read_text())
        require(value['status'] == 'complete' and value['job_id'] == row['job_id'] and value['TEST_accessed'] is False,
                'Reused control fit differs')
    sys.path.insert(0, str(safe(root, 'runtime')))
    from run_fit import validate_job
    for job in jobs: validate_job(job)
    return spec, jobs


def ticks(pid):
    return int(Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19])


def check_peers(allowed):
    require(isinstance(allowed, list) and len(allowed) <= 1 and all(type(r['pid']) is int and r['pid'] > 0 and
            type(r['start_ticks']) is int and r['start_ticks'] > 0 for r in allowed), 'Bind the sole older GPU peer or none')
    active = {int(x.strip()) for x in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines() if x.strip()}
    require(active <= {r['pid'] for r in allowed}, 'Unexpected GPU process; reserved slot unavailable')
    require(all(ticks(r['pid']) == r['start_ticks'] for r in allowed if r['pid'] in active), 'Allowed peer PID was reused')
    return sorted(active)


def terminate(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()


def interrupted(signum, frame):
    raise InterruptedError('Pilot owner received signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--mode', choices=('preflight', 'launch'), default='preflight')
    parser.add_argument('--admission'); parser.add_argument('--admission-sha256')
    args = parser.parse_args(); guard()
    root, here = Path(args.project_root).resolve(), Path(__file__).resolve().parent
    spec, jobs = preflight(root, here)
    if args.mode == 'preflight':
        print(json.dumps({'status': 'preflight_passed_no_launch', 'jobs': 6, 'SPEC': pin(here / 'SPEC.json'), 'TEST_accessed': False})); return 0
    require(args.admission and args.admission_sha256, 'Root publication and resource admission required')
    admission_path = safe(root, args.admission)
    require(pin(admission_path)['sha256'] == args.admission_sha256, 'Root admission changed')
    admission = json.loads(admission_path.read_text())
    require(admission.get('schema') == 'root-reviewed-six-structural-private-risk-fits-v1' and
            admission.get('approved') is True and admission.get('publication_approved') is True and
            admission.get('resource_admission_approved') is True and admission.get('new_fits') == 6 and
            admission.get('maximum_concurrent_new_GPU_workers') == 1 and admission.get('maximum_wall_seconds_per_fit') == 14400 and
            admission.get('allocation_GPU_UUID') == GPU and admission.get('full_campaign_GO') is False and
            admission.get('pilot_manifest_sha256') == pin(here / 'MANIFEST.json')['sha256'] and
            admission.get('SPEC_sha256') == pin(here / 'SPEC.json')['sha256'], 'Exact root scope/source/resource admission required')
    require(all(type(admission.get(key)) is int and admission[key] > 0 for key in
                ('reserved_GPU_bytes', 'reserved_host_bytes')) and admission.get('resource_admission_note'),
            'Root must record explicit reserved GPU/host budgets and resource assessment')
    require(Path(sys.prefix).resolve() == (root / '.venv').resolve(), 'Original repository virtualenv interpreter required')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == admission['publication_commit'],
            'Actual published checkout differs from admitted commit')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0', GPU), 'Only authorized GPU0 may be visible')
    output = safe(root, spec['study_output']); require(not output.exists(), 'Fresh output required; no retry/overwrite/resume')
    lock_path = safe(root, admission['slot_lock_path']); lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_peers(admission['allowed_other_GPU_processes'])
        output.mkdir(parents=True)
        signal.signal(signal.SIGTERM, interrupted)
        outcomes = [{'index': r['index'], 'id': r['id'], 'seed': r['seed'], 'variant': r['variant'],
                     'status': 'not_started'} for r in spec['jobs']]
        child, failure, row = None, None, None
        started = time.monotonic()
        write(output / 'START.json', {'SPEC': pin(here / 'SPEC.json'), 'manifest': pin(here / 'MANIFEST.json'),
            'admission': pin(admission_path), 'owner_pid': os.getpid(), 'owner_start_ticks': ticks(os.getpid()),
            'new_fits': 6, 'maximum_new_GPU_workers': 1, 'wall_limit_per_fit_seconds': 14400,
            'automatic_retries': 0, 'TEST_accessed': False})
        try:
            for binding, job in zip(spec['jobs'], jobs):
                observed = check_peers(admission['allowed_other_GPU_processes'])
                fit = output / binding['fit']
                command = [sys.executable, '-B', '-u', str(safe(root, 'runtime/run_fit.py')), '--project-root', str(root),
                           '--job', str(here / binding['file']), '--output', str(fit)]
                row = outcomes[binding['index']]
                row.update(status='started', command=command, observed_peer_GPU_pids=observed, started_unix=time.time())
                write(output / 'OUTCOMES.json', outcomes, True)
                environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                                   CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1')
                before = time.monotonic()
                try:
                    with (output / ('fit_' + str(binding['index']).zfill(3) + '.log')).open('x') as log:
                        child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        row.update(worker_pid=child.pid, worker_start_ticks=ticks(child.pid))
                        write(output / 'OUTCOMES.json', outcomes, True)
                        try: row['returncode'] = child.wait(timeout=14400)
                        except subprocess.TimeoutExpired:
                            row['timed_out'] = True; terminate(child); row['returncode'] = child.returncode
                finally:
                    terminate(child); row['wall_seconds'] = time.monotonic() - before
                if (fit / 'RESULT.json').is_file():
                    result = json.loads((fit / 'RESULT.json').read_text())
                    row.update(status=result['status'], RESULT=pin(fit / 'RESULT.json'),
                               **{k: result.get(k) for k in ('selected_epoch', 'completed_epochs', 'stop_reason',
                                  'convergence_established', 'optimizer_calls', 'optimizer_calls_this_invocation',
                                  'resources', 'invocation_wall_seconds', 'failure', 'close_error')})
                    require(result['job_hash'] == binding['job_hash'] and result['job_id'] == job['id'] and
                            result['TEST_accessed'] is False, 'Completed child identity/role differs')
                else: row.update(status='missing_RESULT', unreceipted_work='unknown, not zero')
                write(output / 'OUTCOMES.json', outcomes, True)
                require(row['returncode'] == 0 and row['status'] == 'complete' and not row.get('timed_out') and
                        not row.get('close_error'), 'Fit failed/timed out/close failed; later fits remain unstarted')
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(limit=8)}
            if row is not None and row['status'] == 'started':
                row.update(status='owner_interrupted_or_failed', unreceipted_work='unknown, not zero')
            raise
        finally:
            terminate(child); write(output / 'OUTCOMES.json', outcomes, True)
            complete = failure is None and len(outcomes) == 6 and all(r.get('status') == 'complete' and
                r.get('returncode') == 0 and not r.get('timed_out') and r.get('RESULT', {}).get('sha256') for r in outcomes)
            write(output / 'END.json', {'status': 'complete' if complete else 'stopped_no_retry',
                'outcomes': outcomes, 'failure': failure, 'SPEC': pin(here / 'SPEC.json'), 'admission': pin(admission_path),
                'wall_seconds': time.monotonic() - started, 'automatic_retries': 0, 'TEST_accessed': False})
    return 0


if __name__ == '__main__': raise SystemExit(main())
