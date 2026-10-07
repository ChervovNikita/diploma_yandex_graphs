"""Disabled one-shot 24h engineering wait, then exec the exact frozen Mol18 executor."""
import argparse
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

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = PHASE.parents[1]
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
WAIT_HARD = 86400
MINIMUM_FREE = 8 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def inside(relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Phase-relative project path required')
    path = (PHASE / rel).resolve()
    require(path != PHASE.resolve() and path.is_relative_to(PHASE.resolve()), 'Path leaves project phase')
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE.resolve())), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Exact metadata/source binding changed')
    return path


def sealed(row):
    path = bound(row)
    for item in read(path)['files']:
        file = path.parent / item['path']
        require(file.is_file() and sha(file) == item['sha256'] and file.stat().st_size == item['bytes'], 'Sealed source bytes changed')
    return path


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def proc(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()
        return dict(PID=pid, start_ticks=int(fields[19]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(pid, ticks):
    actual = proc(pid)
    return bool(actual and actual['start_ticks'] == ticks and actual['state'] not in ('Z', 'X'))


def pending(paths, owner_live, label):
    missing = [path.name for path in paths if not path.is_file()]
    require(not missing or owner_live, label + ' ended without required receipt(s): ' + ', '.join(missing))
    return bool(missing)


def qualifier(pins):
    spec = pins['qualifier']; root = inside(spec['receipts']); saved = spec['owner']
    alive = live(saved['PID'], saved['start_ticks'])
    if pending([root / 'OWNER.json', root / 'TERMINAL.json'], alive, 'Existing qualifier'):
        return None
    owner, terminal = read(root / 'OWNER.json'), read(root / 'TERMINAL.json')
    require(owner['parent_PID'] == saved['PID'] and owner['parent_start_ticks'] == saved['start_ticks'], 'Exact saved qualifier owner')
    require(terminal.get('complete') is True, 'Qualifier retained failure; no scientific start')
    if pending([root / 'WORK_RECEIPT.json', root / 'INVOCATION.json'], alive, 'Existing qualifier'):
        return None
    work, invocation = read(root / 'WORK_RECEIPT.json'), read(root / 'INVOCATION.json')
    require(work['schema'] == 'allocation-OIPG-MolHIV-GPU-representative-work-v1'
        and work.get('complete') is True and work.get('GPU_execution') is True
        and work.get('scientific_fit') is False and work.get('quality_values_closed') is True and work.get('TEST_scoring') is False
        and work['worker_source_manifest_sha256'] == pins['qualifier_manifest']['sha256']
        and work['worker_program_sha256'] == sha(inside(pins['qualifier_manifest']['path']).parent / 'check.py')
        and invocation['source_manifest_sha256'] == pins['qualifier_manifest']['sha256']
        and invocation['approval_sha256'] == pins['qualifier_approval']['sha256']
        and work['parent_terminal_sha256'] == sha(root / 'TERMINAL.json') and work['parent_costs'] == terminal,
        'Complete source-bound OIPG CUDA work receipt required')
    require(terminal['GPU_child_started'] is True and terminal['exit_code'] == 0 and terminal['reaped'] is True
        and terminal['cap_exceeded'] is False and not terminal['monitor_errors']
        and work['exit_code'] == 0 and work['reaped'] is True and work['identity'] == terminal['identity'], 'Actual completed/reaped qualifier worker')
    identity = terminal['identity']
    require(identity['parent_PID'] == saved['PID'] and identity['parent_start_ticks'] == saved['start_ticks']
        and type(identity['worker_PID']) is int and identity['worker_PID'] > 0
        and type(identity['worker_start_ticks']) is int and identity['worker_start_ticks'] > 0
        and owner['worker_PID'] == identity['worker_PID'] and owner['worker_start_ticks'] == identity['worker_start_ticks'], 'Saved qualifier worker custody')
    expected = pins['adopted_sources']
    fields = {'public_manifest_sha256': 'portable_internal_be_public_interface_20261007_v2',
        'adapter_manifest_sha256': 'public_internal_be_private_steering_adapter_20261007_v1',
        'complete_interface_manifest_sha256': 'public_internal_be_private_steering_complete_interface_20261007_v1',
        'allocation_controls_manifest_sha256': 'public_internal_be_allocation_controls_20261007_v1'}
    require(all(work[key] == expected[name] for key, name in fields.items()) and work['data'] == pins['role_hashes']
        and [case['policy'] for case in work['cases']] == ['O', 'I', 'P', 'G']
        and all(case['actual_CUDA_parameters_gradients_and_Adam'] is True for case in work['cases'])
        and work['runtime']['physical_gpu_uuid'] == GPU and work['runtime']['visible_device_count'] == 1
        and work['runtime']['cuda_initialized'] is True, 'Exact adopted CUDA representative sources/roles')
    admission = work['admission']; closure = inside(pins['Wiki_closure'])
    require(admission['mode'] == 'whole_Wiki24_closed_all_science_terminal'
        and admission['closure_path'] == str(closure) and admission['closure_sha256'] == sha(closure)
        and admission['original_science_processes_live'] == [], 'Qualifier must provide whole-Wiki terminal admission')
    if alive or live(identity['worker_PID'], identity['worker_start_ticks']):
        return None
    return dict(complete=True, owner=saved, worker=dict(PID=identity['worker_PID'], start_ticks=identity['worker_start_ticks']),
        saved_owner_and_worker_terminal=True, terminal=binding(root / 'TERMINAL.json'), work_receipt=binding(root / 'WORK_RECEIPT.json'),
        whole_Wiki_terminal_closure=binding(closure), source_manifest=pins['qualifier_manifest'])


def wiki_analysis(pins):
    spec = pins['Wiki_selected_analysis']; root = inside(spec['receipts']); saved = spec['owner']
    alive = live(saved['PID'], saved['start_ticks'])
    endpoints = [root / name for name in ('COMPLETE.json', 'FAILURE.json') if (root / name).is_file()]
    if pending([root / 'OWNER.json'], alive, 'Existing Wiki selected analysis'):
        return None
    if not endpoints:
        require(alive, 'Wiki selected owner ended without COMPLETE or FAILURE; no restart')
        return None
    require(len(endpoints) == 1, 'Contradictory Wiki selected analysis endpoints')
    owner = read(root / 'OWNER.json')
    require(owner['identity']['PID'] == saved['PID'] and owner['identity']['start_ticks'] == saved['start_ticks']
        and owner['source'] == pins['Wiki_owner_manifest'], 'Exact source-bound Wiki selected owner')
    endpoint = read(endpoints[0]); completed = endpoints[0].name == 'COMPLETE.json'
    require(endpoint.get('complete') is completed, 'Actual Wiki COMPLETE/FAILURE identity required')
    if alive:
        return None
    child_owner, child_terminal, release = (root / name for name in ('CHILD_OWNER.json', 'CHILD_TERMINAL.json', 'COLLECT_RELEASE.json'))
    child = None
    if child_owner.is_file() or child_terminal.is_file() or release.is_file() or completed:
        require(child_terminal.is_file(), 'Wiki child terminal custody missing; no restart')
        terminal = read(child_terminal)
        if terminal['child_started']:
            require(child_owner.is_file() and terminal['reaped'] is True and type(terminal['exit_code']) is int,
                'Any owned Wiki collection child must be actually reaped')
            child = terminal['identity']; recorded = read(child_owner)['identity']
            require(child == recorded and type(child['PID']) is int and child['PID'] > 0
                and type(child['start_ticks']) is int and child['start_ticks'] > 0, 'Exact saved Wiki child custody')
            require(not completed or (terminal['exit_code'] == 0 and not terminal['timed_out'] and not terminal['cap_exceeded']),
                'Wiki COMPLETE must agree with its actual child terminal')
            if live(child['PID'], child['start_ticks']):
                return None
        else:
            require(not child_owner.is_file() and terminal['identity'] is None and not completed, 'No invented absent Wiki child')
    return dict(status='complete' if completed else 'retained_failure', owner=saved, owner_terminal=True,
        owned_child=child, owned_child_reaped=child is not None, endpoint=binding(endpoints[0]),
        child_terminal=binding(child_terminal) if child_terminal.is_file() else None, source_manifest=pins['Wiki_owner_manifest'])


def gpu_observation():
    rows = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
    require(len(rows) == 1 and rows[0].split(',')[0].strip() == GPU, 'Sole authorized physical GPU required')
    free = int(rows[0].split(',')[1].strip()) * 1024**2
    return dict(UTC=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), physical_GPU_uuid=GPU,
        actual_free_GPU_bytes=free, minimum_free_GPU_bytes=MINIMUM_FREE, admitted=free >= MINIMUM_FREE,
        observer='nvidia-smi metadata only; no CUDA context')


def interrupted(signum, frame):
    raise InterruptedError('Owned waiting launcher received signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True); parser.add_argument('--approval-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    require(sys.platform == 'linux' and socket.gethostname() == 'anogena-2-0'
        and Path.cwd().resolve() == REPO.resolve(), 'Authorized normal Linux project host required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Waiting launcher must keep CUDA hidden')
    require(args.approval.resolve().is_relative_to(PHASE.resolve()) and sha(args.approval) == args.approval_sha256, 'Exact root approval inside project phase')
    approval = read(args.approval); pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(approval.get('schema') == 'internal-be-molhiv18-waiting-launcher-approval-v1' and approval.get('enabled') is True
        and approval.get('source_review_approved') is True and approval.get('root_wait_and_scientific_launch_authorized') is True
        and approval.get('TEST_access') is False and approval.get('automatic_retry') is False
        and approval['launcher_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
        and approval['executor_manifest'] == pins['executor_manifest'], 'Disabled pending exact root source review and conditional scientific authority')
    sealed(binding(HERE / 'MANIFEST.json'))
    for key in ('executor_manifest', 'qualifier_manifest', 'Wiki_owner_manifest', 'Wiki_collector_manifest'):
        sealed(pins[key])
    for key in ('qualifier_approval', 'Wiki_owner_approval'):
        bound(pins[key])
    output = inside(approval['waiting_directory']); execution = inside(approval['execution_directory'])
    require(not output.exists() and output.parent.is_dir() and not execution.exists() and execution.parent.is_dir() and execution != output
        and not output.is_relative_to(HERE), 'Fresh separate project waiting/executor directories required')
    output.mkdir(mode=0o700); started = time.monotonic(); exec_attempted = False
    write(output / 'OWNER.json', dict(identity=proc(os.getpid()), source_manifest=binding(HERE / 'MANIFEST.json'),
        root_approval=binding(args.approval), waiting_hard_seconds=WAIT_HARD, CUDA_VISIBLE_DEVICES='', numerical_imports=False,
        automatic_retry=False, other_jobs_modified=False, other_processes_signalled=False))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        while time.monotonic() - started < WAIT_HARD:
            qualified = qualifier(pins); analysis = wiki_analysis(pins)
            gpu = gpu_observation() if qualified is not None and analysis is not None else None
            write(output / 'WAIT.json', dict(elapsed_seconds=time.monotonic() - started,
                qualifier_ready=qualified is not None, Wiki_selected_analysis_terminal=analysis is not None,
                GPU=gpu, scientific_exec_attempted=False, quality_values_opened=False, TEST_access=False))
            if gpu is not None and gpu['admitted']:
                # Fresh engineering checks and physical memory observation before
                # the single exec; missing receipts never trigger a restart.
                qualified, analysis = qualifier(pins), wiki_analysis(pins)
                gpu = gpu_observation() if qualified is not None and analysis is not None else None
                if gpu is not None and gpu['admitted']:
                    break
            time.sleep(min(20, max(0, WAIT_HARD - (time.monotonic() - started))))
        else:
            raise TimeoutError('Single 24h waiting bound expired; no scientific start or retry')
        require(time.monotonic() - started < WAIT_HARD, 'Waiting hard cap expired before exec')
        sealed(pins['executor_manifest']); require(not execution.exists(), 'Executor output ceased to be fresh')
        write(output / 'ADMISSION.json', dict(qualifier=qualified, Wiki_selected_analysis=analysis, GPU=gpu,
            elapsed_wait_seconds=time.monotonic() - started, root_approval=binding(args.approval),
            executor_manifest=pins['executor_manifest'], quality_values_opened=False, TEST_access=False))
        executor_pins = read(inside(pins['executor_manifest']['path']).parent / 'SOURCE_BINDINGS.json')
        release = dict(schema='internal-be-molhiv18-family-release-v1', enabled=True, root_scientific_launch_authorized=True,
            source_review_approved=True, same_host_GPU_readiness_confirmed=True, GPU_memory_admission_confirmed=True,
            TEST_access=False, automatic_retry=False, executor_manifest_sha256=pins['executor_manifest']['sha256'],
            adoption=executor_pins['adoption'], budget_source=executor_pins['budget_source'], execution_directory=approval['execution_directory'],
            conditional_root_approval=binding(args.approval), engineering_admission=binding(output / 'ADMISSION.json'))
        release_path = output / 'EXECUTOR_RELEASE.json'; write(release_path, release)
        command = ['/usr/bin/python3', '-I', '-S', '-B', str(inside(pins['executor_manifest']['path']).parent / 'executor.py'),
            '--release', str(release_path), '--release-sha256', sha(release_path)]
        write(output / 'EXEC_INTENT.json', dict(command=command, exact_generated_release=binding(release_path),
            waiting_owner=proc(os.getpid()), attempt_once=True, intent_is_not_terminal_evidence=True,
            executor_confirmation_path=str(execution / 'PARENT_OWNER.json'), quality_values_opened=False))
        require(time.monotonic() - started < WAIT_HARD, 'Waiting hard cap expired before execve')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=GPU)
        env.pop('PYTHONHOME', None); exec_attempted = True
        os.execve('/usr/bin/python3', command, env)
    except BaseException as error:
        observed_receipts = []
        for spec, names in ((pins['qualifier'], ('TERMINAL.json', 'WORK_RECEIPT.json')),
                            (pins['Wiki_selected_analysis'], ('COMPLETE.json', 'FAILURE.json', 'CHILD_TERMINAL.json'))):
            for name in names:
                path = inside(spec['receipts']) / name
                if path.is_file():
                    observed_receipts.append(binding(path))
        write(output / 'FAILURE.json', dict(complete=False, error_type=type(error).__name__, error=str(error),
            elapsed_wait_seconds=time.monotonic() - started, waiting_hard_seconds=WAIT_HARD,
            wait_cap_exceeded=time.monotonic() - started > WAIT_HARD, scientific_exec_attempted=exec_attempted,
            observed_prerequisite_receipts=observed_receipts,
            automatic_retry=False, other_jobs_modified=False, other_processes_signalled=False,
            quality_values_opened=False, TEST_access=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
