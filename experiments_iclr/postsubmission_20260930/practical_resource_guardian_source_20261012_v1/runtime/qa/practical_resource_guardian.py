"""Finite stdlib guardian for the reviewed node-only zero-update resource roster."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

HELPER_SHA = 'effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0'
MANIFEST_SHA = 'a0bab2ca0188130a24aea84459fb3b119803bcde264ece51516c4fdb4546b497'
MASTER_HASH = '96acea7bdb8bb8be5f5928c348136f3a6e187bff528ad01e3245916768cf9611'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SMOKE = ['eee37c88d0999cd7af6ff49a', '60f08bd7435e48df6a41c859',
         '122eb30df760718c37bbac38', '00ea3ade59db62441a5b501a',
         'd599889325a25b5f805d63d4', '86b0ae48d487ef5fbace9f42']
PHYSICS = {'owner': {'pid': 640426, 'start_ticks': 6055039154, 'pgid': 640426},
           'child': {'pid': 640428, 'start_ticks': 6055039165, 'pgid': 640428}}
PHYSICS_DIRECTORY = 'private_GRU_acquisition_root_20261011_v1/owner_v1'
HELPER = 'practical_resource_qualifier_source_20261012_v2/runtime/qa/practical_zero_update_resources.py'
MANIFEST = 'practical_resource_qualifier_manifest_20261012_v2/MANIFEST.json'
SMOKE_DIRECTORY = 'practical_resource_qualifier_smoke_20261012_v2'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def local(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Project-relative authorities required')
    result = root/path
    if result.is_symlink() or not result.resolve().is_relative_to(root):
        raise ValueError('Authority cannot escape project or use symlinks')
    return result


def binding(root, path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': sha(raw)}


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def process(pid):
    path = Path('/proc')/str(pid)/'stat'
    if not path.exists():
        return None
    fields = path.read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'pgid': int(fields[2]), 'state': fields[0]}


def identity():
    value = process(os.getpid())
    value.update(hostname=socket.gethostname(), gpu_uuid=GPU)
    return value


def verify_allocation():
    if subprocess.check_output(['hostname'], text=True).strip() != 'anogena-2-0':
        raise ValueError('Allocation hostname mismatch')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != GPU:
        raise ValueError('Allocation sole GPU UUID mismatch')


def physics_custody(root, expected_start_binding):
    directory = local(root, PHYSICS_DIRECTORY)
    start_path = directory/'OWNER_START.json'
    if binding(root, start_path) != expected_start_binding:
        raise ValueError('Physics original ownership receipt changed')
    start = json.loads(start_path.read_text())
    if any(start.get(k) != v for k, v in PHYSICS.items()):
        raise ValueError('Physics ownership authorities changed')
    end_path = directory/'OWNER_END.json'
    if end_path.exists():
        # Root authorized this exact terminal receipt as closure regardless of
        # success/failure. Its scientific readout belongs to the existing watcher.
        json.loads(end_path.read_text())
        return {'mode': 'exact_retained_terminal_ownership_closure_only',
                'start_binding': expected_start_binding, 'end_binding': binding(root, end_path),
                'scientific_success_inferred': False}
    values = {}
    for name, expected in PHYSICS.items():
        current = process(expected['pid'])
        if current is None or current['state'] == 'Z' or any(current[k] != v for k, v in expected.items()):
            raise ValueError('Physics exact live identity absent/changed without authorized terminal closure')
        values[name] = current
    return {'mode': 'both_exact_processes_alive', 'start_binding': expected_start_binding, 'identities': values}


def validate_complete(result, case_id):
    if (result.get('status') != 'TRAIN_PROXY_COMPLETE' or result.get('case_id') != case_id
            or result.get('helper_sha256') != HELPER_SHA or result.get('master_hash') != MASTER_HASH):
        raise ValueError('Completed result source authority mismatch')
    for key in ['actual_optimizer_updates', 'actual_optimizer_kernel_calls', 'heldout_original_file_accesses', 'scores_emitted']:
        if result.get(key) != 0:
            raise ValueError('Completed result violates zero-update/no-role/no-score authority')
    for key in ['parameters', 'moments']:
        if result.get(key+'_sha256_before') is None or result[key+'_sha256_before'] != result.get(key+'_sha256_after'):
            raise ValueError('Completed parameter/moment equality proof absent')


def observation(root, case, output, result, exit_code, started, owner):
    fields = ['elapsed_wall_seconds', 'constructor_seconds', 'cold_train_seconds',
              'warm_train_median_seconds', 'TRAIN_only_full_graph_serving_proxy_seconds',
              'parameter_bytes', 'gradient_bytes', 'optimizer_moment_bytes', 'optimizer_objects',
              'CUDA_peak_allocated_bytes', 'CUDA_peak_reserved_bytes', 'process_peak_RSS_KiB',
              'failure_type', 'zero_update_equality_proof_complete']
    value = {k: result[k] for k in fields if k in result}
    value.update(case_id=case['case_id'], status=result['status'], child=owner,
                 process_exit_code=exit_code, guardian_child_wall_seconds=time.monotonic()-started,
                 output=str(output.relative_to(root)), actual_optimizer_updates=result.get('actual_optimizer_updates'),
                 actual_optimizer_kernel_calls=result.get('actual_optimizer_kernel_calls'),
                 original_heldout_accesses=result.get('heldout_original_file_accesses'),
                 scores_emitted=result.get('scores_emitted'), automatic_retry=False)
    if result['status'] == 'TRAIN_PROXY_COMPLETE':
        value.update(parameters_unchanged=True, moments_unchanged=True, zero_update_equality_proof_complete=True)
        value['serialization'] = result['serialization']
        value['allocator_cap'] = result['allocator_cap']
        value['GPU_usage_before_init'] = result['GPU_usage_before_init']
    value['result_binding'] = binding(root, output/('RESULT.json' if result['status'] == 'TRAIN_PROXY_COMPLETE' else 'TERMINAL_FAILURE.json'))
    return value


def run_case(root, case, output, custody, case_index):
    runtime = local(root, HELPER).parents[1]
    command = [str(root.parents[1]/'.venv/bin/python'), '-B', str(local(root, HELPER)),
               '--project-root', str(root), '--mode', 'execute-node-case', '--manifest', MANIFEST,
               '--case-id', case['case_id'], '--reviewed-helper-sha256', HELPER_SHA,
               '--warm-iterations', '3', '--case-timeout-seconds', '600', '--output', str(output.relative_to(root))]
    event = {'case_id': case['case_id'], 'case_index': case_index, 'command': command,
             'physics_custody_before': custody, 'time_unix': time.time(), 'automatic_retry': False}
    receipt_path = output.parent/(output.name+'_COMMAND_RECEIPT.json')
    dump(receipt_path, event)
    started = time.monotonic()
    child = subprocess.Popen(command, cwd=runtime, env=dict(os.environ, PYTHONPATH=str(runtime),
                               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0'),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    owner = process(child.pid)
    if owner is None:
        raise ValueError('Cannot capture fresh owned child identity')
    event['child'] = owner
    dump(receipt_path, event)
    guardian_timeout = False
    try:
        stdout, stderr = child.communicate(timeout=630)
    except subprocess.TimeoutExpired:
        guardian_timeout = True
        current = process(child.pid)
        if current and current['start_ticks'] == owner['start_ticks'] and current['pgid'] == owner['pgid']:
            os.killpg(owner['pgid'], signal.SIGTERM)
        try:
            stdout, stderr = child.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            current = process(child.pid)
            if current and current['start_ticks'] == owner['start_ticks'] and current['pgid'] == owner['pgid']:
                os.killpg(owner['pgid'], signal.SIGKILL)
            stdout, stderr = child.communicate(timeout=10)
    except BaseException:
        # An interruption of the guardian cannot leave its numerical child
        # running. Only its fresh PID/start/PGID authority may be signaled.
        current = process(child.pid)
        if current and current['start_ticks'] == owner['start_ticks'] and current['pgid'] == owner['pgid']:
            os.killpg(owner['pgid'], signal.SIGTERM)
        try:
            stdout, stderr = child.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            current = process(child.pid)
            if current and current['start_ticks'] == owner['start_ticks'] and current['pgid'] == owner['pgid']:
                os.killpg(owner['pgid'], signal.SIGKILL)
            stdout, stderr = child.communicate(timeout=10)
        dump(output.parent/(output.name+'_PROCESS_COMPLETION.json'),
             {'child': owner, 'exit_code': child.returncode, 'guardian_interrupted': True,
              'guardian_child_wall_seconds': time.monotonic()-started, 'stdout': stdout, 'stderr': stderr,
              'automatic_retry': False, 'completed_zero_update_proof_claimed': False})
        raise
    completion = {'child': owner, 'exit_code': child.returncode, 'guardian_timeout': guardian_timeout,
                  'guardian_child_wall_seconds': time.monotonic()-started, 'stdout': stdout, 'stderr': stderr,
                  'automatic_retry': False}
    dump(output.parent/(output.name+'_PROCESS_COMPLETION.json'), completion)
    result_path = output/'RESULT.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if child.returncode != 0:
            raise ValueError('Completed helper receipt with nonzero process exit')
        validate_complete(result, case['case_id'])
    else:
        failure_path = output/'TERMINAL_FAILURE.json'
        if failure_path.exists():
            result = json.loads(failure_path.read_text())
            if result.get('owner', {}).get('pid') != owner['pid'] or result['owner'].get('start_ticks') != owner['start_ticks']:
                raise ValueError('Terminal failure child ownership mismatch')
        else:
            output.mkdir(exist_ok=True)
            result = {'status': 'RESOURCE_DEFERRED' if guardian_timeout else 'ENGINEERING_FAILURE',
                      'failure_type': 'OwnedGuardianTimeoutBeforeRESULT' if guardian_timeout else 'OwnedChildExitedBeforeRESULT',
                      'owner': owner, 'zero_update_equality_proof_complete': False,
                      'actual_optimizer_update_count_verified': None, 'actual_optimizer_updates': None,
                      'elapsed_wall_seconds': time.monotonic()-started, 'scientific_fits': 0,
                      'scores_emitted': 0, 'automatic_retry': False}
            dump(failure_path, result)
    return observation(root, case, output, result, child.returncode, started, owner)


def coverage(output, manifest, observed):
    fields = ['case_id', 'study_id', 'source_spec_hash', 'selection_run_id', 'source_search_run_id',
              'source_input_cohort_run_id', 'method', 'adapter', 'body', 'seed', 'mask', 'TRAIN_rows',
              'shape', 'status', 'CUDA_peak_allocated_bytes', 'CUDA_peak_reserved_bytes',
              'constructor_seconds', 'cold_train_seconds', 'warm_train_median_seconds',
              'serialization_bytes', 'result_path']
    with (output/'ALL_ROSTER_COVERAGE.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields);writer.writeheader()
        for case in manifest['cases']:
            value = observed.get(case['case_id'], {})
            run = case['run']
            row = {'case_id': case['case_id'], 'study_id': case['study_id'], 'source_spec_hash': case['original_spec_hash'],
                   'selection_run_id': case['selection_run_id'], 'source_search_run_id': case['source_search_run_id'],
                   'source_input_cohort_run_id': case['source_input_cohort_run_id'], 'method': run['method'],
                   'adapter': run['adapter'], 'body': run['body'], 'seed': run['seed'], 'mask': case['representative_mask'],
                   'TRAIN_rows': case['actual_TRAIN_rows'], 'shape': canonical(case['representative_dimensions']),
                   'status': value.get('status', 'UNEXECUTED'), 'serialization_bytes': value.get('serialization', {}).get('bytes'),
                   'result_path': value.get('result_binding', {}).get('path')}
            row.update({k: value[k] for k in fields if k in value})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--roster', required=True)
    parser.add_argument('--expected-roster-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify_allocation()  # Before any project read/write.
    root = Path(args.project_root).resolve()
    output = local(root, args.output);output.mkdir(exist_ok=False)
    owner = identity();started = time.monotonic();observed = {}
    dump(output/'OWNER_START.json', {'owner': owner, 'time_unix': time.time(), 'helper_sha256': HELPER_SHA,
                                    'guardian_sha256': sha(Path(__file__).read_bytes()), 'automatic_retry': False})
    manifest = None
    progress = {'status': 'PREPARING', 'owner': owner, 'helper_sha256': HELPER_SHA,
                'master_hash': MASTER_HASH, 'manifest_sha256': MANIFEST_SHA, 'remaining_roster_count': 248,
                'new_attempted': 0, 'new_completed': 0, 'new_deferred': 0, 'new_failed': 0,
                'scientific_fits': 0, 'scientific_launch_enabled': False, 'automatic_retry': False,
                'all_fit_ETA': None, 'whole_master_resource_admission': False}
    def interrupted(signum, frame):
        raise InterruptedError('Finite guardian termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        helper = local(root, HELPER)
        if sha(helper.read_bytes()) != HELPER_SHA:
            raise ValueError('Exact reviewed helper source changed')
        manifest_path = local(root, MANIFEST)
        if sha(manifest_path.read_bytes()) != MANIFEST_SHA:
            raise ValueError('Exact reviewed node manifest changed')
        manifest = json.loads(manifest_path.read_text())
        if manifest['helper_sha256'] != HELPER_SHA or manifest['master_hash'] != MASTER_HASH or len(manifest['cases']) != 254:
            raise ValueError('Reviewed manifest scope differs')
        roster_path = local(root, args.roster)
        if sha(roster_path.read_bytes()) != args.expected_roster_sha256:
            raise ValueError('Reviewed finite roster changed')
        roster = json.loads(roster_path.read_text())
        expected = [c['case_id'] for c in manifest['cases'] if c['case_id'] not in SMOKE]
        if (roster['remaining_case_ids'] != expected or len(expected) != 248
                or roster['helper_sha256'] != HELPER_SHA or roster['manifest_sha256'] != MANIFEST_SHA
                or roster['master_hash'] != MASTER_HASH):
            raise ValueError('Finite roster is not exactly the remaining reviewed cases')
        progress.update(roster_binding=binding(root, roster_path), manifest_binding=binding(root, manifest_path),
                        helper_binding=binding(root, helper), guardian_binding=binding(root, Path(__file__)))
        start_binding = binding(root, local(root, PHYSICS_DIRECTORY)/'OWNER_START.json')
        physics_custody(root, start_binding)
        for index, identifier in enumerate(SMOKE, 1):
            case = next(c for c in manifest['cases'] if c['case_id'] == identifier)
            directory = local(root, SMOKE_DIRECTORY)/('case_'+str(index).zfill(2)+'_'+identifier)
            result = json.loads((directory/'RESULT.json').read_text());validate_complete(result, identifier)
            value = {k: v for k, v in result.items() if k not in ['parameters_sha256_before', 'parameters_sha256_after',
                       'moments_sha256_before', 'moments_sha256_after', 'tensor_load_paths']}
            value['result_binding'] = binding(root, directory/'RESULT.json');observed[identifier] = value
        dump(output/'QUEUE_PROGRESS.json', progress)
        cases = {c['case_id']: c for c in manifest['cases']}
        for index, identifier in enumerate(expected, 1):
            verify_allocation()
            if sha(helper.read_bytes()) != HELPER_SHA:
                raise ValueError('Reviewed helper changed during queue')
            custody = physics_custody(root, start_binding)
            directory = output/('case_'+str(index).zfill(3)+'_'+identifier)
            progress.update(status='RUNNING', current_case_id=identifier, current_case_index=index,
                            new_attempted=index, physics_custody_before=custody,
                            elapsed_wall_seconds=time.monotonic()-started)
            dump(output/'QUEUE_PROGRESS.json', progress)
            value = run_case(root, cases[identifier], directory, custody, index)
            observed[identifier] = value
            dump(output/'OBSERVATIONS.json', observed)
            status = value['status']
            if status == 'TRAIN_PROXY_COMPLETE':
                progress['new_completed'] += 1
            elif status == 'RESOURCE_DEFERRED':
                progress['new_deferred'] += 1
            else:
                progress['new_failed'] += 1
                progress.update(status='STOPPED_ENGINEERING_FAILURE', stopped_case_id=identifier)
            progress.update(elapsed_wall_seconds=time.monotonic()-started, last_terminal_case_id=identifier,
                            last_terminal_status=status)
            dump(output/'QUEUE_PROGRESS.json', progress)
            if status not in {'TRAIN_PROXY_COMPLETE', 'RESOURCE_DEFERRED'}:
                break
        else:
            progress['status'] = 'FINITE_QUEUE_CLOSED'
    except BaseException as error:
        progress.update(status='STOPPED_INTERFACE_ROLE_OR_AUTHORITY_FAILURE', failure_type=type(error).__name__,
                        failure_message=str(error), elapsed_wall_seconds=time.monotonic()-started,
                        zero_update_proof_for_incomplete_cases_claimed=False)
    finally:
        progress.update(elapsed_wall_seconds=time.monotonic()-started, terminal_time_unix=time.time(),
                        total_completed=len([v for v in observed.values() if v['status'] == 'TRAIN_PROXY_COMPLETE']),
                        total_deferred=len([v for v in observed.values() if v['status'] == 'RESOURCE_DEFERRED']),
                        remaining_unexecuted=254-len(observed))
        dump(output/'QUEUE_PROGRESS.json', progress)
        dump(output/'OBSERVATIONS.json', observed)
        if manifest is not None:
            coverage(output, manifest, observed)
        dump(output/'OWNER_END.json', {'owner': owner, 'progress': progress, 'direct_children_waited': True,
                                      'scientific_success_claimed': False, 'automatic_retry': False})
        print(canonical({k: progress[k] for k in ['status', 'new_attempted', 'new_completed', 'new_deferred', 'new_failed',
                                                'total_completed', 'total_deferred', 'remaining_unexecuted', 'elapsed_wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
