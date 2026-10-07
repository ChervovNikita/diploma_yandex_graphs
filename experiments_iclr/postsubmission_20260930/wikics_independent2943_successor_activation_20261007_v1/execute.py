"""Disabled four fixed WikiCS successors; reuse reviewed GPU77 run_fit and helper."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def phase_file(relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Phase-relative file required')
    path = (PHASE / rel).resolve(strict=True)
    require(path.is_relative_to(PHASE.resolve()) and path.is_file(), 'Exact repository phase file required')
    return path


def bound(row):
    path = phase_file(row['path']); require(sha(path) == row['sha256'], 'Existing custody binding changed')
    return path


def binding(path):
    return dict(path=str(Path(path).relative_to(PHASE)), sha256=sha(path))


def module(name, row):
    path = bound(row); spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def physical(cfg):
    require(Path.cwd().resolve() == Path(cfg['repository']).resolve() and socket.gethostname() == cfg['hostname'], 'Exact authorized GPU77 repository/host')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    require(rows == cfg['physical_gpu_inventory'], 'Exact existing two-GPU inventory')


def dependency(lane, cfg, helper):
    terminal = read(bound(lane['prefix_terminal'])); bank_path = bound(lane['prefix_bank']); bank = read(bank_path)
    require(terminal['job_sha256'] == lane['prefix_job']['sha256'] and terminal['exit_code'] == 0
        and terminal['reason'] is None and not terminal['signals_sent'] and terminal['terminal_wait_observed'] is True,
        'Original clean prefix exit/reap custody required')
    bound(lane['prefix_job']); child = terminal['raw_identity_observation']; expected = lane['prefix_child']
    require(child['PID'] == expected['PID'] and child['start_ticks'] == expected['start_ticks'], 'Exact old prefix child')
    actual = helper.identity(child['PID'])
    require(actual is None or actual['start_ticks'] != child['start_ticks'], 'Original prefix child remains present')
    physical_path = phase_file(lane['prefix_physical_terminal_path']); evidence = read(physical_path)
    require(evidence['owned_PID_absent'] is True and evidence['owned_PID_no_CUDA_rows'] is True
        and evidence['terminal_wait_observed'] is True and evidence['child_identity']['PID'] == child['PID']
        and evidence['child_identity']['start_ticks'] == child['start_ticks'], 'Original physical terminal custody required')
    require(bank.get('complete') is True and bank['seed'] == lane['seed'] and bank['arm'] == 'DONOR_INDEPENDENT'
        and bank['members'] == lane['members'] and len(bank['members']) == 4, 'Reuse exact original four-member1100 bank')
    for member, row in enumerate(bank['members']):
        record = read(bound(row))
        require(record['complete'] is True and record['epochs'] == 1100 and record['seed'] == lane['seed'] + 1009 * member
            and record['source_manifest_sha256'] == cfg['source_manifest_sha256'] and record['program_sha256'] == cfg['program_sha256'],
            'Original independently acquired native1100 source/seed/header')
        bound(record['selected']); bound(record['end'])
    return dict(bank_freeze=lane['prefix_bank'], original_exit=lane['prefix_terminal'], physical_terminal=binding(physical_path),
        reused_by=['I_native', 'U_stage'], new_prefix_acquisitions=0, quality_values_read=False)


def completion(output, entry, cfg):
    path = output / 'FREEZE.json'; frozen = read(path)
    require(frozen.get('complete') is True and frozen['arm'] == entry['arm'] and frozen['seed'] == entry['seed']
        and frozen['source_manifest_sha256'] == cfg['source_manifest_sha256'] and frozen['program_sha256'] == cfg['program_sha256']
        and frozen['job_sha256'] == entry['job_sha256'] and frozen['TEST_access'] is False and frozen['retry'] is False,
        'Exact complete fixed endpoint custody')
    selection = frozen['selection']
    if entry['arm'] == 'I_native':
        require(selection['mode'] == 'native_first_max_spanning_source_stages_and_continuation'
            and selection['independent_optimizers'] is True and selection['pooled_training_or_selection'] is False
            and len(selection['members']) == 4
            and all(row['complete'] is True and row['epochs'] == 1200 and row['seed'] == entry['seed'] + 1009 * m
                    for m, row in enumerate(selection['members']))
            and len(frozen['native_costs']) == 4
            and all(row['new_native_epochs'] == 100 for row in frozen['native_costs']), 'Four original native1100+100 continuations required')
    else:
        require(selection['mode'] == 'fixed_final_endpoint' and selection['correction_path_updates'] == 400
            and selection['native_prefix_updates'] == 4400 and selection['cache_only_frozen_donors'] is True
            and selection['all_four_paths_paid'] is True, 'Four original fixed100 private paths required')
    return binding(path)


def lane_work(lane, cfg, output, owner):
    results = [dict(cell_id=row['cell_id'], seed=row['seed'], arm=row['arm'], status='unlaunched') for row in lane['entries']]
    started = time.monotonic(); fatal = None
    try:
        helper = module('existing_Wiki2943_helper_' + str(lane['seed']), cfg['existing_ownership_helper'])
        helper.GPU = lane['physical_gpu_uuid']; source = PHASE / cfg['source_directory']
        context = SimpleNamespace(REPO=Path(cfg['repository']), SOURCE=source, SOURCE_SHA=cfg['source_manifest_sha256'],
            GPU_UUID=lane['physical_gpu_uuid'], GPU_UUIDS=tuple(cfg['physical_gpu_inventory']), phase_file=phase_file,
            physical_host=lambda: physical(cfg), sha=sha, write=write)
        admitted = dependency(lane, cfg, helper); write(output / ('seed' + str(lane['seed']) + '_DEPENDENCY.json'), admitted)
        env = dict(os.environ, **cfg['environment'], CUDA_VISIBLE_DEVICES=lane['physical_gpu_uuid']); env.pop('PYTHONHOME', None)
        for descriptor, result in zip(lane['entries'], results):
            require(cfg['lane_wait_plus_work_bound_seconds'] - (time.monotonic() - started)
                >= descriptor['hard_seconds'] + cfg['resource_limits']['resource_wait_seconds'], 'Finite lane cannot admit another complete cell/wait')
            template = read(bound(descriptor['template'])); job = copy.deepcopy(template)
            job.update(root_execution_authorized=True, source_review_approved=True, fits_authorized=True, VALID_values_access=True)
            job_path = output / 'jobs' / (descriptor['cell_id'] + '.json'); write(job_path, job)
            entry = dict(descriptor, job_relative=str(job_path.relative_to(PHASE)), job_sha256=sha(job_path),
                output_directory=job['output_directory'], argv=[lane['python'], '-B', str(source / 'run.py'),
                    '--job', str(job_path), '--output', job['output_directory']])
            result.update(status='failed', activated_job=binding(job_path), output_directory=job['output_directory'])
            receipt = owner.run_fit(helper, output, entry, cfg, env, Path(job['output_directory']), context)
            child = receipt.get('raw_identity_observation'); absent = child is not None and helper.identity(child['PID']) is None
            rows = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], cfg['resource_limits']['telemetry_timeout_seconds'])
            no_cuda = child is not None and not any(len(parts) >= 2 and parts[1].strip() == str(child['PID']) for parts in (row.split(',') for row in rows))
            terminal_ready = type(receipt['exit_code']) is int and receipt['terminal_wait_observed'] is True and absent and no_cuda
            result.update(exit_code=receipt['exit_code'], terminal=binding(output / 'logs' / (entry['cell_id'] + '.EXIT.json')),
                physical_terminal=dict(child_identity=child, owned_PID_absent=absent, owned_PID_no_CUDA_rows=no_cuda,
                    actual_wait_observed=receipt['terminal_wait_observed']), inclusive_seconds=receipt['elapsed_seconds'])
            require(terminal_ready, 'Owned endpoint terminal custody unavailable; no next child')
            if receipt['exit_code'] == 0 and receipt['reason'] is None and not receipt['signals_sent'] and not (Path(job['output_directory']) / 'FAILURE.json').exists():
                try:
                    result['freeze'] = completion(Path(job['output_directory']), entry, cfg); result['status'] = 'complete'
                except Exception as error:
                    result.update(status='invalid', reason=type(error).__name__ + ': ' + str(error))
            else:
                result['reason'] = receipt['reason'] or 'Retained nonzero endpoint exit ' + str(receipt['exit_code'])
            failure = Path(job['output_directory']) / 'FAILURE.json'
            if failure.is_file(): result['failure'] = binding(failure)
            write(output / ('seed' + str(lane['seed']) + '_PROGRESS.json'), dict(results=results, quality_values_read=False))
    except Exception as error:
        fatal = type(error).__name__ + ': ' + str(error)
    for result in results:
        if result['status'] != 'complete' and not result.get('reason'):
            result['reason'] = fatal or 'Not reached before finite lane closure'
    return dict(seed=lane['seed'], results=results, fatal=fatal, inclusive_seconds=time.monotonic() - started,
        finite_lane_wait_plus_work_seconds=cfg['lane_wait_plus_work_bound_seconds'], quality_values_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True); parser.add_argument('--approval-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    require(sha(args.approval) == args.approval_sha256, 'Exact separate root approval bytes')
    approval = read(args.approval); cfg = read(HERE / 'CONFIG.json')
    require(approval.get('schema') == 'WikiCS_GPU77_four_successor_root_approval_v1' and approval.get('approved') is True
        and approval.get('root_execution_authorized') is True and approval.get('TEST_access') is False and approval.get('retry') is False
        and approval['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Disabled pending root inspection and launch authority')
    for row in read(HERE / 'MANIFEST.json')['files']:
        file = HERE / row['path']; require(sha(file) == row['sha256'] and file.stat().st_size == row['bytes'], 'Prepared activation seal changed')
    physical(cfg); source = PHASE / cfg['source_directory']
    require(sha(source / 'MANIFEST.json') == cfg['source_manifest_sha256'] and sha(source / 'run.py') == cfg['program_sha256'], 'Exact reviewed v4 science source')
    output = PHASE / cfg['execution_root']; require(not output.exists() and output.parent.is_dir(), 'Fresh output only; no retry')
    output.mkdir(); (output / 'logs').mkdir(); (output / 'jobs').mkdir(); (output / 'runs').mkdir()
    owner = module('existing_Wiki2943_GPU77_run_fit', cfg['existing_run_fit'])
    identity_helper = module('existing_Wiki2943_owner_identity', cfg['existing_ownership_helper'])
    for lane in cfg['lanes']:
        lane['python'] = read(bound(read(bound(lane['entries'][0]['template']))['execution_context']))['device_runtime']['python_executable']
    write(output / 'OWNER_STARTED.json', dict(identity=identity_helper.identity(os.getpid()), source_manifest_sha256=sha(HERE / 'MANIFEST.json'),
        exact_root_approval_sha256=args.approval_sha256, fixed_cells=[entry['cell_id'] for lane in cfg['lanes'] for entry in lane['entries']],
        new_prefix_acquisitions=0, quality_values_read=False, TEST_access=False, retry=False))
    closed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(lane_work, lane, cfg, output, owner): lane for lane in cfg['lanes']}
        for future in as_completed(futures):
            closed.append(future.result()); write(output / 'PROGRESS.json', dict(lanes=closed, quality_values_read=False))
    results = [result for lane in closed for result in lane['results']]
    complete = len(results) == 4 and all(row['status'] == 'complete' for row in results)
    write(output / ('COMPLETE.json' if complete else 'FAILURE.json'), dict(complete=complete, closed=True, results=results,
        lanes=closed, all_four_accounted=True, original17_endpoints_preserved=True, original1100_banks_reused=True,
        new_prefix_acquisitions=0, cross_runtime_limit_recorded=True, quality_values_read=False, TEST_access=False, retry=False))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
