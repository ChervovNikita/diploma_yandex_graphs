#!/usr/bin/env python3
"""One detached fixed seed block; unchanged reviewed run_fit and ownership helper."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

ORDER = ('random_signs_all', 'tabm_first_normal', 'warm_identity', 'graph_covariance', 'feature_covariance', 'single', 'independent_warm4')
SOURCE_NAME = 'citeseer_gnnm_initialization_pilot_preparation_20261006_v1'
PROGRAM_SHA = 'b319ab6b2d523aeee60858c16fe1b5d97a71861fd135fc5cc013526d630d071b'
INITIALIZER_SHA = '64c3a064336d523e1dabd968e83a4938c0e5cfaf263cb4e3b6536ceb36adc7bb'
LOOP_REL = 'citeseer_known_ranking_control_gpu77_b0_owned_preparation_20261006_v1/owner.py'
LOOP_SHA = 'a8d36b95fd7faa767f44b3813c6e4f484d97c4dec72cc1c141cf4139ce4e462d'
HELPER_REL = 'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py'
HELPER_SHA = 'e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'
PROVIDERS = {
    'allocation': ('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs', 'anogena-2-0', ('GPU-44039938-fd82-41d2-fefd-de71514e2fac',)),
    'gpu77': ('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git', 'peptide', ('GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced')),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(name, path, expected):
    if sha(path) != expected: raise ValueError('Reviewed dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bound_file(phase, record):
    relative = Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Require phase-relative binding')
    path = (phase / relative).resolve(strict=True)
    if not path.is_relative_to(phase.resolve()) or not path.is_file() or sha(path) != record['sha256']:
        raise ValueError('Exact phase file binding changed: ' + str(relative))
    return path


def absent(helper, pid):
    return helper.identity(pid) is None


def terminal(helper, receipt, expected_job_sha, expected_argv):
    child = receipt.get('child_identity')
    if (receipt.get('exit_code') != 0 or receipt.get('reason') is not None or receipt.get('signals_sent')
        or receipt.get('terminal_wait_observed') is not True or receipt.get('job_sha256') != expected_job_sha
        or child is None or child['argv'] != expected_argv or not absent(helper, child['PID'])):
        raise ValueError('Require exact clean terminal and absent owned child; no retry')
    rows = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], 10)
    if any(len(parts) >= 2 and parts[1].strip() == str(child['PID']) for parts in (row.split(',') for row in rows)):
        raise ValueError('Owned terminal child remains in physical CUDA inventory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256: raise ValueError('Exact root config changed')
    cfg = read(args.config)
    if cfg.get('root_execution_authorized') is not True or cfg.get('TEST_access') is not False or cfg.get('retry') is not False:
        raise ValueError('Root release required; TEST closed and no retry')
    if sha(__file__) != cfg['adapter_sha256']: raise ValueError('Bound adapter source changed')
    repository, hostname, inventory = PROVIDERS[cfg['provider']]
    repo = Path(repository); phase = repo / 'experiments_iclr/postsubmission_20260930'
    gpu = cfg['physical_gpu_uuid']
    if gpu not in inventory or cfg['repository'] != repository: raise ValueError('Exact supported provider required')

    def physical():
        if Path.cwd().resolve() != repo.resolve() or socket.gethostname() != hostname: raise ValueError('Authorized host/cwd changed')
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10)
        if tuple(rows.split()) != inventory: raise ValueError('Physical GPU inventory changed')

    def phase_file(relative):
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts: raise ValueError('Require phase-relative file')
        value = (phase / p).resolve(strict=True)
        if not value.is_relative_to(phase.resolve()) or not value.is_file(): raise ValueError('File leaves phase')
        return value

    physical()
    source = phase / SOURCE_NAME
    if sha(source / 'run.py') != PROGRAM_SHA or sha(source / 'initialize.py') != INITIALIZER_SHA:
        raise ValueError('Original sealed numerical source changed')
    plan_path = bound_file(phase, cfg['plan']); plan = read(plan_path)
    if (plan.get('root_adopted') is not True or plan.get('TEST_closed') is not True
        or tuple(plan['conditions']) != ORDER or (plan['warm_cycles'], plan['post_cycles'], plan['eval_every']) != (20, 60, 5)):
        raise ValueError('Require adopted fixed initialization plan')
    block = next(row for row in plan['blocks'] if row['block'] == cfg['block'])
    owner = load('reviewed_initialization_run_loop', phase_file(LOOP_REL), LOOP_SHA)
    helper = load('initialization_owned_helper', phase_file(HELPER_REL), HELPER_SHA); helper.GPU = gpu
    limits = cfg['resource_limits']; bounds = plan['bounds']
    expected_limits = {'combined_child_log_cap_bytes': bounds['log_per_child_bytes'],
        'minimum_fresh_GPU_free_bytes': bounds['minimum_free_GPU_bytes'], 'own_fit_output_cap_bytes': bounds['output_per_child_bytes'],
        'owned_tree_GPU_memory_cap_bytes': bounds['GPU_bytes'], 'owned_tree_RSS_cap_bytes': bounds['RSS_bytes'],
        'poll_interval_seconds': 15, 'resource_wait_seconds': 3600, 'telemetry_timeout_seconds': 10}
    if limits != expected_limits: raise ValueError('Fixed resource ceilings changed')
    relative_root = Path(cfg['execution_directory_relative'])
    if relative_root.is_absolute() or '..' in relative_root.parts: raise ValueError('Require phase-relative output root')
    root = (phase / relative_root).resolve(strict=True)
    if not root.is_relative_to(phase.resolve()): raise ValueError('Output root leaves phase')
    work = root / (cfg['block'] + '_fit_owner')
    work.mkdir(); (work / 'logs').mkdir()
    ident = helper.identity(os.getpid())
    if ident is None or ident['sid'] != ident['pgid'] or ident['sid'] != os.getpid(): raise ValueError('Detached owner session required')
    write(work / 'OWNER_STARTED.json', {'UTC': helper.now(), 'identity': ident, 'config_sha256': args.config_sha256,
        'plan_sha256': cfg['plan']['sha256'], 'fixed_conditions': list(ORDER), 'TEST_access': False, 'scores_read': False})
    completed = []; started = time.monotonic()
    try:
        warm_job_path = bound_file(phase, cfg['warm_job']); warm_job = read(warm_job_path)
        if (warm_job['phase'] != 'warm' or warm_job['seed'] != block['seed'] or warm_job['factor_seed'] != block['factor_seed']
            or warm_job['plan_sha256'] != cfg['plan']['sha256'] or warm_job['TEST_access'] is not False
            or warm_job['repository'] != repository or warm_job['physical_gpu_uuid'] != gpu): raise ValueError('Same provider/seed warm binding required')
        warm_cfg = read(bound_file(phase, cfg['warm_owner_config']))
        if warm_cfg['job_sha256'] != cfg['warm_job']['sha256']: raise ValueError('Warm owner/job binding differs')
        warm_owner = phase / cfg['warm_owner_directory_relative']
        warm_output = Path(warm_job['output_directory']).resolve()
        if not warm_output.is_relative_to(root) or cfg['warm_wait_seconds'] != bounds['warm']['hard_seconds']:
            raise ValueError('Exact fixed warm output/wait bound required')
        wait_started = time.monotonic()
        while True:
            physical()
            receipt_path = warm_owner / 'TERMINAL.json'; complete_path = warm_owner / 'WARM_COMPLETE.json'
            if receipt_path.is_file():
                receipt = read(receipt_path)
                if receipt.get('exit_code') != 0 or receipt.get('reason') is not None or receipt.get('signals_sent'):
                    raise ValueError('Warm operational failure preserved; no postfit or retry')
                if complete_path.is_file() and absent(helper, cfg['warm_owner_PID']): break
            if time.monotonic() - wait_started >= cfg['warm_wait_seconds']: raise TimeoutError('Fixed warm completion wait exhausted')
            write(work / 'QUEUE_PROGRESS.json', {'UTC': helper.now(), 'phase': 'awaiting_warm_terminal', 'completed': 0, 'total': 7, 'scores_read': False})
            time.sleep(15)
        observed_owner = read(warm_owner / 'OWNER_STARTED.json')['identity']
        if (observed_owner['PID'], observed_owner['start_ticks']) != (cfg['warm_owner_PID'], cfg['warm_owner_start_ticks']):
            raise ValueError('Exact launched warm owner identity differs')
        terminal(helper, receipt, cfg['warm_job']['sha256'], warm_cfg['argv'])
        warm_freeze_path = warm_output / 'FREEZE.json'; frozen = read(warm_freeze_path)
        if (frozen['phase'], frozen['completed_cycles'], frozen['seed']) != ('warm', 20, block['seed']) or frozen['VALID_TEST_access'] is not False:
            raise ValueError('Complete TRAIN-only warm state required')
        if frozen['job_sha256'] != cfg['warm_job']['sha256'] or frozen['plan_sha256'] != cfg['plan']['sha256']:
            raise ValueError('Warm freeze source/job/plan differs')
        if (warm_output / 'FAILURE.json').exists() or sha(warm_output / 'warm_checkpoint.pt') != frozen['checkpoint_sha256']:
            raise ValueError('Warm checkpoint custody failure')
        complete = read(complete_path)
        if complete['freeze_sha256'] != sha(warm_freeze_path) or complete['child_absent'] is not True:
            raise ValueError('Warm completion receipt differs')
        warm_binding = {'path': str(warm_freeze_path.relative_to(phase)), 'sha256': sha(warm_freeze_path)}
        entries = []
        if len(cfg['job_templates']) != 7: raise ValueError('Exactly seven fixed job templates required')
        for condition, binding in zip(ORDER, cfg['job_templates']):
            template = read(bound_file(phase, binding)); job = copy.deepcopy(template)
            if (job['condition'] != condition or job['phase'] != 'fit' or job['seed'] != block['seed']
                or job['factor_seed'] != block['factor_seed'] or job['repository'] != repository or job['physical_gpu_uuid'] != gpu
                or job['program_sha256'] != PROGRAM_SHA or job['initialization_sha256'] != INITIALIZER_SHA
                or job['TEST_access'] is not False or job['retry'] is not False): raise ValueError('Fixed source/seed/condition template differs')
            family = condition if condition in ('single', 'independent_warm4') else 'shared_F4'
            if job['soft_seconds'] != bounds[family]['soft_seconds']: raise ValueError('Fixed fit soft ceiling changed')
            job.update(plan_relative=cfg['plan']['path'], plan_sha256=cfg['plan']['sha256'], warm_freeze_relative=warm_binding['path'],
                warm_freeze_sha256=warm_binding['sha256'], fits_authorized=True, VALID_values_access=True,
                source_review_approved=True, source_review=copy.deepcopy(warm_job['source_review']), external_hard_bound_confirmed=True)
            for key in ('source_manifest_sha256', 'runtime_versions', 'runtime_qualification', 'feature_authority', 'negative_pool_authority'):
                if job[key] != warm_job[key]: raise ValueError('Fit/warm admitted source/runtime/input authority differs')
            path = root / 'jobs' / (cfg['block'] + '_' + condition + '.json')
            output = root / 'runs' / path.stem
            if job['output_directory'] != str(output) or path.exists() or output.exists(): raise ValueError('Fresh exact fixed fit roles required')
            with path.open('x') as stream: stream.write(json.dumps(job, indent=2, sort_keys=True, allow_nan=False) + '\n')
            path.chmod(0o444)
            entries.append({'cell_id': path.stem, 'job_relative': str(path.relative_to(phase)), 'job_sha256': sha(path),
                'hard_seconds': bounds[family]['hard_seconds'], 'argv': [warm_cfg['argv'][0], '-B', str(source / 'run.py'), '--job', str(path), '--output', str(output)]})
        environment = dict(os.environ, **warm_cfg['environment']); environment.pop('PYTHONHOME', None)
        if warm_cfg['environment']['CUDA_VISIBLE_DEVICES'] != gpu: raise ValueError('Warm/fit visible GPU differs')
        context = SimpleNamespace(REPO=repo, SOURCE=source, SOURCE_SHA=PROGRAM_SHA, GPU_UUID=gpu, GPU_UUIDS=inventory,
            physical_host=physical, phase_file=phase_file, sha=sha, write=write)
        write(work / 'BOUND_JOBS.json', {'plan': cfg['plan'], 'warm_freeze': warm_binding, 'entries': entries, 'scores_read': False})
        resource_call = helper.resources; cost = {}; peak = {'RSS_bytes': 0, 'GPU_bytes': 0}

        def measured_resources(rows, resource_limits, remaining):
            rss, gpu_bytes = resource_call(rows, resource_limits, remaining)
            peak['RSS_bytes'] = max(peak['RSS_bytes'], rss); peak['GPU_bytes'] = max(peak['GPU_bytes'], gpu_bytes)
            history = Path(entries[0]['argv'][6]) / 'HISTORY.jsonl'
            if not cost and history.is_file():
                with history.open() as stream: line = stream.readline(4097)
                if len(line) > 4096: raise ValueError('Unexpected first-cycle cost record size')
                if line.endswith('\n'):
                    row = json.loads(line)
                    if row['cycle'] != 1 or 'complete_VALID' in row or 'members' in row or row['counters'] != {'episodes': 61, 'native_Adam_updates': 183}:
                        raise ValueError('First complete TRAIN cycle required; no quality read')
                    seconds = row['cycle_seconds']
                    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0: raise ValueError('Finite positive complete cycle cost required')
                    admitted = (60 * seconds <= bounds['shared_F4']['soft_seconds'] and peak['RSS_bytes'] <= bounds['RSS_bytes'] and peak['GPU_bytes'] <= bounds['GPU_bytes'])
                    cost.update(UTC=helper.now(), first_complete_cycle_seconds=seconds, projected_shared_F4_60_cycle_seconds=60 * seconds,
                        fixed_shared_F4_soft_seconds=bounds['shared_F4']['soft_seconds'], sampled_peak_owned=copy.deepcopy(peak),
                        admitted=admitted, scores_read=False, control_runtime_guarantee=False,
                        scope='Observed shared-F4 full TRAIN cycle only; each fixed condition retains its own enforced complete-fit ceilings.')
                    write(work / 'FIRST_CYCLE_COST_ADMISSION.json', cost)
                    if not admitted: raise TimeoutError('First actual complete cycle exceeds predeclared shared-F4 resource ceiling; preserve and stop whole block')
            return rss, gpu_bytes

        for index, entry in enumerate(entries):
            physical()
            if sha(phase_file(entry['job_relative'])) != entry['job_sha256'] or sha(plan_path) != cfg['plan']['sha256']:
                raise ValueError('Activated exact job/plan changed')
            if sha(source / 'run.py') != PROGRAM_SHA or sha(source / 'initialize.py') != INITIALIZER_SHA: raise ValueError('Numerical source changed')
            output = Path(entry['argv'][6])
            if output.exists(): raise ValueError('No overwrite or retry')
            helper.resources = measured_resources if index == 0 else resource_call
            receipt = owner.run_fit(helper, work, entry, {'resource_limits': limits}, environment, output, context)
            terminal(helper, receipt, entry['job_sha256'], entry['argv'])
            if index == 0 and cost.get('admitted') is not True: raise ValueError('No first-cycle resource admission')
            freeze_path = output / 'FREEZE.json'; frozen = read(freeze_path)
            if (frozen['phase'], frozen['completed_cycles'], frozen['seed'], frozen['condition']) != ('fit', 60, block['seed'], ORDER[index]):
                raise ValueError('Complete fixed sixty-cycle fit required')
            if frozen['job_sha256'] != entry['job_sha256'] or frozen['plan_sha256'] != cfg['plan']['sha256'] or frozen['TEST_access'] is not False or (output / 'FAILURE.json').exists():
                raise ValueError('Fit custody/completion failure')
            completed.append({**receipt, 'freeze_relative': str(freeze_path.relative_to(phase)), 'freeze_sha256': sha(freeze_path), 'owned_PID_absent': True})
            write(work / 'QUEUE_PROGRESS.json', {'UTC': helper.now(), 'phase': 'postfit', 'completed': len(completed), 'total': 7, 'last_cell': entry['cell_id'], 'scores_read': False})
            print('Completed operationally: ' + entry['cell_id'], flush=True)
        write(work / 'BLOCK_FREEZE.json', {'UTC': helper.now(), 'block': cfg['block'], 'completed': completed,
            'selected_block_complete': True, 'physical_postfits': 7, 'plan': cfg['plan'], 'warm_freeze': warm_binding,
            'config_sha256': args.config_sha256, 'inclusive_seconds': time.monotonic() - started, 'TEST_access': False, 'scores_read': False, 'retry': False})
    except BaseException as error:
        write(work / 'QUEUE_FAILURE.json', {'UTC': helper.now(), 'error': type(error).__name__ + ': ' + str(error),
            'completed': completed, 'inclusive_seconds': time.monotonic() - started, 'partial_outputs_preserved': True, 'retry': False, 'scores_read': False})
        raise


if __name__ == '__main__': main()
