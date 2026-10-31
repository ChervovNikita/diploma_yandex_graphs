#!/usr/bin/env python3
"""Fixed supplemental native-head tail after complete initializer seed closure."""
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

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCK_SOURCE = PHASE / 'citeseer_gnnm_initialization_pilot_block_queue_preparation_20261006_v1/block_owner.py'
BLOCK_SHA = 'fd86370c5b51c79e7ac4153936a462e5685e131f5624e1791bcb36df3af5f3b3'
SOURCE = PHASE / 'citeseer_shared_encoder_full_native_heads_supplement_preparation_20261006_v1'
SOURCE_SHA = '0e5123a48b657491f36cd174010f9c810edbac75924b1420b28f9bad94ddc814'
PROGRAM_SHA = '30f34747a5a8d300c940ffbe91db69139e98e72668aceb632df7d571cf68f728'
INITIALIZER_SHA = '24413482ce7f9d0a231ed6648546ca55b8d1038e3169460e31d74326a595d3b8'


def physical():
    if Path.cwd() != REPO or socket.gethostname() != 'anogena-2-0': raise ValueError('Exact allocation host/cwd required')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != [GPU]:
        raise ValueError('Exact single allocation GPU required')


def phase_file(relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts: raise ValueError('Phase-relative path required')
    path = (PHASE / value).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file(): raise ValueError('File leaves authorized phase')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(); physical()
    if hashlib.sha256(BLOCK_SOURCE.read_bytes()).hexdigest() != BLOCK_SHA: raise ValueError('Reviewed owner primitives changed')
    spec = importlib.util.spec_from_file_location('supplement_tail_block_utils', BLOCK_SOURCE)
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    if b.sha(args.config) != args.config_sha256: raise ValueError('Exact root tail release changed')
    cfg = b.read(args.config)
    if (cfg.get('root_execution_authorized') is not True or cfg.get('TEST_access') is not False or cfg.get('retry') is not False
        or cfg.get('adopt_fixed_supplemental_plan') is not True or b.sha(__file__) != cfg['adapter_sha256']):
        raise ValueError('Root must release this exact fixed supplemental tail')
    initial_plan_path = b.bound_file(PHASE, cfg['initialization_plan']); initial_plan = b.read(initial_plan_path)
    serial_cfg_path = b.bound_file(PHASE, cfg['serial_config']); serial_cfg = b.read(serial_cfg_path)
    review = b.read(b.bound_file(PHASE, cfg['source_review']))
    if review['approved'] is not True or review['files']['MANIFEST.json'] != SOURCE_SHA: raise ValueError('Exact source review required')

    def verify_source():
        if b.sha(SOURCE / 'MANIFEST.json') != SOURCE_SHA or b.sha(SOURCE / 'run.py') != PROGRAM_SHA or b.sha(SOURCE / 'initialize.py') != INITIALIZER_SHA:
            raise ValueError('Reviewed supplemental source changed')
        for row in b.read(SOURCE / 'MANIFEST.json')['files']:
            path = (SOURCE / row['path']).resolve(strict=True)
            if not path.is_relative_to(SOURCE.resolve()) or path.stat().st_size != row['bytes'] or b.sha(path) != row['sha256']:
                raise ValueError('Supplemental packet closure changed')

    verify_source(); plan = b.read(b.bound_file(PHASE, cfg['plan_template'])); bounds = plan['bounds']
    if (plan['conditions'] != ['shared_full_heads'] or [row['block'] for row in plan['blocks']] != ['b0', 'b1', 'b2']
        or (plan['warm_cycles'], plan['post_cycles'], plan['eval_every']) != (20, 60, 5)
        or (bounds['soft_seconds'], bounds['hard_seconds']) != (15000, 18000) or plan['TEST_closed'] is not True):
        raise ValueError('Exact separate fixed supplemental plan required')
    expected_limits = {'combined_child_log_cap_bytes': bounds['log_per_child_bytes'], 'minimum_fresh_GPU_free_bytes': bounds['minimum_free_GPU_bytes'],
        'own_fit_output_cap_bytes': bounds['output_per_child_bytes'], 'owned_tree_GPU_memory_cap_bytes': bounds['GPU_bytes'],
        'owned_tree_RSS_cap_bytes': bounds['RSS_bytes'], 'poll_interval_seconds': 15, 'resource_wait_seconds': 3600, 'telemetry_timeout_seconds': 10}
    if cfg['resource_limits'] != expected_limits: raise ValueError('Existing fixed resource ceilings required')
    previous_bounds = initial_plan['bounds']
    block_hard = sum(previous_bounds[c if c in ('single', 'independent_warm4') else 'shared_F4']['hard_seconds'] for c in b.ORDER) + 7 * expected_limits['resource_wait_seconds']
    wait_bound = 3 * block_hard + 2 * (previous_bounds['warm']['hard_seconds'] + expected_limits['resource_wait_seconds'])
    if cfg['serial_wait_seconds'] != wait_bound: raise ValueError('Original plan-derived finite predecessor wait required')
    root = (PHASE / cfg['execution_directory_relative']).resolve(strict=True)
    initial_root = (PHASE / serial_cfg['execution_directory_relative']).resolve(strict=True)
    if not root.is_relative_to(PHASE.resolve()) or root == initial_root: raise ValueError('Separate supplemental output root required')
    o = b.load('reviewed_tail_run_loop', PHASE / b.LOOP_REL, b.LOOP_SHA)
    h = b.load('reviewed_tail_helper', PHASE / b.HELPER_REL, b.HELPER_SHA); h.GPU = GPU
    ident = h.identity(os.getpid())
    if ident is None or ident['sid'] != ident['pgid'] or ident['sid'] != os.getpid(): raise ValueError('Detached tail owner session required')
    work = root / 'tail_owner'; work.mkdir(); (work / 'logs').mkdir()
    b.write(work / 'OWNER_STARTED.json', {'UTC': h.now(), 'identity': ident, 'config_sha256': args.config_sha256,
        'fixed_order': ['b0_shared_full_heads', 'b1_shared_full_heads', 'b2_shared_full_heads'], 'TEST_access': False, 'scores_read': False})
    context = SimpleNamespace(REPO=REPO, SOURCE=SOURCE, SOURCE_SHA=SOURCE_SHA, GPU_UUID=GPU, GPU_UUIDS=(GPU,),
        physical_host=physical, phase_file=phase_file, sha=b.sha, write=b.write)
    completed = []; started = time.monotonic()
    try:
        serial_work = initial_root / 'serial_b1_b2_owner'
        while True:
            physical()
            if (serial_work / 'FAILURE.json').exists(): raise ValueError('Initializer serial chain failed; no supplemental fits or retry')
            if any((initial_root / (block + '_fit_owner') / 'QUEUE_FAILURE.json').exists() for block in ('b0', 'b1', 'b2')):
                raise ValueError('Initializer block failed; no supplemental fits or retry')
            if (serial_work / 'SERIAL_COMPLETE.json').is_file() and h.identity(cfg['serial_owner_PID']) is None: break
            if time.monotonic() - started >= wait_bound: raise TimeoutError('Fixed predecessor completion wait exhausted')
            b.write(work / 'PROGRESS.json', {'UTC': h.now(), 'phase': 'awaiting_all_initializer_blocks_and_serial_terminal', 'completed': 0, 'total': 3, 'scores_read': False})
            time.sleep(15)
        observed = b.read(serial_work / 'OWNER_STARTED.json')['identity']
        if (observed['PID'], observed['start_ticks']) != (cfg['serial_owner_PID'], cfg['serial_owner_start_ticks']): raise ValueError('Exact serial owner identity differs')
        serial_done = b.read(serial_work / 'SERIAL_COMPLETE.json')
        if serial_done['config_sha256'] != cfg['serial_config']['sha256'] or serial_done['blocks'] != ['b1', 'b2']:
            raise ValueError('Serial complete binding differs')
        for receipt in serial_done['completed']:
            b.terminal(h, receipt, receipt['job_sha256'], receipt['child_identity']['argv'])
        predecessor_freezes = []
        for block in ('b0', 'b1', 'b2'):
            config_path = initial_root / (block.upper() + '_BLOCK_OWNER_CONFIG.json')
            freeze_path = initial_root / (block + '_fit_owner') / 'BLOCK_FREEZE.json'; frozen = b.read(freeze_path)
            identity = b.read(freeze_path.parent / 'OWNER_STARTED.json')['identity']
            if (frozen['block'] != block or frozen['config_sha256'] != b.sha(config_path) or frozen['plan'] != cfg['initialization_plan']
                or frozen['selected_block_complete'] is not True or frozen['physical_postfits'] != 7 or frozen['TEST_access'] is not False
                or h.identity(identity['PID']) is not None or [row['cell_id'] for row in frozen['completed']] != [block + '_' + condition for condition in b.ORDER]):
                raise ValueError('Complete clean initializer block required')
            for receipt in frozen['completed']:
                b.terminal(h, receipt, receipt['job_sha256'], receipt['child_identity']['argv'])
                if b.sha(phase_file(receipt['freeze_relative'])) != receipt['freeze_sha256']: raise ValueError('Initializer freeze changed')
            predecessor_freezes.append({'path': str(freeze_path.relative_to(PHASE)), 'sha256': b.sha(freeze_path)})
        plan.update(root_adopted=True, scientific_execution_authorized=True, root_activation_config_sha256=args.config_sha256,
            root_adoption_UTC=h.now())
        plan_path = root / 'PLAN.json'
        if plan_path.exists(): raise ValueError('Fresh separate adopted PLAN required')
        b.write(plan_path, plan); plan_path.chmod(0o444); plan_hash = b.sha(plan_path)
        if len(cfg['job_templates']) != 3: raise ValueError('Exactly three fixed supplemental templates required')
        entries = []
        for block, binding in zip(plan['blocks'], cfg['job_templates']):
            job = b.read(b.bound_file(PHASE, binding)); seed = block['seed']; cell = block['block'] + '_shared_full_heads'
            if (job['seed'], job['factor_seed'], job['condition'], job['soft_seconds']) != (seed, block['factor_seed'], 'shared_full_heads', 15000):
                raise ValueError('Fixed supplemental seed/condition/cap differs')
            warm_path = phase_file(job['warm_freeze_relative']); warm = b.read(warm_path)
            warm_job_path = initial_root / 'jobs' / (block['block'] + '_warm.json'); warm_job = b.read(warm_job_path)
            if ((warm['phase'], warm['completed_cycles'], warm['seed']) != ('warm', 20, seed) or warm['VALID_TEST_access'] is not False
                or warm['job_sha256'] != b.sha(warm_job_path) or warm['plan_sha256'] != cfg['initialization_plan']['sha256']
                or b.sha(warm_path.parent / 'warm_checkpoint.pt') != warm['checkpoint_sha256']): raise ValueError('Same exact successful warm state required')
            for key in ('source_manifest_sha256', 'runtime_versions', 'runtime_qualification', 'feature_authority', 'negative_pool_authority'):
                if job[key] != warm_job[key]: raise ValueError('Same admitted warm source/runtime/input authority required')
            job.update(plan_relative=str(plan_path.relative_to(PHASE)), plan_sha256=plan_hash, warm_freeze_sha256=b.sha(warm_path),
                fits_authorized=True, VALID_values_access=True, external_hard_bound_confirmed=True, native_pointer_gradient_check_authorized=True,
                source_review_approved=True, source_review={'approved': True, 'evidence': [cfg['source_review']]}, supplement_manifest_sha256=SOURCE_SHA)
            job_path = root / 'jobs' / (cell + '.json'); output = root / 'runs' / cell
            if job['output_directory'] != str(output) or job['program_sha256'] != PROGRAM_SHA or job['initialization_sha256'] != INITIALIZER_SHA:
                raise ValueError('Exact separate source/output bindings required')
            if job_path.exists() or output.exists(): raise ValueError('No overwrite or retry')
            b.write(job_path, job); job_path.chmod(0o444)
            entries.append({'cell_id': cell, 'job_relative': str(job_path.relative_to(PHASE)), 'job_sha256': b.sha(job_path),
                'hard_seconds': 18000, 'argv': [cfg['python_executable'], '-B', str(SOURCE / 'run.py'), '--job', str(job_path), '--output', str(output)]})
        b.write(work / 'BOUND_JOBS.json', {'plan_sha256': plan_hash, 'entries': entries, 'predecessor_freezes': predecessor_freezes, 'scores_read': False})
        environment = dict(os.environ, **cfg['environment']); environment.pop('PYTHONHOME', None)
        resource_call = h.resources
        for entry in entries:
            physical(); verify_source(); cost = {}; peak = {'GPU_bytes': 0, 'RSS_bytes': 0}; output = Path(entry['argv'][6])
            if b.sha(phase_file(entry['job_relative'])) != entry['job_sha256'] or b.sha(plan_path) != plan_hash: raise ValueError('Activated job/plan changed')

            def measured_resources(rows, limits, remaining):
                sample_started = time.monotonic(); rss, gpu_bytes = resource_call(rows, limits, remaining)
                peak['RSS_bytes'] = max(peak['RSS_bytes'], rss); peak['GPU_bytes'] = max(peak['GPU_bytes'], gpu_bytes)
                history = output / 'HISTORY.jsonl'
                if not cost and history.is_file():
                    with history.open() as stream: line = stream.readline(4097)
                    if len(line) > 4096: raise ValueError('Unexpected first-cycle cost record size')
                    if line.endswith('\n'):
                        row = json.loads(line); seconds = row['cycle_seconds']
                        if row['cycle'] != 1 or 'complete_VALID' in row or 'members' in row or row['counters'] != {'episodes': 61, 'native_Adam_updates': 183}:
                            raise ValueError('First complete TRAIN cycle required; no quality read')
                        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0: raise ValueError('Finite positive cycle cost required')
                        inclusive = 18000 - remaining + time.monotonic() - sample_started
                        projected = inclusive + 59 * seconds
                        admitted = projected <= 15000 and peak['RSS_bytes'] <= bounds['RSS_bytes'] and peak['GPU_bytes'] <= bounds['GPU_bytes']
                        cost.update(UTC=h.now(), first_complete_TRAIN_cycle_seconds=seconds, observed_owned_seconds_through_cycle1=inclusive,
                            projected_charged_60_cycle_seconds=projected, fixed_soft_seconds=15000, sampled_peak_owned=copy.deepcopy(peak),
                            admitted=admitted, native_gate_artifact_sha256=b.sha(output / 'NATIVE_POINTER_GRADIENT_QUALIFICATION.json'), scores_read=False,
                            scope='Prefit native gate/setup and first cycle are charged; future VALID work remains inside the enforced full-fit cap. No predictive runtime guarantee.')
                        b.write(work / 'logs' / (entry['cell_id'] + '.FIRST_CYCLE_COST_ADMISSION.json'), cost)
                        if not admitted: raise TimeoutError('Actual first complete-cycle cost/resource ceiling failed; preserve and stop tail')
                return rss, gpu_bytes

            h.resources = measured_resources
            receipt = o.run_fit(h, work, entry, {'resource_limits': cfg['resource_limits']}, environment, output, context)
            h.resources = resource_call; b.terminal(h, receipt, entry['job_sha256'], entry['argv'])
            if cost.get('admitted') is not True: raise ValueError('Required complete-cycle resource admission absent')
            freeze_path = output / 'FREEZE.json'; frozen = b.read(freeze_path)
            if ((frozen['phase'], frozen['completed_cycles'], frozen['condition']) != ('fit', 60, 'shared_full_heads')
                or frozen['job_sha256'] != entry['job_sha256'] or frozen['plan_sha256'] != plan_hash or frozen['supplement_manifest_sha256'] != SOURCE_SHA
                or frozen['TEST_access'] is not False or (output / 'FAILURE.json').exists()): raise ValueError('Exact complete supplemental fit freeze required')
            completed.append({**receipt, 'freeze_relative': str(freeze_path.relative_to(PHASE)), 'freeze_sha256': b.sha(freeze_path)})
            b.write(work / 'PROGRESS.json', {'UTC': h.now(), 'phase': 'supplemental_fit', 'completed': len(completed), 'total': 3, 'last_cell': entry['cell_id'], 'scores_read': False})
            print('Completed operationally: ' + entry['cell_id'], flush=True)
        b.write(work / 'TAIL_COMPLETE.json', {'UTC': h.now(), 'completed': completed, 'plan_sha256': plan_hash,
            'config_sha256': args.config_sha256, 'inclusive_seconds': time.monotonic() - started, 'TEST_access': False, 'scores_read': False, 'retry': False})
    except BaseException as error:
        b.write(work / 'FAILURE.json', {'UTC': h.now(), 'error': type(error).__name__ + ': ' + str(error), 'completed': completed,
            'partial_outputs_preserved': True, 'retry': False, 'scores_read': False})
        raise


if __name__ == '__main__': main()
