#!/usr/bin/env python3
"""After clean b0 closure, run b1 warm/block then b2 warm/block serially."""
import argparse
import copy
import hashlib
import importlib.util
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


def physical():
    if Path.cwd() != REPO or socket.gethostname() != 'anogena-2-0': raise ValueError('Exact allocation host/cwd required')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != [GPU]:
        raise ValueError('Exact one-GPU allocation required')


def phase_file(relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Phase-relative path required')
    path = (PHASE / p).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file(): raise ValueError('File leaves authorized phase')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(); physical()
    if hashlib.sha256(BLOCK_SOURCE.read_bytes()).hexdigest() != BLOCK_SHA: raise ValueError('Reviewed block owner changed')
    spec = importlib.util.spec_from_file_location('serial_initialization_block_utils', BLOCK_SOURCE)
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    if b.sha(args.config) != args.config_sha256: raise ValueError('Exact serial release changed')
    cfg = b.read(args.config)
    if (cfg.get('root_execution_authorized') is not True or cfg.get('TEST_access') is not False
        or cfg.get('retry') is not False or b.sha(__file__) != cfg['adapter_sha256']): raise ValueError('Exact root serial release required')
    root = (PHASE / cfg['execution_directory_relative']).resolve(strict=True)
    if not root.is_relative_to(PHASE.resolve()): raise ValueError('Root leaves authorized phase')
    plan_path = b.bound_file(PHASE, cfg['plan']); plan = b.read(plan_path); bounds = plan['bounds']
    if not plan['root_adopted'] or plan['TEST_closed'] is not True or tuple(plan['conditions']) != b.ORDER:
        raise ValueError('Original adopted fixed design required')
    block_hard = sum(bounds[c if c in ('single', 'independent_warm4') else 'shared_F4']['hard_seconds'] for c in b.ORDER) + 7 * cfg['resource_limits']['resource_wait_seconds']
    warm_hard = bounds['warm']['hard_seconds'] + cfg['resource_limits']['resource_wait_seconds']
    if cfg['previous_block_wait_seconds'] != block_hard or [row['block'] for row in cfg['later_blocks']] != ['b1', 'b2']:
        raise ValueError('Fixed serial order and existing plan-derived ceilings required')
    o = b.load('reviewed_serial_loop', PHASE / b.LOOP_REL, b.LOOP_SHA)
    h = b.load('reviewed_serial_helper', PHASE / b.HELPER_REL, b.HELPER_SHA); h.GPU = GPU
    ident = h.identity(os.getpid())
    if ident is None or ident['sid'] != ident['pgid'] or ident['sid'] != os.getpid(): raise ValueError('Detached serial owner required')
    work = root / 'serial_b1_b2_owner'; work.mkdir(); (work / 'logs').mkdir()
    b.write(work / 'OWNER_STARTED.json', {'UTC': h.now(), 'identity': ident, 'config_sha256': args.config_sha256,
        'fixed_order': ['b1_warm', 'b1_block', 'b2_warm', 'b2_block'], 'TEST_access': False, 'scores_read': False})
    context = SimpleNamespace(REPO=REPO, SOURCE=BLOCK_SOURCE.parent, SOURCE_SHA=BLOCK_SHA, GPU_UUID=GPU, GPU_UUIDS=(GPU,),
        physical_host=physical, phase_file=phase_file, sha=b.sha, write=b.write)
    completed = []; started = time.monotonic()

    def block_closed(block, owner_PID, owner_ticks, config_hash):
        owner_root = root / (block + '_fit_owner')
        if (owner_root / 'QUEUE_FAILURE.json').exists(): raise ValueError('Previous block failed; preserve and stop serial queue')
        path = owner_root / 'BLOCK_FREEZE.json'
        if not path.is_file(): return False
        frozen = b.read(path)
        if (frozen['block'] != block or frozen['config_sha256'] != config_hash or frozen['selected_block_complete'] is not True
            or frozen['physical_postfits'] != 7 or frozen['plan'] != cfg['plan'] or frozen['TEST_access'] is not False
            or [row['cell_id'] for row in frozen['completed']] != [block + '_' + condition for condition in b.ORDER]):
            raise ValueError('Exact complete preceding block binding differs')
        observed = b.read(owner_root / 'OWNER_STARTED.json')['identity']
        if (observed['PID'], observed['start_ticks']) != (owner_PID, owner_ticks): raise ValueError('Preceding block owner identity differs')
        if h.identity(owner_PID) is not None: return False
        for row in frozen['completed']:
            b.terminal(h, row, row['job_sha256'], row['child_identity']['argv'])
            if b.sha(phase_file(row['freeze_relative'])) != row['freeze_sha256']: raise ValueError('Preceding block freeze changed')
        return True

    def run_owner(cell, path, config_hash, argv, output, hard_seconds):
        entry = {'cell_id': cell, 'job_relative': str(path.relative_to(PHASE)), 'job_sha256': config_hash,
            'argv': argv, 'hard_seconds': hard_seconds}
        environment = dict(os.environ, **cfg['owner_environment']); environment.pop('PYTHONHOME', None)
        receipt = o.run_fit(h, work, entry, {'resource_limits': cfg['resource_limits']}, environment, output, context)
        b.terminal(h, receipt, config_hash, argv)
        completed.append(receipt)
        return receipt

    try:
        prior = cfg['previous_block']; b.bound_file(PHASE, prior['config'])
        while not block_closed(prior['block'], prior['owner_PID'], prior['owner_start_ticks'], prior['config']['sha256']):
            physical()
            if time.monotonic() - started >= cfg['previous_block_wait_seconds']: raise TimeoutError('Fixed preceding block wait exhausted')
            b.write(work / 'PROGRESS.json', {'UTC': h.now(), 'phase': 'awaiting_b0_clean_block', 'completed_owner_operations': 0, 'scores_read': False})
            time.sleep(15)
        for row in cfg['later_blocks']:
            block = row['block']; physical()
            warm_source = b.bound_file(PHASE, cfg['warm_owner_source'])
            warm_cfg = b.read(b.bound_file(PHASE, row['warm_config_preview'])); warm_cfg['root_execution_authorized'] = True
            warm_path = root / (block.upper() + '_WARM_OWNER_CONFIG.json')
            if warm_path.exists(): raise ValueError('No warm owner config overwrite or retry')
            b.write(warm_path, warm_cfg); warm_path.chmod(0o444)
            warm_hash = b.sha(warm_path)
            warm_argv = ['/usr/bin/python3', '-I', '-S', '-B', str(warm_source), '--config', str(warm_path), '--config-sha256', warm_hash]
            warm_receipt = run_owner(block + '_warm_owner', warm_path, warm_hash, warm_argv, root / ('warm_' + block + '_owner'), warm_hard)
            warm_ident = warm_receipt['child_identity']
            block_cfg = b.read(b.bound_file(PHASE, row['block_config_preview']))
            block_cfg.update(root_execution_authorized=True, warm_owner_PID=warm_ident['PID'], warm_owner_start_ticks=warm_ident['start_ticks'],
                warm_owner_config={'path': str(warm_path.relative_to(PHASE)), 'sha256': warm_hash})
            block_path = root / (block.upper() + '_BLOCK_OWNER_CONFIG.json')
            if block_path.exists(): raise ValueError('No block config overwrite or retry')
            b.write(block_path, block_cfg); block_path.chmod(0o444); block_hash = b.sha(block_path)
            block_argv = ['/usr/bin/python3', '-I', '-S', '-B', str(BLOCK_SOURCE), '--config', str(block_path), '--config-sha256', block_hash]
            block_receipt = run_owner(block + '_block_owner', block_path, block_hash, block_argv, root / (block + '_fit_owner'), block_hard)
            owner_ident = block_receipt['child_identity']
            if not block_closed(block, owner_ident['PID'], owner_ident['start_ticks'], block_hash): raise ValueError('Required serial block terminal closure absent')
            b.write(work / 'PROGRESS.json', {'UTC': h.now(), 'last_complete_block': block, 'completed_owner_operations': len(completed), 'scores_read': False})
            print('Completed operationally: ' + block, flush=True)
        b.write(work / 'SERIAL_COMPLETE.json', {'UTC': h.now(), 'completed': completed, 'blocks': ['b1', 'b2'],
            'plan': cfg['plan'], 'config_sha256': args.config_sha256, 'inclusive_seconds': time.monotonic() - started,
            'TEST_access': False, 'scores_read': False, 'retry': False})
    except BaseException as error:
        b.write(work / 'FAILURE.json', {'UTC': h.now(), 'error': type(error).__name__ + ': ' + str(error), 'completed': completed,
            'partial_outputs_preserved': True, 'retry': False, 'scores_read': False})
        raise


if __name__ == '__main__': main()
