#!/usr/bin/env python3
"""Parameterize the existing allocation warm owner for the fixed later seeds."""
import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCK_SOURCE = PHASE / 'citeseer_gnnm_initialization_pilot_block_queue_preparation_20261006_v1/block_owner.py'
BLOCK_SHA = 'fd86370c5b51c79e7ac4153936a462e5685e131f5624e1791bcb36df3af5f3b3'


def main():
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    if Path.cwd() != REPO or socket.gethostname() != 'anogena-2-0': raise ValueError('Exact allocation host/cwd required')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != [GPU]:
        raise ValueError('Exact one-GPU allocation required')
    if hashlib.sha256(BLOCK_SOURCE.read_bytes()).hexdigest() != BLOCK_SHA: raise ValueError('Reviewed block owner changed')
    spec = importlib.util.spec_from_file_location('initialization_block_utils', BLOCK_SOURCE)
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    if b.sha(args.config) != args.config_sha256: raise ValueError('Exact warm config changed')
    cfg = b.read(args.config)
    if cfg.get('root_execution_authorized') is not True or cfg['adapter_sha256'] != b.sha(__file__): raise ValueError('Root-released exact warm owner required')
    job_path = b.bound_file(PHASE, {'path': cfg['job_relative'], 'sha256': cfg['job_sha256']}); job = b.read(job_path)
    plan_path = b.bound_file(PHASE, {'path': job['plan_relative'], 'sha256': job['plan_sha256']}); plan = b.read(plan_path)
    block = next(row for row in plan['blocks'] if row['block'] == cfg['block'])
    if ((job['seed'], job['factor_seed']) != (block['seed'], block['factor_seed']) or job['phase'] != 'warm'
        or job['TEST_access'] is not False or job['VALID_values_access'] is not False or job['fits_authorized'] is not False
        or job['soft_seconds'] != plan['bounds']['warm']['soft_seconds'] or plan['warm_cycles'] != 20 or not plan['root_adopted']):
        raise ValueError('Exact fixed TRAIN-only warm design required')
    source = PHASE / b.SOURCE_NAME
    if b.sha(source / 'run.py') != b.PROGRAM_SHA or b.sha(source / 'initialize.py') != b.INITIALIZER_SHA: raise ValueError('Sealed source changed')
    o = b.load('reviewed_warm_loop', PHASE / b.LOOP_REL, b.LOOP_SHA)
    h = b.load('reviewed_warm_helper', PHASE / b.HELPER_REL, b.HELPER_SHA); h.GPU = GPU
    output = Path(job['output_directory'])
    expected = [cfg['argv'][0], '-B', str(source / 'run.py'), '--job', str(job_path), '--output', str(output)]
    if cfg['argv'] != expected or output.exists(): raise ValueError('Exact fresh warm argv/output required')
    work = PHASE / cfg['owner_directory_relative']; work.mkdir(); (work / 'logs').mkdir()
    ident = h.identity(os.getpid())
    if ident is None or ident['sid'] != ident['pgid'] or ident['sid'] != os.getpid(): raise ValueError('Detached warm owner required')
    b.write(work / 'OWNER_STARTED.json', {'UTC': h.now(), 'identity': ident, 'TRAIN_only': True, 'TEST_closed': True})
    context = SimpleNamespace(REPO=REPO, SOURCE=source, SOURCE_SHA=b.PROGRAM_SHA, GPU_UUID=GPU, GPU_UUIDS=(GPU,),
        physical_host=lambda: physical(), phase_file=lambda rel: phase_file(rel), sha=b.sha, write=b.write)
    environment = dict(os.environ, **cfg['environment']); environment.pop('PYTHONHOME', None)
    entry = {'cell_id': cfg['block'] + '_warm', 'job_relative': cfg['job_relative'], 'job_sha256': cfg['job_sha256'],
        'argv': cfg['argv'], 'hard_seconds': plan['bounds']['warm']['hard_seconds']}
    receipt = o.run_fit(h, work, entry, {'resource_limits': cfg['resource_limits']}, environment, output, context)
    b.write(work / 'TERMINAL.json', receipt)
    b.terminal(h, receipt, cfg['job_sha256'], cfg['argv'])
    freeze = b.read(output / 'FREEZE.json')
    if (freeze['phase'], freeze['completed_cycles'], freeze['seed']) != ('warm', 20, block['seed']) or freeze['VALID_TEST_access'] is not False:
        raise ValueError('Exact complete TRAIN-only warm state required')
    b.write(work / 'WARM_COMPLETE.json', {'UTC': h.now(), 'freeze_sha256': b.sha(output / 'FREEZE.json'),
        'checkpoint_sha256': freeze['checkpoint_sha256'], 'native_updates': freeze['counters']['native_Adam_updates'],
        'TEST_closed': True, 'child_absent': True})


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


if __name__ == '__main__': main()
