"""Run a frozen full family on the sole authorized route, preserving each child.

Local coordination only. Training and all model/data operations are performed
by the exact sealed remote runner under its unchanged bounded supervisor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

PHASE = Path(__file__).resolve().parents[1]
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE = REPO + '/experiments_iclr/postsubmission_20260930'
PYTHON = REPO + '/.venv/bin/python'
DESTINATION = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', DESTINATION]
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def ref(p):
    return dict(path=str(p.relative_to(PHASE)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def confined(p):
    p = p.resolve()
    if not p.is_relative_to(PHASE):
        raise ValueError('Coordinator input/output escapes local phase')
    return p


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--validation-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-prefix', required=True)
    args = parser.parse_args()
    binding_path = confined(args.binding)
    freeze_path = confined(args.validation_freeze)
    output = confined(args.output)
    if output.exists() or not args.job_prefix.replace('_', '').isalnum():
        raise ValueError('New exclusive coordinator output and simple job prefix required')
    binding_ref, freeze_ref = ref(binding_path), ref(freeze_path)
    binding = json.loads(binding_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (binding['scientific_training_authorized'] is not True
            or binding['test_target_export_or_evaluation_authorized'] is not False
            or binding['gpu_uuid'] != UUID or freeze['binding'] != binding_ref):
        raise ValueError('Actual frozen root-authorized study required')
    order = binding['fit_order']
    if len(order) != 30 or len({json.dumps(c, sort_keys=True) for c in order}) != 30:
        raise ValueError('Complete frozen 30-cell family required')
    source_folder = binding_path.parent
    manifest_ref = ref(source_folder / 'SOURCE_MANIFEST_v1.json')
    if manifest_ref['sha256'] != binding['source_manifest_sha256']:
        raise ValueError('Frozen exact runner source differs')
    output.mkdir(parents=True, exist_ok=False)
    start = dict(UTC=utc(), source=ref(Path(__file__)), binding=binding_ref,
                 validation_freeze=freeze_ref, manifest=manifest_ref, ssh_destination=DESTINATION,
                 operation_order='all30fits_then_all30replays_then_family_lock',
                 failed_children_preserved_and_stop_without_retry=True)
    (output / 'START.json').write_text(json.dumps(start, indent=2)+'\n')
    operations = [('fit', c) for c in order] + [('replay', c) for c in order] + [('lock-family', None)]
    ledger = output / 'LAUNCHES.jsonl'
    for index, (operation, cell) in enumerate(operations):
        if ref(binding_path) != binding_ref or ref(freeze_path) != freeze_ref:
            raise ValueError('Study binding/freeze changed during coordination')
        idle = []
        for repetition in range(2):
            q = subprocess.run(SSH+['nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True, timeout=30)
            row = [v.strip() for v in q.stdout.strip().split(',')]
            if len(row) != 4 or row[1] != UUID or int(row[2]) != 0 or int(row[3]) != 0:
                raise RuntimeError('Stable idle guard denied; no scientific child launched')
            idle.append(dict(UTC=utc(), query=q.stdout))
            if repetition == 0:
                time.sleep(2)
        job = args.job_prefix + '_%02d_' % index + operation.replace('-', '_')
        inner = REMOTE + '/protocols/logged_runs/' + job
        cap = binding['whole_process_policy']['per_child_wall_cap_seconds']
        child = [PYTHON, REMOTE+'/'+str(source_folder.relative_to(PHASE))+'/runner.py', operation,
                 '--binding', REMOTE+'/'+binding_ref['path'], '--validation-freeze', REMOTE+'/'+freeze_ref['path'],
                 '--supervisor', inner]
        if cell is not None:
            child += ['--task', cell['task'], '--arm', cell['arm'], '--seed', str(cell['seed'])]
        argv = [PYTHON, REMOTE+'/protocols/bounded_run_v1.py', '--output', REMOTE+'/protocols/bounded_runs/'+job,
                '--request', REMOTE+'/'+binding_ref['path'], '--cap-seconds', str(cap), '--grace-seconds', '5',
                '--', PYTHON, REMOTE+'/protocols/run_authorized_v2.py', '--output', inner, '--require-idle-gpu', '--', *child]
        command = ('cd '+shlex.quote(REPO)+' && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh'
                   +' && export GNNM_SSH_DESTINATION='+shlex.quote(DESTINATION)
                   +' CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 && '+shlex.join(argv))
        began = utc()
        result = subprocess.run(SSH+[command], capture_output=True, text=True)
        record = dict(index=index, operation=operation, cell=cell, job=job, start_UTC=began, terminal_UTC=utc(),
                      stable_idle=idle, remote_argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        with ledger.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(dict(index=index, operation=operation, cell=cell, exit_code=result.returncode)), flush=True)
        if result.returncode:
            raise RuntimeError('Child did not complete normally; original evidence retained; automatic retry refused')
    (output/'COMPLETED.json').write_text(json.dumps(dict(UTC=utc(), children=61, all_launches_ref=ref(ledger),
                                                      scientific_evidence_still_requires_fetch_and_independent_audit=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
