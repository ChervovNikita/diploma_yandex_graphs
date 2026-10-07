"""Small allocation-only launcher; unchanged frozen workers perform all learning.

No resource acquisition, staging, resume, retry, result opening or77 routing.
Root supplies complete same-seed resource bindings and a separate family release.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def seal(folder, expected=None):
    if expected is not None and sha(folder / 'MANIFEST.json') != expected:
        raise ValueError('Exact immutable source seal')
    for row in json.loads((folder / 'MANIFEST.json').read_text())['files']:
        path = (folder / row['path']).resolve(strict=True)
        if not path.is_relative_to(folder.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Source changed: ' + row['path'])


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def setup(task, release_path):
    plan = json.loads((ROOT / (task + '.json')).read_text())
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    for key in ('root_scientific_fit_authorized', 'fixed_family_adopted', 'root_launcher_source_approved'):
        if release.get(key) is not True: raise ValueError('Disabled family release: ' + key)
    if release.get('TEST_access') is not False or release.get('automatic_retry') is not False or release.get('predictive_opening_authorized') is not False:
        raise ValueError('Closed family scope')
    if release.get('task') != task or release.get('launcher_manifest_sha256') != sha(ROOT / 'MANIFEST.json'):
        raise ValueError('Exact task/launcher release')
    seal(ROOT)
    suite = ROOT.parent / plan['suite_directory']; seal(suite, plan['source_manifest_sha256'])
    spec = importlib.util.spec_from_file_location('frozen_' + task + '_runtime', suite / 'runtime.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    runtime.allocation()  # Exact existing host, sole UUID, interpreter, cwd and CUDA visibility.
    if not ROOT.is_relative_to(runtime.PHASE) or not release_path.is_relative_to(runtime.PHASE):
        raise ValueError('Exact allocation phase only')
    config = json.loads(runtime.bound(runtime.PHASE, plan['config']).read_text())
    roster = [(arm, seed) for seed in config['pilot_seeds'] for arm in config['arms']]
    if len(roster) != 24 or config['training']['epochs'] != 100:
        raise ValueError('Frozen complete eight-arm, three-seed, 100-epoch family')
    approval = json.loads(runtime.bound(runtime.PHASE, plan['data_export_review']).read_text())
    if approval.get('approved') is not True or approval.get('source_manifest_sha256') != plan['source_manifest_sha256'] or approval.get('data_manifest') != plan['data_manifest']:
        raise ValueError('Exact audited data approval')
    for row in plan['source_review_evidence']:
        review = json.loads(runtime.bound(runtime.PHASE, row).read_text())
        if review.get('approved') is not True or review.get('source_manifest_sha256') != plan['source_manifest_sha256']:
            raise ValueError('Exact frozen suite approval')
    qualifier = ROOT.parent / plan['qualifier_directory']; seal(qualifier, plan['qualifier_manifest_sha256'])
    qreview = json.loads(runtime.bound(runtime.PHASE, plan['qualifier_source_review']).read_text())
    if qreview.get('approved') is not True or qreview.get('qualifier_manifest_sha256') != plan['qualifier_manifest_sha256']:
        raise ValueError('Exact existing qualifier source inspection')
    resources = release.get('resources', {})
    if set(resources) != {arm + '_' + str(seed) for arm, seed in roster}:
        raise ValueError('Supply all 24 same-arm/seed resource bindings before adopting fits')
    for arm, seed in roster:
        base = {'task': task, 'arm': arm, 'seed': seed, 'config': plan['config'],
                'data_manifest': plan['data_manifest'], 'source_manifest_sha256': plan['source_manifest_sha256']}
        receipt = json.loads(runtime.bound(runtime.PHASE, resources[arm + '_' + str(seed)]).read_text())
        if receipt.get('passed') is not True or receipt.get('seed') != seed or receipt.get('cell_identity') != runtime.cell_identity(base) or receipt.get('qualifier_manifest_sha256') != plan['qualifier_manifest_sha256']:
            raise ValueError('Actual same-seed/task/arm/source/data/runtime resource pass')
        if receipt.get('schema') != 'internal-be-resource-qualification-v2' or receipt.get('status') != 'complete' or receipt.get('exit_code') != 0:
            raise ValueError('Completed supervisor-issued resource evidence')
        receipt_path = runtime.bound(runtime.PHASE, resources[arm + '_' + str(seed)])
        terminal_path = receipt_path.with_name(receipt_path.name.replace('_LIVE_RESOURCE.json', '_LIVE_TERMINAL.json'))
        terminal = json.loads(terminal_path.read_text())
        if sha(terminal_path) != receipt.get('terminal_receipt_sha256') or terminal.get('status') != 'complete' or terminal.get('owned_worker_completed') is not True or terminal.get('reap_observed') is not True or terminal.get('cap_exceeded') is not False:
            raise ValueError('Actual completed/reaped/in-cap resource parent')
        runtime.resource_measurements(receipt)
    out = (runtime.PHASE / release['execution_directory']).resolve()
    if not out.is_relative_to(runtime.PHASE) or out.exists(): raise ValueError('Fresh allocation output only')
    return plan, release, config, roster, suite, runtime, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('collab', 'molhiv'), required=True)
    parser.add_argument('--release')
    parser.add_argument('--list', action='store_true', help='Print fixed metadata only; no package/data/model load')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(json.loads((ROOT / (args.task + '.json')).read_text())['cells'])); return
    if not args.release: parser.error('Separate root release required; launch disabled')
    started = time.monotonic(); os.umask(0o077)
    plan, release, config, roster, suite, runtime, out = setup(args.task, args.release)
    out.mkdir()
    for name in ('jobs', 'receipts', 'outputs', 'logs'): (out / name).mkdir()
    rows = []; fatal = False
    for arm, seed in roster:
        name = arm + '_' + str(seed)
        row = {'arm': arm, 'seed': seed, 'cell': name, 'status': 'not_launched', 'scores_opened': False}
        appended = False
        try:
            evidence = release['resources'][name]
            measured = json.loads(runtime.bound(runtime.PHASE, evidence).read_text())
            raw = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if len(raw) != 2 or raw[0].strip() != runtime.GPU: raise ValueError('Exact allocation memory observation')
            free = int(raw[1].strip()) * 1024**2
            if free < measured['peak_GPU_bytes'] + 2 * 1024**3:
                row['status'] = 'retained_memory_admission_failure'; continue
            job = {'schema': 'internal-be-predictive-cell-v1', 'task': args.task, 'arm': arm, 'seed': seed,
                'root_execution_authorized': True, 'source_review_approved': True, 'fixed_protocol_adopted': True,
                'source_manifest_sha256': plan['source_manifest_sha256'], 'source_review_evidence': plan['source_review_evidence'],
                'config': plan['config'], 'data_manifest': plan['data_manifest'], 'data_export_review': plan['data_export_review'],
                'resource_qualification_evidence': [evidence], 'external_hard_bound_confirmed': True,
                'soft_seconds': config['budget']['cell_soft_seconds'], 'hard_seconds': config['budget']['cell_hard_seconds'],
                'active_compute_seconds': config['budget']['cell_active_compute_seconds'], 'cleanup_grace_seconds': 10,
                'output_directory': str(out / 'outputs' / name),
                'supervisor_receipt_path': str((out / 'receipts' / (name + '_LIVE.json')).relative_to(runtime.PHASE)),
                'TEST_access': False, 'automatic_retry': False}
            job_path = out / 'jobs' / (name + '.json'); write_new(job_path, job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=runtime.GPU,
                PYTHONPATH=str(runtime.PHASE / 'native_ncn_dependency_overlay_20261005_v1') + ':' + str(runtime.REPO / '.venv/lib/python3.11/site-packages'))
            env.pop('PYTHONHOME', None)
            with (out / 'logs' / (name + '.log')).open('x') as log:
                process = subprocess.Popen([str(runtime.PYTHON), '-B', str(suite / 'supervise.py'), '--job', str(job_path)],
                    cwd=runtime.REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    owner_ticks = runtime.proc_start(process.pid)
                finally:
                    code = process.wait()  # Always reap the unchanged finite source supervisor.
            terminal_path = out / 'receipts' / (name + '_LIVE_TERMINAL.json')
            terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
            freeze_path = out / 'outputs' / name / 'FREEZE.json'
            freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else {}  # Engineering fields only.
            live_path = out / 'receipts' / (name + '_LIVE.json')
            live = json.loads(live_path.read_text()) if live_path.exists() else {}
            expected = runtime.cell_identity(job)
            if code == 0 and (live.get('supervisor_pid') != process.pid or live.get('supervisor_start_ticks') != owner_ticks or live.get('job_sha256') != sha(job_path) or live.get('cell_identity') != expected or terminal.get('cell_identity') != expected or terminal.get('live_receipt_sha256') != sha(live_path)):
                raise ValueError('Actual unchanged scientific parent/job custody')
            complete = code == 0 and terminal.get('admission_success') is True and freeze.get('complete') is True
            complete = complete and freeze.get('epochs') == 100 and freeze.get('steps') == plan['expected_full_steps']
            complete = complete and all(freeze.get(k) == v for k, v in {'task': args.task, 'arm': arm, 'seed': seed,
                'source_manifest_sha256': plan['source_manifest_sha256'], 'scores_closed': True, 'TEST_access': False}.items())
            if complete and sha(out / 'outputs' / name / 'selected.pt') != freeze.get('selected_sha256'):
                raise ValueError('Immutable selected endpoint hash')
            row.update(status='complete' if complete else 'retained_fit_failure', exit_code=code,
                checkpoint_sha256=freeze.get('selected_sha256') if complete else None,
                full_epochs=freeze.get('epochs'), full_steps=freeze.get('steps'))
        except Exception as error:
            row.update(status='retained_launcher_or_custody_failure', error_type=type(error).__name__)
            # A source/custody failure stops admission rather than silently retrying.
            rows.append(row); appended = True; fatal = True
            write_new(out / 'receipts' / (name + '_CELL.json'), row)
            for next_arm, next_seed in roster[len(rows):]:
                rows.append({'arm': next_arm, 'seed': next_seed, 'cell': next_arm + '_' + str(next_seed),
                             'status': 'not_launched_launcher_failure', 'scores_opened': False})
            break
        finally:
            if not appended:
                rows.append(row); write_new(out / 'receipts' / (name + '_CELL.json'), row)
    write_new(out / 'FAMILY_CLOSURE.json', {'task': args.task, 'cells': rows, 'closed': len(rows) == 24,
        'complete_cells': sum(x['status'] == 'complete' for x in rows), 'scores_opened': False,
        'TEST_access': False, 'automatic_retry': False, 'wall_seconds': time.monotonic() - started})
    print(json.dumps({'task': args.task, 'closed': len(rows) == 24, 'scores_opened': False,
                      'complete_cells': sum(x['status'] == 'complete' for x in rows)}))
    if fatal: raise SystemExit(1)


if __name__ == '__main__': main()
