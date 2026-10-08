"""Finish two never-started original Wiki12 cells; metadata only, no retry."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
CELLS = ('6307_residual_only', '6307_combined')
STOP_REQUESTED = False
EXPECTED_WORK = dict(shadow_member_forwards=8800, replay_member_forwards=8800,
    output_cotangent_collections=1100, member_reverse_collections=8800,
    optimizer_bank_updates=1100, exact_member_RNG_endpoint_checks=1100)
ORIGINAL_LIMITS = dict(owned_tree_GPU_memory_cap_bytes=34359738368,
    owned_tree_RSS_cap_bytes=34359738368, combined_child_log_cap_bytes=8388608,
    own_fit_output_cap_bytes=4294967296, minimum_fresh_GPU_free_bytes=38654705664,
    resource_wait_seconds=1800, poll_interval_seconds=5, telemetry_timeout_seconds=10)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def inside(relative):
    path = (PHASE / relative).resolve()
    require(path.is_relative_to(PHASE) and path != PHASE, 'Project phase only')
    return path


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'], 'Changed input: ' + row['path'])
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Changed input size')
    return path


def load(name, row):
    spec = importlib.util.spec_from_file_location(name, bound(row))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def gone(helper, pid, ticks):
    actual = helper.identity(pid)
    return actual is None or actual['start_ticks'] != ticks or actual['state'] == 'Z'


def no_cuda(helper, pid):
    rows = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory',
                         '--format=csv,noheader,nounits'], 10)
    return not any(len(parts) > 1 and parts[1].strip() == str(pid)
                   for parts in (row.split(',') for row in rows))


def old_custody(protocol, original, helper):
    snapshot = read(bound(protocol['old_terminal_custody']))
    require(snapshot['old_owner_absent'] and snapshot['quality_scores_read'] is False
            and snapshot['checkpoint_loaded'] is False, 'Original no-quality terminal custody')
    for relative, binding in snapshot['files'].items():
        bound(dict(binding, path=relative))
    parent = protocol['original_parent']
    require(gone(helper, parent['pid'], parent['start_ticks']), 'Original owner still live')
    old_root = inside(protocol['original_activation'])
    actual_parent = read(old_root / 'PARENT_OWNER.json')
    require(actual_parent['PID'] == parent['pid'] and actual_parent['start_ticks'] == parent['start_ticks'],
            'Exact original owner identity')
    require(read(old_root / 'FAMILY_CLOSURE.json')['complete'] is False,
            'Original failed closure must remain complete:false')
    failure = read(old_root / 'LANE_0_FAILURE.json')
    require(failure['error'] == 'TimeoutError: Fixed fresh-GPU resource window exhausted before child launch',
            'Original failure was not the never-started resource wait')
    roster = {row['cell_id']: row for lane in original.bundle['lanes'].values() for row in lane}
    completed = snapshot['completed']
    require(len(completed) == 10 and {r['cell_id'] for r in completed} == set(roster) - set(CELLS),
            'Exactly the original ten cells are required')
    for row in completed:
        require(row['release'] == roster[row['cell_id']], 'Old cell release changed')
        bound(row['release']); bound(row['completion']); exit_path = bound(row['exit'])
        require(row['completion']['epochs'] == row['completion']['steps'] == 1100
                and row['exit']['child_absent'] and row['exit']['child_no_CUDA_rows'],
                'Original whole-horizon terminal custody')
        receipt = read(exit_path); child = receipt['raw_identity_observation']
        require(receipt['exit_code'] == 0 and receipt['reason'] is None
                and receipt['terminal_wait_observed'] and not receipt['signals_sent']
                and receipt['job_sha256'] == row['release']['sha256']
                and child['PID'] == row['exit']['pid']
                and gone(helper, child['PID'], child['start_ticks']), 'Old fit terminal/source mismatch')
    require([r['release']['cell_id'] for r in snapshot['unused']] == list(CELLS)
            and all(r['scientific_child_started'] is False and r['output_absent']
                    for r in snapshot['unused']), 'Two original unused releases only')
    for row in snapshot['unused']:
        require(row['release'] == roster[row['release']['cell_id']]
                and read(bound(row['release']))['output'] == row['output'], 'Unused original roster/output binding')
    preflight = old_root / 'logs' / (CELLS[0] + '.PREFLIGHT.json')
    status = read(preflight)
    require(status['scientific_child_started'] is False and status['elapsed_seconds'] >= 1800
            and status['GPU_UUID'] == protocol['gpu_uuid']
            and status['GPU_free_bytes'] < protocol['limits']['minimum_fresh_GPU_free_bytes'],
            'Preserve the original failed 1800s admission and cost')
    for cell in CELLS:
        require(not (old_root / 'logs' / (cell + '.CHILD_STARTED.json')).exists(),
                'Original supposedly unused cell has a child-start record')
    return snapshot, dict(path=str(preflight.relative_to(PHASE)), sha256=sha(preflight),
                         bytes=preflight.stat().st_size, observation=status)


def supcon_terminal(protocol, helper):
    parent = protocol['supcon_parent']
    if not gone(helper, parent['pid'], parent['start_ticks']):
        return None
    admission = read(bound(protocol['supcon_admission']))
    path = inside(admission['output_directory']) / 'CLOSURE.json'
    if not path.is_file():
        return None
    record = read(path)
    require(record['family_accounted'] is True and record['owner']['pid'] == parent['pid']
            and record['owner']['start_ticks'] == parent['start_ticks']
            and record['owner']['group'] == record['owner']['session'] == parent['pid']
            and record['controller_manifest_sha256'] == admission['controller_manifest_sha256']
            and record['protocol_sha256'] == admission['protocol_sha256'], 'Exact SupCon family closure')
    require(len(record['rows']) == 3 and {r['seed'] for r in record['rows']} == {6101, 6203, 6307},
            'Whole SupCon roster, including failures')
    for row in record['rows']:
        require(row['status'] not in ('running', 'unlaunched'), 'SupCon still scheduled/running')
        child = row.get('owner')
        if child is not None:
            require(row.get('actual_exit_and_reap') is True and row.get('owned_child_after') is None
                    and gone(helper, child['pid'], child['start_ticks']) and no_cuda(helper, child['pid']),
                    'SupCon owned child not terminal/cleaned')
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path),
                all_new_fits_complete=record['all_new_fits_complete'],
                terminal_rows=[dict(seed=r['seed'], status=r['status']) for r in record['rows']])


def schedule(protocol, original, helper, output, deadline, frozen_supcon=None):
    while True:
        require(not STOP_REQUESTED, 'Stop requested; no next fit launched')
        original.physical()
        terminal = supcon_terminal(protocol, helper)
        if frozen_supcon is not None:
            require(terminal == frozen_supcon, 'SupCon terminal custody changed')
        rows = helper.query(['--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], 10)
        selected = [r.split(',') for r in rows if r.split(',')[0].strip() == protocol['gpu_uuid']]
        require(len(selected) == 1 and selected[0][1].strip().isdigit(), 'Fresh original GPU0 telemetry')
        available = int(selected[0][1].strip()) * 1024 ** 2
        original.write(output / 'SCHEDULING_WAIT.json', dict(UTC=helper.now(),
            supcon_terminal=terminal, fresh_free_GPU_bytes=available,
            required_free_GPU_bytes=protocol['limits']['minimum_fresh_GPU_free_bytes'],
            scheduling_seconds_remaining=max(0, deadline - time.monotonic()),
            scientific_child_started=False, fit_admission_wait_not_started=True))
        if terminal is not None and available >= protocol['limits']['minimum_fresh_GPU_free_bytes']:
            return terminal
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Separate scheduling window exhausted; no fit admission or retry')
        time.sleep(min(protocol['scheduling_poll_seconds'], remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    require(sha(args.admission) == args.admission_sha256, 'Exact root admission')
    admission = read(args.admission)
    require(admission['enabled'] and admission['root_launch_authorized'], 'Preparation is inactive')
    require(sha(HERE / 'controller.py') == admission['controller_sha256'], 'Reviewed controller')
    protocol = read(bound(admission['protocol']))
    require(protocol['cells'] == list(CELLS) and protocol['automatic_retry'] is False
            and protocol['TEST_access'] is False, 'Fixed two never-started cells')
    require(protocol['limits'] == ORIGINAL_LIMITS, 'Original fit caps/admission wait unchanged')
    original = load('original_wiki12_owner', protocol['original_owner_source'])
    original.pins = read(bound(protocol['original_source_bindings']))
    original.bundle = read(bound(protocol['original_bundle']))
    require(original.bundle['owner_sha256'] == protocol['original_owner_source']['sha256'], 'Original owner binding')
    original.physical()
    require(str(Path(os.sys.executable).absolute()) == original.pins['python'], 'Original runtime interpreter')
    helper = original.module('unused2_owned', original.pins['reviewed_ownership_helper'])
    helper.GPU = protocol['gpu_uuid']
    supervisor = original.module('unused2_runfit', original.pins['reviewed_owned_fit_helper'])
    owner = helper.identity(os.getpid())
    require(owner is not None and owner['pgid'] == owner['sid'] == owner['PID'], 'Fresh detached owner session')
    output = inside(protocol['continuation_output']); output.mkdir(exist_ok=False)
    (output / 'logs').mkdir()
    original.write(output / 'OWNER.json', dict(owner=owner, admission_sha256=args.admission_sha256,
        protocol=admission['protocol'], quality_scores_read=False, automatic_retry=False))
    completed = []; old_verified = False; old = None; preflight = None; supcon = None; fatal = None
    def interrupted(signum, frame):
        # Do not interrupt the reviewed helper between Popen and its custody
        # observation. A live fit remains under its unchanged finite bounds.
        global STOP_REQUESTED
        STOP_REQUESTED = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        old, preflight = old_custody(protocol, original, helper); old_verified = True
        for row in old['unused']:
            require(not inside(row['output']).exists(), 'Both original unused output directories must be absent')
        original.write(output / 'OLD_CUSTODY.json', dict(snapshot=protocol['old_terminal_custody'],
            preserved_failed_preflight=preflight, old_complete_false_immutable=True))
        scheduling_remaining = protocol['scheduling_wait_seconds']
        for row in old['unused']:
            began = time.monotonic()
            supcon = schedule(protocol, original, helper, output,
                              began + scheduling_remaining, supcon)
            scheduling_remaining -= time.monotonic() - began
            old_custody(protocol, original, helper)
            require(sha(inside(preflight['path'])) == preflight['sha256'], 'Historical preflight changed')
            item = row['release']; path = bound(item); cfg = read(path)
            require(cfg['seed'] == 6307 and item['cell_id'] in CELLS
                    and cfg['physical_gpu_uuid'] == protocol['gpu_uuid'], 'Original seed/GPU release')
            target = inside(cfg['output']); require(not target.exists(), 'No trained-cell retry')
            source = inside(protocol['original_scientific_source'])
            require(sha(source / 'MANIFEST.json') == original.bundle['source_manifest_sha256']
                    == cfg['source_manifest_sha256'], 'Original scientific source')
            entry = dict(cell_id=item['cell_id'], job_relative=item['path'],
                job_sha256=item['sha256'], hard_seconds=cfg['external_active_seconds'],
                argv=[original.pins['python'], '-B', str(source / 'run_cell.py'),
                      '--release', str(path), '--release-sha256', item['sha256']])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=protocol['gpu_uuid'], PYTHONPATH='',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            environment.pop('PYTHONHOME', None)
            context = SimpleNamespace(REPO=REPO, SOURCE=source,
                SOURCE_SHA=original.bundle['source_manifest_sha256'], GPU_UUID=protocol['gpu_uuid'],
                GPU_UUIDS=tuple(original.pins['physical_gpu_inventory']), phase_file=original.phase_file,
                physical_host=original.physical, sha=original.sha, write=original.write)
            require(not STOP_REQUESTED, 'Stop requested; no next fit launched')
            receipt = supervisor.run_fit(helper, output, entry,
                {'resource_limits': protocol['limits']}, environment, target, context)
            child = receipt['raw_identity_observation']
            require(receipt['exit_code'] == 0 and receipt['reason'] is None
                    and receipt['terminal_wait_observed'] and not receipt['signals_sent']
                    and child is not None and helper.identity(child['PID']) is None
                    and no_cuda(helper, child['PID']) and not (target / 'FAILURE.json').exists(),
                    'New cell lacks exact owned terminal closure; no retry')
            endpoint = read(target / 'COMPLETE.json')
            require(endpoint['complete'] and endpoint['epochs'] == endpoint['steps'] == 1100
                    and endpoint['seed'] == cfg['seed']
                    and endpoint['mechanism_ablation']['condition'] == cfg['condition']
                    and endpoint['execution_accounting'] == EXPECTED_WORK, 'Original full1100 cell required')
            completed.append(dict(cell_id=item['cell_id'], complete=True,
                completion_sha256=sha(target / 'COMPLETE.json'), release=item,
                exit_sha256=sha(output / 'logs' / (item['cell_id'] + '.EXIT.json')),
                exit_receipt=receipt, child_absent=True, child_no_CUDA_rows=True))
            original.write(output / 'PROGRESS.json', dict(completed=completed, quality_scores_read=False))
        old_custody(protocol, original, helper)
        require(sha(inside(preflight['path'])) == preflight['sha256'], 'Historical preflight changed')
    except (Exception, KeyboardInterrupt) as error:
        fatal = type(error).__name__ + ': ' + str(error)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    closure = dict(schema='Wiki12-old10-plus-never-started2-union-closure-v1',
        complete=old_verified and len(completed) == 2 and fatal is None,
        original_fixed_cells=12, original_completed_cells=10 if old_verified else None,
        old_terminal_custody=protocol['old_terminal_custody'],
        original_complete_false_closure_immutable=True, preserved_original_failed_preflight=preflight,
        supcon_terminal_custody=supcon, new_completed=completed, fatal_error=fatal,
        original_completed_cells_not_repeated=True, never_started_cells_only=True,
        automatic_retry=False, quality_scores_read=False, TEST_access=False,
        comparison_opening_authorized=False, collection_must_bind_this_union_closure=True)
    original.write(output / 'UNION_CLOSURE.json', closure)
    return 0 if closure['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
