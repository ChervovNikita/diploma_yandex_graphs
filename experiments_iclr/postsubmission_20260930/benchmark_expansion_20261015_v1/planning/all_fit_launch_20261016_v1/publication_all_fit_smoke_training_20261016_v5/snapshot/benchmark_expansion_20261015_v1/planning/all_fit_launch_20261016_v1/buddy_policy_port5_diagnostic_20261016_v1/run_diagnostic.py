"""One exact failed-cell repair smoke; never global coverage or full training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
E = HERE.parents[2]
PARENT = E / 'campaign_execution_20261016_v5'
SMOKE = E / 'qualification/all_fit_smoke_20261016_v5'
PORT = E / 'server51_campaign_port_v5'
WORKER = 'server51/gpu1'
FIT_ID = 'collab-buddy/long_convergence/split-0/seed-2/wrapper-boundary_private_half_pool/normalization-route-private-v1'
PARENT_SHA = '1960d3a07693e9813e00789a4c8eb54393f197639cd89ee66cbe91076b99184c'
PORT_SHA = '3fa47ca1ec206990fd25e37aa2f273e97aae8ec9e37c47800e9cfeda75654d13'
SMOKE_SHA = '45e9e3bff02cc8fe0517c6c820aa56170d8758425e7cafc5972a8d1c8e16f8f8'
ORIGINAL_SHA = 'ee7420e7e86af6603be01ccafbd28d498521cfdd7c5874423233de5709174d87'
sys.path.insert(0, str(PARENT))
import common as c
from worker import Worker


def verify_package(folder, expected):
    source = c.pin(folder / 'MANIFEST.json')
    c.require(source['sha256'] == expected, 'Source manifest differs: ' + str(folder))
    for row in c.read(folder / 'MANIFEST.json')['files']:
        c.verify({'path': str(folder / row['path']), **{k: row[k] for k in ('bytes', 'sha256')}}, folder)
    return source


def diagnostic_inventory(original, row, smoke_source, port_source, entry):
    c.require(row['id'] == FIT_ID and row['worker'] == WORKER, 'Only exact GPU1 failed cell allowed')
    result = copy.deepcopy(original)
    new_row = copy.deepcopy(row)
    new_row['runtime_entrypoint'] = copy.deepcopy(entry)
    result.update(source_manifest=smoke_source, producer_manifest=port_source,
        fits=[new_row], counts={WORKER: 1}, diagnostic_only=True,
        global_smoke_coverage_eligible=False, full_training_admitted=False,
        diagnostic_scope='Exact original failed descriptor with actual port5 helper-metadata repair')
    return result


class DiagnosticWorker(Worker):
    def invocation(self, task, phase, directory):
        c.require(phase == 'smoke' and task['id'] == FIT_ID, 'Diagnostic cannot launch other work')
        result = super().invocation(task, phase, directory)
        c.require(result['argv'][3] == str(c.SMOKE / 'run_one.py'), 'Original smoke CLI changed')
        result['argv'][3] = str(SMOKE / 'run_one.py')
        config = PORT / 'hosts/server51_gpu1.json'
        result['environment'].update(BENCHMARK_HOST_CONFIG=str(config),
            BENCHMARK_HOST_CONFIG_SHA256=c.pin(config)['sha256'])
        result.update(actual_scheduler_producer=self.actual_source,
            diagnostic_only=True, global_smoke_coverage_eligible=False)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', required=True)
    parser.add_argument('--parent-plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    config, config_pin = c.environment(WORKER)
    root = Path(config['project_root'])
    c.require(os.environ.get('TMUX') and HERE.is_relative_to(root), 'Actual project/tmux owner required')
    source = verify_package(HERE, args.source_sha256)
    c.require(c.source_manifest()['sha256'] == PARENT_SHA, 'Execution5 helper changed')
    port_source = verify_package(PORT, PORT_SHA)
    smoke_source = verify_package(SMOKE, SMOKE_SHA)
    parent_pin = c.pin(args.parent_plan)
    c.require(parent_pin['sha256'] == args.parent_plan_sha256, 'Original parent plan changed')
    plan = c.read(args.parent_plan)
    c.require(plan['worker'] == WORKER and plan['host_config'] == config_pin, 'Wrong original worker plan')
    original_path = HERE.parent / 'BUDDY_POLICY_FAILED_CELL_ORIGINAL.json'
    c.require(c.pin(original_path)['sha256'] == ORIGINAL_SHA, 'Frozen failure metadata changed')
    original = c.read(original_path)
    c.require(plan['inventory'] == original['original_inventory'], 'Wrong original inventory')
    old_inventory = c.read(c.verify(original['original_inventory'], root))
    row = next(r for r in old_inventory['fits'] if r['id'] == FIT_ID)
    c.require(row == original['row'], 'Original exact inventory row changed')
    job = c.read(c.verify(row['descriptor'], root))
    c.require(job == original['descriptor'] and c.pin(row['descriptor']['path']) == row['descriptor'], 'Original descriptor bytes changed')
    job_hash = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    c.require(job_hash == row['runtime_job_hash'], 'Original job identity changed')
    selected = [t for t in plan['tasks'] if t['id'] == FIT_ID]
    c.require(len(selected) == 1 and selected[0]['kind'] == 'fit' and selected[0]['descriptor'] == row['descriptor'], 'Exact original task required')
    old_state = Path(plan['state_root']) / 'smokes' / c.token(FIT_ID) / 'STATE.json'
    failed = c.read(old_state)
    c.require(failed['status'] == 'failed', 'Original failure must remain preserved')
    failure_result = c.read(c.verify(failed['native_result'], root))
    c.require(failure_result['TRAIN_updates'] == 0 and 'policy_receipt' in failure_result['failure']['message'], 'Wrong original failure')
    run = HERE / 'run'
    c.require(not run.exists(), 'Fresh diagnostic only; preserve every prior outcome')
    run.mkdir()
    entry = c.pin(PORT / 'composed/runtime/run_fit.py')
    inventory = diagnostic_inventory(old_inventory, row, smoke_source, port_source, entry)
    inventory['diagnostic_original_inventory'] = original['original_inventory']
    inventory_path = run / 'INVENTORY.json'; c.write(inventory_path, inventory)
    task = copy.deepcopy(selected[0]); task['runtime_entrypoint'] = entry
    diagnostic_plan = copy.deepcopy(plan)
    diagnostic_plan.update(inventory=c.pin(inventory_path), smoke_source_manifest=smoke_source,
        tasks=[task], state_root=str(run / 'state'), diagnostic_only=True,
        global_smoke_coverage_eligible=False, diagnostic_parent_plan=parent_pin)
    plan_path = run / 'PLAN.json'; c.write(plan_path, diagnostic_plan)
    worker_args = argparse.Namespace(worker=WORKER, plan=str(plan_path), plan_sha256=c.pin(plan_path)['sha256'],
        retry_failed=False, poll_seconds=2, smoke_wall_seconds=1800)
    worker = DiagnosticWorker(worker_args)
    worker.actual_source = {'manifest': source, 'unchanged_parent_manifest': c.source_manifest(),
        'native_smoke_source': smoke_source, 'native_port_source': port_source,
        'diagnostic_only': True, 'global_smoke_coverage_eligible': False, 'full_training_supported': False}
    owner = c.read(worker.root / 'OWNER.json'); owner['actual_scheduler_producer'] = worker.actual_source
    c.write(worker.root / 'OWNER.json', owner)
    # Nonblocking custody: root must first gracefully stop GPU1's ordinary queue
    # and wait for its children. No signals or queue mutation occur here.
    for name in ('slot_lock', 'physical_lock', 'worker_lock'):
        handle = c.lock(plan[name])
        c.require(handle is not None, 'GPU1 queue or native child still owns ' + name)
        worker.leases.append(handle)
    started = time.time()
    c.write(run / 'START.json', {'identity': c.identity(), 'producer': worker.actual_source,
        'original_failed_state': c.pin(old_state), 'original_native_failure': failed['native_result'],
        'exact_descriptor': row['descriptor'], 'TRAIN_update_cap': 2, 'child_wall_cap_seconds': 1800,
        'TEST_accessed': False, 'started': started})
    passed = worker.run_phase('smoke')
    state = worker.states[FIT_ID]
    result = {'schema': 'excluded-one-cell-port5-repair-diagnostic-v1', 'passed': passed,
        'id': FIT_ID, 'state': state, 'producer': worker.actual_source,
        'diagnostic_only': True, 'global_smoke_coverage_eligible': False,
        'scientific_scores_claimed': False, 'full_training_launched': False,
        'TEST_accessed': False, 'owner_wall_seconds': time.time() - started}
    worker.phase = 'excluded_diagnostic_finished'; worker.report(**result)
    c.write(run / 'DIAGNOSTIC_RESULT.json', result)
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
