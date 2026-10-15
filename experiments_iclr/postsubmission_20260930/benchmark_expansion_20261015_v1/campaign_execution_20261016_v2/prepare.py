"""Bind the real deployment and fixed ownership to the all-fit execution queue."""
import argparse
from pathlib import Path
from common import *


def prepare(args):
    config, host_pin = environment(args.worker)
    root = Path(config['project_root'])
    own_source = source_manifest()
    inventory_pin = pin(args.inventory)
    require(inventory_pin['sha256'] == args.inventory_sha256, 'Inventory hash differs')
    inventory = read(verify(inventory_pin, root))
    require(inventory['host'] == args.worker.split('/')[0], 'Wrong inventory host')
    require(inventory['source_closure_verified_once'] is True, 'Prepared source closure required')
    go_pin = pin(args.authorization)
    require(go_pin['sha256'] == args.authorization_sha256, 'Authorization hash differs')
    go = read(verify(go_pin, root))
    require(go['training_authorized_after_all_fit_smokes_pass'] is True and go['expected_training_fits'] == 6460,
            'Exact conditional user GO required')
    require(args.worker in go['authorized_workers'], 'Unauthorized worker')
    release = read(verify(inventory['release'], root))
    candidates = {row['id']: row for row in read(verify(inventory['candidates'], root))}
    registry = read(verify(inventory['reporting_index'], root))
    presets = {}
    for receipt in registry['preset_files']:
        preset = read(verify(receipt, root))
        for kind, field in (('fit', 'fit_specs'), ('aggregate_I4_exports', 'I4_specs')):
            for task_id, spec in preset[field].items():
                require(task_id not in presets, 'Duplicate reporting task')
                presets[task_id] = (receipt, spec)
    resources = module(RESOURCES / 'resources.py', '_allfit_resources_prepare')
    fits = {row['id']: row for row in inventory['fits'] if row['worker'] == args.worker}
    require(len(fits) == COUNTS[args.worker], 'Exact worker fit count required')
    tasks = []
    for task in release['tasks']:
        if task['worker'] != args.worker:
            continue
        candidate = candidates[task['id']]
        preset_pin, spec = presets[task['id']]
        row = dict(task)
        row['preset'] = preset_pin
        require(candidate['worker'] == task['worker'], 'Candidate ownership differs')
        if task['kind'] == 'fit':
            smoke = fits[task['id']]
            require(smoke['runtime_job_hash'] == task['runtime_job_hash'], 'Smoke/full descriptor differs')
            require(spec['descriptor'] == smoke['descriptor'] and spec['runtime_entrypoint'] == smoke['runtime_entrypoint'], 'Reporter source differs')
            row['descriptor'] = smoke['descriptor']
            row['smoke_order'] = smoke['smoke_order']
            row['runtime_entrypoint'] = smoke['runtime_entrypoint']
            job = read(verify(row['descriptor'], root))
        else:
            require(task['kind'] == 'aggregate_I4_exports', 'Unexpected campaign task')
            row['runtime_entrypoint'] = spec['aggregator_source']
            job = None
        row['reservation'] = resources.reserve_for(row, job)
        tasks.append(row)
    index = {row['id']: row for row in tasks}
    require(set(fits) == {row['id'] for row in tasks if row['kind'] == 'fit'}, 'Assigned fit set differs')
    for row in tasks:
        require(all(dep in index for dep in row.get('dependencies', [])), 'Dependency crosses fixed worker')
    actual_map = read(E / 'planning/PUBMED_THREE_GPU_ACTUAL_MAP.json')['workers'][args.worker]
    state_root = (HERE / 'runs' / args.worker.replace('/', '_')).resolve()
    output = Path(args.output).resolve()
    require(output.is_relative_to(HERE) and not output.exists(), 'Fresh own execution plan required')
    prerequisites = E / 'planning/all_fit_launch_20261016_v1/ALLOCATION_PREREQUISITES.json'
    plan = {'schema': 'all-fit-campaign-execution-plan-v1', 'worker': args.worker, 'project_root': str(root),
        'host_config': host_pin, 'inventory': inventory_pin, 'source_manifest': own_source,
        'smoke_source_manifest': inventory['source_manifest'], 'authorization': go_pin,
        'canonical_fit_ids_sha256': inventory['canonical_fit_ids_sha256'],
        'reporting_entrypoint': pin(PORT / 'reporting/runtime_entry.py'),
        'resource_source': pin(RESOURCES / 'resources.py'), 'resource_catalogue': pin(RESOURCES / 'RESERVATIONS.json'),
        'state_root': str(state_root), 'global_smoke_pass': str(HERE / 'GLOBAL_SMOKE_PASS.json'),
        'prerequisites': str(prerequisites) if args.worker == 'allocation/gpu0' else None,
        'slot_lock': str(root / actual_map['slot_lock_relative']),
        'physical_lock': str(root / config['phase_relative'] / '.expanded_dispatcher_leases' / (config['physical_gpu_uuid'] + '.lock')),
        'worker_lock': str(root / config['phase_relative'] / '.campaign_worker_leases' / (args.worker.replace('/', '_') + '.lock')),
        'host_ledger': str(HERE / 'runs' / 'host_ledger'), 'tasks': tasks, 'TEST_access': False,
        'smoke_updates': 2, 'full_training_uses_smoke_checkpoint': False}
    write(output, plan)
    return {'plan': pin(output), 'fits': len(fits), 'tasks': len(tasks), 'state_root': str(state_root)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('worker', 'inventory', 'inventory-sha256', 'authorization', 'authorization-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    print(json.dumps(prepare(p.parse_args()), sort_keys=True))
