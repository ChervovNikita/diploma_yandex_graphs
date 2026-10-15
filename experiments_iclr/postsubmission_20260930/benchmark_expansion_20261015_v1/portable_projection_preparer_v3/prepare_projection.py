"""Guarded finite compiler4 -> immutable owner and unadmitted host projections.

Default owner is allocation for every task. Optional exact group placement may
prepare future 77 metadata, but cannot enable it or invent any qualification.
All fields are checked/relocated explicitly; canonical scientific JOBs remain
unchanged. This preparer opens JSON and source bytes only, never tensors.
Metadata-loop successor: use each published host configuration pin explicitly.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import sys
PREPARER_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PREPARER_HERE.parent / 'portable_dispatcher_v2'))
from custody import digest, pinned, read, require, write
from descriptor_binding import rebind_job
from host_binding import (HERE, ALLOCATION_ROOT, PHASE_RELATIVE, WORKERS, config_path,
                          load_config, guard, pin, verify)
from ownership import CAMPAIGN, validate

PARENT_MANIFEST_SHA256 = '158dbd047d0d929d68e4930f5544f9bc071f8d06c973158a1f5caaeaf0f22ade'
ALLOCATION_BOOTSTRAP_SHA256 = '009f2ae98a0bdee33414b219ad3c52e340cb409c0dd0745b79da327517bbabc6'


def metadata_host_configs(package):
    """Read only bundled metadata with its own published pin; keep execution env."""
    records = {row['path']: row for row in package['files']}
    configs = {}
    for worker in WORKERS:
        path = config_path(worker)
        expected = records[str(path.relative_to(HERE))]
        config, actual = load_config(path, expected_sha256=expected['sha256'])
        require(actual['bytes'] == expected['bytes'], 'Host metadata length changed')
        configs[worker] = config
    return configs


def relocate_path(path, host):
    source = PurePosixPath(ALLOCATION_ROOT)
    phase = source / PHASE_RELATIVE
    value = PurePosixPath(path)
    require(value.is_relative_to(source) and '..' not in value.parts, 'Canonical path outside allocation project')
    if value.is_relative_to(phase):
        return str(PurePosixPath(host['project_root']) / host['phase_relative'] / value.relative_to(phase))
    return str(PurePosixPath(host['project_root']) / value.relative_to(source))


def relocate_pin(value, host):
    return dict(value, path=relocate_path(value['path'], host))


def project_task(task, worker, host, original):
    value = copy.deepcopy(task)
    value['worker'] = worker
    value['canonical_runtime_job_hash'] = task.get('runtime_job_hash')
    value['runtime_binding'] = None
    if task['kind'] == 'fit':
        require(digest(original) == task['runtime_job_hash'], 'Canonical JOB hash differs')
        bound, receipt = rebind_job(original, ALLOCATION_ROOT, PHASE_RELATIVE, host['project_root'], host['phase_relative'])
        value['source_descriptor'] = relocate_pin(task['source_descriptor'], host)
        value['runtime_job_hash'] = digest(bound)
        value['runtime_binding'] = receipt
    return value


def build_ownership(tasks, hosts, placement=None):
    groups = {t['group_id'] for t in tasks}
    placement = placement or {g: 'allocation/gpu0' for g in groups}
    require(set(placement) == groups and set(placement.values()).issubset(WORKERS), 'Exact complete cohort placement required')
    value = {'schema': 'portable-global-ownership-v2', 'campaign': CAMPAIGN,
             'stop_issuer': 'reviewed_campaign_operator', 'hosts': hosts,
             'assignments': [{'id': t['id'], 'worker': placement[t['group_id']],
                              'canonical_runtime_job_hash': t.get('runtime_job_hash'),
                              'science_contract_hash': t['science_contract_hash'], 'dependencies': t['dependencies'],
                              'output_relative': t['output_relative'], 'cohort': t['group_id']} for t in tasks]}
    validate(value, tasks)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--placement', help='Optional reviewed group_to_worker JSON; no native admission')
    parser.add_argument('--placement-sha256')
    args = parser.parse_args()
    config, config_pin = load_config(config_path('allocation/gpu0'), expected_sha256=ALLOCATION_BOOTSTRAP_SHA256)
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)
    require(PREPARER_HERE == HERE.parent / 'portable_projection_preparer_v3', 'Use the canonical successor source copy')
    package = pinned(HERE / 'MANIFEST.json', PARENT_MANIFEST_SHA256)
    parents = read(HERE / 'PARENT_PINS.json')
    summary = pinned(HERE / 'PARENT_GENERATION.json', parents['parent_summary_sha256'])
    require(summary['compiler_readiness']['sha256'] == parents['compiler4_readiness_sha256'], 'Use compiler4')
    require(summary['runtime_link_v9_manifest']['sha256'] == parents['runtime_link_v9_manifest_sha256'], 'Use runtime v9')
    for name in ('compiler_readiness', 'runtime_link_v9_manifest', 'skeleton', 'bindings', 'execution_candidates'):
        verify(summary[name], config['project_root'])
    readiness = read(summary['compiler_readiness']['path'])
    require(readiness['artifacts']['index']['sha256'] == parents['compiler4_full_index_sha256'], 'Wrong full compiler4 index')
    skeleton = read(summary['skeleton']['path'])
    bindings = read(summary['bindings']['path'])
    require(not bindings['tasks'], 'Never transfer old execution admission')
    tasks = skeleton['tasks']
    configs = metadata_host_configs(package)
    hosts = {h: {'host': h, 'project_root': configs[h+'/gpu0']['project_root'],
                 'phase_relative': configs[h+'/gpu0']['phase_relative'], 'maximum_children_per_GPU': 2}
             for h in ('allocation', 'server77')}
    placement = pinned(args.placement, args.placement_sha256)['group_to_worker'] if args.placement else None
    ownership = build_ownership(tasks, hosts, placement)
    owners = {r['id']: r['worker'] for r in ownership['assignments']}
    output = HERE / 'generated_v3_config_pin'
    require(not output.exists(), 'Preserve prior generation')
    # Host execution will perform this check again, including on77 after reauthorization.
    campaign_outputs = HERE.parent / 'campaign_outputs'
    ids = set(owners)
    for claim in campaign_outputs.glob('*/control/tasks/*/CLAIM.json'):
        require(read(claim).get('id') not in ids, 'Existing claim forbids a fresh ownership/output namespace')
    jobs = {}
    for task in tasks:
        if task['kind'] == 'fit':
            verify(task['source_descriptor'], config['project_root'])
            jobs[task['id']] = read(task['source_descriptor']['path'])
            require(digest(jobs[task['id']]) == task['runtime_job_hash'], 'Canonical JOB hash differs')
    package_sources = []
    for row in package['files']:
        value = pin(HERE / row['path'])
        require(value['sha256'] == row['sha256'] and value['bytes'] == row['bytes'], 'Portable package changed')
        package_sources.append(value)
    package_sources.append(pin(HERE / 'MANIFEST.json'))
    successor = read(PREPARER_HERE / 'MANIFEST.json')
    for row in successor['files']:
        value = pin(PREPARER_HERE / row['path'])
        require(value['sha256'] == row['sha256'] and value['bytes'] == row['bytes'], 'Preparer successor changed')
        package_sources.append(value)
    package_sources.append(pin(PREPARER_HERE / 'MANIFEST.json'))
    output.mkdir()
    write(output / 'OWNERSHIP.json', ownership)
    owner_pin = pin(output / 'OWNERSHIP.json')
    for host, host_config in hosts.items():
        folder = output / host
        folder.mkdir()
        with (folder / 'CANONICAL_PLAN.json').open('xb') as handle:
            handle.write(Path(summary['skeleton']['path']).read_bytes())
        projected, candidates = [], []
        for task in tasks:
            worker = owners[task['id']]
            if worker.split('/')[0] != host:
                continue
            value = project_task(task, worker, host_config, jobs.get(task['id']))
            value.update(binding=None, execution_status='blocked_missing_qualification')
            projected.append(value)
            family = task.get('family') if task['kind'] == 'fit' else 'I4'
            runtime = 'runtime_link/run_fit.py' if family == 'link' else 'runtime_I4/aggregate_selected.py' if family == 'I4' else 'runtime_node_molecular/run_fit.py'
            job_pin = None
            if task['kind'] == 'fit':
                bound, _ = rebind_job(jobs[task['id']], ALLOCATION_ROOT, PHASE_RELATIVE, host_config['project_root'], host_config['phase_relative'])
                job_dir = folder / 'jobs'
                job_dir.mkdir(exist_ok=True)
                job_path = job_dir / (digest(task['id']) + '.json')
                write(job_path, bound)
                job_pin = relocate_pin(pin(job_path), host_config)
            candidates.append({'id': task['id'], 'worker': worker, 'job_descriptor': job_pin,
                               'runtime_entrypoint': relocate_pin(pin(HERE / runtime), host_config),
                               'canonical_runtime_job_hash': value['canonical_runtime_job_hash'],
                               'runtime_job_hash': value['runtime_job_hash'], 'admitted': False})
        closure = {v['path']: v for v in bindings['source_files'] + package_sources}
        sources = [relocate_pin(v, host_config) for v in closure.values()]
        release = {'schema': 'portable-host-production-release-v2', 'host': host,
                   'canonical_skeleton': relocate_pin(pin(folder / 'CANONICAL_PLAN.json'), host_config),
                   'ownership': relocate_pin(owner_pin, host_config), 'tasks': projected,
                   'host_config_pins': {w: relocate_pin(pin(config_path(w)), host_config) for w in WORKERS if w.startswith(host+'/')},
                   'source_files': sources, 'source_bundle_sha256': digest({v['path']: v['sha256'] for v in sources}),
                   'legacy_supervisor_sources': {k: relocate_pin(v, host_config) for k, v in skeleton['legacy_supervisor_sources'].items()},
                   'output_root': str(Path(host_config['project_root']) / host_config['phase_relative'] / 'benchmark_expansion_20261015_v1/campaign_outputs' / CAMPAIGN),
                   'all_tasks_qualified': False, 'human_GO': False,
                   'ordinary_failure_policy': 'retain_failure_continue_other_admissible_tasks', 'TEST_access': False}
        write(folder / 'RELEASE_UNADMITTED.json', release)
        write(folder / 'CANDIDATES.json', candidates)
    write(output / 'PREPARATION.json', {'schema': 'portable-host-projection-preparation-v3',
          'preparer_source': pin(__file__), 'preparer_manifest': pin(PREPARER_HERE / 'MANIFEST.json'),
          'ownership': owner_pin, 'parent_summary': pin(HERE / 'PARENT_GENERATION.json'),
          'tasks': 6730, 'fits': 6460, 'I4': 270, 'native_jobs_started': 0,
          'server77_enabled': False, 'destination_data_or_resource_qualification': False, 'human_GO': False})
    print(json.dumps(pin(output / 'PREPARATION.json')))


if __name__ == '__main__':
    main()
