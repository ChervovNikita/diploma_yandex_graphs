"""Allocation-only metadata successor: bind portable runtime candidates, zero children.

This prepares a candidate source closure and a fully blocked release. Root must
attach actual existing qualification/resource/GO authorities in a later release.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
from host_binding import HERE, config_path, guard, load_config, pin, require
from descriptor_binding import audit_runtime_paths


def main():
    config, config_pin = load_config(config_path('allocation/gpu0'))
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)
    root = Path(config['project_root'])
    E = root / config['phase_relative'] / 'benchmark_expansion_20261015_v1'
    previous = E / 'expanded_dispatcher_v2'
    sys.path.insert(0, str(previous))
    import contracts
    require(Path(contracts.__file__).resolve() == previous / 'contracts.py', 'Unexpected dispatcher module')
    summary_pin = pin(HERE / 'PARENT_GENERATION.json')
    require(summary_pin['sha256'] == '09a9da861c6d3d8204823bca71fc7031f90c5cf6ee1402e88509d9d438805662', 'Compiler3 source summary changed')
    summary = json.loads(Path(summary_pin['path']).read_text())
    for name in ('skeleton', 'bindings', 'execution_candidates'):
        contracts.verify(summary[name])
    skeleton_pin = summary['skeleton']
    skeleton = json.loads(Path(skeleton_pin['path']).read_text())
    bindings = json.loads(Path(summary['bindings']['path']).read_text())
    candidates = json.loads(Path(summary['execution_candidates']['path']).read_text())
    require(len(skeleton['tasks']) == 6730 and len(candidates['candidates']) == 6730, 'Keep exact inventory')
    require(all(t['worker'] == 'allocation/gpu0' for t in skeleton['tasks']), 'Allocation successor cannot reassign tasks')
    require(not bindings['tasks'], 'Preparation never copies execution admission')
    descriptor_count = 0
    for task in skeleton['tasks']:
        if task['kind'] != 'fit':
            continue
        contracts.verify(task['source_descriptor'])
        job = json.loads(Path(task['source_descriptor']['path']).read_text())
        require(contracts.digest(job) == task['runtime_job_hash'], 'Canonical runtime job changed')
        require(contracts.digest(job['expanded_contract']) == task['science_contract_hash'], 'Scientific contract changed')
        audit_runtime_paths(job, config['project_root'], config['phase_relative'])
        descriptor_count += 1
    require(descriptor_count == 6460, 'Audit every descriptor before producing allocation candidates')
    ids = {t['id'] for t in skeleton['tasks']}
    for claim_path in (E / 'campaign_outputs').glob('*/control/tasks/*/CLAIM.json'):
        claim = json.loads(claim_path.read_text())
        require(claim.get('id') not in ids, 'An existing permanent campaign claim forbids a fresh namespace: ' + str(claim_path))
    package = json.loads((HERE / 'MANIFEST.json').read_text())
    closure = {r['path']: r for r in bindings['source_files']}
    for record in package['files']:
        target = HERE / record['path']
        actual = pin(target)
        require(actual['sha256'] == record['sha256'] and actual['bytes'] == record['bytes'], 'Portable source changed')
        closure[actual['path']] = actual
    package_pin = pin(HERE / 'MANIFEST.json')
    closure[package_pin['path']] = package_pin
    runtime_by_family = {'node': 'runtime_node_molecular/run_fit.py', 'ogb': 'runtime_node_molecular/run_fit.py',
                         'link': 'runtime_link/run_fit.py', 'I4': 'runtime_I4/aggregate_selected.py'}
    tasks = {t['id']: t for t in skeleton['tasks']}
    for candidate in candidates['candidates']:
        task = tasks[candidate['id']]
        before = copy.deepcopy(candidate)
        family = task['family'] if task['kind'] == 'fit' else 'I4'
        candidate['execution']['runtime_entrypoint'] = pin(HERE / runtime_by_family[family])
        require(candidate['execution']['job_descriptor'] == before['execution']['job_descriptor'], 'Do not rewrite allocation jobs')
        require(candidate['science_contract_hash'] == before['science_contract_hash'] and candidate['runtime_job_hash'] == before['runtime_job_hash'], 'Keep all scientific/job identities')
        candidate['admitted'] = False
        candidate['portable_host_guard_delta_requires_root_source_review'] = True
    bindings['source_files'] = list(closure.values())
    output = HERE / 'generated_allocation_v1'
    require(not output.exists(), 'Preserve prior metadata generation')
    output.mkdir()
    def write(name, value):
        path = output / name
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
        return pin(path)
    candidate_pin = write('EXECUTION_CANDIDATES.json', candidates)
    binding_pin = write('BINDINGS.json', bindings)
    result = contracts.compile_release(skeleton_pin, binding_pin['path'], output / 'RELEASE.json',
        E / 'campaign_outputs/portable_allocation_v1_REQUIRES_REVIEW')
    write('PREPARATION.json', {'schema': 'portable-allocation-metadata-preparation-v1', 'parent_summary': summary_pin,
        'candidates': candidate_pin, 'bindings': binding_pin, 'release': result, 'host_config': config_pin,
        'same_6460_scientific_and_runtime_job_hashes': True, 'same_270_I4_dependency_sets': True,
        'same_permanent_worker_ownership': True, 'native_jobs_started': 0, 'human_GO': False, 'TEST_accessed': False})
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
