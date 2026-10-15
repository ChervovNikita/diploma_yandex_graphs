"""One reviewed command per physical GPU; disabled hosts exit before any probe."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from host_binding import config_path, load_config, guard, require, confined, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'), required=True)
    parser.add_argument('--run-spec', required=True)
    parser.add_argument('--run-spec-sha256', required=True)
    args = parser.parse_args()
    config, config_pin = load_config(config_path(args.worker))
    require(config['host_enabled'], 'Host disabled before probes; explicit reauthorization and host qualification required')
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)
    root = Path(config['project_root'])
    spec_pin = {'path': str(confined(args.run_spec, root)), 'sha256': args.run_spec_sha256}
    spec = verify(spec_pin, root)
    require(spec['schema'] == 'portable-reviewed-run-v2' and spec['worker'] == args.worker and spec['host_config'] == config_pin, 'Exact run-spec worker/config binding required')
    from contracts import authority, pin, read, verify as verify_local
    from backend import PortableBackend
    from engine import Engine
    release, release_pin, ownership, budget, recheck = authority(spec, args.worker)
    extra = [spec_pin]
    dispositions = {}
    for field, schema in (('resumes', 'expanded-reviewed-resume-map-v1'), ('reconciliations', 'expanded-reviewed-reconciliation-map-v1')):
        dispositions[field] = {}
        if spec.get(field):
            value = verify(spec[field], root)
            require(value['schema'] == schema and value['manifest_sha256'] == release_pin['sha256'], 'Wrong reviewed disposition map')
            require(set(value['tasks']).issubset({t['id'] for t in release['tasks'] if t['worker'] == args.worker}), 'Foreign disposition IDs')
            dispositions[field] = value['tasks']
            extra.append(spec[field])
    continuation = spec.get('continuation')
    if continuation:
        verify(continuation, root)
        extra.append(continuation)
    def full_recheck(task):
        recheck(task)
        for value in extra:
            verify(value, root, parse_json=False)
    # Forbid a second output namespace even if a prior engine claimed the same IDs.
    own_ids = {r['id'] for r in ownership['assignments'] if r['worker'].split('/')[0] == release['host']}
    output = Path(release['output_root'])
    for claim in output.parent.glob('*/control/tasks/*/CLAIM.json'):
        if claim.is_relative_to(output):
            continue
        require(read(claim).get('id') not in own_ids, 'Prior permanent claim in another namespace')
    transform = reporting_entry = None
    require(spec.get('reporting') == release.get('reporting'), 'Reporting hook must be bound in reviewed release')
    if spec.get('reporting'):
        import importlib.util
        reporting = spec['reporting']
        require(set(reporting) == {'bridge', 'entrypoint', 'registry', 'manifest'}, 'Exact reporting closure binding required')
        require(args.worker == 'allocation/gpu0', 'Reporting observer is currently allocation-only')
        expected = root / config['phase_relative'] / 'benchmark_expansion_20261015_v1/benchmark_reporting_integration_v2'
        require(reporting['bridge']['path'] == str(expected / 'backend_bridge.py') and reporting['entrypoint']['path'] == str(expected / 'runtime_entry.py'), 'Fixed reviewed reporting integration required')
        closure = {p['path']: p['sha256'] for p in release['source_files']}
        for value in reporting.values():
            verify(value, root, parse_json=False)
            require(closure.get(value['path']) == value['sha256'], 'Reporting source/registry missing from reviewed closure')
            extra.append(value)
        require(reporting['manifest']['path'] == str(expected / 'MANIFEST.json'), 'Fixed reporting manifest required')
        for row in read(reporting['manifest']['path'])['files']:
            source_path = str(root / config['phase_relative'] / row['path_relative_to_phase'])
            require(closure.get(source_path) == row['sha256'], 'Reporting dependency absent from release closure')
        module_spec = importlib.util.spec_from_file_location('_portable_reporting_bridge', reporting['bridge']['path'])
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        transform = module.ObservedArgvTransform(reporting['registry']['path'], reporting['registry']['sha256'], root)
        reporting_entry = reporting['entrypoint']
    backend = PortableBackend(config, ownership, budget, full_recheck, verify_local, transform, reporting_entry)
    backend.output_root = output
    if spec.get('ledger_reconciliations'):
        from backend import identity_status
        for value in spec['ledger_reconciliations']:
            backend.ledger.reconcile(value, verify_local, identity_status)
            extra.append(value)
    phase = root / config['phase_relative']
    leases = [phase / '.expanded_dispatcher_leases' / (config['physical_gpu_uuid'] + '.lock'),
              phase / '.campaign_worker_leases' / (args.worker.replace('/', '_') + '.lock')]
    tasks = [t for t in release['tasks'] if t['worker'] == args.worker]
    result = Engine(tasks, output, release_pin['sha256'], args.worker, backend, leases,
                    resumes=dispositions['resumes'], continuation=continuation,
                    reconciliations=dispositions['reconciliations'], allowed_root=root).run()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['status'] in ('complete', 'complete_with_failed_outcomes') else 2)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(exc), 'children_started': 0}))
        raise SystemExit(2)
