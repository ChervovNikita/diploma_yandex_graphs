"""One reviewed command per physical GPU; disabled hosts exit before any probe."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from host_binding import config_path, load_config, guard, require, confined, verify, WORKERS


def reporting_source_authority(root, phase_relative, reporting, registry, closure):
    """Check the actual current reporter/scorer manifests without launching work."""
    expected = Path(root) / phase_relative / 'benchmark_expansion_20261015_v1/server51_campaign_port_v5/reporting'
    require(expected == Path(__file__).resolve().parent / 'reporting', 'Exact current reporting deployment required')
    require(reporting['bridge']['path'] == str(expected / 'backend_bridge.py')
            and reporting['entrypoint']['path'] == str(expected / 'runtime_entry.py'),
            'Fixed successor reporter required')
    require(reporting['manifest']['path'] == str(expected / 'MANIFEST.json'), 'Exact successor reporting manifest required')
    manifest = verify(reporting['manifest'], root)
    require(manifest['schema'] == 'allocation-server51-reporting-source-v5', 'Wrong reporting source schema')
    for row in manifest['files']:
        require(closure.get(str(expected / row['path'])) == row['sha256'], 'Reporting dependency absent from source closure')
    scorer = registry['selected_serving_source']
    expected_scorer = expected.parent / 'scoring/MANIFEST.json'
    require(scorer['path'] == str(expected_scorer), 'Exact current successor scorer required')
    scoring_manifest = verify(scorer, root)
    require(scoring_manifest['schema'] == 'server51-composed-final-scoring-source-v5', 'Wrong successor scorer schema')
    require(closure.get(scorer['path']) == scorer['sha256'], 'Scorer source absent from closure')
    require(scoring_manifest['parent_source_manifest']['sha256'] == '683bd900f1e1870d701b53457491a63d7e46f31b9ab1847a1e96e9227a545234',
            'Keep exact port3 scorer ancestry')
    return scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=WORKERS, required=True)
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
    require(spec.get('reporting') is not None, 'Normalization campaign requires the exact rebound reporting hook')
    require(spec.get('reporting') == release.get('reporting'), 'Reporting hook must be bound in reviewed release')
    if spec.get('reporting'):
        import importlib.util
        reporting = spec['reporting']
        require(set(reporting) == {'bridge', 'entrypoint', 'registry', 'manifest'}, 'Exact reporting closure binding required')
        closure={p['path']:p['sha256'] for p in release['source_files']}
        for value in reporting.values():
            verify(value,root,parse_json=False)
            require(closure.get(value['path'])==value['sha256'],'Reporting source/registry absent from reviewed closure')
            extra.append(value)
        registry=read(reporting['registry']['path']);marker=registry.get('host_projection',{})
        require(marker.get('schema')=='allocation-server51-reporting-registry-projection-v1' and marker.get('host')==release['host'],'Wrong reporting projection host')
        require(marker.get('canonical_registry')==release['server51_successor']['canonical_registry_origin'] and marker['canonical_registry']['sha256']=='cd2bc64047e630452eb7b4ce13a0829b205327876c5183d7e28853f6d4445db6','Canonical reporting identity changed')
        scorer = reporting_source_authority(root, config['phase_relative'], reporting, registry, closure)
        extra.append(scorer)
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
