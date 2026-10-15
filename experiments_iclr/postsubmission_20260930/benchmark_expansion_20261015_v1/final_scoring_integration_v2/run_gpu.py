"""One final-scoring command per physical GPU; original guards and custody."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    expansion = Path(__file__).resolve().parent.parent
    portable_path = expansion / 'portable_dispatcher_v3'
    sys.path.insert(0, str(portable_path))
    from host_binding import config_path, guard, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, choices=('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'))
    parser.add_argument('--run-spec', required=True); parser.add_argument('--run-spec-sha256', required=True)
    args = parser.parse_args()
    config, config_pin = load_config(config_path(args.worker))
    if not config['host_enabled']:
        raise ValueError('Host disabled before probes; reauthorization and qualification required')
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)
    sys.path.insert(0, str(expansion))
    from final_scoring_integration_v2.common import digest, pin, require, verify
    from final_scoring_integration_v2.authority import authority
    from final_scoring_integration_v2.backend import backend_class
    from final_scoring_integration_v2.engine import Engine
    from final_scoring_integration_v2.phase_ledger import ledger_class
    from final_scoring_integration_v2.preparation import prepare_worker
    import backend as portable_backend
    import contracts as training_contracts
    import custody
    from host_ledger import Ledger
    from ownership import claim_host
    root = Path(config['project_root'])
    spec_pin = pin(args.run_spec, root)
    require(spec_pin['sha256'] == args.run_spec_sha256, 'Pinned final run spec changed')
    spec = verify(spec_pin, root, json_metadata=True)
    release, ownership, budget, recheck, parents = authority(spec, args.worker, training_contracts, config)
    train, training_owner, training_spec, seal, go = parents
    require(spec['training_go'] == training_spec['go'], 'Training+scoring must use identical local GO view')
    roster = verify(release['roster'], root, json_metadata=True)
    index = verify(roster['registry_index'], root, json_metadata=True)
    profiles = {verify(value, root, json_metadata=True)['registry']['presets'][0]['id']: value for value in index['preset_files']}
    phase = root / config['phase_relative']
    def verify_terminal_seal():
        verify(spec['training_seal'], root, json_metadata=True)
        for task in train['tasks']:
            for field in ('state', 'terminal', 'training_result'):
                value = seal['outcomes'][task['id']][field]
                if value:
                    verify(value, root, json_metadata=True)
    handoff = {'training_release': release['training_release'], 'final_release': spec['release'],
               'training_seal': spec['training_seal'], 'GO_view': spec['training_go'],
               'global_decision': spec['global_decision'], 'global_contract': spec['global_contract']}
    ledger = ledger_class(Ledger, custody, claim_host)(phase / '.portable_campaign_host_v2', ownership,
        release['host'], training_ownership=training_owner, handoff=handoff, verify_terminal_seal=verify_terminal_seal)
    preparation_pin = prepare_worker(spec_pin, spec, release, train, ownership, config_pin, root)
    backend = backend_class(portable_backend)(config, ownership, budget, recheck, ledger, spec, spec_pin, release, seal, go, profiles,
        {row['id']: row for row in train['tasks']}, preparation_pin)
    leases = [phase / '.expanded_dispatcher_leases' / (config['physical_gpu_uuid'] + '.lock'),
              phase / '.campaign_worker_leases' / (args.worker.replace('/', '_') + '.lock')]
    tasks = [task for task in release['tasks'] if task['worker'] == args.worker]
    result = Engine(tasks, release['output_root'], spec['release']['sha256'], args.worker, backend, leases,
                    continuation=spec.get('continuation'), allowed_root=root).run()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['status'] in ('complete', 'complete_with_failed_outcomes') else 2)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error), 'children_started': 0}))
        raise SystemExit(2)
