"""One combined command: existing TRAIN → offline barrier → existing final.

No scientific work is implemented here. Existing supervisors retain native
GPU scheduling, reservations, STOP/drain and result authority. Preparation
ships no production-valid run spec or GO; final source readiness is pending.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys


def main():
    here = Path(__file__).resolve().parent
    expansion = here.parent
    sys.path.insert(0, str(expansion / 'portable_dispatcher_v3'))
    from host_binding import config_path, guard, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, choices=('allocation/gpu0', 'server77/gpu0', 'server77/gpu1'))
    parser.add_argument('--run-spec', required=True); parser.add_argument('--run-spec-sha256', required=True)
    args = parser.parse_args()
    config, config_pin = load_config(config_path(args.worker))
    if not config['host_enabled']:
        raise ValueError('Host disabled before probes; existing route reauthorization/qualification required')
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)
    sys.path.insert(0, str(expansion))
    from final_scoring_integration_v2.common import digest, pin, require, verify
    from final_scoring_integration_v2.offline_seal import build_host_shard
    from combined_campaign_orchestration_v3.contracts import derive_final_spec, validate_actual, validate_delivery, validate_host_templates
    from combined_campaign_orchestration_v3.stages import (confirm_worker_terminal, exclusive, lock_worker,
        run_stage, same_or_create, wait_for_files)
    from combined_campaign_orchestration_v3.local_ready import ensure_local_ready
    from qualified_python_entry import validate_runtime_environment
    from backend import identity
    root = Path(config['project_root'])
    spec_pin = pin(args.run_spec, root)
    require(spec_pin['sha256'] == args.run_spec_sha256, 'Combined run spec changed')
    spec = verify(spec_pin, root, json_metadata=True)
    require(spec['worker'] == args.worker and spec['host_config'] == config_pin, 'Wrong actual worker/config')
    template = verify(spec['execution_template'], root, json_metadata=True)
    training_spec = verify(spec['training_run_spec'], root, json_metadata=True)
    go = verify(spec['training_go'], root, json_metadata=True)
    validate_actual(spec, template, go, training_spec, root)
    validate_host_templates(template, {worker: verify(value, root, json_metadata=True)
        for worker, value in go['combined_execution']['execution_templates'].items()})
    verify(go['human_instruction_evidence'], root)
    decision = verify(spec['global_decision'], root, json_metadata=True)
    contract = verify(spec['global_contract'], root, json_metadata=True)
    require(decision['schema'] == 'portable-global-human-decision-v1' and decision['user_explicit_GO'] is True
            and decision['human_instruction_evidence'] == go['human_instruction_evidence']
            and decision['global_final_contract_sha256'] == digest(contract)
            and go['parent_global_decision_sha256'] == digest(decision), 'One exact parent campaign decision required')
    manifest = verify(spec['source_manifest'], root, json_metadata=True)
    require(spec['source_manifest']['path'] == str(here / 'MANIFEST.json')
            and manifest['schema'] == 'combined-campaign-orchestration-source-v3', 'Exact sequencer source closure required')
    for row in manifest['files']:
        verify({'path': str(here / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    parents = json.loads((here / 'SOURCE_BINDINGS.json').read_text())
    for row in parents['files']:
        verify({'path': str(expansion / row['path_relative_to_expansion']), 'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    for value in (template['training_entry'], template['final_entry'], template['scorer_source_manifest'], template['final_integration_manifest']):
        verify(value, root)
    final_manifest = verify(template['final_integration_manifest'], root, json_metadata=True)
    require(template['final_integration_manifest']['path'] == str(Path(template['final_entry']['path']).parent / 'MANIFEST.json')
            and any(row['path'] == 'run_gpu.py' and row['sha256'] == template['final_entry']['sha256'] for row in final_manifest['files']), 'Final entry absent from same GO-bound integration closure')
    environment = verify(template['supervisor_python_environment'], root, json_metadata=True)
    validate_runtime_environment(environment)
    # The sequencer ignores terminal hangup; supervisors additionally start new
    # sessions, use file logs and /dev/null stdin. No live Mac is required.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    control = Path(template['orchestration_output_root']) / args.worker.replace('/', '_')
    control.mkdir(parents=True, exist_ok=True)
    lease = lock_worker(control / 'SEQUENCER.lock')
    stops = root / config['phase_relative'] / '.portable_campaign_host_v2/stops'
    claim_path = control / 'CLAIM.json'
    if claim_path.exists():
        claim = verify(pin(claim_path, root), root, json_metadata=True)
        require(claim['run_spec'] == spec_pin, 'Combined worker cannot change its permanent run spec')
    else:
        claim = {'schema': 'combined-permanent-worker-claim-v1', 'worker': args.worker, 'run_spec': spec_pin,
                 'baseline_STOPs': [pin(path, root) for path in sorted(stops.glob('*.json'))], 'automatic_retries': 0}
        exclusive(claim_path, claim)
    def check_stop():
        for value in claim['baseline_STOPs']:
            verify(value, root)
        require({str(path) for path in stops.glob('*.json')} == {value['path'] for value in claim['baseline_STOPs']},
                'A new retained host STOP halts the combined sequencer; original review/continuation policy applies')
    try:
        train_return = run_stage('TRAIN', control, sys.executable, template['training_entry'], args.worker,
                                 spec['training_run_spec'], root, identity)
        training = verify(spec['training_release'], root, json_metadata=True)
        ack = confirm_worker_terminal(spec['training_release'], training, args.worker, root)
        ack.update(GO_view=spec['training_go'], global_decision=spec['global_decision'], supervisor_return=train_return)
        sync = Path(template['host_sync_root'])
        (sync / 'workers').mkdir(parents=True, exist_ok=True)
        own_ack = sync / 'workers' / (args.worker.replace('/', '_') + '.json')
        same_or_create(own_ack, ack, root)
        local_workers = sorted({task['worker'] for task in training['tasks']})
        require(all(worker.split('/')[0] == config['worker'].split('/')[0] for worker in local_workers), 'Training release contains foreign host workers')
        ack_paths = [sync / 'workers' / (worker.replace('/', '_') + '.json') for worker in local_workers]
        wait_for_files(ack_paths, check_stop)
        shard_path = sync / 'HOST_TERMINAL_SHARD.json'
        if args.worker == local_workers[0]:
            all_states, acknowledgments = {}, {}
            for worker, path in zip(local_workers, ack_paths):
                local_ack = verify(pin(path, root), root, json_metadata=True)
                require(local_ack['worker'] == worker and local_ack['training_release'] == spec['training_release']
                        and local_ack['GO_view'] == spec['training_go'] and local_ack['global_decision'] == spec['global_decision'], 'Local terminal ack belongs to a changed campaign')
                require(not (set(all_states) & set(local_ack['states'])), 'Local workers duplicated a training task')
                all_states.update(local_ack['states']); acknowledgments[worker] = pin(path, root)
            ownership = verify(training_spec['ownership'], root, json_metadata=True)
            expected = {'host': training['host'], 'training_release': spec['training_release'],
                'scoring_roster': template['scoring_roster'], 'state_pins': all_states,
                'training_ownership_sha256': digest(ownership)}
            ensure_local_ready(shard_path, sync / 'LOCAL_READY.json', expected, acknowledgments,
                spec['global_contract'], root,
                lambda: build_host_shard(spec['training_release'], template['scoring_roster'], all_states, ownership, root))
        wait_for_files([shard_path, sync / 'LOCAL_READY.json'], check_stop)
        same_or_create(control / 'WAITING_OFFLINE_SYNC.json', {'schema': 'combined-offline-barrier-wait-v1',
            'local_host_shard': pin(shard_path, root), 'expected_delivery': template['offline_delivery_path'],
            'global_contract': spec['global_contract'], 'new_user_action_required': False,
            'meaning': 'exchange all immutable host shards and derive local global seal; no TEST before final authority'}, root)
        delivery_path = Path(template['offline_delivery_path'])
        wait_for_files([delivery_path], check_stop)
        delivery = verify(pin(delivery_path, root), root, json_metadata=True)
        validate_delivery(delivery, spec, template, root)
        final_spec = derive_final_spec(template, spec, delivery)
        final_path = control / 'FINAL_RUN_SPEC.json'
        final_pin = same_or_create(final_path, final_spec, root)
        same_or_create(control / 'OFFLINE_DELIVERY_CONSUMED.json', {'delivery': pin(delivery_path, root),
            'derived_final_run_spec': final_pin, 'new_human_action_claimed': False}, root)
        check_stop()
        final_return = run_stage('FINAL', control, sys.executable, template['final_entry'], args.worker,
                                 final_pin, root, identity)
        same_or_create(control / 'COMPLETE.json', {'schema': 'combined-worker-completion-v1', 'worker': args.worker,
            'TRAIN_supervisor_return': train_return, 'FINAL_supervisor_return': final_return,
            'scientific_outcomes_authority': 'actual native and final RESULTs, not this stage receipt',
            'automatic_retries': 0, 'new_user_action_claimed': False}, root)
    finally:
        lease.close()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error), 'automatic_retries': 0}))
        raise SystemExit(2)
