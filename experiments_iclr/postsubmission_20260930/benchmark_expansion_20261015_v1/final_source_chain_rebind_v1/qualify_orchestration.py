"""Bounded real allocation launcher qualification; no scientific execution.

Source preparation does not execute this file. Root may later run it with
--execute-engineering and a fresh project output directory. It launches only
three explicitly metadata-only fixture supervisors and two refusing CLIs.
"""
import argparse
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    here = Path(__file__).resolve().parent
    expansion = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-engineering', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not args.execute_engineering:
        raise ValueError('Explicit engineering invocation required; this is not campaign GO')
    sys.path.insert(0, str(expansion / 'portable_dispatcher_v3'))
    from host_binding import config_path, guard, load_config
    config, config_pin = load_config(config_path('allocation/gpu0'))
    os.environ['BENCHMARK_HOST_CONFIG'] = config_pin['path']
    os.environ['BENCHMARK_HOST_CONFIG_SHA256'] = config_pin['sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['physical_gpu_uuid']
    guard(config)  # literal authorized allocation before project metadata
    sys.path.insert(0, str(expansion))
    from final_scoring_integration_v2.common import digest, inside, pin, require, verify
    from combined_campaign_orchestration_v3.stages import exclusive, run_stage, wait_for_files
    from combined_campaign_orchestration_v3.check_local_ready_faultpoint import run_faultpoint_check
    from final_scoring_integration_v2.phase_ledger import ledger_class
    from host_ledger import Ledger
    from ownership import claim_host
    from backend import identity
    import custody
    import subprocess
    import check_chain
    root = Path(config['project_root'])
    own_manifest = json.loads((here / 'MANIFEST.json').read_text())
    require(own_manifest['schema'] == 'coherent-final-source-chain-rebind-package-v1', 'Reviewed engineering source package required')
    for row in own_manifest['files']:
        source = (here / row['path']).resolve()
        require(source.is_relative_to(here), 'Engineering source closure escapes package')
        raw = source.read_bytes()
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Engineering source changed after review')
    out = inside(args.output, root)
    require(not out.exists(), 'Fresh engineering output required; preserve previous attempts')
    out.mkdir(parents=True)
    started = time.monotonic()
    result = {'schema': 'real-allocation-orchestration-qualification-v1', 'status': 'running', 'cases': [],
              'models': 0, 'scientific_fits': 0, 'optimizer_updates': 0, 'TEST_accessed': False,
              'network_operations': 0, 'campaign_GO_created': False, 'native_runtime_readiness': 'pending'}
    def save(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        exclusive(path, body)
        return pin(path, root)
    def must_refuse(function, *values):
        try:
            function(*values)
        except ValueError:
            return
        raise AssertionError('Expected retained refusal')
    try:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            check_chain.main()
        chain_check = json.loads(capture.getvalue())
        require(chain_check['status'] == 'PASS', 'Coherent source chain required')
        chain = json.loads((here / 'SOURCE_CHAIN.json').read_text())
        roster = verify(chain['existing_actual_allocation_artifacts']['roster'], root, json_metadata=True)
        require(len(roster['tasks']) == 6730 and roster['counts']['additional_training_fits'] == 0, 'Use actual unchanged release14 roster')
        result['actual_roster'] = chain['existing_actual_allocation_artifacts']['roster']
        result['source_check'] = chain_check
        pending = save(out / 'pending.json', {'schema': 'engineering-pending-authority-v1', 'status': 'pending', 'user_explicit_GO': False})
        false_go = save(out / 'honest_false_GO.json', {'schema': 'portable-human-GO-v2', 'user_explicit_GO': False})
        training_spec = save(out / 'pending_training_spec.json', {'go': false_go, 'release': pending})
        def absolute(value):
            return pin(expansion.parent / value['path_relative_to_phase'], root)
        template = {'schema': 'prepared-combined-worker-execution-template-v1', 'worker': 'allocation/gpu0',
            'training_entry': pin(expansion / 'portable_dispatcher_v3/run_gpu.py', root),
            'final_entry': pin(expansion / 'final_scoring_integration_v2/run_gpu.py', root),
            'final_static_fields': {'release': pending, 'readiness': pending},
            'scorer_source_manifest': absolute(chain['source_manifests']['final_selected_scoring_v4']),
            'final_integration_manifest': absolute(chain['source_manifests']['final_scoring_integration_v2']),
            'original_paper_reruns': False, 'new_human_action_at_barrier': False, 'maximum_native_children_per_GPU': 2,
            'scoring_roster': result['actual_roster'], 'supervisor_python_environment': pending,
            'orchestration_output_root': str(out / 'not_admitted'), 'host_sync_root': str(out / 'not_admitted_sync'),
            'offline_delivery_path': str(out / 'not_admitted_delivery/DELIVERY.json')}
        template_pin = save(out / 'negative_template.json', template)
        negative = save(out / 'negative_combined_spec.json', {'schema': 'combined-portable-campaign-run-v1',
            'worker': 'allocation/gpu0', 'host_config': config_pin, 'execution_template': template_pin,
            'training_run_spec': training_spec, 'training_go': false_go, 'training_release': pending,
            'source_manifest': absolute(chain['source_manifests']['combined_campaign_orchestration_v3']),
            'global_decision': pending, 'global_contract': pending})
        entry = expansion / 'combined_campaign_orchestration_v3/run_gpu.py'
        for worker, expected in [('allocation/gpu0', 'Actual existing campaign GO required'),
                                 ('server77/gpu0', 'Host disabled before probes')]:
            argv = [sys.executable, '-B', '-u', str(entry), '--worker', worker,
                    '--run-spec', negative['path'], '--run-spec-sha256', negative['sha256']]
            completed = subprocess.run(argv, cwd=root, env=dict(os.environ), capture_output=True, text=True, timeout=40)
            save(out / (worker.replace('/', '_') + '_REFUSAL.json'), {'argv': argv, 'returncode': completed.returncode,
                'stdout': completed.stdout, 'stderr': completed.stderr})
            require(completed.returncode != 0 and expected in completed.stdout, 'Expected real production admission refusal')
            result['cases'].append({'case': worker + '_admission', 'status': 'PASS', 'expected_refusal': expected})
        require(not (out / 'not_admitted').exists(), 'False GO must not create stage control or children')
        fixture = pin(here / 'metadata_supervisor_fixture.py', root)
        worker = 'allocation/gpu0'
        specs = {}
        for case in ('success', 'failure', 'briefly_live'):
            specs[case] = save(out / (case + '_spec.json'), {'schema': 'metadata-only-supervisor-engineering-fixture-v1',
                'case': case, 'root': str(out), 'output': str(out / (case + '_child'))})
        args_success = ('success', out, sys.executable, fixture, worker, specs['success'], root, identity)
        first = run_stage(*args_success)
        second = run_stage(*args_success)
        require(first == second, 'Successful restart must reuse exact RETURN')
        result['cases'].append({'case': 'success_then_restart', 'status': 'PASS', 'actual_fixture_children': 1})
        args_failure = ('failure', out, sys.executable, fixture, worker, specs['failure'], root, identity)
        must_refuse(run_stage, *args_failure); must_refuse(run_stage, *args_failure)
        result['cases'].append({'case': 'failure_then_restart', 'status': 'PASS', 'actual_fixture_children': 1})
        live_errors = []
        args_live = ('briefly_live', out, sys.executable, fixture, worker, specs['briefly_live'], root, identity)
        def live():
            try:
                run_stage(*args_live)
            except BaseException as error:
                live_errors.append(str(error))
        thread = threading.Thread(target=live)
        thread.start()
        deadline = time.monotonic() + 15
        while not (out / 'briefly_live/PID.json').is_file() and time.monotonic() < deadline:
            time.sleep(.01)
        require((out / 'briefly_live/PID.json').is_file() and not (out / 'briefly_live/RETURN.json').exists(), 'Observe actual in-flight supervisor')
        must_refuse(run_stage, *args_live)
        thread.join(15)
        require(not thread.is_alive() and not live_errors, 'Original engineering supervisor must finish/reap')
        result['cases'].append({'case': 'inflight_duplicate_refused', 'status': 'PASS', 'actual_fixture_children': 1})
        result['cases'].append({'case': 'shard_ready_faultpoint', **run_faultpoint_check()})
        delivery = out / 'delayed_delivery.json'
        def deliver():
            time.sleep(.2); save(delivery, {'engineering_delivery': True})
        delivery_thread = threading.Thread(target=deliver); delivery_thread.start()
        wait_for_files([delivery], lambda: None, interval_seconds=1); delivery_thread.join()
        stop = out / 'new_STOP.json'; save(stop, {'engineering_STOP': True})
        def check_stop():
            require(not stop.exists(), 'new retained STOP')
        must_refuse(wait_for_files, [out / 'never_delivery.json'], check_stop)
        result['cases'].append({'case': 'offline_delivery_and_STOP', 'status': 'PASS', 'final_children': 0})
        ledger_root = out / 'isolated_ledger'
        parent_owner = {'hosts': {'allocation': {'project_root': str(root)}}, 'assignments': [], 'engineering': 'parent'}
        final_owner = {'hosts': parent_owner['hosts'], 'assignments': [], 'engineering': 'final'}
        parent = Ledger(ledger_root, parent_owner, 'allocation'); parent.establish()
        owner_before = pin(ledger_root / 'OWNER.json', root)
        final = ledger_class(Ledger, custody, claim_host)(ledger_root, final_owner, 'allocation',
            training_ownership=parent_owner, handoff={'engineering_only': True}, verify_terminal_seal=lambda: None)
        final.establish(); final.load(); must_refuse(parent.load)
        require(pin(ledger_root / 'OWNER.json', root) == owner_before, 'Original owner bytes must remain unchanged')
        require((ledger_root / 'TRAINING_STATE_AT_HANDOFF.json').is_file(), 'Retain original STATE bytes')
        result['cases'].append({'case': 'isolated_actual_PhaseLedger', 'status': 'PASS', 'live_campaign_ledger_touched': False})
        require(len(result['cases']) == 8, 'Fixed eight-case suite only')
        require(sum(1 for path in out.glob('*_child/ENGINEERING_CHILD.json')) == 3, 'Exactly three metadata-only supervisors')
        result.update(status='PASS', actual_metadata_fixture_children=3, negative_CLI_invocations=2)
    except BaseException as error:
        result.update(status='FAIL', failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        result['elapsed_wall_seconds'] = time.monotonic() - started
        save(out / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'result': pin(out / 'RESULT.json', root),
                      'scientific_fits': 0, 'TEST_accessed': False}))


if __name__ == '__main__':
    main()
