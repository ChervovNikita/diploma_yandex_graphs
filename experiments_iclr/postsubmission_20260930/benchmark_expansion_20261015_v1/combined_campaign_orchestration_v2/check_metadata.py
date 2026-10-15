"""Source/pure-metadata checks; never starts a supervisor, host probe or model."""
import ast
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from combined_campaign_orchestration_v2.contracts import derive_final_spec, validate_actual, validate_template, validate_host_templates
from combined_campaign_orchestration_v2.stages import supervisor_argv, wait_for_files
from combined_campaign_orchestration_v2.check_local_ready_faultpoint import run_faultpoint_check


def refuses(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError('Expected source/metadata refusal')


def main():
    checks = []
    parsed = {}
    for path in HERE.glob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        parsed[path.name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not item.name.startswith(('torch', 'numpy', 'torch_geometric')) for item in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('torch', 'numpy', 'torch_geometric'))
    checks.append('all modules parse; no model/data library imports')
    def receipt(path):
        return {'path': str(path), 'bytes': 1, 'sha256': 'a'*64}
    root = HERE.parent.parent.parent
    base = HERE.parent
    template = {'schema': 'prepared-combined-worker-execution-template-v1', 'worker': 'allocation/gpu0',
        'training_entry': receipt(base / 'portable_dispatcher_v3/run_gpu.py'),
        'final_entry': receipt(base / 'final_scoring_integration_v2/run_gpu.py'),
        'final_static_fields': {'release': receipt(HERE / 'fixture_final_release.json'), 'readiness': receipt(HERE / 'fixture_readiness.json')},
        'scorer_source_manifest': receipt(base / 'final_selected_scoring_v3/MANIFEST.json'),
        'final_integration_manifest': receipt(base / 'final_scoring_integration_v2/MANIFEST.json'),
        'scoring_roster': receipt(HERE / 'fixture_roster.json'),
        'original_paper_reruns': False, 'new_human_action_at_barrier': False, 'maximum_native_children_per_GPU': 2,
        'orchestration_output_root': str(HERE / 'absent_fixture_control'), 'host_sync_root': str(HERE / 'absent_fixture_sync'),
        'offline_delivery_path': str(HERE / 'absent_fixture_delivery/DELIVERY.json')}
    validate_template(template)
    pending = copy.deepcopy(template); pending['scorer_source_manifest'] = None
    refuses(validate_template, pending)
    extra = copy.deepcopy(template); extra['final_static_fields']['arm_override'] = 'forbidden'
    refuses(validate_template, extra)
    bad_entry = copy.deepcopy(template); bad_entry['training_entry']['path'] = '/arbitrary/train.py'
    refuses(validate_template, bad_entry)
    checks.append('typed existing entries accepted; pending source/arm override/arbitrary entry refused')
    go_pin, contract_pin = receipt(HERE / 'fixture_GO.json'), receipt(HERE / 'fixture_contract.json')
    spec = {'schema': 'combined-portable-campaign-run-v1', 'worker': 'allocation/gpu0',
            'source_manifest': receipt(HERE / 'MANIFEST.json'), 'execution_template': receipt(HERE / 'fixture_template.json'),
            'training_go': go_pin, 'training_release': receipt(HERE / 'fixture_training_release.json'),
            'training_run_spec': receipt(HERE / 'fixture_training_run.json'),
            'global_decision': receipt(HERE / 'fixture_decision.json'), 'global_contract': contract_pin}
    go = {'schema': 'portable-human-GO-v2', 'user_explicit_GO': True, 'allowed_workers': ['allocation/gpu0'],
          'combined_execution': {'source_manifest': spec['source_manifest'], 'execution_templates': {'allocation/gpu0': spec['execution_template']}},
          'final_scoring': {'source_manifest': template['scorer_source_manifest']},
          'final_execution': {'source_manifest': template['final_integration_manifest'], **template['final_static_fields']},
          'host_releases': {'allocation': spec['training_release']}}
    training = {'go': go_pin, 'release': spec['training_release']}
    validate_actual(spec, template, go, training, root)
    wrong = dict(training, go=receipt(HERE / 'different_GO.json'))
    refuses(validate_actual, spec, template, go, wrong, root)
    pending_go = dict(go, user_explicit_GO=False)
    refuses(validate_actual, spec, template, pending_go, training, root)
    server_spec = dict(spec, worker='server77/gpu1')
    server_template = dict(template, worker='server77/gpu1')
    server_go = copy.deepcopy(go)
    server_go['allowed_workers'] = ['server77/gpu0', 'server77/gpu1']
    server_go['host_releases'] = {'server77': spec['training_release']}
    server_go['combined_execution']['execution_templates'] = {'server77/gpu0': receipt(HERE / 'gpu0_template.json'), 'server77/gpu1': spec['execution_template']}
    validate_actual(server_spec, server_template, server_go, training, root)
    peer_template = dict(server_template, worker='server77/gpu0')
    validate_host_templates(server_template, {'server77/gpu0': peer_template, 'server77/gpu1': server_template})
    broken_peer = dict(peer_template, host_sync_root='/different/barrier')
    refuses(validate_host_templates, server_template, {'server77/gpu0': broken_peer, 'server77/gpu1': server_template})
    checks.append('same existing GO/release and two-worker77template map accepted; changed or pending GO refused')
    delivery = {'schema': 'offline-final-host-delivery-v1', 'host': 'allocation', 'global_contract': contract_pin,
                'training_seal': receipt(HERE / 'fixture_seal.json'), 'new_human_action_claimed': False}
    final = derive_final_spec(template, spec, delivery)
    assert final['training_go'] == go_pin and final['release'] == template['final_static_fields']['release']
    assert set(final) == {'schema', 'worker', 'release', 'readiness', 'training_run_spec', 'training_go', 'global_decision', 'global_contract', 'training_seal'}
    refuses(derive_final_spec, template, spec, dict(delivery, host='server77'))
    refuses(derive_final_spec, template, spec, dict(delivery, new_human_action_claimed=True))
    checks.append('offline seal adds deterministic parent fields only; foreign host/new-action delivery refused')
    argv = supervisor_argv('/fixture/python', template['training_entry'], 'allocation/gpu0', spec['training_run_spec'])
    assert argv == ['/fixture/python', '-B', '-u', template['training_entry']['path'], '--worker', 'allocation/gpu0',
                    '--run-spec', spec['training_run_spec']['path'], '--run-spec-sha256', spec['training_run_spec']['sha256']]
    popens = [node for node in ast.walk(parsed['stages.py']) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Popen']
    assert len(popens) == 1 and not any(k.arg == 'shell' for k in popens[0].keywords)
    checks.append('fixed argv only; one supervisor Popen implementation, no shell/arbitrary commands')
    wait_node = next(node for node in ast.walk(parsed['stages.py']) if isinstance(node, ast.FunctionDef) and node.name == 'wait_for_files')
    calls = [node.func.id for node in ast.walk(wait_node) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert 'verify' not in calls and 'confirm_worker_terminal' not in calls and 'build_host_shard' not in calls
    observed = []
    wait_for_files([HERE / 'check_metadata.py'], lambda: observed.append('STOP_snapshot'), interval_seconds=1)
    assert observed == ['STOP_snapshot']
    checks.append('ready-file wait checks one STOP snapshot; poll path does not hash/scan campaign tasks')
    faultpoint = run_faultpoint_check()
    assert faultpoint['builder_calls_on_recovery'] == 0 and faultpoint['readiness_recreated']
    main_source = (HERE / 'run_gpu.py').read_text()
    assert 'and not shard_path.exists()' not in main_source and 'ensure_local_ready(' in main_source
    checks.append('fault after shard commit recovers LOCAL_READY with0builder calls; byte-exact reuse/idempotence; changed shard refused')
    package_status = json.loads((HERE / 'PREPARATION_STATUS.json').read_text())
    assert package_status['scorer_binding'] is None and package_status['final_runtime_binding'] is None
    assert package_status['actual_GO_artifacts_generated'] == package_status['training_or_final_supervisors_started'] == 0
    assert 'torch' not in sys.modules
    print(json.dumps({'schema': 'combined-orchestration-source-metadata-checks-v2', 'status': 'PASS', 'checks': checks, 'focused_faultpoint': faultpoint,
        'supervisors_started': 0, 'new_training_fits': 0, 'models': 0, 'forwards': 0, 'optimizer_updates': 0,
        'remote_calls': 0, 'host_probes': 0, 'TEST_accessed': False, 'scorer_runtime_bound': False,
        'full_campaign_GO': False, 'native_or_orchestration_execution_qualified': False}, indent=2))


if __name__ == '__main__':
    main()
