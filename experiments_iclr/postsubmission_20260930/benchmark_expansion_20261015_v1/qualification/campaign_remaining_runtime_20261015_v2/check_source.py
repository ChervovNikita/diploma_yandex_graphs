"""Finite stdlib-only checks; no native constructor, arrays, data or prediction payload."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from witness_support import GPU, GiB, digest, limits, receipt
from runtime_guards import projection_host
from regression_checks import run as regression_checks

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def refuses(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected fail-closed refusal')


def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    for path in HERE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    inputs = json.loads((HERE/'INPUTS.json').read_text())
    prefix = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/'
    verified = 0
    for pin in [*inputs['source_packages'], *inputs['source_files']]:
        p = E/pin['path'].removeprefix(prefix)
        assert p.is_relative_to(E)
        value = receipt(p)
        assert all(value[k] == pin[k] for k in ('bytes', 'sha256'))
        verified += 1
        if p.name == 'MANIFEST.json':
            for row in json.loads(p.read_text())['files']:
                member = (p.parent/row['path']).resolve()
                assert member.is_relative_to(p.parent.resolve())
                actual = receipt(member)
                assert all(actual[k] == row[k] for k in ('bytes', 'sha256'))
                verified += 1
    prior = json.loads((E/'NORMALIZATION_MOLBACE_OUTCOME_21.json').read_text())
    assert prior['result']['status'] == 'passed' and prior['result']['TRAIN_updates'] == 38
    original = prior['result']['invocations'][-1]['result']
    pins = inputs['molbace']
    assert original['job_hash'] == pins['job_sha256']
    assert original['selected_checkpoint'] == pins['selected_checkpoint']
    assert original['resumed_checkpoint'] == pins['recovery_checkpoint']
    assert original['selected_VALID_export'] == pins['selected_VALID_export']
    evidence = pins['evidence_local_receipt']
    actual = receipt(E/evidence['path_relative_to_expansion'])
    assert all(actual[k] == evidence[k] for k in ('bytes', 'sha256'))
    book = load(E/'composable_normalization_runtime_v1/runtime/bookkeeping.py', '_remaining_stopped_selector')
    policy = book.Policy.from_dict(original['policy'], original['family'])
    selector = book.Selector(policy)
    selector.load_state({'policy': dict(policy.__dict__), 'completed_epoch': 1, 'selected_epoch': 1,
        'best_metric': original['best_validation_metric'], 'stale': 0})
    assert selector.stopping_status()['stop'] is True
    refuses(lambda: selector.observe(2, {'rocauc': .5}))
    supervisor = load(HERE/'supervise.py', '_remaining_supervisor_checks')
    peer = {'pid': 640428, 'start_ticks': '6055039165', 'peak_host_RSS_bytes': 16*GiB, 'peak_GPU_bytes': 32*GiB}
    state = {'uuid': GPU, 'free_bytes': 70*GiB, 'processes': [{'pid': peer['pid'], 'start_ticks': peer['start_ticks'], 'gpu_memory_bytes': 10*GiB}]}
    assert supervisor.gpu_budget(state, [peer], [], 40*GiB) == (22*GiB, 0)
    short = copy.deepcopy(state); short['free_bytes'] = 60*GiB
    refuses(lambda: supervisor.gpu_budget(short, [peer], [], 40*GiB))
    wrong = copy.deepcopy(state); wrong['processes'][0]['start_ticks'] = 'different'
    refuses(lambda: supervisor.gpu_budget(wrong, [peer], [], 4*GiB))
    third = copy.deepcopy(state); third['processes'] *= 3
    refuses(lambda: supervisor.gpu_budget(third, [peer], [{'pid': 8, 'start_ticks': '9'}], 4*GiB))
    oversized = copy.deepcopy(state); oversized['processes'] = [{'pid': 8, 'start_ticks': '9', 'gpu_memory_bytes': 5*GiB}]
    refuses(lambda: supervisor.gpu_budget(oversized, [], [{'pid': 8, 'start_ticks': '9'}], 4*GiB))
    reporter = E/'server51_campaign_port_v1/reporting'
    successor = reporter.parent
    sys.path.insert(0, str(successor))
    projection = load(successor/'reporting_projection.py', '_remaining_projection_check')
    sys.path.remove(str(successor))
    assert Path(sys.modules['descriptor_binding'].__file__).resolve() == successor/'descriptor_binding.py'
    del sys.modules['descriptor_binding']
    fixture_job = {'id': 'scalar-projection-fixture', 'family': 'ogb',
        'normalization_composition': original['source_metadata']['normalization_composition']['declaration'],
        'callback_sha256': original['runtime_binding']['callback_sha256'],
        'session_transform': {'sha256': original['runtime_binding']['extension_sha256']}}
    binding = {'native_source': 'native-scalar-fixture', 'data': 'data-scalar-fixture',
               'controller': 'controller-scalar-fixture', 'evaluator': 'evaluator-scalar-fixture'}
    spec = {'descriptor': receipt(HERE/'INPUTS.json'), 'job_hash': digest(fixture_job),
        'identity': {'scope': 'scalar-only-fixture'}, 'bindings': binding,
        'wrapper_authority': {'descriptor': receipt(HERE/'INPUTS.json')}, 'observed_input_files': []}
    profile = {'fit_specs': {fixture_job['id']: spec}, 'I4_specs': {}, 'registry': {}, 'contrast_plan': {}}
    candidate = {'_job': fixture_job, 'job_descriptor': receipt(HERE/'INPUTS.json'),
                 'runtime_entrypoint': receipt(successor/'composed/runtime/run_fit.py')}
    task = {'runtime_job_hash': digest(fixture_job), 'runtime_binding': {'scope': 'scalar-fixture'}}
    def source_pin(path, expected=None):
        actual = receipt(path)
        assert expected is None or actual == expected
        return actual
    actual_host = json.loads((successor/'hosts/allocation_gpu0.json').read_text())
    assert 'host' not in actual_host
    projected = projection.project_profile(profile, receipt(HERE/'INPUTS.json'), {fixture_job['id']: task},
        {fixture_job['id']: candidate}, projection_host(actual_host), source_pin)
    projected_spec = projected['fit_specs'][fixture_job['id']]
    assert projected_spec['identity'] == spec['identity'] and projected_spec['bindings'] == binding
    authority = load(reporter/'composed_authority.py', '_remaining_new_authority_check')
    assert projected['host_projection']['host'] == 'allocation'
    assert projected_spec['composed_runtime_authority']['project_root'] == actual_host['project_root']
    # Check local byte closure separately; the actual host root is remote and is
    # not claimed to be destination-verified by this local metadata fixture.
    authority.verify_sources({**projected_spec['composed_runtime_authority'], 'project_root': str(E.parent.parent)})
    refuses(lambda: authority.runtime_witness(fixture_job, projected_spec, original['runtime_binding']))
    assert projected_spec['composed_runtime_authority']['producer_manifest'] == receipt(successor/'composed/MANIFEST.json')
    bridge = load(reporter/'backend_bridge.py', '_remaining_bridge_checks')
    with tempfile.TemporaryDirectory(prefix='remaining-runtime-metadata-') as temporary:
        root = Path(temporary)
        # Metadata markers only. These are never presented as model checkpoints/exports.
        descriptor = root/'descriptor.json'; descriptor.write_text('{}\n')
        recovery = root/'metadata_marker.txt'; recovery.write_text('metadata-only fixture; never deserialized\n')
        task = {'id': 'metadata-only', 'kind': 'fit', 'worker': 'allocation/gpu0', 'cell_id': 'fixture',
            'binding': {'execution': {'python_executable': 'python', 'runtime_entrypoint': receipt(reporter/'runtime_entry.py'),
                                     'job_descriptor': receipt(descriptor)}}}
        spec = {'runtime_entrypoint': task['binding']['execution']['runtime_entrypoint'], 'descriptor': receipt(descriptor)}
        profile = root/'fixture.json'; profile.write_text(json.dumps({'fit_specs': {task['id']: spec}, 'I4_specs': {}}))
        index = root/'INDEX.json'; index.write_text(json.dumps({'preset_files': [receipt(profile)]}))
        transform = bridge.ObservedArgvTransform(index, receipt(index)['sha256'], root)
        original_argv = ['python', spec['runtime_entrypoint']['path']]
        argv = transform(task, root/'output', receipt(recovery), {}, original_argv)
        assert argv[3] == str(reporter/'runtime_entry.py') and argv[-2:] == ['--resume', str(recovery.resolve())]
        bad = copy.deepcopy(task); bad['worker'] = 'server77/gpu0'
        refuses(lambda: transform(bad, root/'output', receipt(recovery), {}, original_argv))
        stale = receipt(recovery); recovery.write_text('changed\n')
        refuses(lambda: transform(task, root/'output', stale, {}, original_argv))
    for case in ('bound_observer_molbace', 'buddy_selected_VALID'):
        admission = json.loads((HERE/('ROOT_ADMISSION_TEMPLATE_'+case+'.json')).read_text())
        assert admission['status'] == 'TEMPLATE_NOT_ADMITTED' and admission['root_reviewed_source'] is False
        assert admission['resources'] == limits(case) and admission['TEST_calls'] == 0
        assert admission['TRAIN_updates'] == (38 if case == 'bound_observer_molbace' else 0)
    child = ast.parse((HERE/'bound_observer.py').read_text())
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ('train_epoch', 'train_update', 'backward', 'step') for node in ast.walk(child))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    regressions = regression_checks()
    return {'status': 'PASS', 'pinned_source_receipts_verified': verified, 'review_regressions': regressions,
        'retained_original_receipts_and_stopped_selector_verified': True,
        'peer_growth_PID_reuse_third_worker_and_overbudget_rejected': True,
        'genuine_argv_bridge_resume_and_stale_receipt_checks': True,
        'actual_successor_profile_projection_preserves_engineering_identity': True,
        'old_checkpoint_producer_rejected_by_successor_authority': True,
        'unadmitted_templates': 2, 'native_execution': 0, 'model_array_tensor_execution': 0,
        'selected_prediction_payloads_constructed': 0, 'TRAIN_updates': 0, 'TEST_calls': 0,
        'runtime_witness_remains_pending': True}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
