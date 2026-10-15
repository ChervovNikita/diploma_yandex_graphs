"""Compact scalar/metadata checks; no model, data, arrays, or exported predictions."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import rebind_registry as b
from composed_authority import runtime_witness, verify_sources
from observe_exports import observe


def refuses(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected metadata authority refusal')


def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    for path in b.HERE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    parent = b.E / 'benchmark_reporting_integration_v2'
    for name in ('runtime_entry.py', 'backend_bridge.py'):
        assert (b.HERE / name).read_bytes() == (parent / name).read_bytes()
    generator = b.module(b.E / 'normalization_campaign_migration_v1/generate.py', '_reporting_fixture_generator')
    contracts = generator.api()
    reporter = b.module(b.E / 'benchmark_reporting_implementation_v1/reporter.py', '_accepted_reporter_fixture')
    pin = lambda name: {'path': '/fixture/' + name + '.json', 'bytes': 1, 'sha256': 'a' * 64}
    bindings = {key: 'b' * 64 for key in (*b.COMMON, 'readout')}
    specs, i4, expected, control_rows, cohorts = {}, {}, [], [], []
    methods = ['native_M1', 'native_body_1', 'native_body_2', 'native_body_3',
               'gnnm_boundary_native', 'gnnm_private_bn_state_v3',
               'aggregation_mean_globalJ_v1', 'interaction_input_conditionalJ']
    old_contrasts = []
    def contrast(lhs, rhs, readout='actual'):
        return {'id': lhs + '__vs__' + rhs + '__' + readout,
                'lhs': {'method_id': lhs, 'readout': 'actual'}, 'rhs': {'method_id': rhs, 'readout': readout}}
    for pair in [(methods[4], 'native_M1'), (methods[4], 'native_I4'), (methods[5], methods[4]),
                 (methods[6], methods[4]), (methods[7], methods[4]), (methods[7], methods[6])]:
        old_contrasts.append(contrast(*pair))
    old_contrasts += [contrast(methods[7], methods[7], 'same_selected_bank_mean'), contrast('native_M1', 'native_I4')]
    preserved = {}
    for seed in range(10):
        cohort = 'fixture/cohort-' + str(seed)
        cohorts.append({'cohort_id': cohort, 'seed': seed, 'split_index': 0, 'literal_split_seed': None})
        ids = {method: cohort + '/' + method for method in methods}
        mapping = {ids[m]: ids[m] + contracts.SUFFIX for m in methods[6:]}
        for method in methods:
            fid = ids[method]
            identity = {'dataset': 'fixture', 'preset_id': 'fixture', 'track': 'matched_extended',
                'cohort_id': cohort, 'cohort_seed': seed, 'split_index': 0, 'literal_split_seed': None,
                'scientific_method_id': method, 'runtime_procedure_id': method, 'readout': 'actual',
                'fit_id': fid, 'actual_seed': seed, 'endpoint': {'name': 'rocauc', 'units': 'fraction', 'direction': 'max'},
                'bindings': copy.deepcopy(bindings)}
            spec = {'identity': identity, 'bindings': copy.deepcopy(bindings), 'descriptor': pin(fid),
                'job_hash': 'c' * 64, 'runtime_entrypoint': pin('runtime'), 'output_relative': 'fits/' + b.digest(fid),
                'wrapper_authority': {'descriptor': pin(fid), 'callback': pin('callback'), 'extension': None},
                'observed_input_files': []}
            if method == methods[7]:
                mean = copy.deepcopy(identity); mean['readout'] = 'same_selected_bank_mean'
                spec['same_bank_mean_identity'] = mean
            if method in methods[6:]:
                job = {'id': mapping[fid], 'family': 'ogb', 'callback_sha256': 'd' * 64,
                    'policy': {'fixture': 'scalar only'}, 'expanded_contract_hash': 'e' * 64,
                    'normalization_composition': {'policy': contracts.POLICY, 'fixture': 'not a native descriptor'}}
                update = generator.reporting_delta(spec, job, pin(mapping[fid]),
                    b.receipt(b.E / 'composable_normalization_runtime_v1/runtime/run_fit.py'),
                    b.receipt(b.E / 'composable_normalization_v2/MANIFEST.json'),
                    b.receipt(b.E / 'composable_normalization_runtime_v1/MANIFEST.json'),
                    b.receipt(b.E / 'private_bn_state_v3/banks.py'), ids[methods[5]])
                dispatch = generator.dispatch_delta({'id': fid, 'kind': 'fit', 'binding': None},
                    {'id': fid, 'admitted': False}, job, update['descriptor'], update['runtime_entrypoint'])
                authority = b.composed_authority(job, update)
                before = copy.deepcopy(spec)
                spec = b.apply_fit(spec, update, job, dispatch, authority)
                assert before['identity']['fit_id'] == fid and spec['identity']['fit_id'] == mapping[fid]
                assert all(spec['bindings'][key] == before['bindings'][key] for key in b.COMMON)
                assert spec['bindings']['readout'] != before['bindings']['readout']
                if seed == 0 and method == methods[7]:
                    observer_job, observer_spec = job, spec
                    bad = copy.deepcopy(update); bad['parent_spec_sha256'] = '0' * 64
                    refuses(lambda: b.apply_fit(before, bad, job, dispatch, authority))
            else:
                preserved[fid] = copy.deepcopy(spec)
            specs[spec['identity']['fit_id']] = spec
            if method not in methods[1:4]:
                expected.append(copy.deepcopy(spec['identity']))
            if 'same_bank_mean_identity' in spec:
                expected.append(copy.deepcopy(spec['same_bank_mean_identity']))
        body_ids = [ids[method] for method in methods[:4]]
        aggregate = copy.deepcopy(specs[body_ids[0]]['identity'])
        aggregate.update(scientific_method_id='native_I4', runtime_procedure_id='aggregate_I4_exports',
                         fit_id=cohort + '/I4', actual_seed=None, component_fit_ids=body_ids)
        i4[aggregate['fit_id']] = {'identity': aggregate, 'component_fit_ids': body_ids, 'aggregator_source': pin('I4')}
        expected.append(copy.deepcopy(aggregate))
        for method in methods[4:]:
            links = []
            for c in [row for row in old_contrasts if row['lhs']['method_id'] == method]:
                target = c['rhs']['method_id']
                if c['rhs']['readout'] == 'same_selected_bank_mean':
                    links.append({'kind': 'diagnostic_same_selected_fit', 'fit_ids': [ids[method]]})
                else:
                    links.append({'kind': 'logical_native_bank' if target.startswith('native_') else 'paired_shared_fit',
                                  'resolved_control': target, 'fit_ids': body_ids if target == 'native_I4' else [ids[target]]})
            control = {'fit_id': ids[method], 'scientific_method_id': method, 'controls': links}
            control_rows.append(contracts.migrated_controls(control, mapping, ids[methods[5]]))
    profile = {'fit_specs': specs, 'I4_specs': i4,
        'registry': {'schema': 'expanded-ten-cohort-report-registry-v1', 'presets': [
            {'id': 'fixture', 'matched_cohorts': cohorts, 'contrasts': copy.deepcopy(old_contrasts)}], 'expected_predictors': expected},
        'contrast_plan': {'schema': 'expanded-prescribed-contrast-plan-v1', 'presets': {'fixture': copy.deepcopy(old_contrasts)},
            'cohort_keys': {'fixture': [{k: c[k] for k in ('seed', 'split_index', 'literal_split_seed')} for c in cohorts]}}}
    original = copy.deepcopy(profile)
    assert b.rebuild_controls(profile, control_rows, pin('controls')) == len(old_contrasts) + 2
    assert profile['I4_specs'] == original['I4_specs'] and all(profile['fit_specs'][fid] == spec for fid, spec in preserved.items())
    assert sum('primary_ingredient_attribution' in row.get('comparison_roles', [])
               for row in profile['registry']['presets'][0]['contrasts']) == 2
    report = reporter.report(profile['registry'], [], [], profile['contrast_plan'])
    assert all(row['n_score_pairs'] == 0 and row['descriptive_marginal_95pct_paired_percentile_interval'] is None
               for row in report['contrasts'])
    refuses(lambda: b.rebuild_controls(copy.deepcopy(original), control_rows[:-1], pin('controls')))
    duplicate = copy.deepcopy(control_rows); duplicate[0]['controls'].append(copy.deepcopy(duplicate[0]['controls'][0]))
    refuses(lambda: b.rebuild_controls(copy.deepcopy(original), duplicate, pin('controls')))
    crossed = copy.deepcopy(control_rows); crossed[2]['controls'][0]['fit_ids'] = [cohorts[1]['cohort_id'] + '/' + methods[4]]
    refuses(lambda: b.rebuild_controls(copy.deepcopy(original), crossed, pin('controls')))

    authority = observer_spec['composed_runtime_authority']
    verify_sources(authority)
    helper = {k: copy.deepcopy(authority[k]) for k in ('callback_sha256', 'extension_sha256', 'policy_sha256')}
    helper.update(runtime_sha256=authority['helper_runtime_sha256'], dependency_sha256={
        str(b.E / 'composable_normalization_v2/contracts.py'): b.receipt(b.E / 'composable_normalization_v2/contracts.py')['sha256']})
    actual = {**copy.deepcopy(helper), 'schema': 'composed-normalization-genuine-runtime-binding-v1',
        'runtime_sha256': authority['runtime_sha256'], 'composition_producer': {
            'source_manifest': authority['producer_manifest'], 'entrypoint': authority['entrypoint'],
            'normalization_source': authority['normalization_source'], 'unchanged_helper_runtime': copy.deepcopy(helper)}}
    witness = runtime_witness(observer_job, observer_spec, actual)
    for field in ('schema', 'runtime_sha256', 'callback_sha256', 'dependency_sha256', 'composition_producer'):
        bad = copy.deepcopy(actual); bad[field] = None
        refuses(lambda bad=bad: runtime_witness(observer_job, observer_spec, bad))
    # Invoke the actual observer only on a failed scalar invocation with no export.
    # No selected prediction/checkpoint payload is fabricated or serialized.
    def forbidden_writer(*args):
        raise AssertionError('Scalar check must not call a native/tensor writer')
    runtime = SimpleNamespace(atomic_torch=forbidden_writer)
    with tempfile.TemporaryDirectory(prefix='normalization-reporting-scalar-') as temporary:
        with observe(runtime, observer_job, Path(temporary) / 'observation', observer_spec) as obs:
            invocation = {'job_id': observer_job['id'], 'job_hash': observer_spec['job_hash'],
                'runtime_binding': actual, 'policy': observer_job['policy'], 'status': 'failed',
                'invocation_wall_seconds': 1.0, 'selected_VALID_export': None}
            good = obs.record_invocation(invocation, 'scalar_fixture_only')
            assert good['composed_runtime_witness'] == witness and not good['binding_unresolved_reasons']
            bad = copy.deepcopy(invocation); bad['runtime_binding'] = helper
            rejected = obs.record_invocation(bad, 'legacy_callback_only_fixture')
            assert rejected['bindings'] == {} and rejected['binding_unresolved_reasons']
            assert rejected['actual_cost'] == good['actual_cost']
        assert runtime.atomic_torch is forbidden_writer
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return {'status': 'PASS', 'metadata_profiles': 1, 'cohorts': 10, 'changed_spec_fixtures': 20,
        'matched_BN_contrasts_added': 2, 'old_contrasts_preserved': len(old_contrasts),
        'native_I4_and_unchanged_specs_preserved': True, 'missing_duplicate_cross_cohort_controls_rejected': True,
        'stale_parent_spec_rejected': True, 'malformed_runtime_identities_rejected': 5,
        'actual_observer_callback_only_binding_withheld_and_costs_preserved': True,
        'accepted_reporter_all_missing_intervals_withheld': True, 'runtime_entry_and_argv_bridge_byte_identical': True,
        'model_array_tensor_or_TEST_execution': 0, 'exports_constructed': 0, 'remote_checks_executed_by_this_check': 0}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
