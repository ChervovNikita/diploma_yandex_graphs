"""Finite allocation-only metadata rebind; no native work, release or GO."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from composed_authority import RUNTIME_NAMES, digest, receipt, require, verify_sources

HERE = Path(__file__).resolve().parent
E, PHASE = HERE.parent, HERE.parent.parent
ROOT = PHASE.parents[1]
ALLOCATION = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
COMMON = ('native_source', 'data', 'controller', 'evaluator')
COUNTS = {'physical_fits': 6460, 'I4_exports': 270, 'expected_predictor_rows': 6190,
          'source_reference_rows': 270, 'same_bank_mean_rows': 270}
SCORER_SHA = 'af692038bec60bbb54e9ed45d659289af98a0cb844ab0637c8923ee6ce70b67b'


def read(pin):
    path = Path(pin['path']).resolve()
    require(path.is_relative_to(ROOT) and path.suffix == '.json' and receipt(path) == pin,
            'Exact project-confined JSON receipt required')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sources(expected):
    pin = receipt(HERE / 'MANIFEST.json')
    require(pin['sha256'] == expected, 'Exact root-reviewed reporting source required')
    manifest = read(pin)
    require(manifest['schema'] == 'normalization-reporting-rebind-source-v1', 'Wrong reporting source schema')
    for row in manifest['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE), 'Source member escapes package')
        now = receipt(path)
        require(all(now[key] == row[key] for key in ('bytes', 'sha256')), 'Reporting member changed')
    for row in json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())['files']:
        path = (PHASE / row['path_relative_to_phase']).resolve()
        require(path.is_relative_to(ROOT), 'Reporting dependency escapes project')
        now = receipt(path)
        require(all(now[key] == row[key] for key in ('bytes', 'sha256')), 'Frozen reporting dependency changed')
    return pin


def composed_authority(job, update):
    producer = E / 'composable_normalization_runtime_v1'
    helper = E / 'portable_dispatcher_v3' / ('runtime_link' if job['family'] == 'link' else 'runtime_node_molecular')
    runtime_pins = [receipt(producer / 'runtime' / name) for name in RUNTIME_NAMES]
    helper_pins = [receipt(helper / name) for name in RUNTIME_NAMES]
    supplied = update['normalization_sources']
    require(supplied == {'composition_manifest': receipt(E / 'composable_normalization_v2/MANIFEST.json'),
                        'producer_manifest': receipt(producer / 'MANIFEST.json'),
                        'bank_law': receipt(E / 'private_bn_state_v3/banks.py')}
            and update['runtime_entrypoint'] == runtime_pins[0], 'Wrong declared genuine normalization sources')
    pins = [*runtime_pins, *helper_pins, *supplied.values(), receipt(producer / 'runtime_bridge.py')]
    return {'schema': 'normalization-reporting-runtime-authority-v1', 'project_root': str(ROOT),
            'producer_manifest': supplied['producer_manifest'], 'entrypoint': update['runtime_entrypoint'],
            'normalization_source': supplied['composition_manifest'],
            'normalization_composition': copy.deepcopy(job['normalization_composition']),
            'runtime_sha256': {Path(p['path']).name: p['sha256'] for p in runtime_pins},
            'helper_runtime_sha256': {Path(p['path']).name: p['sha256'] for p in helper_pins},
            'callback_sha256': job['callback_sha256'],
            'extension_sha256': (job.get('session_transform') or {}).get('sha256'),
            'policy_sha256': job.get('link_policy_sha256') if job['family'] == 'link' else None,
            'source_files': list({p['path']: p for p in pins}.values())}


def apply_fit(spec, update, job, dispatch, authority):
    require(update['parent_spec_sha256'] == digest(spec) and update['parent_fit_id'] == spec['identity']['fit_id'],
            'Reporting delta must bind the exact parent spec')
    require(update['fit_id'] == job['id'] and update['job_hash'] == digest(job)
            and update['parent_observed_input_files'] == spec['observed_input_files']
            and update['parent_wrapper_authority'] == spec['wrapper_authority'], 'Reporting delta parent/job mismatch')
    require(update['preserved_common_bindings'] == {k: spec['bindings'][k] for k in COMMON}
            and all(update['identity']['bindings'][k] == spec['bindings'][k] for k in COMMON),
            'Common native/data/controller/evaluator identities changed')
    task = dispatch['successor_task']
    require(task['id'] == job['id'] and task['source_descriptor'] == update['descriptor']
            and task['runtime_job_hash'] == update['job_hash'] and task['binding'] is None
            and dispatch['successor_candidate']['admitted'] is False
            and task['output_relative'] == 'fits/' + hashlib.sha256(job['id'].encode()).hexdigest(),
            'Exact unadmitted successor dispatch output required')
    result = copy.deepcopy(spec)
    result.update(descriptor=update['descriptor'], job_hash=update['job_hash'], identity=copy.deepcopy(update['identity']),
                  runtime_entrypoint=update['runtime_entrypoint'], output_relative=task['output_relative'],
                  bindings=copy.deepcopy(update['identity']['bindings']), composed_runtime_authority=authority)
    result['normalization_parent_spec_sha256'] = digest(spec)
    result['wrapper_authority']['descriptor'] = update['descriptor']
    result['wrapper_authority']['normalization_composition'] = copy.deepcopy(job['normalization_composition'])
    pins = spec['observed_input_files'] + authority['source_files']
    unique = {}
    for pin in pins:
        require(pin['path'] not in unique or unique[pin['path']] == pin, 'Conflicting observed source receipt')
        unique[pin['path']] = pin
    result['observed_input_files'] = list(unique.values())
    if 'same_bank_mean_identity' in spec:
        result['same_bank_mean_identity'] = copy.deepcopy(update['same_bank_mean_identity'])
    return result


def rebuild_controls(profile, control_rows, control_pin):
    """Use exact successor fit links; retain the accepted reporter's contrast format."""
    specs, i4 = profile['fit_specs'], profile['I4_specs']
    old = profile['contrast_plan']['presets'][profile['registry']['presets'][0]['id']]
    old_by_id = {row['id']: row for row in old}
    groups, pairs = {}, []
    cohort_ids = {row['cohort_id'] for row in profile['registry']['presets'][0]['matched_cohorts']}
    for row in control_rows:
        lhs = specs[row['fit_id']]['identity']
        require(lhs['scientific_method_id'] == row['scientific_method_id'], 'Control method alias differs')
        for link in row['controls']:
            if link['kind'] == 'diagnostic_same_selected_fit':
                require(link['fit_ids'] == [lhs['fit_id']], 'Mean diagnostic changed selected fit')
                rhs = specs[lhs['fit_id']]['same_bank_mean_identity']
            elif link['kind'] == 'logical_native_bank' and link['resolved_control'] == 'native_I4':
                matches = [s['identity'] for s in i4.values() if s['component_fit_ids'] == link['fit_ids']]
                require(len(matches) == 1, 'Exact I4 four-body identity missing or duplicated')
                rhs = matches[0]
            else:
                require(len(link['fit_ids']) == 1, 'One exact paired control required')
                rhs = specs[link['fit_ids'][0]]['identity']
                require(rhs['scientific_method_id'] == link['resolved_control'], 'Control resolved method differs')
            require(all(lhs[k] == rhs[k] for k in ('preset_id', 'track', 'cohort_id', 'cohort_seed', 'split_index', 'literal_split_seed'))
                    and all(lhs['bindings'][k] == rhs['bindings'][k] for k in COMMON), 'Control pairing authorities differ')
            cid = lhs['scientific_method_id'] + '__vs__' + rhs['scientific_method_id'] + '__' + rhs['readout']
            contrast = {'id': cid, 'lhs': {'method_id': lhs['scientific_method_id'], 'readout': 'actual'},
                        'rhs': {'method_id': rhs['scientific_method_id'], 'readout': rhs['readout']}}
            group = groups.setdefault(cid, {'contrast': contrast, 'cohorts': set(), 'roles': set()})
            require(group['contrast'] == contrast and lhs['cohort_id'] not in group['cohorts'], 'Duplicate paired contrast in one cohort')
            group['cohorts'].add(lhs['cohort_id'])
            if link.get('comparison_role'):
                group['roles'].add(link['comparison_role'])
            pairs.append({'contrast_id': cid, 'lhs_fit_id': lhs['fit_id'], 'rhs_fit_id': rhs['fit_id'],
                          'control_component_fit_ids': copy.deepcopy(link['fit_ids']), 'cohort_id': lhs['cohort_id'],
                          'comparison_role': link.get('comparison_role', 'prescribed_ingredient_control')})
        if 'normalization_attribution_reference' in row:
            matched = row['normalization_attribution_reference']
            require(matched in row['controls'] and matched['resolved_control'] == 'gnnm_private_bn_state_v3'
                    and matched['comparison_role'] == 'primary_ingredient_attribution', 'Private-BN attribution missing')
    native = 'native_M1__vs__native_I4__actual'
    require(native in old_by_id and set(old_by_id) - {native} <= set(groups), 'Existing prescribed contrasts were dropped')
    rows = []
    for cid in [row['id'] for row in old] + sorted(set(groups) - set(old_by_id)):
        if cid == native:
            rows.append(copy.deepcopy(old_by_id[cid])); continue
        group = groups[cid]
        require(group['cohorts'] == cohort_ids, 'Every prescribed control needs all exact cohorts')
        if cid in old_by_id:
            require(group['contrast'] == {k: old_by_id[cid][k] for k in ('id', 'lhs', 'rhs')}, 'Existing contrast definition changed')
            contrast = copy.deepcopy(old_by_id[cid])
        else:
            require(group['contrast']['rhs']['method_id'] == 'gnnm_private_bn_state_v3'
                    and group['roles'] == {'primary_ingredient_attribution'}, 'Unprescribed new contrast')
            contrast = group['contrast']
        if group['roles']:
            contrast['comparison_roles'] = sorted(group['roles'])
        rows.append(contrast)
    preset_id = profile['registry']['presets'][0]['id']
    profile['registry']['presets'][0]['contrasts'] = rows
    profile['contrast_plan']['presets'][preset_id] = copy.deepcopy(rows)
    profile['contrast_plan']['successor_control_links'] = control_pin
    profile['normalization_control_attribution'] = {'source': control_pin, 'paired_fit_links': pairs,
        'private_BN_is_primary_ingredient_reference': True, 'original_GNNM_total_comparisons_retained': True}
    return len(rows)


def prepare(preparation_pin, output, source_pin):
    bound = json.loads((HERE / 'INPUTS.json').read_text())
    require(preparation_pin == bound['migration_preparation'], 'Use the exact completed root-bound migration')
    generation = read(preparation_pin)
    require(generation['schema'] == 'normalization-campaign-migration-preparation-v1'
            and generation['status'] == 'metadata_generated_not_admitted'
            and generation['source_manifest'] == receipt(E / 'normalization_campaign_migration_v1/MANIFEST.json')
            and generation['full_campaign_GO'] is False and generation['models_constructed'] == 0
            and generation['counts']['changed'] == 2890 and generation['counts']['fits'] == 6460,
            'Exact completed metadata migration required')
    artifacts = generation['artifacts']
    delta, controls = read(artifacts['reporting_delta']), read(artifacts['controls'])
    dispatch, descriptors = read(artifacts['dispatcher_delta']), read(artifacts['index'])
    require(delta['schema'] == 'normalization-reporting-rebind-delta-v1'
            and delta['future_control_links'] == artifacts['controls']
            and controls['schema'] == 'expanded-paired-control-links-v1' and len(controls['entries']) == 5110,
            'Actual successor reporting/control artifacts required')
    scorer = delta['selected_serving_source_rebind_pending']['required_manifest']
    require(scorer == receipt(E / 'composable_normalization_final_scoring_v2/MANIFEST.json')
            and scorer['sha256'] == SCORER_SHA, 'Bind the exact repaired composed scorer V2')
    scorer_value = read(scorer)
    for row in scorer_value['files']:
        now = receipt(E / 'composable_normalization_final_scoring_v2' / row['path'])
        require(all(now[k] == row[k] for k in ('bytes', 'sha256')), 'Repaired scorer member changed')
    parent = read(delta['parent_index'])
    require(delta['parent_index'] == generation['parent_inputs']['reporting_index'], 'Migration parent registry differs')
    require(receipt(E / 'benchmark_reporting_implementation_v1/reporter.py') == parent['parent_reporter_source'],
            'Accepted reporter source changed')
    require(parent['counts'] == delta['parent_counts_retained'] == COUNTS
            and parent['preset_files'] == delta['parent_profiles'] and len(parent['preset_files']) == 27,
            'Exact complete parent reporting inventory required')
    updates = {u['parent_fit_id']: u for u in delta['fit_updates']}
    tasks = {d['parent_task']['id']: d for d in dispatch['changed_tasks']}
    require(len(updates) == len(delta['fit_updates']) == 2890 and set(tasks) == set(updates), 'Missing or duplicate fit/dispatch updates')
    by_control = {row['fit_id']: row for row in controls['entries']}
    require(len(by_control) == 5110, 'Duplicate successor control row')
    generator = module(E / 'normalization_campaign_migration_v1/generate.py', '_reporting_migration_source')
    profiles, seen, preserved, totals, contrasts = [], set(), set(), {k: 0 for k in COUNTS}, 0
    source_families = set()
    output.mkdir()
    try:
        for parent_pin in parent['preset_files']:
            old = read(parent_pin)
            profile = copy.deepcopy(old)
            profile['fit_specs'] = {}
            replacements = {}
            for fid, spec in old['fit_specs'].items():
                require(fid not in seen, 'Duplicate parent physical fit')
                seen.add(fid)
                if fid not in updates:
                    profile['fit_specs'][fid] = copy.deepcopy(spec); preserved.add(fid); continue
                update = updates[fid]
                job = read(update['descriptor'])
                src = update['normalization_sources']
                expected = generator.reporting_delta(spec, job, update['descriptor'], update['runtime_entrypoint'],
                    src['composition_manifest'], src['producer_manifest'], src['bank_law'], update['matched_control_fit_id'])
                require(update == expected, 'Actual reporting delta differs from its unchanged generator')
                authority = composed_authority(job, update)
                if job['family'] not in source_families:
                    verify_sources(authority)
                    source_families.add(job['family'])
                successor = apply_fit(spec, update, job, tasks[fid], authority)
                profile['fit_specs'][update['fit_id']] = successor
                replacements[(fid, 'actual')] = successor['identity']
                if 'same_bank_mean_identity' in successor:
                    replacements[(fid, 'same_selected_bank_mean')] = successor['same_bank_mean_identity']
                control = by_control[update['fit_id']]['normalization_attribution_reference']
                require(control['fit_ids'] == [update['matched_control_fit_id']], 'Reporting/private-BN control mismatch')
            profile['registry']['expected_predictors'] = [copy.deepcopy(replacements.get((row['fit_id'], row['readout']), row))
                                                        for row in old['registry']['expected_predictors']]
            require(profile['I4_specs'] == old['I4_specs'], 'Native I4/body0 reuse changed')
            relevant = [by_control[fid] for fid in profile['fit_specs'] if fid in by_control]
            contrasts += rebuild_controls(profile, relevant, artifacts['controls'])
            profile['selected_serving_source'] = scorer
            profile['normalization_rebind'] = {'schema': 'normalization-reporting-profile-rebind-v1',
                'parent_profile': parent_pin, 'reporting_delta': artifacts['reporting_delta'], 'source_manifest': source_pin,
                'actual_data_bytes_reverified': False, 'native_qualification_granted': False, 'human_GO': False}
            preset_id = profile['registry']['presets'][0]['id']
            profiles.append(write(output / (preset_id + '.json'), profile))
            totals['physical_fits'] += len(profile['fit_specs']); totals['I4_exports'] += len(profile['I4_specs'])
            rows = profile['registry']['expected_predictors']
            totals['expected_predictor_rows'] += len(rows)
            totals['source_reference_rows'] += sum(row['track'] == 'source_reference' for row in rows)
            totals['same_bank_mean_rows'] += sum(row['readout'] == 'same_selected_bank_mean' for row in rows)
        require(totals == COUNTS and preserved == set(delta['unchanged_fit_ids']) and len(preserved) == 3570,
                'Reporting rebind changed the frozen inventory')
        new_ids = {updates[fid]['fit_id'] if fid in updates else fid for fid in seen}
        require(len(new_ids) == 6460 and new_ids == {row['id'] for row in descriptors['fits']}
                and set(by_control) <= new_ids, 'Reporting/new descriptor/control coverage differs')
        index = copy.deepcopy(parent)
        index.update(preset_files=profiles, selected_serving_source=scorer,
            normalization_rebind={'schema': 'normalization-reporting-registry-rebind-v1', 'source_manifest': source_pin,
                'migration_preparation': preparation_pin, 'reporting_delta': artifacts['reporting_delta'],
                'control_links': artifacts['controls'], 'parent_registry': delta['parent_index'],
                'observer_source': receipt(HERE / 'observe_exports.py'), 'collector_source': receipt(HERE / 'collect_campaign.py'),
                'runtime_entry_source': receipt(HERE / 'runtime_entry.py'), 'composed_authority_source': receipt(HERE / 'composed_authority.py'),
                'selected_serving_source_bound': True, 'contrast_count': contrasts,
                'changed_specs': 2890, 'unchanged_specs': 3570, 'native_qualification_granted': False,
                'campaign_output_root_requires_release_custody_review': True, 'release_created': False, 'human_GO': False})
        index_pin = write(output / 'INDEX.json', index)
        result = {'schema': 'normalization-reporting-rebind-preparation-v1', 'status': 'metadata_generated_not_admitted',
            'source_manifest': source_pin, 'index': index_pin, 'counts': totals, 'contrast_count': contrasts,
            'selected_serving_source': scorer, 'remaining_dependencies': [
                'root_review_actual_reporting_generation', 'normalized_producer_native_qualification',
                'observer_composed_native_witness_with_existing_authorized_work',
                'dispatcher_release_and_ownership_rebind', 'existing_claim_and_checkpoint_custody_review',
                'measured_target_resource_cost_and_pair_admission', 'exact_updated_campaign_human_GO'],
            'model_tensor_array_execution': 0, 'TEST_payload_accessed': False, 'native_work_added': 0,
            'source_predecessors_modified': False, 'release_created': False, 'human_GO': False}
        write(output / 'PREPARATION.json', result)
        return result
    except BaseException as error:
        write(output / 'FAILED.json', {'status': 'failed', 'error': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(limit=8), 'completed_profiles': len(profiles), 'native_work_added': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--migration-preparation', required=True)
    parser.add_argument('--migration-preparation-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0', 'Wrong allocation host; no project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=10).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong sole allocation GPU')
    require(ROOT == ALLOCATION, 'Exact allocation deployment required')
    source = sources(args.source_manifest_sha256)
    preparation = receipt(args.migration_preparation)
    require(preparation['sha256'] == args.migration_preparation_sha256, 'Exact root-bound generated migration required')
    output = Path(args.output).resolve()
    require(output.parent == HERE and not output.exists(), 'Fresh direct child output required')
    result = prepare(preparation, output, source)
    require('torch' not in sys.modules and 'numpy' not in sys.modules, 'Metadata generation imported a numerical library')
    print(json.dumps({'preparation': receipt(output / 'PREPARATION.json'), 'counts': result['counts'],
                      'contrast_count': result['contrast_count'], 'native_work_added': 0}))


if __name__ == '__main__':
    main()
