"""Rebind reporting metadata to the exact allocation portable projection.

No launch, model imports, dataset/checkpoint reads, scalar reconstruction or Git.
The caller must supply pins from the actually generated, reviewed projection.
The original registry and all reporting implementation sources remain intact.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
EXPANSION = PHASE / 'benchmark_expansion_20261015_v1'
PORTABLE = EXPANSION / 'portable_dispatcher_v2'
PARENT = EXPANSION / 'benchmark_reporting_integration_v2'
CAMPAIGN = 'expanded6460_compiler4_portable_v2'
EXPECTED_COUNTS = {'physical_fits': 6460, 'I4_exports': 270,
                   'expected_predictor_rows': 6190, 'source_reference_rows': 270,
                   'same_bank_mean_rows': 270}
COMMON = ('native_source', 'data', 'controller', 'evaluator')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def confined(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Metadata/source path escapes allocation project')
    return path


def pin(path):
    path = confined(path)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path, expected_sha256=None):
    path = confined(path)
    require(path.suffix == '.json', 'Only explicit JSON metadata inputs are parsed')
    raw = path.read_bytes()
    if expected_sha256 is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Pinned metadata changed: ' + str(path))
    return json.loads(raw)


def verified(receipt, parse_json=True):
    require(pin(receipt['path']) == receipt, 'Exact source/metadata receipt differs: ' + receipt['path'])
    return read(receipt['path']) if parse_json else receipt


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def common_projection(spec):
    return {key: spec['identity']['bindings'][key] for key in COMMON}


def expected_binding(job):
    return {'schema': 'explicit-runtime-descriptor-path-binding-v1', 'id': job['id'],
            'canonical_runtime_job_hash': digest(job), 'bound_runtime_job_hash': digest(job),
            'science_contract_hash': job.get('expanded_contract_hash'), 'field_changes': [],
            'source_hashes_or_data_receipts_rewritten': False, 'scientific_fields_changed': False,
            'destination_source_or_data_bytes_verified': False, 'launch_admitted': False}


def main():
    require(socket.gethostname() == 'anogena-2-0', 'Wrong allocation hostname; no project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                    text=True, timeout=10).splitlines() == [
                'GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong sole allocation GPU; no project reads')
    here = Path(__file__).resolve().parent
    require(here == EXPANSION / 'benchmark_reporting_portable_rebind_v1', 'Exact deployment folder required')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-index', 'preparation', 'candidates', 'release', 'portable-manifest'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--projection-directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = confined(args.output)
    require(output.is_relative_to(here) and not output.exists(), 'Fresh own metadata output required')
    projection = confined(args.projection_directory)
    require(projection.is_relative_to(PORTABLE) and projection != PORTABLE,
            'Explicit own portable projection directory required')
    sources = {
        'parent_index': PARENT / 'generated_v1/INDEX.json',
        'preparation': projection / 'PREPARATION.json',
        'candidates': projection / 'allocation/CANDIDATES.json',
        'release': projection / 'allocation/RELEASE_UNADMITTED.json',
        'portable_manifest': PORTABLE / 'MANIFEST.json'}
    source_values = {key: read(path, getattr(args, key + '_sha256'))
                     for key, path in sources.items()}
    parent, preparation = source_values['parent_index'], source_values['preparation']
    candidates, release = source_values['candidates'], source_values['release']
    manifest = source_values['portable_manifest']
    require(parent['schema'] == 'real-compiler-reporting-registry-index-v2'
            and parent['compiler_generation'] == parent['dispatcher_generation'] == 4
            and parent['counts'] == EXPECTED_COUNTS and len(parent['preset_files']) == 27,
            'Canonical registry inventory changed')
    verified(parent['parent_reporter_source'], parse_json=False)
    require(parent['parent_reporter_source']['sha256'] ==
            'e2f26a3133d21189fdf88fbbc0c775e76279a3f3f2ade50e5c51916d091c979b', 'Wrong accepted reporter')
    require(preparation['schema'] == 'portable-host-projection-preparation-v2'
            and preparation['tasks'] == 6730 and preparation['fits'] == 6460 and preparation['I4'] == 270
            and preparation['native_jobs_started'] == 0 and preparation['server77_enabled'] is False
            and preparation['human_GO'] is False, 'Wrong actual projection preparation scope')
    require(release['schema'] == 'portable-host-production-release-v2' and release['host'] == 'allocation'
            and release['ownership'] == preparation['ownership'] and release['all_tasks_qualified'] is False
            and release['human_GO'] is False and release['TEST_access'] is False,
            'Only the actual unadmitted allocation projection can be rebound')
    expected_root = EXPANSION / 'campaign_outputs' / CAMPAIGN
    require(confined(release['output_root']) == expected_root, 'Wrong portable campaign output root')
    require(manifest['schema'] == 'portable-dispatcher-source-package-v2'
            and manifest['server77_enabled'] is False and manifest['human_GO'] is False,
            'Wrong reviewed allocation portable source package')
    package_files = {}
    for row in manifest['files']:
        path = confined(PORTABLE / row['path'])
        require(path.is_relative_to(PORTABLE), 'Portable source receipt escapes its package')
        actual = pin(path)
        require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'Portable source bytes changed')
        require(row['path'] not in package_files, 'Duplicate portable source receipt')
        package_files[row['path']] = actual
    delta = verified(package_files['RUNTIME_DELTA.json'])
    require(delta['schema'] == 'portable-runtime-source-delta-v2'
            and delta['native_training_selection_checkpoint_export_functions_AST_unchanged'] is True
            and delta['native_execution_performed'] is False, 'Reviewed native function projection missing')
    delta_files = {row['target']: row for row in delta['files']}
    for row in delta['files']:
        original = pin(EXPANSION / row['source'])
        require(original['sha256'] == row['source_sha256']
                and package_files[row['target']]['sha256'] == row['target_sha256'],
                'Portable runtime delta authority differs')
    # This verifies expected input metadata only. It does not open graph,
    # checkpoint, tensor or prediction data; the production observer does that
    # source/data byte verification before any future native invocation.
    canonical_source = next(row for row in parent['source_inputs']
                            if row['path'].endswith('/expanded_dispatcher_v2/generated_v4/PLAN.json'))
    require(canonical_source['sha256'] ==
            '42973f8c9573b3d782879eff0da7aee3546d1416dccce31f6f3ab0c120059351', 'Wrong canonical plan')
    canonical = verified(canonical_source)
    require(verified(release['canonical_skeleton']) == canonical, 'Portable skeleton differs from canonical plan')
    ownership = verified(preparation['ownership'])
    require(ownership['schema'] == 'portable-global-ownership-v2' and ownership['campaign'] == CAMPAIGN,
            'Wrong permanent ownership generation')
    by_id = {row['id']: row for row in candidates}
    tasks = {row['id']: row for row in release['tasks']}
    originals = {row['id']: row for row in canonical['tasks']}
    owners = {row['id']: row for row in ownership['assignments']}
    require(len(by_id) == len(candidates) == len(tasks) == len(release['tasks']) ==
            len(originals) == len(owners) == len(ownership['assignments']) == 6730
            and set(by_id) == set(tasks) == set(originals) == set(owners), 'Exact complete task inventory required')
    for task_id, task in tasks.items():
        candidate, original, owner = by_id[task_id], originals[task_id], owners[task_id]
        require(candidate['worker'] == task['worker'] == owner['worker'] == 'allocation/gpu0'
                and candidate['admitted'] is False, 'No77 or execution admission is permitted by this rebind')
        require(set(candidate) == {'id', 'worker', 'job_descriptor', 'runtime_entrypoint',
                                   'canonical_runtime_job_hash', 'runtime_job_hash', 'admitted'}, 'Unknown candidate fields')
        require(candidate['canonical_runtime_job_hash'] == original.get('runtime_job_hash')
                == owner['canonical_runtime_job_hash'], 'Canonical runtime identity changed')
        for key in ('science_contract_hash', 'dependencies', 'output_relative'):
            require(owner[key] == original[key], 'Ownership changed canonical ' + key)
        require(owner['cohort'] == original['group_id'], 'Ownership changed cohort')
        require(candidate['runtime_job_hash'] == original.get('runtime_job_hash'),
                'Allocation projection must preserve the exact canonical job hash')
    output.mkdir()
    profiles, all_seen = [], set()
    counts = {key: 0 for key in EXPECTED_COUNTS}
    for parent_pin in parent['preset_files']:
        old = verified(parent_pin)
        require(old['schema'] == 'real-compiler-reporting-preset-v2', 'Wrong canonical reporting profile')
        profile = copy.deepcopy(old)
        preset_id = old['registry']['presets'][0]['id']
        for task_id, spec in profile['fit_specs'].items():
            prior = old['fit_specs'][task_id]
            task, canonical_task, candidate = tasks[task_id], originals[task_id], by_id[task_id]
            job, projected = verified(prior['descriptor']), verified(candidate['job_descriptor'])
            require(job == projected and digest(job) == prior['job_hash'] == candidate['runtime_job_hash']
                    and canonical_task['source_descriptor'] == prior['descriptor'],
                    'Projected allocation descriptor differs from canonical science')
            require(job['id'] == task_id and job['expanded_contract_hash'] == canonical_task['science_contract_hash']
                    == digest(job['expanded_contract']), 'Scientific contract differs')
            expected_task = copy.deepcopy(canonical_task)
            expected_task.update(worker='allocation/gpu0', canonical_runtime_job_hash=prior['job_hash'],
                                 runtime_binding=expected_binding(job), binding=None,
                                 execution_status='blocked_missing_qualification')
            require(task == expected_task, 'Projected task changed canonical fields or admission')
            family = 'runtime_link' if job['family'] == 'link' else 'runtime_node_molecular'
            target = family + '/run_fit.py'
            require(candidate['runtime_entrypoint'] == package_files[target], 'Wrong exact portable fit entrypoint')
            row = delta_files[target]
            require(prior['runtime_entrypoint'] == pin(EXPANSION / row['source']),
                    'Canonical fit runtime is not the reviewed portable parent')
            spec['canonical_execution'] = {key: copy.deepcopy(prior[key])
                                           for key in ('descriptor', 'job_hash', 'runtime_entrypoint')}
            spec['canonical_execution']['science_contract_hash'] = task['science_contract_hash']
            spec['descriptor'] = candidate['job_descriptor']
            spec['job_hash'] = candidate['runtime_job_hash']
            spec['runtime_entrypoint'] = candidate['runtime_entrypoint']
            spec['wrapper_authority']['descriptor'] = candidate['job_descriptor']
            require(common_projection(spec) == common_projection(prior)
                    and spec['bindings'] == prior['bindings'] == spec['identity']['bindings']
                    and spec['observed_input_files'] == prior['observed_input_files']
                    and digest(job['policy']) == spec['bindings']['controller'],
                    'Common native/data/controller/evaluator projections changed')
            all_seen.add(task_id)
        replacements = {}
        for task_id, spec in profile['I4_specs'].items():
            prior = old['I4_specs'][task_id]
            task, canonical_task, candidate = tasks[task_id], originals[task_id], by_id[task_id]
            expected_task = copy.deepcopy(canonical_task)
            expected_task.update(worker='allocation/gpu0', canonical_runtime_job_hash=canonical_task.get('runtime_job_hash'),
                                 runtime_binding=None, binding=None, execution_status='blocked_missing_qualification')
            require(task == expected_task and task['dependencies'] == spec['component_fit_ids']
                    and candidate['job_descriptor'] is None, 'I4 science or components changed')
            target = 'runtime_I4/aggregate_selected.py'
            require(candidate['runtime_entrypoint'] == package_files[target]
                    and prior['aggregator_source'] == pin(EXPANSION / delta_files[target]['source']),
                    'Wrong exact portable I4 aggregator or canonical parent')
            spec['canonical_execution'] = {'aggregator_source': prior['aggregator_source'],
                                           'readout_binding': prior['identity']['bindings']['readout']}
            spec['aggregator_source'] = candidate['runtime_entrypoint']
            source = {key: spec['aggregator_source'][key] for key in ('bytes', 'sha256')}
            spec['identity']['bindings']['readout'] = digest({
                'source': source, 'aggregation': 'mean_raw_logits_after_four_own_selectors'})
            require(common_projection(spec) == common_projection(prior), 'I4 common projections changed')
            replacements[task_id] = spec['identity']
            all_seen.add(task_id)
        profile['registry']['expected_predictors'] = [copy.deepcopy(replacements.get(row['fit_id'], row))
                                                     for row in old['registry']['expected_predictors']]
        require(profile['registry']['presets'] == old['registry']['presets']
                and profile['contrast_plan'] == old['contrast_plan'], 'Cohorts or prescribed contrasts changed')
        profile['portable_rebind'] = {'schema': 'allocation-reporting-profile-execution-rebind-v1',
                                     'canonical_profile': parent_pin, 'worker': 'allocation/gpu0',
                                     'science_and_common_projections_unchanged': True,
                                     'actual_input_bytes_reverified_by_this_metadata_build': False}
        path = output / (preset_id + '.json')
        write(path, profile)
        profiles.append(pin(path))
        counts['physical_fits'] += len(profile['fit_specs'])
        counts['I4_exports'] += len(profile['I4_specs'])
        rows = profile['registry']['expected_predictors']
        counts['expected_predictor_rows'] += len(rows)
        counts['source_reference_rows'] += sum(row['track'] == 'source_reference' for row in rows)
        counts['same_bank_mean_rows'] += sum(row['readout'] == 'same_selected_bank_mean' for row in rows)
    require(counts == EXPECTED_COUNTS and all_seen == set(tasks), 'Rebind lost or duplicated campaign tasks')
    index = copy.deepcopy(parent)
    index.update(preset_files=profiles, campaign_output_root=release['output_root'],
                 canonical_registry_index=pin(sources['parent_index']),
                 portable_rebind={'schema': 'allocation-reporting-registry-execution-rebind-v1',
                                  'worker': 'allocation/gpu0', 'inputs': {k: pin(v) for k, v in sources.items()},
                                  'source': pin(__file__), 'canonical_science_unchanged': True,
                                  'common_native_data_controller_evaluator_bindings_unchanged': True,
                                  'I4_readout_source_binding_recomputed_canonical_binding_retained': True,
                                  'observed_input_authority_arrays_unchanged': True,
                                  'actual_data_bytes_reverified': False, 'native_jobs_started': 0,
                                  'server77_accessed': False, 'native_or_resource_admission': False,
                                  'human_GO': False})
    write(output / 'INDEX.json', index)
    print(json.dumps({'index': pin(output / 'INDEX.json'), 'counts': counts,
                      'canonical_tasks_verified': len(all_seen), 'native_jobs_started': 0,
                      'actual_data_bytes_reverified': False, 'human_GO': False}))


if __name__ == '__main__':
    main()
