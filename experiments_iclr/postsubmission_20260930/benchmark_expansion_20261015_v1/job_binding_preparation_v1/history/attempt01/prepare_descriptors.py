"""Prepare preserved predecessor fit descriptors; never constructs or launches a fit.

Stdlib only. Execution is restricted to the guarded allocation and writes only
to a new child of this package. Full descriptors remain on that allocation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT/'experiments_iclr/postsubmission_20260930'
EXPANSION = PHASE/'benchmark_expansion_20261015_v1'
DESIGN = PHASE/'reference_wrapper_redesign_20261013_v1'
HERE = EXPANSION/'job_binding_preparation_v1'
PINS = {
    'PLAN.json': 'f60564b4b5e20a629b65c2de1dd8e37c3a0eca3906db432c0b28e247aede0ea6',
    'ROSTER.json': 'f6d0e8c6abd67ba8ce63c72bf41b8909fb5c13ce09f3daa6a9aa199bcebd622a',
    'smoke_20261014/MANIFEST.json': 'f95d4ddc652becbe7f2379e19ef4ec2fd09fe0f6f3026b080c1bfd9410f28371',
}
RNG_AUDIT = EXPANSION/'qualification/runtime_v2_source_rng_v1.json'
RNG_AUDIT_SHA = 'd8be44fce4ac8d531689f67547e2d4069fd354af8dc571fbf5cffcfe12bbad4c'
LINK_CALLBACK_SHA = '791c06dbacd0e9a53ad584fa3591a4a3f50b05e600959db4db06deff17fbef24'
LINK_POLICY_SHA = '732eaf774347b0c1eb48d30f436a7315af4d8cc32bd3b7b43685facfcb818fbc'


def guard():
    assert socket.gethostname() == 'anogena-2-0', 'Wrong allocation; no project access'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong GPU identity; no project access'
    assert Path(__file__).resolve().parent == HERE, 'Run the allocation copy of this package'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT), 'Path escapes actual repository: '+str(path)
    return path


def receipt(path):
    path = inside(path)
    if not path.is_file():
        return {'path': str(path), 'exists': False}
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block); size += len(block)
    return {'path': str(path), 'exists': True, 'bytes': size, 'sha256': h.hexdigest()}


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def family(recipe):
    if recipe['dataset'].startswith('ogbg-'):
        return 'ogb'
    return 'link' if recipe['dataset'] == 'ogbl-collab' else 'node'


def make_policy(kind, source):
    cap, minimum, patience, metric = {
        ('node', True): (2000, 0, 250, 'accuracy'),
        ('node', False): (10000, 2000, 100, 'accuracy'),
        ('ogb', True): (100, 100, None, 'rocauc'),
        ('ogb', False): (1000, 100, 50, 'rocauc'),
        ('link', True): (100, 100, None, 'hits@50'),
        ('link', False): (1000, 100, 50, 'hits@50'),
    }[(kind, source)]
    result = {'track': 'source_reference' if source else 'matched_extended',
              'maximum_epochs': cap, 'minimum_epochs': minimum, 'patience': patience,
              'metric': metric, 'direction': 'max', 'ties': 'first',
              'evaluate_TRAIN_each_epoch': kind == 'ogb' and source}
    if kind == 'link':
        result.update(family='link', evaluation_mode='VALID_only_disclosed_adaptation')
    return result


def data_paths(case):
    recipe, root = case['recipe'], Path(case['data_root'])
    kind = family(recipe)
    if kind == 'link':
        assert set(recipe['data_files']) == {'graph', 'train', 'valid'}
        return copy.deepcopy(recipe['data_files'])
    if kind == 'ogb':
        dataset = root/recipe['dataset'].replace('-', '_')
        return {'shared_processed_graph_container': str(dataset/'processed/geometric_data_processed.pt'),
                'TRAIN_indices': str(dataset/'split/scaffold/train.csv.gz'),
                'VALID_indices': str(dataset/'split/scaffold/valid.csv.gz')}
    name = {'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'coauthor-cs': 'CS', 'coauthor-physics': 'Physics'}[recipe['dataset']]
    return {'shared_processed_graph_container': str(root/recipe['dataset']/name/'processed/data.pt'),
            'native_polynomial_basis_cache': str(DESIGN/'smoke_20261014/working_data/node_bases'/(recipe['id']+'.pt'))}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def prepare(output_name):
    guard()  # Must precede every project read/write/import.
    assert output_name.replace('_', '').isalnum() and '/' not in output_name
    output = HERE/output_name
    assert not output.exists(), 'Fresh output required; existing evidence is preserved'
    inputs = {}
    for relative, expected in PINS.items():
        pin = receipt(DESIGN/relative)
        assert pin.get('sha256') == expected, 'Frozen source changed: '+relative
        inputs[relative] = pin
    plan = json.loads((DESIGN/'PLAN.json').read_text())
    roster = json.loads((DESIGN/'ROSTER.json').read_text())
    smoke = json.loads((DESIGN/'smoke_20261014/MANIFEST.json').read_text())
    fits = [row for row in plan['jobs'] if row['kind'] == 'fit']
    assert len(fits) == 2480 and len(roster['recipes']) == 27
    recipes = {row['id']: row for row in roster['recipes']}
    assert len(recipes) == 27 and len({row['id'] for row in fits}) == 2480
    assert Counter(row['contract']['track'] for row in fits) == {'native_replica': 270, 'long_convergence': 2210}
    cases = {(row['recipe']['id'], row['procedure']): row for row in smoke['cases']}
    groups = {job_id: group for group in plan['groups'] for job_id in group['job_ids']}
    assert all(row['id'] in groups for row in fits)
    native_manifest = json.loads((DESIGN/'native_sources/MANIFEST.json').read_text())
    source_receipts = [receipt(DESIGN/'native_sources/MANIFEST.json')]
    for row in native_manifest['files']:
        actual = receipt(PHASE/row['snapshot_path'])
        assert actual.get('sha256') == row['sha256'] and actual.get('bytes') == row['bytes'], row['snapshot_path']
        source_receipts.append(actual)
    extra_paths = list((EXPANSION/'runtime_v2').glob('*.py'))
    extra_paths += list((EXPANSION/'link_native_integration_v1').glob('*.py'))
    extra_paths += [EXPANSION/'link_policy_v1/policy.py', EXPANSION/'link_policy_v1/BINDINGS.json']
    extra_paths += list((DESIGN/'smoke_20261014/ogb_port').glob('*.py'))
    extra_paths += list((DESIGN/'smoke_20261014/link_port').glob('*.py'))
    extra_paths += [DESIGN/'smoke_20261014/node_port.py', DESIGN/'runtime/training_rules.py', DESIGN/'runtime/boundary_core.py']
    source_receipts += [receipt(path) for path in sorted(set(extra_paths))]
    assert all(row['exists'] for row in source_receipts)
    pins_by_path = {row['path']: row for row in source_receipts}
    assert pins_by_path[str(EXPANSION/'link_native_integration_v1/session.py')]['sha256'] == LINK_CALLBACK_SHA
    assert pins_by_path[str(EXPANSION/'link_policy_v1/policy.py')]['sha256'] == LINK_POLICY_SHA
    audit_pin = receipt(RNG_AUDIT)
    assert audit_pin.get('sha256') == RNG_AUDIT_SHA
    audit = json.loads(RNG_AUDIT.read_text())
    assert audit['schema'] == 'omitted-source-test-iterator-cpu-rng-audit-v1'
    assert audit['TEST_accessed'] is False and audit['model_calls'] == 0
    assert len(audit['semantic_check']['cases']) == 36
    # These validators import stdlib only; neither adapter nor create_session runs.
    sys.path.insert(0, str(EXPANSION/'runtime_v2'))
    runtime = module(EXPANSION/'runtime_v2/run_fit.py', '_predecessor_metadata_validator')
    link_policy = module(EXPANSION/'link_policy_v1/policy.py', '_predecessor_link_policy')
    link_policy.source_bindings(ROOT)
    cell_bindings, data_cache = {}, {}
    for recipe_id, recipe in recipes.items():
        case = cases[(recipe_id, 'native_single')]
        assert all(case['recipe'].get(key) == value for key, value in recipe.items())
        paths = data_paths(case)
        for path in paths.values():
            if path not in data_cache:
                data_cache[path] = receipt(path)
        kind = family(recipe)
        callback = EXPANSION/'link_native_integration_v1/session.py' if kind == 'link' else Path(case['callback_module'])
        gaps = []
        for role, path in paths.items():
            if not data_cache[path]['exists']:
                gaps.append('missing_data_role:'+role+':'+path)
        if kind == 'node':
            gaps.append('per_split_mask_materialization_not_verified_by_this_metadata_preparation')
        cell_bindings[recipe_id] = {
            'family': kind, 'dataset': recipe['dataset'], 'native_fixed_config': copy.deepcopy(recipe['native_fixed_config']),
            'callback': receipt(callback), 'smoke_callback': receipt(case['callback_module']),
            'smoke_case_id': case['id'], 'smoke_manifest': inputs['smoke_20261014/MANIFEST.json'],
            'data_root': str(inside(case['data_root'])), 'data_role_receipts': {role: data_cache[path] for role, path in paths.items()},
            'official_split_bindings': copy.deepcopy(recipe['official_split_bindings']),
            'native_seed_offsets': copy.deepcopy(recipe['native_seed_offsets']),
            'gaps': gaps, 'data_receipt_scope': 'Streaming byte hashes only; no tensor, task dataset, split loader, TEST iterator or model constructed',
        }
    output.mkdir()
    jobs_root = output/'jobs'; jobs_root.mkdir()
    index, counts = [], Counter()
    for source_job in fits:
        contract = source_job['contract']; recipe_id = contract['recipe_id']; row = cell_bindings[recipe_id]
        assert digest(contract) == source_job['contract_hash']
        recipe = copy.deepcopy(cases[(recipe_id, 'native_single')]['recipe'])
        split, seed, body = contract['split'], contract['initialization_seed'], contract['body']
        assert recipe['official_split_bindings'][str(split)] == contract['role_bindings']
        assert contract['native_config'] == recipe['native_fixed_config']
        assert {'seed': contract['cohort_seed'], 'split': split} in recipe['cohorts_by_track'][contract['track']]
        source = contract['track'] == 'native_replica'; kind = row['family']
        if contract['model_procedure'] == 'native_single':
            assert seed == contract['cohort_seed'] + recipe['native_seed_offsets'][body]
        recipe['_repository_root'] = str(ROOT)
        recipe['_redesign_root'] = str(DESIGN)
        recipe['_seed'] = seed; recipe['seed'] = seed
        job = {'schema': 'native-full-epoch-fit-v1', 'id': source_job['id'],
               'cohort_id': groups[source_job['id']]['id'], 'family': kind, 'recipe_id': recipe_id,
               'recipe': recipe, 'callback_module': row['callback']['path'],
               'callback_sha256': row['callback']['sha256'], 'data_root': row['data_root'],
               'procedure': contract['model_procedure'], 'base_procedure': contract['model_procedure'],
               'seed': seed, 'cohort_seed': contract['cohort_seed'], 'split': split, 'body': body,
               'device': 'cuda:0', 'torch_threads': 2, 'TEST_access': False,
               'policy': make_policy(kind, source), 'source_plan_entry': copy.deepcopy(source_job),
               'source_plan_contract_hash': source_job['contract_hash'],
               'descriptor_preparation': {'scientific_launch_enabled': False, 'resource_admission': False,
                   'final_expanded_campaign_count': False, 'descriptor_custody': 'authorized_allocation_only',
                   'historical_plan_worker': groups[source_job['id']]['worker'],
                   'historical_assignment_is_runtime_binding': False,
                   'server77_root_environment_assignment': 'blocked_unverified_no_access',
                   'cell_binding_authority': str(output/'CELL_BINDINGS.json'),
                   'source_plan': inputs['PLAN.json'], 'source_roster': inputs['ROSTER.json']}}
        if kind == 'link':
            job['link_policy_module'] = str(EXPANSION/'link_policy_v1/policy.py')
            job['link_policy_sha256'] = LINK_POLICY_SHA
            recipe['_native_callback'] = row['smoke_callback']['path']
            recipe['_native_callback_sha256'] = row['smoke_callback']['sha256']
        if kind == 'ogb' and source:
            job['source_rng_emulation'] = {'mode': audit['mode'], 'audit': str(RNG_AUDIT), 'audit_sha256': RNG_AUDIT_SHA}
        runtime.validate_job(job, ROOT)
        target = jobs_root/(hashlib.sha256(job['id'].encode()).hexdigest()+'.json')
        file_pin = write(target, job)
        index.append({'id': job['id'], 'recipe_id': recipe_id, 'family': kind, 'track': contract['track'],
                      'contract_hash': source_job['contract_hash'], 'runtime_job_hash': digest(job),
                      'descriptor': file_pin, 'historical_worker': groups[source_job['id']]['worker']})
        counts[kind+'/'+contract['track']] += 1
    ids = {row['id']: row for row in fits}
    main_m1 = main_i4 = 0
    for bank in plan['logical_banks']:
        assert all(job_id in ids for job_id in bank['fit_ids'])
        if bank['id'].endswith('/native-I4'):
            assert bank['track'] == 'long_convergence' and len(bank['fit_ids']) == 4
            assert len(set(bank['fit_ids'])) == 4 and bank['exact_M1_body0_reuse'] is True
            own = [ids[job_id]['contract'] for job_id in bank['fit_ids']]
            assert [x['body'] for x in own] == [0, 1, 2, 3]
            assert len({x['initialization_seed'] for x in own}) == 4
            m1_id = bank['id'].removesuffix('native-I4')+'native-M1'
            m1 = next(row for row in plan['logical_banks'] if row['id'] == m1_id)
            assert m1['fit_ids'] == bank['fit_ids'][:1]
            main_i4 += 1
        if bank['track'] == 'long_convergence' and bank['id'].endswith('/native-M1'):
            main_m1 += 1
    assert main_m1 == main_i4 == 270
    files = {}
    files['cells'] = write(output/'CELL_BINDINGS.json', {'schema': 'predecessor-cell-bindings-v1', 'cells': cell_bindings})
    files['sources'] = write(output/'SOURCE_RECEIPTS.json', {'inputs': inputs, 'source_files': source_receipts, 'source_rng_audit': audit_pin})
    files['index'] = write(output/'DESCRIPTOR_INDEX.json', {'schema': 'predecessor-fit-descriptor-index-v1', 'fits': index})
    files['logical_banks'] = write(output/'LOGICAL_BANKS.json', {'source_plan': inputs['PLAN.json'], 'logical_banks': plan['logical_banks']})
    files['statistical_jobs'] = write(output/'STATISTICAL_JOBS.json', {'source_plan': inputs['PLAN.json'], 'jobs': [row for row in plan['jobs'] if row['kind'] != 'fit']})
    assert not any(name in sys.modules for name in ('torch', 'torch_geometric', 'ogb', 'numpy'))
    gaps = [
        'measured_per_cell_method_resource_and_cost_bindings_missing',
        'full_epoch_runtime_and_numerical_qualification_not_established_for_all_27_cells_and_methods',
        'giant_campaign_human_verification_and_GO_missing',
        'server77_root_environment_device_mapping_and_runtime_receipts_blocked_unverified',
        'expanded_method_roster_and_final_expanded_fit_count_not_frozen_by_this_predecessor_package',
    ]
    readiness = {'schema': 'predecessor-runtime-descriptor-preparation-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'descriptors_prepared_not_admitted',
        'scope': 'Preserved predecessor fits only; not final expanded campaign count',
        'allocation_guard': {'hostname': socket.gethostname(), 'sole_gpu_uuid': 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'},
        'source_inputs': inputs, 'remote_output': str(output), 'descriptor_count': len(index),
        'separate_source_reference_fits': 270, 'main_fits': 2210, 'statistical_evaluate_roster_jobs': 540,
        'counts_by_family_and_track': dict(counts), 'cells': len(cell_bindings),
        'main_M1_banks': main_m1, 'main_I4_banks': main_i4,
        'exact_M1_body0_reuse_verified': True, 'four_independent_own_native_fit_ids_verified': True,
        'preserved_original_PLAN_entries_and_contract_hashes': True,
        'historical_worker_fit_counts': dict(Counter(row['historical_worker'] for row in index)),
        'all_descriptors_held_on_allocation': True,
        'server77_accessed': False, 'server77_runtime_binding': None,
        'metadata_validate_job_passes': len(index), 'models_constructed': 0, 'fits_started': 0,
        'GPU_compute_started': False, 'TEST_provider_loaded': False, 'TEST_predictions_or_metrics': False,
        'source_OGB_iterator_emulation_descriptors': 160, 'source_rng_audit': audit_pin,
        'whole_paper_replication_claim': False, 'scientific_launch_enabled': False,
        'resources_measured': False, 'runtime_admitted': False, 'compiled_campaign': False,
        'human_GO_received': False, 'final_expanded_campaign_count': False,
        'missing_deployment_roles': gaps,
        'cell_gaps': {key: value['gaps'] for key, value in cell_bindings.items() if value['gaps']},
        'artifacts': files}
    ready_pin = write(output/'READINESS.json', readiness)
    return {'readiness': readiness, 'readiness_receipt': ready_pin}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', default='generated_v1')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_name), sort_keys=True))
