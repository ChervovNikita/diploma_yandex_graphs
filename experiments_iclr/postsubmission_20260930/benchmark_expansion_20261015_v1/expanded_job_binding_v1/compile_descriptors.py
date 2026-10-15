"""Compile all proposed physical fit intentions from frozen source/JSON metadata.

Allocation-only, stdlib-only. No model/transform imports, data/tensor reads,
resource allocation, scientific fits, TEST provider, or numerical computation.
Full fit JSON stays on allocation. Missing source/context emits a blocked record.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT/'experiments_iclr/postsubmission_20260930'
EXPANSION = PHASE/'benchmark_expansion_20261015_v1'
DESIGN = PHASE/'reference_wrapper_redesign_20261013_v1'
HERE = EXPANSION/'expanded_job_binding_v1'
PREDECESSOR = EXPANSION/'job_binding_preparation_v1/generated_v2'
PINS = {
    'PLAN.json': 'f60564b4b5e20a629b65c2de1dd8e37c3a0eca3906db432c0b28e247aede0ea6',
    'ROSTER.json': 'f6d0e8c6abd67ba8ce63c72bf41b8909fb5c13ce09f3daa6a9aa199bcebd622a',
    'smoke_20261014/MANIFEST.json': 'f95d4ddc652becbe7f2379e19ef4ec2fd09fe0f6f3026b080c1bfd9410f28371',
    'native_sources/MANIFEST.json': '27bbb9ef1a0f2ca13aa28d426707cfd7dd724af4af9ed77b66742e95de278467',
}
PREDECESSOR_INDEX_SHA = 'a57b3ec822f1b47509293bb9782b3d092b20cdc9870c55b97adc6e75751ebcfb'
ALIASES = {'boundary_global_half_pool': 'aggregation_mean_globalJ_v1'}
FORBIDDEN_IMPORTS = ('torch', 'torch_geometric', 'ogb', 'numpy', 'scipy')


def guard():
    assert socket.gethostname() == 'anogena-2-0', 'Wrong allocation; no project access'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong GPU; no project access'
    assert Path(__file__).resolve().parent == HERE


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def metadata_path(path):
    path = Path(path).resolve()
    assert path.is_relative_to(PHASE), 'Escaping research phase: '+str(path)
    assert path.suffix in {'.json', '.py', '.md', '.txt', '.xml'}, 'Only source/text/JSON metadata may be read: '+str(path)
    assert not ({'working_data', 'datasets', 'processed', 'raw', 'results', '.cache'} & set(path.relative_to(PHASE).parts)), 'Data/result access excluded: '+str(path)
    return path


def receipt(path):
    path = metadata_path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def authenticate(path, expected):
    actual = receipt(path)
    assert actual['sha256'] == expected['sha256'], 'SHA mismatch: '+str(path)
    if 'bytes' in expected:
        assert actual['bytes'] == expected['bytes'], 'Size mismatch: '+str(path)
    return actual


def read_json(path, expected=None):
    path = metadata_path(path)
    if expected:
        authenticate(path, expected)
    return json.loads(path.read_text())


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)+'\n').encode()
    with path.open('xb') as out:
        out.write(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def exported(pin):
    path = metadata_path(PHASE/pin['path_relative_to_phase'])
    actual = authenticate(path, pin)
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert pin['function'] in names, 'Missing exact export: '+str(path)+':'+pin['function']
    return {**actual, 'function': pin['function']}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, metadata_path(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def make_policy(kind, source):
    cap, minimum, patience, metric = {
        ('node', True): (2000, 0, 250, 'accuracy'), ('node', False): (10000, 2000, 100, 'accuracy'),
        ('ogb', True): (100, 100, None, 'rocauc'), ('ogb', False): (1000, 100, 50, 'rocauc'),
        ('link', True): (100, 100, None, 'hits@50'), ('link', False): (1000, 100, 50, 'hits@50'),
    }[(kind, source)]
    value = {'track': 'source_reference' if source else 'matched_extended', 'maximum_epochs': cap,
             'minimum_epochs': minimum, 'patience': patience, 'metric': metric, 'direction': 'max',
             'ties': 'first', 'evaluate_TRAIN_each_epoch': kind == 'ogb' and source}
    if kind == 'link':
        value.update(family='link', evaluation_mode='VALID_only_disclosed_adaptation')
    return value


def check_fixed_controller(policy, row, source):
    value = row['source_reference_controller' if source else 'matched_controller']
    assert value['maximum'] == policy['maximum_epochs']
    assert value['optimizer'] == row['native_optimizer']
    assert value['selection']['ties'] == policy['ties']
    assert value['selection']['components'] == [{'metric': policy['metric'], 'direction': 'max'}]
    if not source:
        assert value['stopping']['minimum_units'] == policy['minimum_epochs']
        assert value['stopping']['patience_validation_checks'] == policy['patience']
    elif policy['patience'] is not None:
        assert value['native_stopping']['patience'] == policy['patience']
    return copy.deepcopy(value)


def compile_all(output_name, registry_name):
    guard()  # Must precede project metadata reads/writes/imports.
    assert output_name.replace('_', '').isalnum() and '/' not in output_name
    assert Path(registry_name).name == registry_name and registry_name.endswith('.json')
    output = HERE/output_name
    assert not output.exists(), 'Fresh generation required; preserve existing evidence'
    registry_path = HERE/registry_name
    registry_pin = receipt(registry_path)
    registry = read_json(registry_path)
    assert registry['schema'] == 'expanded-fit-source-registry-v1' and registry['runnable'] is False
    source_inputs = {}
    for name, expected in PINS.items():
        source_inputs[name] = authenticate(DESIGN/name, {'sha256': expected})
    plan = read_json(DESIGN/'PLAN.json')
    native_roster = read_json(DESIGN/'ROSTER.json')
    recipes = {r['id']: r for r in native_roster['recipes']}
    for role in ('proposed_roster', 'native_settings', 'input_audit', 'inherited_cell_bindings', 'runtime_manifest'):
        pin = registry[role]
        source_inputs[role] = authenticate(PHASE/pin['path_relative_to_phase'], pin)
    proposed = read_json(source_inputs['proposed_roster']['path'])
    settings = {r['preset']: r for r in read_json(source_inputs['native_settings']['path'])['rows']}
    cells = {r['id']: r for r in proposed['cells']}
    inherited_cells = read_json(source_inputs['inherited_cell_bindings']['path'])['cells']
    assert len(recipes) == len(settings) == len(cells) == len(inherited_cells) == 27
    assert proposed['totals'] == {'matched_native':1080, 'physical_fits':6460, 'shared':5110, 'source':270}
    assert proposed['parent_PLAN_sha256'] == PINS['PLAN.json']
    source_receipts = []
    for pin in registry['source_files']:
        source_receipts.append(authenticate(PHASE/pin['path_relative_to_phase'], pin))
    # Verify the actual native-source snapshot, restricted to source files only.
    snapshot = read_json(DESIGN/'native_sources/MANIFEST.json')
    for pin in snapshot['files']:
        if Path(pin['snapshot_path']).suffix == '.py':
            source_receipts.append(authenticate(PHASE/pin['snapshot_path'], pin))
    resolved = {}
    for row in registry['entries']:
        key = (row['cell_id'], row['method_id'])
        assert key not in resolved and row['method_id'] in cells[row['cell_id']]['shared_method_ids']
        value = copy.deepcopy(row)
        if row.get('callback'):
            value['callback'] = exported(row['callback'])
            if row.get('session_transform'):
                value['session_transform'] = exported(row['session_transform'])
            if cells[row['cell_id']]['task_family'] == 'link':
                assert value['callback']['function'] == 'create_adapter' and not value.get('session_transform')
            else:
                assert value['callback']['function'] == 'create_session'
            if row.get('context_manifest_receipt'):
                pin = row['context_manifest_receipt']
                source_receipts.append(authenticate(pin['path'], pin))  # JSON metadata; never its tensor references.
        resolved[key] = value
    assert len(resolved) == 511
    runtime_path = metadata_path(PHASE/registry['runtime_package']/'run_fit.py')
    sys.path.insert(0, str(runtime_path.parent))
    runtime = module(runtime_path, '_expanded_descriptor_metadata_validator')
    assert not any(name in sys.modules for name in FORBIDDEN_IMPORTS)
    predecessor_index_pin = authenticate(PREDECESSOR/'DESCRIPTOR_INDEX.json', {'sha256':PREDECESSOR_INDEX_SHA})
    predecessor_index = read_json(predecessor_index_pin['path'])['fits']
    old_index = {r['id']:r for r in predecessor_index}
    source_jobs = {r['id']:r for r in plan['jobs'] if r['kind']=='fit'}
    assert len(old_index) == len(source_jobs) == 2480 and old_index.keys() == source_jobs.keys()
    original_jobs, shared_lookup, templates = {}, {}, {}
    for job_id, index in old_index.items():
        path = metadata_path(index['descriptor']['path'])
        assert path.parent == PREDECESSOR/'jobs'
        value = read_json(path, index['descriptor'])
        assert digest(value) == index['runtime_job_hash'] and value['id'] == job_id
        source_job = source_jobs[job_id]
        assert source_job == value['source_plan_entry']
        assert digest(source_job['contract']) == source_job['contract_hash'] == index['contract_hash']
        original_jobs[job_id] = value
        contract = source_job['contract']
        if contract['model_procedure'] != 'native_single':
            scientific = ALIASES.get(contract['model_procedure'], contract['model_procedure'])
            shared_lookup[(contract['recipe_id'], contract['split'], contract['cohort_seed'], scientific)] = job_id
        if contract['model_procedure'] == 'gnnm_boundary_native':
            templates[(contract['recipe_id'], contract['split'], contract['cohort_seed'])] = job_id
    assert len(shared_lookup) == 1130 and len(templates) == 270
    output.mkdir()
    (output/'jobs').mkdir(); (output/'blocked').mkdir()
    index_rows, job_by_id, blocked_rows = [], {}, []
    counts, validation_counts, block_counts, changes = Counter(), Counter(), Counter(), Counter()
    method_definitions = {r['id']:r for r in proposed['methods']}
    intentions = []
    for job_id, old in original_jobs.items():
        contract = source_jobs[job_id]['contract']
        if contract['model_procedure'] == 'native_single':
            intentions.append((job_id, contract['recipe_id'], contract['split'], contract['cohort_seed'], 'native_single', old))
    for cell_id, cell in cells.items():
        cohorts = recipes[cell_id]['cohorts_by_track']['long_convergence']
        assert len(cohorts) == cell['cohorts'] == 10
        assert recipes[cell_id]['native_seed_offsets'] == [0,100003,200003,300007]
        assert settings[cell_id]['native_config'] == recipes[cell_id]['native_fixed_config']
        for cohort in cohorts:
            split, seed = cohort['split'], cohort['seed']
            if cell['task_family'] == 'node':
                assert split == seed
            else:
                assert split == 0
            for method in cell['shared_method_ids']:
                key = (cell_id, split, seed, method)
                job_id = shared_lookup.get(key, f'{cell_id}/long_convergence/split-{split}/seed-{seed}/wrapper-{method}')
                template_id = templates[(cell_id, split, seed)]
                intentions.append((job_id, cell_id, split, seed, method, original_jobs.get(job_id, original_jobs[template_id])))
    assert len(intentions) == 6460 and len({r[0] for r in intentions}) == 6460
    assert set(old_index).issubset({r[0] for r in intentions})
    for job_id, cell_id, split, cohort_seed, method, template in intentions:
        cell, recipe = cells[cell_id], recipes[cell_id]
        source = '/native_replica/' in job_id
        family = cell['task_family']
        ancestor = old_index.get(job_id)
        binding = resolved.get((cell_id, method)) if method != 'native_single' else None
        blocked_reason = None
        if binding and not binding.get('callback'):
            blocked_reason = binding.get('missing_reason', 'native_source_binding_missing')
        elif binding and isinstance(binding.get('supported_seeds'), list) and cohort_seed not in binding['supported_seeds']:
            blocked_reason = binding.get('unsupported_seed_reason', 'seed_context_not_authenticated')
        job = copy.deepcopy(template)
        job.update(id=job_id, recipe_id=cell_id, family=family, cohort_seed=cohort_seed, split=split,
                   scientific_method_id=method, execution_mode='production_gpu', device='cuda:0', TEST_access=False)
        if ancestor is None:
            job.pop('source_plan_entry', None); job.pop('source_plan_contract_hash', None)
            job['seed'] = cohort_seed; job['body'] = None
        assert job['recipe']['native_fixed_config'] == settings[cell_id]['native_config']
        assert job['recipe']['official_split_bindings'][str(split)] == recipe['official_split_bindings'][str(split)]
        assert job['recipe']['_seed'] == job['seed'] and job['recipe']['seed'] == job['seed']
        if method == 'native_single':
            assert job['seed'] == cohort_seed + recipe['native_seed_offsets'][job['body']]
        else:
            assert job['seed'] == cohort_seed
        policy = make_policy(family, source)
        assert job['policy'] == policy
        native_controller = check_fixed_controller(policy, settings[cell_id], source)
        old_procedure = job['procedure']
        if binding and binding.get('callback'):
            callback = binding['callback']
            job.update(callback_module=callback['path'], callback_sha256=callback['sha256'],
                       procedure=binding['runtime_procedure'], base_procedure=binding['base_procedure'],
                       method_settings=copy.deepcopy(binding['method_settings']))
            job.pop('session_transform', None)
            if binding.get('session_transform'):
                transform = binding['session_transform']
                job['session_transform'] = {k:transform[k] for k in ('path','sha256','function')}
        expanded_contract = {
            'schema':'expanded-physical-fit-contract-v1', 'id':job_id, 'cell_id':cell_id, 'family':family,
            'track':'native_replica' if source else 'long_convergence', 'scientific_method_id':method,
            'method_definition': {'id':'native_single', 'independent_native_body':True} if method=='native_single' else method_definitions[method],
            'split':split, 'role_bindings':copy.deepcopy(recipe['official_split_bindings'][str(split)]),
            'cohort_seed':cohort_seed, 'initialization_seed':job['seed'], 'body':job['body'],
            'native_settings':copy.deepcopy(settings[cell_id]), 'native_controller':native_controller,
            'own_optimizer':True, 'own_selection':True, 'historical_state_inherited':False,
            'whole_paper_replication_claim':False,
            'source_registry':registry_pin, 'binding_entry_sha256':None if binding is None else digest(registry['entries'][next(i for i,r in enumerate(registry['entries']) if (r['cell_id'],r['method_id'])==(cell_id,method))]),
            'ancestor_contract_hash':None if ancestor is None else ancestor['contract_hash'],
        }
        job['expanded_contract'] = expanded_contract
        job['expanded_contract_hash'] = digest(expanded_contract)
        job['descriptor_preparation'] = {
            'scientific_launch_enabled':False, 'resource_admission':False, 'human_GO_received':False,
            'descriptor_custody':'authorized_allocation_only', 'registry':registry_pin,
            'runtime_manifest':source_inputs['runtime_manifest'], 'runtime_metadata_validation_only':True,
            'runtime_selected_serving_qualified':False, 'runtime_link_successor_required':family=='link',
            'link_runtime_gaps': ['v4_link_selected_mean_export_missing','v4_outer_extension_receipt_and_dependency_authority_incomplete'] if family=='link' else [],
            'all_native_epoch_lifecycle_qualified':False,
            'native_settings_authority':source_inputs['native_settings'], 'data_receipt_authority':source_inputs['inherited_cell_bindings'],
            'data_receipts_refreshed':False, 'data_or_tensor_files_opened_by_compiler':False,
            'historical_plan_worker':None if ancestor is None else ancestor['historical_worker'],
            'historical_assignment_is_runtime_binding':False, 'resource_binding':None, 'server77_accessed':False,
            'ancestor_descriptor':None if ancestor is None else ancestor['descriptor'],
            'predecessor_template_descriptor':old_index[template['id']]['descriptor'],
            'ancestor_contract_preserved':ancestor is not None,
            'runtime_binding_migration': None if binding is None else {'from_predecessor_procedure':old_procedure,
                'to_runtime_procedure':job['procedure'], 'scientific_method_id':method,
                'alias_preserves_one_physical_fit': method=='aggregation_mean_globalJ_v1' and ancestor is not None},
            'pending': [] if binding is None else binding.get('pending', []),
        }
        row = {'id':job_id, 'cell_id':cell_id, 'family':family,
               'track':expanded_contract['track'], 'method_id':method, 'split':split, 'cohort_seed':cohort_seed,
               'initialization_seed':job['seed'], 'body':job['body'], 'ancestor':ancestor is not None,
               'ancestor_contract_hash':None if ancestor is None else ancestor['contract_hash'],
               'expanded_contract_hash':job['expanded_contract_hash'], 'launch_admitted':False}
        filename = hashlib.sha256(job_id.encode()).hexdigest()+'.json'
        if blocked_reason:
            block_counts[blocked_reason] += 1
            row.update(status='blocked', blocked_reason=blocked_reason, descriptor=None, runtime_job_hash=None,
                       runtime_metadata_validation='not_applicable_missing_source_or_context')
            blocked = {'schema':'expanded-blocked-fit-intention-v1', 'index':row,
                       'expanded_contract':expanded_contract, 'binding':binding, 'source_or_context_required':blocked_reason,
                       'native_hyperparameters_changed':False, 'fallback_or_seed_cache_substitution':False,
                       'scientific_launch_enabled':False}
            row['blocked_record'] = write(output/'blocked'/filename, blocked)
            blocked_rows.append(row)
        else:
            runtime.validate_job(job, ROOT)
            assert not any(name in sys.modules for name in FORBIDDEN_IMPORTS)
            validation_counts['pass'] += 1
            row.update(status='descriptor_prepared_not_admitted', descriptor=write(output/'jobs'/filename, job),
                       runtime_job_hash=digest(job), runtime_metadata_validation='pass_metadata_only')
            if ancestor and old_procedure != job['procedure']:
                changes['preserved_ancestor_id_with_runtime_procedure_alias'] += 1
        counts[family+'/'+expanded_contract['track']+'/'+method] += 1
        index_rows.append(row); job_by_id[job_id] = row
    assert len(index_rows) == 6460 and len(job_by_id) == 6460
    assert sum(r['ancestor'] for r in index_rows) == 2480
    assert sum(r['method_id']=='native_single' and r['track']=='native_replica' for r in index_rows)==270
    assert sum(r['method_id']=='native_single' and r['track']=='long_convergence' for r in index_rows)==1080
    assert sum(r['method_id']!='native_single' for r in index_rows)==5110
    # Physical ownership and symbolic controls resolve to exact paired cohort IDs.
    banks = []
    for bank in plan['logical_banks']:
        if bank['id'].endswith(('/native-M1','/native-I4')):
            banks.append(copy.deepcopy(bank))
    main_bank_ids = {r['id']:r for r in banks}
    controls = []
    for cell_id, cell in cells.items():
        for cohort in recipes[cell_id]['cohorts_by_track']['long_convergence']:
            split, seed = cohort['split'], cohort['seed']
            prefix = f'{cell_id}/long_convergence/split-{split}/seed-{seed}'
            m1, i4 = main_bank_ids[prefix+'/native-M1'], main_bank_ids[prefix+'/native-I4']
            assert i4['fit_ids'][:1] == m1['fit_ids'] and len(set(i4['fit_ids']))==4 and i4['exact_M1_body0_reuse'] is True
            own = [job_by_id[x] for x in i4['fit_ids']]
            assert [x['body'] for x in own]==[0,1,2,3]
            assert [x['initialization_seed'] for x in own] == [seed+v for v in [0,100003,200003,300007]]
            assert all(x['method_id']=='native_single' and not x['launch_admitted'] for x in own)
            method_ids = {r['method_id']:r['id'] for r in index_rows if r['cell_id']==cell_id and r['track']=='long_convergence' and r['split']==split and r['cohort_seed']==seed and r['method_id']!='native_single'}
            for method, fit_id in method_ids.items():
                banks.append({'id':prefix+'/'+method, 'fit_ids':[fit_id], 'scientific_method_id':method,
                              'track':'long_convergence', 'selection':'own selected state', 'extra_fits':0,
                              'fit_status':job_by_id[fit_id]['status']})
                links = []
                for symbolic in method_definitions[method]['controls']:
                    resolved_control = {'task_geometry_alignment_variance_or_link_endpoint': 'link_same_query_endpoint' if cell['task_family']=='link' else 'task_geometry_alignment_variance',
                        'structural_class_quartile_private_or_link_endpoint_v2': 'link_endpoint_responsibility_v2' if cell['task_family']=='link' else 'structural_class_quartile_private_v1'}.get(symbolic,symbolic)
                    if resolved_control == 'same_selected_bank_fixed_mean':
                        links.append({'symbolic_control':symbolic, 'kind':'diagnostic_same_selected_fit', 'fit_ids':[fit_id],
                                      'extra_fits':0, 'selection_changed':False, 'runtime_selected_export_qualification':'pending'})
                    elif resolved_control in {'native_M1','native_I4'}:
                        bank = m1 if resolved_control=='native_M1' else i4
                        links.append({'symbolic_control':symbolic, 'resolved_control':resolved_control, 'kind':'logical_native_bank', 'fit_ids':bank['fit_ids']})
                    else:
                        assert resolved_control in method_ids, (cell_id, method, symbolic)
                        other = method_ids[resolved_control]
                        links.append({'symbolic_control':symbolic, 'resolved_control':resolved_control, 'kind':'paired_shared_fit',
                                      'fit_ids':[other], 'fit_status':job_by_id[other]['status']})
                controls.append({'fit_id':fit_id,'scientific_method_id':method,'controls':links})
    assert len(banks)==5920 and len({r['id'] for r in banks})==5920 and len(controls)==5110
    assert all(x in job_by_id for b in banks for x in b['fit_ids'])
    stat_jobs = [r for r in plan['jobs'] if r['kind']!='fit']
    assert len(stat_jobs)==540
    artifacts = {}
    artifacts['index'] = write(output/'DESCRIPTOR_INDEX.json', {'schema':'expanded-fit-descriptor-index-v1', 'fits':index_rows})
    artifacts['blocked'] = write(output/'BLOCKED_FITS.json', {'schema':'expanded-blocked-fit-index-v1','fits':blocked_rows})
    artifacts['banks'] = write(output/'LOGICAL_BANKS.json', {'schema':'expanded-logical-banks-v1','logical_banks':banks})
    artifacts['controls'] = write(output/'CONTROL_LINKS.json', {'schema':'expanded-paired-control-links-v1','entries':controls})
    artifacts['statistics_ancestry'] = write(output/'STATISTICAL_ANCESTRY.json', {'source_plan':source_inputs['PLAN.json'],
        'predecessor_jobs':stat_jobs, 'job_count':540, 'expanded_decisions_frozen':False,
        'scope':'Historical 540 statistical/evaluate jobs only; prospective expanded statistical/access protocol remains pending'})
    artifacts['inherited_cells'] = write(output/'INHERITED_CELL_BINDINGS.json', {'schema':'expanded-inherited-cell-metadata-v1',
        'authority':source_inputs['inherited_cell_bindings'], 'cells':inherited_cells, 'data_receipts_refreshed':False,
        'scope':'Copied JSON receipts from predecessor only; no data/tensor file opened or hashed'})
    artifacts['sources'] = write(output/'SOURCE_RECEIPTS.json', {'registry':registry_pin,'inputs':source_inputs,
        'predecessor_index':predecessor_index_pin,'source_and_context_metadata':source_receipts,
        'model_source_imports':False,'data_receipts_refreshed':False})
    readiness = {'schema':'expanded-6460-source-descriptor-readiness-v1', 'UTC':datetime.now(timezone.utc).isoformat(),
        'status':'all_fit_intentions_accounted_source_prepared_or_explicitly_blocked', 'remote_output':str(output),
        'registry':registry_pin, 'allocation_guard':{'hostname':socket.gethostname(),'sole_gpu_uuid':'GPU-44039938-fd82-41d2-fefd-de71514e2fac'},
        'total_physical_fit_intentions':6460, 'source_reference_fits':270, 'matched_native_body_fits':1080,
        'shared_method_fit_intentions':5110, 'concrete_descriptors':validation_counts['pass'], 'blocked_fit_intentions':len(blocked_rows),
        'blocked_by_reason':dict(block_counts), 'counts_by_family_track_method':dict(counts),
        'preserved_ancestor_fit_ids_and_contract_hashes':2480, 'added_fit_intentions':3980,
        'runtime_procedure_alias_migrations':dict(changes), 'logical_banks':len(banks), 'paired_control_rows':len(controls),
        'predecessor_statistical_evaluate_jobs_retained_as_history':540, 'expanded_statistical_decisions_frozen':False,
        'exact_main_M1_body0_reuse_verified':True, 'I4_four_independent_fit_ids_seeds_optimizers_selectors_verified':True,
        'source_track_and_matched_extended_controllers_separate':True, 'native_fixed_settings_and_literal_role_bindings_preserved':True,
        'runtime_metadata_validation_passes':validation_counts['pass'], 'runtime_metadata_validation_rejections':0,
        'runtime_metadata_validation_is_execution_qualification':False, 'runtime_link_successor_required':True,
        'all_selected_serving_lifecycles_qualified':False, 'source_receipt_files':len(source_receipts),
        'inherited_data_receipts_refreshed':False, 'data_or_tensor_files_opened':0, 'native_models_constructed':0,
        'scientific_fits_started':0, 'GPU_compute_started':False, 'TEST_provider_loaded':False, 'TEST_metrics_or_predictions':False,
        'resource_bindings':None,'resources_measured':False,'scientific_launch_enabled':False,'runtime_admitted':False,
        'human_GO_received':False,'server77_accessed':False,'full_descriptors_allocation_only':True,
        'remaining_global_gaps':['runtime link selected-state/mean-export successor and native lifecycle qualification',
            'complete epoch/selection/resume/CUDA qualification for all actual method/preset contexts',
            'all literal node split providers and native seed-context/provider qualification',
            'measured resource/cost and concurrency admission', 'prospective statistics/access-ledger freeze', 'human giant-campaign verification and GO'],
        'artifacts':artifacts}
    assert readiness['concrete_descriptors']+readiness['blocked_fit_intentions']==6460
    readiness_pin = write(output/'READINESS.json', readiness)
    assert not any(name in sys.modules for name in FORBIDDEN_IMPORTS)
    return {'readiness_receipt':readiness_pin, 'counts':{k:readiness[k] for k in ('total_physical_fit_intentions','concrete_descriptors','blocked_fit_intentions','blocked_by_reason','logical_banks','runtime_metadata_validation_passes')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', default='generated_v1')
    parser.add_argument('--registry', default='REGISTRY_v1.json')
    args = parser.parse_args()
    print(json.dumps(compile_all(args.output_name, args.registry), sort_keys=True))
