"""Compile a disabled exact-coordinate source packet on verified allocation.

Only standard-library metadata/arithmetic is used. No source model import,
provider array, training, score, endpoint target reader or GO receipt is created.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.coordinate import ARMS, COORDINATE_SHA256, NATIVE_SHA256, SOURCE_CONTRACTS
from compile_native_matrix import bind_custody, file_binding, product_union, resource_for, RUNTIME_FILES
from partition import assign
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

SEARCH_SEEDS = (71101, 71103)
COMPARATIVE_SEEDS = (71201, 71203, 71207)
METHODS = {'original': 'original_boundary_GNNM', 'single': 'ordinary_native_single',
    'untied': 'ordinary_joint_untied4', 'gt_sep_single': 'ordinary_native_single',
    **{arm: 'coordinate_family_' + arm for arm in ('factor', 'coordinate', 'permutation', 'bias_only', 'heads')}}
TEMPLATES = [(arm, arm, 1 if arm in {'single', 'gt_sep_single'} else 4) for arm in ARMS]
TEMPLATES += [(arm + '_m1', arm, 1) for arm in ('factor', 'coordinate', 'permutation', 'bias_only')]
I4_PARENTS = ('single', 'gt_sep_single', 'factor_m1', 'coordinate_m1', 'permutation_m1', 'bias_only_m1')
CANONICAL_DATASET = {'amazon-photo': 'AmazonPhoto', 'coauthor-cs': 'CoauthorCS'}
SPECIFICATION_SHA256 = '6da6a44bbc2b55118296b8e0de007330fd5d83983b57fb3d17fdd093e0cef44d'


def count_function(source):
    """Extract exact source's pure arithmetic without importing its Torch module."""
    tree = ast.parse(Path(source).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parameter_count_formula']
    if len(functions) != 1:
        raise ValueError('Expected the exact coordinate source parameter-count function')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['parameter_count_formula']


def model_bytes(formula, arm, features, width, classes, depth, members=4):
    value = 4 * formula(arm, features, width, classes, members=members, num_layers=depth, hidden_dim_multiplier=1)
    if arm in {'coordinate', 'permutation'}:
        value += 8 * members * depth * (9 * width)
    return value


def matching_width(formula, arm, features, candidate_width, classes, depth):
    target = model_bytes(formula, 'coordinate', features, candidate_width, classes, depth)
    low, high = 1, max(2, candidate_width * 4)
    while model_bytes(formula, arm, features, high, classes, depth) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if model_bytes(formula, arm, features, middle, classes, depth) < target:
            low = middle + 1
        else:
            high = middle
    choices = {low, max(1, low - 1)}
    return min(choices, key=lambda width: (abs(model_bytes(formula, arm, features, width, classes, depth) - target), width))


def validate_recipes(specification):
    recipes = specification['expected_recipes']
    expected = {dataset + '_' + arm + '_stage1' for dataset in CANONICAL_DATASET.values() for arm in ARMS}
    if set(recipes) != expected:
        raise ValueError('Bind the actual 18 source-default stage1 recipes, without silently filling missing rows')
    for recipe in recipes.values():
        if (recipe['min_epochs'], recipe['max_epochs'], recipe['patience'], recipe['eval_every']) != (1000, 2000, 200, 10):
            raise ValueError('Preserve the adequate source 1000–2000 cycle policy and cadence')
        if (recipe['checkpoint_metric'] != 'validation_nll' or recipe['checkpoint_ties'] != 'earliest'
                or recipe['scheduler'] != 'constant' or recipe['optimizer']['name'] != 'AdamW'
                or recipe['optimizer']['betas'] != [.9, .999] or recipe['optimizer']['eps'] != 1e-8
                or recipe['model_kwargs'] != {'hidden_dim_multiplier': 1, 'normalization': 'layer', 'num_heads': 8}):
            raise ValueError('Source-default recipe/optimizer/selector locks changed')
    return recipes


def numeric_domain(dataset, arm, members, graph, formula, recipes):
    # 80 broad points plus 20 source-anchor points. Constructor defaults are
    # documented separately; these are the actual full source training defaults.
    points = product_union([
        {'learning_rate': [.00025, .0005, .001, .002, .004], 'width': [128, 256],
         'depth': [2, 3], 'dropout': [.2, .4], 'weight_decay': [0., .0001]},
        {'learning_rate': [.00003, .0003], 'width': [256, 512], 'depth': [1, 2, 3, 4, 5],
         'dropout': [.2], 'weight_decay': [0.]},
    ])
    resolved = []
    for point in points:
        point = dict(point)
        if arm in {'single', 'untied', 'heads'}:
            point['width'] = matching_width(formula, arm, graph['features'], point['width'], graph['classes'], point['depth'])
        resolved.append(point)
    source_name = CANONICAL_DATASET.get(dataset)
    anchors = ([recipes[source_name + '_' + arm + '_stage1']] if source_name else
               [recipes[name + '_' + arm + '_stage1'] for name in CANONICAL_DATASET.values()])
    # Source defaults always survive HPO even if current provider shape makes
    # the historic storage-matched width differ from newly matched widths.
    for anchor in anchors:
        resolved.append({'learning_rate': anchor['optimizer']['lr'], 'weight_decay': anchor['optimizer']['weight_decay'],
            'width': anchor['width'], 'depth': anchor['num_layers'], 'dropout': anchor['dropout']})
    # JSON distinguishes 0 from0.0 despite identical resolved optimizer values.
    # Normalize numeric types before fingerprinting/deduplicating configurations.
    normalized = [{key: int(value) if key in {'width', 'depth'} else float(value)
                   for key, value in point.items()} for point in resolved]
    return [json.loads(key) for key in sorted({canonical(point) for point in normalized})]


def make_study(root, custody, data_dir, sources, profile, recipes, formula, runtime, coordinate_source, native_source, validator=validate):
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    dataset = custody['dataset_id']
    study_id = 'coordinate-v2-' + dataset
    splits = {row['split_id']: row for row in custody['splits']}
    runs, selections, banks, comparative_ids, domains = [], {}, [], [], {}
    def paths(row, field):
        return {key: str((data_dir / value).resolve().relative_to(root)) for key, value in row[field].items()}
    def add(identifier, kind, method, split, seed, body, adapter, config, parents=()):
        runs.append({'run_id': identifier, 'kind': kind, 'method': method, 'split': split,
            'seed': seed, 'body': body, 'adapter': adapter, 'config': config,
            'budget': {'unit': 'epochs', 'policy': 'full_budget', 'maximum': 2000} if kind == 'model_fit' else {},
            'selection': {'metric': 'pooled_nll', 'direction': 'min', 'ties': 'first',
                          'evaluation_interval': 10, 'checkpoint0': False} if kind == 'model_fit' else {},
            'parents': list(parents)})
        return identifier
    def common(split, arm, members, stage):
        return {'source_arm': arm, 'backbone': 'GT-sep' if arm == 'gt_sep_single' else 'SAGE',
            'classes': custody['model_graph']['classes'], 'target_law': 'multiclass_ce',
            'members': members, 'source_requested_members': 4 if arm in {'single', 'gt_sep_single'} else members,
            'coordinate_source': coordinate_source, 'native_model_source': native_source,
            'node_inputs': paths(splits[split], 'node_inputs'), 'ffn_multiplier': 1, 'heads': 8,
            'normalization': 'LayerNorm', 'permutation_seed_offset': 1000,
            'source_minimum_cycles': 1000, 'source_patience_cycles': 200,
            'fusion': 'mean_probabilities', 'stage': stage, 'is_original_gnnm': arm == 'original',
            'source_optimizer_count': 4 if arm == 'untied' else 1,
            'recipe_scope': 'exact_coordinate_mechanism_on_standardized_node_roles_full2000',
            'reservation_mib': profile['reservation_mib'],
            'estimated_seconds': profile['estimated_seconds_per_update'] * 2000 * members}
    for name, arm, members in TEMPLATES:
        domain = numeric_domain(dataset, arm, members, custody['model_graph'], formula, recipes)
        domains[name] = domain
        candidates = []
        for point in domain:
            config_id = fingerprint(point)[:20]
            ids = [add('search-' + name + '-' + config_id + '-s' + str(seed), 'model_fit', METHODS[arm], 0, seed, 0,
                runtime + '/adapters/coordinate.py:make_adapter', {**common(0, arm, members, 'search'), **point}) for seed in SEARCH_SEEDS]
            candidates.append({'config_id': config_id, 'parameters': point, 'run_ids': ids})
        selections[name] = add('select-' + name, 'selection', name, 0, 0, 0,
            runtime + '/adapters/search.py:select_configuration', {'stage': 'search_selection',
                'configurations': candidates, 'ranking_metric': 'pooled_nll', 'ranking_direction': 'min',
                'reservation_mib': 1, 'estimated_seconds': 1}, [i for c in candidates for i in c['run_ids']])
    for split in sorted(splits):
        for seed in COMPARATIVE_SEEDS:
            fitted = {}
            for name, arm, members in TEMPLATES:
                selection = selections[name]
                fitted[name] = [add('compare-sp' + str(split) + '-' + name + '-s' + str(seed), 'model_fit', METHODS[arm], split, seed, 0,
                    runtime + '/adapters/coordinate.py:make_adapter', {**common(split, arm, members, 'comparative'),
                        **domains[name][0], 'selected_config_from': selection}, [selection])]
                comparative_ids.extend(fitted[name])
            for name in I4_PARENTS:
                original = next(r for r in runs if r['run_id'] == fitted[name][0])
                fitted[name + '_I4'] = [original['run_id']]
                for body in (1, 2, 3):
                    identifier = add(original['run_id'] + '-I4-b' + str(body), 'model_fit', original['method'], split, seed, body,
                        original['adapter'], copy.deepcopy(original['config']), original['parents'])
                    fitted[name + '_I4'].append(identifier)
                    comparative_ids.append(identifier)
            endpoint_inputs = paths(splits[split], 'endpoint_inputs')
            for name, fits in fitted.items():
                first = next(r for r in runs if r['run_id'] == fits[0])
                bank = 'bank-sp' + str(split) + '-' + name + '-s' + str(seed)
                export = add(bank, 'ensemble_assembly', name, split, seed, 0, runtime + '/adapters/node_endpoints.py:export_bank',
                    {'stage': 'bank_export', 'fit_parents': fits, 'endpoint_inputs': endpoint_inputs,
                     'fusion': 'mean_probabilities', 'reservation_mib': profile['reservation_mib'], 'estimated_seconds': 1}, fits)
                cal = add(bank + '-cal', 'calibration', name, split, seed, 0, runtime + '/adapters/node_endpoints.py:scalar_calibration',
                    {'stage': 'calibration', 'export_parent': export, 'endpoint_inputs': endpoint_inputs,
                     'fusion': 'mean_probabilities', 'placement': 'after_serving_pool', 'steps': 500,
                     'learning_rate': .01, 'reservation_mib': 512, 'estimated_seconds': 5}, [export])
                evaluation = add(bank + '-evaluate', 'evaluation', name, split, seed, 0, runtime + '/adapters/node_endpoints.py:evaluate_bank',
                    {'stage': 'evaluation', 'export_parent': export, 'calibration_parent': cal,
                     'endpoint_inputs': endpoint_inputs, 'fusion': 'mean_probabilities', 'reservation_mib': 512,
                     'estimated_seconds': 5}, [export, cal])
                banks.append({'arm': name, 'split': split, 'seed': seed, 'fit_parents': fits,
                    'routes': 4 if len(fits) == 4 else first['config']['members'], 'body0_reused': len(fits) == 4,
                    'export': export, 'calibration': cal, 'evaluation': evaluation})
    roles = {role: {str(split): paths(row, 'node_inputs' if role in {'train', 'select'} else 'endpoint_inputs')[role if role in {'train', 'select'} else role + '_targets']
            for split, row in sorted(splits.items())} for role in ('train', 'select', 'cal', 'evaluate')}
    study = {'schema_version': 1, 'study_id': study_id, 'stage': 'development', 'dataset': dataset,
        'hypothesis': 'Exact source coordinate/bias/factor/permutation mechanisms with competent source controls and separately fitted I4 references.',
        'data': {'task': 'node_classification', 'preprocessing': custody['request']['graph_policy'], 'roles': roles,
            'prior_exposure': custody['request']['evaluation_exposure'], 'evidence': {'custody_request_hash': custody['request_hash'],
                'model_graph': custody['model_graph'], 'provider_files': custody['provider_files'], 'search_split': 0,
                'source_historical_data_recipe_reproduction': False, 'source_contract': SOURCE_CONTRACTS}},
        'source': sources, 'runs': runs, 'comparisons': [{'references': ['single', 'single_I4', 'original', 'untied', 'heads', 'gt_sep_single', 'gt_sep_single_I4'],
            'candidates': ['factor', 'coordinate', 'permutation', 'bias_only'],
            'equipped_I4': [n + '_I4' for n in I4_PARENTS if n.endswith('_m1')],
            'scope': 'all declared splits and three paired new seeds; no acceptance guarantee'}],
        'release': {'policy': 'whole_roster', 'primary_endpoint': 'all_closed_raw_and_calibrated_bank_evaluations', 'global_plan_gates_required': True},
        'resources': {'wall_seconds': profile['wall_seconds'], 'profile': profile, 'status': 'DISABLED_REQUIRES_COORDINATE_SOURCE_DATA_RESOURCE_QA_AND_HUMAN_GO'}}
    validator(study)
    fits = [r for r in runs if r['kind'] == 'model_fit']
    counts = {'atomic_fit_run_groups_by_stage': dict(Counter(r['config']['stage'] for r in fits)),
        'paid_optimizer_starts_by_stage': dict(Counter()), 'physical_bodies_by_stage': dict(Counter()),
        'full_budget_optimizer_step_calls': sum(r['config']['source_optimizer_count'] * 2000 for r in fits),
        'full_budget_source_cycles': 2000 * len(fits), 'search_configuration_counts': {k: len(v) for k, v in domains.items()},
        'search_selection_jobs': len(selections), 'comparative_banks': len(banks), 'exports': len(banks),
        'scalar_CAL_fits': len(banks), 'evaluation_jobs': len(banks),
        'genuine_I4_banks_and_reused_member0': sum(b['body0_reused'] for b in banks),
        'served_route_views': sum(b['routes'] for b in banks), 'all_run_records': len(runs),
        'source_acquisition_fits': 0, 'continuations': 0, 'extensions': 0, 'actual_optimizer_updates': 0}
    for field, selector in [('paid_optimizer_starts_by_stage', lambda r: r['config']['source_optimizer_count']),
                            ('physical_bodies_by_stage', lambda r: 4 if r['config']['source_arm'] == 'untied' else 1)]:
        values = Counter()
        for r in fits:
            values[r['config']['stage']] += selector(r)
        counts[field] = dict(values)
    return study, counts, banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'runtime-package', 'coordinate-source', 'native-model-source', 'source-specification', 'resources', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--custody', nargs='+', required=True)
    parser.add_argument('--fixture-receipts', nargs='+', required=True)
    parser.add_argument('--frozen-runtime', action='store_true', required=True)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    verify_allocation()
    root = Path(args.project_root).resolve()
    def local(value):
        path = Path(value)
        return confined(root, str(path.resolve().relative_to(root)) if path.is_absolute() else str(path))
    output = local(args.output)
    if output.exists():
        raise ValueError('Preserve prior packets; fresh output required')
    runtime, coordinate, native, specification, resource_path = [local(getattr(args, name)) for name in
        ('runtime_package', 'coordinate_source', 'native_model_source', 'source_specification', 'resources')]
    if runtime.is_relative_to(root / 'training_pipeline_v2_20261012') or coordinate == root / 'coordinate_ensemble_source_v1/models.py' or native == root / 'common_wrapper_native_source_20261010_v1/models.py':
        raise ValueError('Final compiler requires frozen copies outside live runtime/model directories')
    if file_binding(root, coordinate)['sha256'] != COORDINATE_SHA256 or file_binding(root, native)['sha256'] != NATIVE_SHA256:
        raise ValueError('Exact audited coordinate and native source fingerprints required')
    if file_binding(root, specification)['sha256'] != SPECIFICATION_SHA256:
        raise ValueError('Exact source stage1 specification required for the preserved numeric defaults')
    recipes = validate_recipes(json.loads(specification.read_text()))
    formula = count_function(coordinate)
    resources = json.loads(resource_path.read_text())
    capacities = resources['worker_capacities_cuda_bytes']
    if not capacities or any(k not in {'allocation/gpu0', 'server77/gpu0', 'server77/gpu1'} or type(v) is not int or v < 1 for k, v in capacities.items()):
        raise ValueError('Explicit positive planning capacities required; no remote verification is implied')
    fixture_paths = [local(p) for p in args.fixture_receipts]
    source_paths = [runtime / name for name in RUNTIME_FILES] + [runtime / 'adapters/coordinate.py', coordinate, native, specification, resource_path, *fixture_paths]
    bindings = [file_binding(root, p) for p in dict.fromkeys(source_paths)]
    schema_spec = importlib.util.spec_from_file_location('coordinate_frozen_schema', runtime / 'spec.py')
    schema = importlib.util.module_from_spec(schema_spec)
    schema_spec.loader.exec_module(schema)
    commit = args.source_commit or subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    datasets = [bind_custody(root, local(path)) for path in args.custody]
    if len({c['dataset_id'] for c, _, _ in datasets}) != len(datasets):
        raise ValueError('Duplicate dataset custody')
    prepared = []
    for custody, directory, data_bindings in datasets:
        source = {'git_commit': commit, 'files': bindings + data_bindings, 'working_tree_fingerprints_authoritative': True,
                  'frozen_runtime_package': str(runtime.relative_to(root)), 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'official_author_protocol_replication': False}
        profile = resource_for(resources['profiles'], custody['dataset_id'], 'coordinate-SAGE-source')
        prepared.append(make_study(root, custody, directory, source, profile, recipes, formula,
            str(runtime.relative_to(root)), str(coordinate.relative_to(root)), str(native.relative_to(root)), schema.validate))
    groups = [{'id': s['study_id'], 'estimated_seconds': sum(r['config']['estimated_seconds'] for r in s['runs']),
               'maximum_cuda_bytes': max(r['config']['reservation_mib'] for r in s['runs']) * 1024**2,
               'run_keys': [s['study_id'] + '/' + r['run_id'] for r in s['runs']]} for s, _, _ in prepared]
    placement = assign(groups, capacities)
    prefix = str(output.relative_to(root))
    plan = {'schema_version': 1, 'slots_per_gpu': 2, 'torch_threads_per_fit': 2,
        'output_root': prefix + '/execution', 'runtime_launcher': str((runtime / 'launcher.py').relative_to(root)),
        'runtime_package': str(runtime.relative_to(root)), 'studies': [], 'groups': placement['groups'],
        'estimated_gpu_seconds': placement['estimated_gpu_seconds'], 'estimates_are_measurements': False,
        'scientific_launch_enabled': False, 'human_GO_required': True, 'global_gates': {}, 'barriers': {},
        'host_bindings': {'allocation': {'status': 'verified_at_compilation', 'hostname': 'anogena-2-0',
            'gpu_uuids': ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}, 'server77': {'status': 'pending_live_verification_before_launch'}}}
    hosts = sorted({row['gpu_assignment'].split('/')[0] for row in placement['groups']})
    for name, phases, stages in [('whole_search', ['search', 'search_selection'], ['comparative']),
        ('whole_comparative', ['comparative'], ['bank_export', 'calibration', 'evaluation']),
        ('whole_calibration', ['calibration'], ['evaluation'])]:
        plan['global_gates'][name] = {'path': prefix + '/gate_receipts/' + name + '.json',
            'source_stages': phases, 'required_for_stages': stages, 'required_hosts': hosts,
            'receipt_contract': 'status=complete;plan_hash=current_plan_hash;closed_phase=gate_name;host_receipts_keys=required_hosts'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'studies').mkdir()
    summaries = []
    for study, counts, banks in prepared:
        path = output / 'studies' / (study['study_id'] + '.json.gz')
        path.write_bytes(gzip.compress((canonical(study) + '\n').encode(), mtime=0))
        plan['studies'].append({'path': str(path.relative_to(root)), 'spec_hash': fingerprint(study)})
        summaries.append({'study_id': study['study_id'], 'dataset_id': study['dataset'], 'counts': counts, 'banks': banks})
    for name, value in [('PLAN.json', plan), ('SUMMARY.json', {'status': 'DISABLED_REQUIRES_COORDINATE_QA_AND_GO',
        'source_contract': SOURCE_CONTRACTS, 'studies': summaries, 'full_249_predictive_inventory_covered': False,
        'blockers': ['SAGE contexts only; GT-sep ordinary single is distinct', 'binary/sampler/heterogeneous providers unsupported',
            'LastFMAsia has no prepared v2 provider', 'source historical as_pyg edges/roles differ from standardized custody',
            'resource/optimizer/serializer/endpoint QA pending', 'outer all-packet global phase-gate integration belongs to root']})]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    with (output / 'SOURCE_COVERAGE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_alias', 'canonical_method', 'operator', 'port_status', 'blocker'])
        writer.writeheader()
        for arm in ARMS:
            writer.writerow({'source_alias': 'coordinate_family_' + arm, 'canonical_method': METHODS[arm],
                'operator': 'GT-sep ordinary single' if arm == 'gt_sep_single' else 'SAGE',
                'port_status': 'disabled_exact_source_adapter_prepared_root_QA_pending',
                'blocker': 'full input/resource/controller/endpoint QA and outer stage gates before GO'})
        writer.writerow({'source_alias': 'LastFMAsia_coordinate_scope', 'canonical_method': 'coordinate_family',
            'operator': 'SAGE', 'port_status': 'provider_blocked', 'blocker': 'No v2 prepared LastFMAsia provider'})
    print(canonical({'status': 'disabled', 'output': str(output), 'datasets': len(prepared), 'fits': 0, 'scores': 0}))


if __name__ == '__main__':
    main()
