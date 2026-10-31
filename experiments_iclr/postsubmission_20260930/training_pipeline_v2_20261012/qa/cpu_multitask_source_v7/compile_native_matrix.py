"""Compile a disabled native study packet from real allocation data custody.

No Torch, provider arrays, models, training or scoring are imported here. Each
dataset/backbone searches split0 once, then applies its frozen choice to every
prepared official/custom split. Global phase gates are collected by the root.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition import assign
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, normalized_id, validate, ORIGINAL_PAPER_DATASETS


ENGINE = 'training_pipeline_v2_20261012'
NATIVE_SOURCE = 'common_wrapper_native_source_20261010_v1/models.py'
SEARCH_SEEDS = (70101, 70103)
COMPARATIVE_SEEDS = (70201, 70203, 70207)
BACKBONES = ('ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'TAG')
BASIC = (
    ('single', 'ordinary_native_single', 'single', 'own', {}, 1),
    ('original', 'original_boundary_GNNM', 'original_gnnm', 'own', {}, 4),
    ('allmap_shared4', 'allmap_factor_shared4', 'internal_be', 'own', {}, 4),
    ('allmap_single', 'allmap_factor_single', 'factorized_single', 'own', {}, 1),
    ('joint_untied4', 'ordinary_joint_untied4', 'joint_untied', 'own', {}, 4),
)
SCALARS = (
    ('gncl_shared4', 'gncl_supcon_shared4_gncl', 'internal_be', 'gncl', {'pool_weight': .5}, 4),
    ('supcon_shared4', 'gncl_supcon_shared4_own_supcon', 'internal_be', 'within_route_supcon', {'supcon_weight': .05, 'temperature': .2}, 4),
    ('gncl_supcon_shared4', 'gncl_supcon_shared4_gncl_supcon', 'internal_be', 'gncl_within_route_supcon', {'pool_weight': .5, 'supcon_weight': .05, 'temperature': .2}, 4),
    ('supcon_single', 'gncl_supcon_ordinary_m1_supcon', 'single', 'within_route_supcon', {'supcon_weight': .05, 'temperature': .2}, 1),
    ('supcon_allmap_single', 'gncl_supcon_factorized_m1_supcon', 'factorized_single', 'within_route_supcon', {'supcon_weight': .05, 'temperature': .2}, 1),
    ('two_view_class_full', 'standardized_sage_two_view_class_full_cross_view', 'internal_be', 'two_view_class_full_cross_view', {'alignment_weight': .05, 'temperature': .2}, 4),
    ('two_view_supcon_eq2', 'standardized_sage_two_view_supcon_eq2', 'internal_be', 'two_view_supcon_eq2', {'supcon_weight': .05, 'temperature': .2}, 4),
    ('two_view_alignment_residual', 'standardized_sage_two_view_alignment_residual', 'internal_be', 'two_view_internal_be', {'alignment_weight': .05, 'residual_weight': .05, 'temperature': .2}, 4),
)


def file_binding(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError('Missing, symlinked or unconfined source/input: ' + str(path))
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def product_union(blocks):
    rows = {}
    for block in blocks:
        keys = sorted(block)
        for values in itertools.product(*(block[k] for k in keys)):
            value = dict(zip(keys, values))
            rows[canonical(value)] = value
    return [rows[key] for key in sorted(rows)]


def coefficients(native, mode):
    if not native:
        return [{}]
    if mode == 'native_fixed':
        return [copy.deepcopy(native)]
    domains = {key: ([.25, .5, .75] if key == 'pool_weight' else
                     [.1, .2, .4] if key == 'temperature' else [.025, .05, .1]) for key in native}
    return product_union(domains and [domains])


def bind_custody(root, path):
    path = confined(root, str(Path(path).resolve().relative_to(root)))
    value = json.loads(path.read_text())
    if value.get('schema') != 'node-provider-role-custody-v1' or value.get('status') != 'prepared':
        raise ValueError('An actual complete node preparation CUSTODY.json is required')
    if normalized_id(value['dataset_id']) in ORIGINAL_PAPER_DATASETS:
        raise ValueError('Original-paper datasets may not enter this native new-dataset packet')
    if value['model_graph']['classes'] < 3:
        raise ValueError('Binary/native BCE cells require a separate packet and adapter')
    if value.get('benchmark_fits') != 0 or value.get('predictive_scores') != 0:
        raise ValueError('Preparation receipt unexpectedly contains fitting/scoring work')
    split_ids = [r['split_id'] for r in value['splits']]
    if 0 not in split_ids or len(set(split_ids)) != len(split_ids):
        raise ValueError('Search split0 and a unique full prepared split roster are required')
    files = [file_binding(root, path)]
    for row in value['files']:
        observed = file_binding(root, path.parent / row['path'])
        if observed['sha256'] != row['sha256'] or observed['bytes'] != row['bytes']:
            raise ValueError('Prepared node file changed after custody')
        files.append(observed)
    for row in value['splits']:
        if row['evaluate_purpose'] != 'development':
            raise ValueError('Protected confirmation needs its separately frozen confirmation packet')
        if set(row['node_inputs']) != {'graph', 'train', 'select'} or set(row['endpoint_inputs']) != {
                'cal_ids', 'evaluate_ids', 'cal_targets', 'evaluate_targets'}:
            raise ValueError('Custody must separate node inputs, label-free endpoint IDs and heldout targets')
    return value, path.parent, files


def resource_for(profiles, dataset, backbone):
    value = profiles.get(dataset + '/' + backbone, profiles.get('*'))
    if not isinstance(value, dict):
        raise ValueError('Missing explicit resource profile for ' + dataset + '/' + backbone)
    for key in ('reservation_mib', 'wall_seconds'):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError('Positive integer resource declaration required: ' + key)
    if not isinstance(value.get('estimated_seconds_per_update'), (int, float)) or value['estimated_seconds_per_update'] <= 0:
        raise ValueError('Positive explicit scheduling estimate required')
    if not value.get('evidence'):
        raise ValueError('Resource estimates require their declared source/assumption evidence')
    return value


def make_study(root, custody, data_dir, backbone, recipe, sources, profile, mode):
    dataset = custody['dataset_id']
    study_id = 'native-v2-' + dataset + '-' + backbone.lower().replace('-', '_')
    split_rows = {r['split_id']: r for r in custody['splits']}
    native_domain = product_union(recipe['numeric_search_blocks'])
    if len(native_domain) != 100:
        raise ValueError('Every native backbone requires the exact deduplicated 100-point base domain')
    budget = {'unit': 'optimizer_updates', 'policy': 'full_budget', 'maximum': 5000}
    selector = {'metric': 'accuracy', 'direction': 'max', 'ties': 'first',
                'evaluation_interval': 1, 'checkpoint0': False, 'patience_steps': 5000}
    runs, search_ids, selections, banks = [], [], {}, []
    templates = list(BASIC) + (list(SCALARS) if backbone == 'SAGE' else [])
    def paths(row, field):
        return {key: str((data_dir / value).resolve().relative_to(root)) for key, value in row[field].items()}
    def common(row, architecture, variant, native_coefficients, members, stage):
        value = {'architecture': architecture, 'backbone': backbone, 'classes': custody['model_graph']['classes'],
                 'target_law': 'multiclass_ce', 'members': members, 'ffn_multiplier': 1, 'heads': 8,
                 'normalization': 'LayerNorm', 'native_model_source': NATIVE_SOURCE,
                 'node_inputs': paths(row, 'node_inputs'), 'fusion': 'mean_logits' if architecture == 'original_gnnm' else 'mean_probabilities',
                 'objective_variant': variant, 'objective_coefficients': copy.deepcopy(native_coefficients),
                 'stage': stage, 'is_original_gnnm': architecture == 'original_gnnm',
                 'reservation_mib': profile['reservation_mib'],
                 'estimated_seconds': profile['estimated_seconds_per_update'] * 5000 * members,
                 'recipe_scope': 'standardized_new_dataset_native_operator_comparison'}
        if variant != 'own':
            value['recipe_scope'] = 'new_standardized_SAGE_loss_transfer; tensor_formula_port_not_archived_recipe_reproduction'
        if variant.startswith('two_view_'):
            value['auxiliary_max_rows'] = 512
        return value
    def run(identifier, kind, method, split, seed, body, adapter, config, parents=()):
        value = {'run_id': identifier, 'kind': kind, 'method': method, 'split': split,
                 'seed': seed, 'body': body, 'adapter': adapter, 'config': config,
                 'budget': copy.deepcopy(budget) if kind == 'model_fit' else {},
                 'selection': copy.deepcopy(selector) if kind == 'model_fit' else {}, 'parents': list(parents)}
        runs.append(value)
        return identifier
    for arm, method, architecture, variant, native_coefficients, members in templates:
        candidates = []
        for point in native_domain:
            for coeff in coefficients(native_coefficients, mode):
                parameters = {**point, 'objective_coefficients': coeff}
                config_id = fingerprint(parameters)[:20]
                ids = []
                for seed in SEARCH_SEEDS:
                    identifier = 'search-' + arm + '-' + config_id + '-s' + str(seed)
                    config = {**common(split_rows[0], architecture, variant, coeff, members, 'search'), **point}
                    ids.append(run(identifier, 'model_fit', method, 0, seed, 0, ENGINE + '/adapters/node.py:make_adapter', config))
                candidates.append({'config_id': config_id, 'parameters': parameters, 'run_ids': ids})
                search_ids.extend(ids)
        select_id = 'select-' + arm
        selections[arm] = run(select_id, 'selection', method, 0, 0, 0, ENGINE + '/adapters/search.py:select_configuration',
                  {'stage': 'search_selection', 'ranking_metric': 'accuracy', 'ranking_direction': 'max',
                   'configurations': candidates, 'reservation_mib': 256, 'estimated_seconds': 1},
                  [identifier for candidate in candidates for identifier in candidate['run_ids']])
    comparative_ids = []
    for split, row in sorted(split_rows.items()):
        endpoint_inputs = paths(row, 'endpoint_inputs')
        for seed in COMPARATIVE_SEEDS:
            fitted = {}
            for arm, method, architecture, variant, native_coefficients, members in templates:
                identifier = 'compare-sp' + str(split) + '-' + arm + '-s' + str(seed) + '-b0'
                config = common(row, architecture, variant, native_coefficients, members, 'comparative')
                config['selected_config_from'] = selections[arm]
                fitted[arm] = [run(identifier, 'model_fit', method, split, seed, 0, ENGINE + '/adapters/node.py:make_adapter', config, [selections[arm]])]
                comparative_ids.extend(fitted[arm])
            for member_arm, bank_arm, method in [('single', 'ordinary_I4', 'ordinary_genuine_independent4'),
                    ('allmap_single', 'allmap_I4', 'allmap_factor_genuine_independent4')] + (
                    [('supcon_allmap_single', 'supcon_allmap_I4', 'gncl_supcon_genuine_factorized_i4_supcon')] if backbone == 'SAGE' else []):
                original = next(r for r in runs if r['run_id'] == fitted[member_arm][0])
                fits = list(fitted[member_arm])
                for body in (1, 2, 3):
                    identifier = 'compare-sp' + str(split) + '-' + bank_arm + '-s' + str(seed) + '-b' + str(body)
                    fits.append(run(identifier, 'model_fit', original['method'], split, seed, body,
                                    original['adapter'], copy.deepcopy(original['config']), original['parents']))
                    comparative_ids.append(identifier)
                fitted[bank_arm] = fits
            for arm, fits in fitted.items():
                fit = next(r for r in runs if r['run_id'] == fits[0])
                fusion = fit['config']['fusion']
                bank_id = 'bank-sp' + str(split) + '-' + arm + '-s' + str(seed)
                export_id = run(bank_id, 'ensemble_assembly', arm, split, seed, 0, ENGINE + '/adapters/node_endpoints.py:export_bank',
                    {'stage': 'bank_export', 'fit_parents': fits, 'endpoint_inputs': endpoint_inputs,
                     'fusion': fusion, 'reservation_mib': profile['reservation_mib'],
                     'estimated_seconds': profile['estimated_seconds_per_update'] * max(1, len(fits))}, fits)
                cal_id = run(bank_id + '-cal', 'calibration', arm, split, seed, 0, ENGINE + '/adapters/node_endpoints.py:scalar_calibration',
                    {'stage': 'calibration', 'export_parent': export_id, 'endpoint_inputs': endpoint_inputs,
                     'fusion': fusion, 'placement': 'after_serving_pool', 'steps': 500, 'learning_rate': .01,
                     'reservation_mib': 512, 'estimated_seconds': 5}, [export_id])
                eval_id = run(bank_id + '-evaluate', 'evaluation', arm, split, seed, 0, ENGINE + '/adapters/node_endpoints.py:evaluate_bank',
                    {'stage': 'evaluation', 'export_parent': export_id, 'calibration_parent': cal_id,
                     'endpoint_inputs': endpoint_inputs, 'fusion': fusion, 'reservation_mib': 512,
                     'estimated_seconds': 5}, [export_id, cal_id])
                banks.append({'split': split, 'seed': seed, 'arm': arm, 'fit_parents': fits,
                              'export': export_id, 'calibration': cal_id, 'evaluation': eval_id,
                              'routes': 4 if len(fits) == 4 else fit['config']['members'],
                              'body0_reused': len(fits) == 4})
    roles = {role: {str(split): paths(row, 'node_inputs' if role in {'train', 'select'} else 'endpoint_inputs')[role if role in {'train', 'select'} else role + '_targets']
                    for split, row in sorted(split_rows.items())} for role in ('train', 'select', 'cal', 'evaluate')}
    spec = {'schema_version': 1, 'study_id': study_id, 'stage': 'development',
        'hypothesis': 'Comprehensive declared native mechanisms on new data; competent native, original GNNM and independently fitted equipped I4 remain explicit references.',
        'dataset': dataset, 'data': {'task': 'node_classification', 'preprocessing': custody['request']['graph_policy'],
          'roles': roles, 'prior_exposure': custody['request']['evaluation_exposure'],
          'evidence': {'custody_request_hash': custody['request_hash'], 'model_graph': custody['model_graph'],
                       'provider_files': custody['provider_files'], 'split_roster': sorted(split_rows),
                       'search_split': 0, 'source_native_author_protocol_replication': False}},
        'source': sources, 'runs': runs,
        'comparisons': [{'scope': 'every_registered_split_and_three_new_paired_seeds',
            'references': ['single', 'original', 'ordinary_I4', 'allmap_single', 'allmap_I4'],
            'candidates': [t[0] for t in templates if t[0] not in {'single', 'original', 'allmap_single'}],
            'report': 'all arm/seed/split outcomes and paired deltas; overlapping masks are not independent datasets',
            'acceptance_guarantee': False}],
        'release': {'policy': 'whole_roster', 'primary_endpoint': 'all_closed_calibrated_and_raw_bank_evaluations',
                    'global_plan_gates_required': True},
        'resources': {'wall_seconds': profile['wall_seconds'], 'profile': profile,
                      'status': 'PREPARED_ESTIMATED_RESOURCE_PROFILE_REQUIRES_RUNTIME_QUALIFICATION'}}
    validate(spec)
    fit_rows = [r for r in runs if r['kind'] == 'model_fit']
    counts = {'optimizer_starts_by_stage': dict(Counter(r['config']['stage'] for r in fit_rows)),
      'initial_training_segments': len(fit_rows), 'continuation_segments': 0, 'extension_segments': 0,
      'physical_body_instances_by_stage': dict(Counter()), 'coupled_optimizer_fits_by_stage': dict(Counter()),
      'selected_search_banks': len(search_ids), 'comparative_served_banks': len(banks),
      'bank_exports': len(banks), 'scalar_CAL_fits': len(banks), 'evaluation_jobs': len(banks),
      'search_selection_jobs': len(selections), 'fresh_source_stage_fits': 0,
      'reused_I4_body0_artifacts': sum(b['body0_reused'] for b in banks),
      'genuine_I4_banks': sum(len(b['fit_parents']) == 4 for b in banks),
      'comparative_served_prediction_views': sum(b['routes'] for b in banks),
      'full_budget_optimizer_updates': len(fit_rows) * 5000,
      'all_run_records': len(runs), 'search_split': 0, 'comparative_split_count': len(split_rows)}
    for field in ('physical_body_instances_by_stage', 'coupled_optimizer_fits_by_stage'):
        values = Counter()
        for r in fit_rows:
            architecture = r['config']['architecture']
            values[r['config']['stage']] += (4 if architecture == 'joint_untied' else 1) if field.startswith('physical') else int(r['config']['members'] > 1)
        counts[field] = dict(values)
    return spec, counts, banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--custody', nargs='+', required=True)
    parser.add_argument('--resources', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--host-bindings')
    parser.add_argument('--scalar-coefficients', choices=['neighbors', 'native_fixed'], default='neighbors')
    parser.add_argument('--plain-json', action='store_true')
    args = parser.parse_args()
    verify_allocation()
    root = Path(args.project_root).resolve()
    def local(value):
        path = Path(value)
        return confined(root, str(path.resolve().relative_to(root)) if path.is_absolute() else str(path))
    output = local(args.output)
    if output.exists():
        raise ValueError('Compiler output already exists; no overwrite or implicit replacement')
    profile_path = local(args.resources)
    profiles = json.loads(profile_path.read_text())['profiles']
    matrix_path = root / ENGINE / 'docs/MATRIX_AXES_DRAFT.json'
    matrix = json.loads(matrix_path.read_text())
    recipes = {r['native_constructor_name'].split('(')[-1].rstrip(')'): r
               for r in matrix['baseline_recipe_catalogue']['rows'] if r['readiness']['adapter_available']}
    if set(recipes) != set(BACKBONES):
        raise ValueError('Exactly eight native Model/TABMModel recipe interfaces required')
    inventory_path = root / ENGINE / 'docs/METHOD_INVENTORY.csv'
    inventory = list(csv.DictReader(inventory_path.open()))
    source_paths = sorted(p for p in (root / ENGINE).rglob('*.py') if '__pycache__' not in p.parts)
    source_paths += [root / NATIVE_SOURCE, matrix_path, inventory_path,
                     root / ENGINE / 'docs/METHOD_SOURCE_ALIASES.csv', profile_path]
    source_paths += sorted((root / ENGINE / 'qa').glob('*/SOURCE_INVENTORY.json'))
    common_bindings = [file_binding(root, p) for p in dict.fromkeys(source_paths)]
    commit = args.source_commit or subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if not commit:
        raise ValueError('Verified source commit is required alongside dirty-file fingerprints')
    dataset_inputs = [bind_custody(root, local(p)) for p in args.custody]
    if len({c['dataset_id'] for c, _, _ in dataset_inputs}) != len(dataset_inputs):
        raise ValueError('A dataset may have only one prepared custody roster in this packet')
    prepared = []
    for custody, data_dir, data_bindings in dataset_inputs:
        source = {'git_commit': commit, 'files': common_bindings + data_bindings,
                  'working_tree_fingerprints_authoritative': True,
                  'official_author_protocol_replication': False}
        for backbone in BACKBONES:
            profile = resource_for(profiles, custody['dataset_id'], backbone)
            spec, counts, banks = make_study(root, custody, data_dir, backbone, recipes[backbone], source, profile, args.scalar_coefficients)
            prepared.append((spec, counts, banks))
    groups = [{'id': spec['study_id'], 'estimated_seconds': sum(r['config']['estimated_seconds'] for r in spec['runs']),
               'maximum_cuda_bytes': max(r['config']['reservation_mib'] for r in spec['runs']) * 1024**2,
               'run_keys': [spec['study_id'] + '/' + r['run_id'] for r in spec['runs']]} for spec, _, _ in prepared]
    assignments = assign(groups)
    relative_output = str(output.relative_to(root))
    plan = {'schema_version': 1, 'slots_per_gpu': 2, 'torch_threads_per_fit': 2,
      'output_root': relative_output + '/execution', 'studies': [], 'groups': assignments['groups'],
      'estimated_gpu_seconds': assignments['estimated_gpu_seconds'], 'estimates_are_measurements': False,
      'scientific_launch_enabled': False, 'human_GO_required': True,
      'host_bindings': json.loads(local(args.host_bindings).read_text()) if args.host_bindings else {
          'allocation': {'status': 'verified_at_compilation', 'hostname': 'anogena-2-0',
                         'gpu_uuids': ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']},
          'server77': {'status': 'pending_live_verification_before_launch', 'historical_receipts_are_not_live_verification': True}},
      'global_gates': {}, 'barriers': {}}
    required_hosts = sorted({r['gpu_assignment'].split('/')[0] for r in plan['groups']})
    for name, sources, required in [('whole_search', ['search', 'search_selection'], ['comparative']),
          ('whole_comparative', ['comparative'], ['bank_export', 'calibration', 'evaluation']),
          ('whole_calibration', ['calibration'], ['evaluation'])]:
        plan['global_gates'][name] = {'path': relative_output + '/gate_receipts/' + name + '.json',
            'source_stages': sources, 'required_for_stages': required, 'required_hosts': required_hosts,
            'receipt_contract': 'status=complete;plan_hash=current_plan_hash;closed_phase=gate_name;host_receipts_keys=required_hosts'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'studies').mkdir()
    summaries = []
    for spec, counts, banks in prepared:
        suffix = '.json' if args.plain_json else '.json.gz'
        relative = relative_output + '/studies/' + spec['study_id'] + suffix
        raw = (canonical(spec) + '\n').encode()
        path = root / relative
        path.write_bytes(raw if args.plain_json else gzip.compress(raw, mtime=0))
        plan['studies'].append({'path': relative, 'spec_hash': fingerprint(spec)})
        summaries.append({'study_id': spec['study_id'], 'dataset_id': spec['dataset'],
                          'backbone': next(r['config']['backbone'] for r in spec['runs'] if r['kind'] == 'model_fit'),
                          'counts': counts, 'served_banks': banks})
    totals = Counter()
    for row in summaries:
        for key, value in row['counts'].items():
            if type(value) is int and key not in {'search_split', 'comparative_split_count'}:
                totals[key] += value
        for stage, value in row['counts']['optimizer_starts_by_stage'].items():
            totals['optimizer_starts_' + stage] += value
        for stage, value in row['counts']['physical_body_instances_by_stage'].items():
            totals['physical_body_instances_' + stage] += value
        for stage, value in row['counts']['coupled_optimizer_fits_by_stage'].items():
            totals['coupled_optimizer_fits_' + stage] += value
    base_covered = {r[1] for r in BASIC} | {'ordinary_genuine_independent4', 'allmap_factor_genuine_independent4'}
    transfers = {r[1] for r in SCALARS} | {'gncl_supcon_genuine_factorized_i4_supcon'}
    with (output / 'INVENTORY_COVERAGE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['canonical_method_id', 'grid_eligibility', 'packet_status', 'source_exact_recipe_ready', 'blocker'])
        writer.writeheader()
        for row in inventory:
            method = row['canonical_method_id']
            status = 'native_operator_packet' if method in base_covered else 'SAGE_formula_transfer_only' if method in transfers else 'uncovered_in_this_native_packet'
            writer.writerow({'canonical_method_id': method, 'grid_eligibility': row['grid_eligibility'],
              'packet_status': status, 'source_exact_recipe_ready': False,
              'blocker': ('source-specific task/recipe remains distinct; this packet has new standardized roles/HPO' if method in base_covered else
                          'SAGE-only standardized tensor-formula transfer; not source-exact archived reproduction' if method in transfers else row['migration_work'])})
    (output / 'PLAN.json').write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    summary = {'schema': 'native-matrix-preparation-summary-v1',
       'status': 'PREPARED_ESTIMATED_RESOURCE_PROFILE_REQUIRES_RUNTIME_QUALIFICATION',
       'plan_hash': fingerprint(plan), 'studies': len(prepared), 'datasets': len(dataset_inputs),
       'native_backbones': list(BACKBONES), 'base_configurations_per_search_cell': 100,
       'scalar_coefficient_policy': args.scalar_coefficients, 'search_split': 0,
       'search_seed_ids': list(SEARCH_SEEDS), 'comparative_seed_ids': list(COMPARATIVE_SEEDS),
       'required_hosts': required_hosts, 'fresh_fit_counts': dict(totals), 'per_study': summaries,
       'canonical_inventory_rows': len(inventory), 'all_249_predictive_candidates_covered': False,
       'confirmation_author_replication_and_other_method_packets': 'not materialized by this native compiler',
       'training_started': False, 'GO_receipt_created': False, 'optional_halving_and_extensions': 'absent'}
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_study'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
