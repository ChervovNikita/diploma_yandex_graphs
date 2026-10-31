"""Prepare a disabled six-cell source-native wrapper packet from metadata only.

This local compiler reads copied source and custody metadata. It neither reads
provider tensors/outcomes nor contacts an allocation, invokes Git, imports
numerics, starts model fits, or assigns final resources. No source recipe
replication or scientific-roster approval is claimed.
"""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path

from adapters.common_native_wrapper import ARMS, BACKBONES, CONTRACT, DATASETS, I4, SOURCES, SOURCE_ANCHOR, bound, receipt, render_sources
from spec import canonical, confined, fingerprint, validate

DOMAIN = [{'learning_rate': lr, 'dropout': dropout} for lr in (.0005, .001, .002) for dropout in (.1, .2, .4)]
SEARCH_SEEDS = (88101, 88103)
COMPARATIVE_SEEDS = (88201, 88203, 88207)
METHODS = {
    'ordinary_M1': 'ordinary_native_single', 'ordinary_I4': 'ordinary_genuine_independent4',
    'factorized_allmap_M1': 'allmap_factor_single', 'equipped_I4': 'allmap_factor_genuine_independent4',
    'shared4_unchanged': 'allmap_factor_shared4', 'separable_equal_size': 'common_native_wrapper_separable_equal_size',
    'exchange': 'common_native_wrapper_exchange', 'original_gnnm': 'original_boundary_GNNM',
}


def bind_metadata(root, custody_path, receipt_path):
    """Verify raw copied metadata and preserve original payload receipts unread."""
    custody_path, receipt_path = Path(custody_path), Path(receipt_path)
    original = json.loads(receipt_path.read_text())
    raw = custody_path.read_bytes()
    if len(raw) != original['bytes'] or hashlib.sha256(raw).hexdigest() != original['sha256']:
        raise ValueError('Copied custody bytes differ from allocation receipt')
    custody = json.loads(raw)
    dataset = custody['dataset_id']
    expected = {'wikics': (11701, 300, 10, 442907), 'amazon-photo': (7650, 745, 8, 245812)}
    if (dataset not in DATASETS or custody.get('schema') != 'node-provider-role-custody-v1' or custody.get('status') != 'prepared'
            or custody.get('benchmark_fits') != 0 or custody.get('predictive_scores') != 0
            or tuple(custody['model_graph'][k] for k in ('nodes', 'features', 'classes', 'edges')) != expected[dataset]
            or custody['model_graph']['self_edges'] != expected[dataset][0]
            or custody['provider_pyg_version'] != '2.7.0'
            or sorted(row['split_id'] for row in custody['splits']) != list(range(DATASETS[dataset]))):
        raise ValueError('Exact prepared homogeneous complete WikiCS20/Photo10 provider roster required')
    original_path = Path(original['path'])
    if original_path != Path('training_pipeline_v2_20261012/prepared_data') / dataset / 'CUSTODY.json':
        raise ValueError('Original allocation custody path differs')
    data_dir = original_path.parent
    files = custody['files']
    if len({row['path'] for row in files}) != len(files):
        raise ValueError('Ambiguous payload receipt paths')
    for row in files:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or type(row['bytes']) is not int or row['bytes'] < 1 or len(row['sha256']) != 64:
            raise ValueError('Invalid metadata payload receipt')
    bindings = [receipt(root, custody_path), receipt(root, receipt_path)]
    # These records are never verified by touching payloads during preparation.
    deferred = [{'path': str(data_dir / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']} for row in files]
    return custody, data_dir, bindings, deferred


def make_study(custody, data_dir, backbone, source, sources, runtime):
    dataset = custody['dataset_id']
    split_rows = {row['split_id']: row for row in custody['splits']}
    files = {row['path']: row for row in custody['files']}
    def inputs(split, field):
        return {key: {'path': str(data_dir / rel), 'bytes': files[rel]['bytes'], 'sha256': files[rel]['sha256']}
                for key, rel in split_rows[split][field].items()}
    def common(arm, split, stage):
        members = 4 if arm in ('shared4_unchanged', 'separable_equal_size', 'exchange', 'original_gnnm') else 1
        return {'wrapper_arm': arm, 'backbone': backbone, 'architecture': 'original_gnnm' if arm == 'original_gnnm' else
                'single' if arm == 'ordinary_M1' else 'factorized_single' if members == 1 else 'internal_be',
            'members': members, 'physical_slow_bodies': 1, 'source_optimizer_count': 1,
            'depth': 2, 'width': 128, 'rank': 16, 'classes': custody['model_graph']['classes'], 'weight_decay': 0.,
            'provider_graph': custody['model_graph'], 'wrapper_sources': sources, 'node_inputs': inputs(split, 'node_inputs'),
            'is_original_gnnm': arm == 'original_gnnm', 'fusion': 'mean_logits' if arm == 'original_gnnm' else 'mean_probabilities',
            'stage': stage, 'source_contract': CONTRACT, 'reservation_mib': 32768,
            'estimated_seconds': 1, 'scheduling_weight_status': 'unmeasured placeholder in legacy seconds field; no ETA or admission claim',
            'resource_evidence': 'root-owned resource qualification and three-GPU assignment pending',
            'recipe_scope': 'literal source model/operator/initialization/objective; standardized provider/controller/search adaptation'}
    runs, selections, banks = [], {}, []
    def add(identifier, kind, arm, split, seed, body, config, parents=()):
        entry = ('common_native_wrapper.py:make_adapter' if kind == 'model_fit' else 'search.py:select_configuration' if kind == 'selection'
                 else 'node_endpoints.py:' + {'ensemble_assembly': 'export_bank', 'calibration': 'scalar_calibration', 'evaluation': 'evaluate_bank'}[kind])
        row = {'run_id': identifier, 'kind': kind, 'method': METHODS[arm], 'split': split, 'seed': seed, 'body': body,
            'adapter': runtime + '/adapters/' + entry, 'config': config, 'parents': list(parents),
            'budget': {'unit': 'optimizer_updates', 'policy': 'full_budget', 'maximum': 5000} if kind == 'model_fit' else {},
            'selection': {'metric': 'accuracy', 'direction': 'max', 'ties': 'first', 'evaluation_interval': 1,
                'checkpoint0': False, 'patience_steps': 5000} if kind == 'model_fit' else {}}
        runs.append(row)
        return row['run_id']
    for arm in ARMS:
        candidates = []
        for point in DOMAIN:
            config_id = fingerprint(point)[:20]
            ids = [add('search-' + arm + '-' + config_id + '-s' + str(seed), 'model_fit', arm, 0, seed, 0,
                {**common(arm, 0, 'search'), **point}) for seed in SEARCH_SEEDS]
            candidates.append({'config_id': config_id, 'parameters': point, 'run_ids': ids})
        selections[arm] = add('select-' + arm, 'selection', arm, 0, 0, 0,
            {'stage': 'search_selection', 'configurations': candidates, 'ranking_metric': 'accuracy', 'ranking_direction': 'max',
                'reservation_mib': 1, 'estimated_seconds': 1}, [identifier for row in candidates for identifier in row['run_ids']])
    for split in sorted(split_rows):
        for seed in COMPARATIVE_SEEDS:
            fits = {}
            for arm in ARMS:
                fits[arm] = [add('compare-sp' + str(split) + '-' + arm + '-s' + str(seed), 'model_fit', arm, split, seed, 0,
                    {**common(arm, split, 'comparative'), **SOURCE_ANCHOR, 'selected_config_from': selections[arm]}, [selections[arm]])]
            for bank_arm, arm in I4.items():
                # All four are freshly trained, distinct from the paired M1 fit.
                fits[bank_arm] = [add('compare-sp' + str(split) + '-' + bank_arm + '-s' + str(seed) + '-b' + str(body),
                    'model_fit', arm, split, seed, body,
                    {**common(arm, split, 'comparative'), **SOURCE_ANCHOR, 'selected_config_from': selections[arm]}, [selections[arm]]) for body in range(4)]
            for arm, parents in fits.items():
                single_arm = I4.get(arm, arm)
                fusion = common(single_arm, split, 'comparative')['fusion']
                endpoint_inputs = {key: row['path'] for key, row in inputs(split, 'endpoint_inputs').items()}
                prefix = 'bank-sp' + str(split) + '-' + arm + '-s' + str(seed)
                export = add(prefix, 'ensemble_assembly', arm, split, seed, 0,
                    {'stage': 'bank_export', 'fit_parents': parents, 'endpoint_inputs': endpoint_inputs, 'fusion': fusion,
                        'reservation_mib': 32768, 'estimated_seconds': 1}, parents)
                calibration = add(prefix + '-cal', 'calibration', arm, split, seed, 0,
                    {'stage': 'calibration', 'export_parent': export, 'endpoint_inputs': endpoint_inputs, 'fusion': fusion,
                        'placement': 'after_serving_pool', 'steps': 500, 'learning_rate': .01, 'reservation_mib': 512, 'estimated_seconds': 1}, [export])
                evaluation = add(prefix + '-evaluate', 'evaluation', arm, split, seed, 0,
                    {'stage': 'evaluation', 'export_parent': export, 'calibration_parent': calibration, 'endpoint_inputs': endpoint_inputs,
                        'fusion': fusion, 'reservation_mib': 512, 'estimated_seconds': 1}, [export, calibration])
                banks.append({'arm': arm, 'split': split, 'seed': seed, 'fit_parents': parents, 'export': export,
                    'calibration': calibration, 'evaluation': evaluation, 'genuine_I4': arm in I4, 'body0_reused': False,
                    'served_views': 4 if arm in I4 else common(single_arm, split, 'comparative')['members']})
    roles = {role: {str(split): inputs(split, 'node_inputs' if role in ('train', 'select') else 'endpoint_inputs')[
        role if role in ('train', 'select') else role + '_targets'] for split in sorted(split_rows)} for role in ('train', 'select', 'cal', 'evaluate')}
    study = {'schema_version': 1, 'study_id': 'common-native-wrapper-v1-' + dataset + '-' + backbone.lower(), 'stage': 'development', 'dataset': dataset,
        'hypothesis': 'Native aligned route exchange can change served generalization versus same-size separable and unchanged shared blocks, with ordinary/equipped genuine independent and original GNNM references.',
        'data': {'task': 'node_classification', 'preprocessing': custody['request']['graph_policy'], 'roles': roles,
            'prior_exposure': custody['request']['evaluation_exposure'], 'evidence': {'custody_request_hash': custody['request_hash'],
                'model_graph': custody['model_graph'], 'split_roster': sorted(split_rows), 'search_split': 0, 'source_exact_training_recipe': False}},
        'source': source, 'runs': runs, 'comparisons': [{'references': [METHODS[arm] for arm in METHODS if arm != 'exchange'],
            'candidates': [METHODS['exchange']], 'scope': 'every complete registered mask and fixed comparison seed; all outcomes/costs/failures retained',
            'interpretation': 'development evidence; masks/seed views are not independent graphs; no accuracy or acceptance guarantee'}],
        'release': {'policy': 'whole_roster', 'primary_endpoint': 'all raw and after-pool CAL500 banks', 'global_plan_gates_required': True},
        'resources': {'wall_seconds': 1, 'status': 'UNMEASURED_DISABLED_CANDIDATE', 'wall_seconds_is_placeholder': True}}
    validate(study)
    fits = [row for row in runs if row['kind'] == 'model_fit']
    stage_counts = Counter(row['config']['stage'] for row in fits)
    backwards = sum(4 if row['config']['wrapper_arm'] == 'original_gnnm' else 1 for row in fits) * 5000
    train_views = sum(row['config']['members'] for row in fits) * 5000
    counts = {'optimizer_starts_by_stage': dict(stage_counts), 'atomic_fit_groups': len(fits), 'paid_optimizer_objects': len(fits),
        'physical_body_instances': len(fits), 'full_budget_optimizer_updates': len(fits) * 5000, 'logical_backward_calls': backwards,
        'full_view_native_TRAIN_forwards': train_views, 'search_selection_jobs': len(selections), 'comparative_banks': len(banks),
        'bank_exports': len(banks), 'scalar_CAL_fits': len(banks), 'scalar_CAL_optimizer_updates': len(banks) * 500,
        'evaluation_jobs': len(banks), 'genuine_I4_banks': sum(row['genuine_I4'] for row in banks),
        'reused_I4_body0': 0, 'all_run_records': len(runs), 'actual_optimizer_updates': 0, 'actual_predictive_scores': 0}
    return study, counts, banks


def compile_packet(root, packet, output):
    root, packet, output = Path(root).resolve(), Path(packet).resolve(), Path(output).resolve()
    if not packet.is_relative_to(root) or not output.is_relative_to(packet) or output.exists():
        raise ValueError('Owned fresh local preparation output required')
    bindings = json.loads((packet / 'SOURCE_BINDINGS.json').read_text())
    runtime = str((packet / 'runtime').relative_to(root))
    source_files = [receipt(root, path) for path in sorted((packet / 'runtime').rglob('*.py'))]
    source_files += list(bindings.values()) + [receipt(root, packet / 'SOURCE_BINDINGS.json')]
    output.mkdir()
    (output / 'studies').mkdir()
    studies, prepared, banks, totals, stages, adaptations = [], [], [], Counter(), Counter(), []
    for dataset in DATASETS:
        custody, data_dir, metadata_bindings, deferred = bind_metadata(root, packet / 'metadata' / dataset / 'CUSTODY.json', packet / 'metadata' / dataset / 'RECEIPT.json')
        rendered = render_sources(root, bindings, custody['model_graph']['features'], custody['model_graph']['classes'])
        adaptations.append({'dataset': dataset, **rendered})
        source = {'git_commit': 'PENDING_ROOT_SOURCE_COMMIT_BINDING', 'source_commit_status': 'pending_root_owned',
            'files': source_files + metadata_bindings + deferred, 'working_tree_fingerprints_authoritative': True,
            'frozen_runtime_package': runtime, 'official_author_protocol_replication': False,
            'payload_bindings_status': 'original custody declarations; payloads unread/unverified locally; root allocation admission required'}
        for backbone in BACKBONES:
            study, counts, study_banks = make_study(custody, data_dir, backbone, copy.deepcopy(source), bindings, runtime)
            rel = str((output / 'studies' / (study['study_id'] + '.json.gz')).relative_to(root))
            (root / rel).write_bytes(gzip.compress((canonical(study) + '\n').encode(), mtime=0))
            studies.append({'path': rel, 'spec_hash': fingerprint(study)})
            prepared.append({'study_id': study['study_id'], 'dataset': dataset, 'backbone': backbone, 'counts': counts})
            banks.extend({'study_id': study['study_id'], **row} for row in study_banks)
            totals.update({key: value for key, value in counts.items() if type(value) is int})
            stages.update(counts['optimizer_starts_by_stage'])
    totals['optimizer_starts_by_stage'] = dict(stages)
    if (len(studies) != 6 or dict(stages) != {'search': 648, 'comparative': 3780} or totals['atomic_fit_groups'] != 4428
            or totals['comparative_banks'] != 2160 or totals['genuine_I4_banks'] != 540 or totals['reused_I4_body0'] != 0):
        raise ValueError('Conservative complete six-study fixed-count design differs')
    prefix = str(output.relative_to(root))
    groups = [{'id': row['study_id'], 'gpu_assignment': 'allocation/gpu0', 'allowed_workers': ['allocation/gpu0'],
        'maximum_cuda_bytes': 32768 * 1024**2, 'estimated_seconds': row['counts']['all_run_records'],
        'assignment_status': 'preview placeholder; root-owned final three-GPU staging pending',
        'run_keys': [row['study_id'] + '/' + run['run_id'] for run in json.loads(gzip.decompress((root / path['path']).read_bytes()))['runs']]}
        for row, path in zip(prepared, studies)]
    plan = {'schema_version': 1, 'slots_per_gpu': 2, 'torch_threads_per_fit': 2, 'output_root': prefix + '/execution',
        'runtime_launcher': runtime + '/launcher.py', 'runtime_package': runtime, 'worker_capacities_cuda_bytes': {'allocation/gpu0': 78 * 1024**3},
        'studies': studies, 'groups': groups, 'global_gates': {}, 'barriers': {}, 'scientific_launch_enabled': False, 'human_GO_required': True,
        'scientific_roster_approved': False, 'resource_admission_status': 'pending_root_qualification', 'source_commit_status': 'pending_root_owned',
        'preparation_scope': 'candidate common-native wrapper port; accepted count proposal is not an approved scientific roster',
        'resource_estimates_are_measurements': False, 'scheduling_weights_are_ETA': False,
        'host_bindings': {'allocation': {'status': 'root_owned_live_verification_pending', 'hostname': 'anogena-2-0',
            'gpu_uuids': ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']}}}
    for name, source_stages, target in [('whole_search', ['search', 'search_selection'], ['comparative']),
            ('whole_comparative', ['comparative'], ['bank_export', 'calibration', 'evaluation']),
            ('whole_calibration', ['calibration'], ['evaluation'])]:
        plan['global_gates'][name] = {'path': prefix + '/gate_receipts/' + name + '.json', 'source_stages': source_stages,
            'required_for_stages': target, 'required_hosts': ['allocation'],
            'receipt_contract': 'status=complete;plan_hash=current_plan_hash;closed_phase=gate_name;host_receipts_keys=required_hosts'}
    summary = {'schema': 'common-native-wrapper-local-preparation-v1', 'status': 'DISABLED_CANDIDATE_ROOT_SOURCE_RESOURCE_MASTER_ROSTER_AND_GO_PENDING',
        'plan_hash': fingerprint(plan), 'counts': dict(totals), 'studies': prepared, 'banks': banks,
        'domain': DOMAIN, 'source_anchor': SOURCE_ANCHOR, 'source_anchor_domain_membership_count': DOMAIN.count(SOURCE_ANCHOR),
        'search_seed_ids': list(SEARCH_SEEDS), 'comparative_seed_ids': list(COMPARATIVE_SEEDS), 'source_contract': CONTRACT,
        'newly_scoped_canonical_methods': 2, 'common_backbone_source_scope': list(BACKBONES), 'body0_artifact_reuse': False,
        'scientific_training_started': False, 'GO_created': False, 'real_tensors_opened': False,
        'remote_operations': False, 'git_operations': False, 'source_exact_recipe': False}
    for name, value in [('PLAN.json', plan), ('SUMMARY.json', summary), ('PROVIDER_AST_BINDINGS.json', adaptations)]:
        (output / name).write_text(canonical(value) + '\n')
    return {'counts': dict(totals), 'plan_hash': fingerprint(plan), 'studies': len(studies), 'scientific_fits': 0, 'GO': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'packet', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    def local(path):
        value = Path(path)
        return confined(root, str(value.resolve().relative_to(root)) if value.is_absolute() else str(value))
    print(canonical(compile_packet(root, local(args.packet), local(args.output))))


if __name__ == '__main__':
    main()
