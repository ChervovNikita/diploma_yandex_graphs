"""Complete native SAGE continuous context affinity reader; stdlib until full custody.

No native model, loader, checkpoint deserialization, forward, selector or TEST.
Exactly75 admitted original calibration endpoints are reused without fitting;
105 fresh fixed endpoint attempts finish before any score or contrast.
"""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PILOT = 'SAGE_continuous_context_affinity_pilot_root_20261011_v2'
LEARNING = 'native_SAGE_continuous_context_affinity_source_20261011_v2'
PRIOR = 'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
PRIOR_READER = 'SAGE_GNCL_SupCon_2x2_complete_reader_source_20261010_v2/analysis.py'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
REFERENCE = 'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
REFERENCE_SHA = 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
CALIBRATOR = 'common_wrapper_graph_reliability_source_20261010_v2/operators.py'
SEEDS = (7301, 7403, 7507)
NEW = ('ordinaryM1_weighted_mixture', 'factorizedM1_weighted_mixture',
       'genuinefactorizedI4_weighted_mixture', 'genuinefactorizedI4_weighted_assigned', 'shared4_weighted_private',
       'shared4_weighted_mixture', 'shared4_uniform_allclassmate')
REFS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
OLD = REFS + ('separable_equal_size', 'exchange')
ARMS = NEW + REFS
O1C, F1C, F4C, F4A, PRIVATE, MIX, UNIFORM = NEW
O1, O4, F1, F4, BASE = REFS
CONTEXT_MODES = {O1C: 'weighted_mixture', F1C: 'weighted_mixture', F4C: 'weighted_mixture',
                 F4A: 'weighted_assigned', PRIVATE: 'weighted_private', MIX: 'weighted_mixture', UNIFORM: 'uniform_allclassmate'}
CONTEXT_NAMES = ('X', 'X-PX', 'PX', 'P^2X')
TARGET_KEYS = {'train_ids', 'train_y', 'single_order', 'weights', 'uniform_weights', 'positive', 'full_allowed'}
OBJECTIVE_CONTRACT = {'target_policy': 'continuous_all_classmate_full_denominator_v1',
    'coefficient': .05, 'student_temperature': .2, 'teacher_temperature': .2,
    'objective': 'outside_log_weighted_CE_all_classmates_all_nonself_TRAIN',
    'mixture': 'mean_of_four_complete_objectives_for_each_body_or_route',
    'assigned_independent_body': 'body_m_uses_Q_m_with_explicit_context_index',
    'native_factual_forwards_per_route_update': 1}
VALID_HASHES = {'ids': '48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801',
                'labels': '2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'}
EXPECTED_PROTOCOL = {'schema': 'native_SAGE_continuous_context_affinity_complete_reader_protocol_v2', 'production_protocol': {'scope': 'one_backbone_SAGE_encountered_development', 'new_banks': 21, 'new_optimizer_native_fits': 39, 'original_anchor_banks': 15, 'original_selected_fit_records': 33, 'total_selected_fit_records': 72, 'total_comparative_banks': 36, 'calibration_endpoints': 180, 'reused_endpoints': 75, 'new_endpoints': 105, 'candidate': 'shared4_weighted_private', 'raw_gain_pp': {'shared4_unchanged': 0.2, 'genuinefactorizedI4_weighted_mixture': 0.2, 'genuinefactorizedI4_weighted_assigned': 0.2, 'shared4_weighted_mixture': 0.1, 'shared4_uniform_allclassmate': 0.1}, 'per_seed_nonnegative': True, 'at_least_two_positive': True, 'NLL_mean_harm_max': 0.02, 'NLL_per_seed_harm_max': 0.05, 'TEST_access': False, 'primitive_novelty_claim': False, 'original_scores_unchanged': True, 'all_roster_before_interpretation': True}, 'calibration_policy': {'temperature_parameter': 'one bank-global logT; positive exp(logT)', 'folds': 5, 'fold_order': 'CPU randperm seed11709; position modulo5', 'dtype': 'float64 CPU', 'optimizer': 'Adam', 'learning_rate': 0.01, 'steps_per_endpoint': 500, 'endpoint': 'final step', 'calibrated_banks': 36, 'scalar_fits': 180, 'scalar_steps': 90000, 'operations_source': 'common_wrapper_graph_reliability_source_20261010_v2/operators.py', 'equally_applied_to_all_banks': True, 'TEST_access': False, 'all_VALID_refit': False, 'scope': 'postselection encountered development', 'risk_rule': 'DECISION.md'}, 'calibration': {'folds': 5, 'fold_seed': 11709, 'global_log_temperature_initial': 0, 'optimizer': 'CPU float64 Adam', 'learning_rate': 0.01, 'betas': [0.9, 0.999], 'epsilon': 1e-08, 'weight_decay': 0, 'updates': 500, 'endpoint': 'fixed final update', 'reused_endpoints': 75, 'new_fit_attempts': 105, 'all_terminal_records': 180, 'new_requested_scalar_updates': 52500, 'original_scalar_updates_already_paid': 37500, 'total_comparative_requested_scalar_updates': 90000, 'retries': 0, 'grid': False, 'whole_VALID_refit': False}, 'calibrated_accuracy_diagnostic_only': True, 'member_competence_diagnostic_not_extra_veto': True, 'both_equipped_singles_reported_as_capable_references': True, 'stronger_single_limits_broad_superiority': True, 'raw_and_protected_findings_separate': True, 'target_preparation_replayed': False, 'native_model_or_checkpoint_deserialization': False, 'TEST_access': False, 'automatic_launch': False, 'scope': 'One native SAGE backbone on encountered WikiCS development; complete fitted/stopped/selected procedures, not same-state causal isolation or unused confirmation.', 'fixed_target_policy': 'continuous_all_classmate_full_denominator_v1', 'teacher_temperature': 0.2, 'collapse_prevention_or_realizability_guarantee': False, 'assigned_independent_target_index_bound': True}

BASE_KEYS = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}
LIMIT, CHUNK_BUDGET = 2000000, 1800000

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def scoped(path, parent):
    value = Path(path).resolve()
    require(value.is_relative_to(parent.resolve()), 'Out-of-scope artifact: ' + str(value))
    return value

def load_source(name, path, digest):
    require(sha(path) == digest, 'Immutable source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def strict_mask(np, logits, labels):
    truth = np.take_along_axis(logits, labels[None, :, None], axis=2)
    return (logits > truth).all(0).any(1)

def strict_counts(np, mask, served, labels):
    return {'nodes': int(mask.sum()), 'corrected': int((mask & served).sum()),
            'classes': [{'class': k, 'nodes': int((mask & (labels == k)).sum()),
                         'corrected': int((mask & served & (labels == k)).sum())} for k in range(10)]}

def encode(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'

def chunk_documents(category, ordered_rows):
    """Deterministic roster order; split before a small document can overflow."""
    documents, current, offset = [], [], 0
    for row in ordered_rows:
        candidate = {'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current + [row]}
        if len(encode(candidate)) >= CHUNK_BUDGET and current:
            documents.append((category + '_part' + str(len(documents) + 1) + '.json',
                              {'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current}))
            offset += len(current)
            current = []
        current.append(row)
        require(len(encode({'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current})) < CHUNK_BUDGET,
                'An individual fixed row exceeds bounded export budget')
    if current:
        documents.append((category + '_part' + str(len(documents) + 1) + '.json',
                          {'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current}))
    return documents


def retained_json(value):
    if isinstance(value, dict):
        return {k: retained_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [retained_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    return value


def check_group_costs(row):
    units, costs = row['fits'], row['costs']
    keys = set(units[0]['costs'])
    require(all(set(u['costs']) == keys for u in units), 'Consistent native fit cost fields')
    for unit in units:
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                    for v in unit['costs'].values()), 'Finite nonnegative complete native costs')
    require(set(costs) == keys | {'selected_pool_seconds', 'selected_serving_readout_seconds'}, 'Complete group cost schema')
    for key in keys:
        expected = max(u['costs'][key] for u in units) if 'peak' in key else sum(u['costs'][key] for u in units)
        require(costs[key] == expected, 'Original sum/peak group accounting: ' + key)
    require(math.isfinite(costs['selected_pool_seconds']) and costs['selected_pool_seconds'] >= 0
            and costs['selected_serving_readout_seconds'] == costs['selected_forward_and_metrics_seconds'] + costs['selected_pool_seconds'],
            'Pool timing and inclusive selected serving accounting')


def admit_reuse(root, here, freeze, inputs, helper, expected, reference_admission):
    """Close all75 original endpoints and prediction bytes before any array."""
    path = here / 'CALIBRATION_REUSE_ADMISSION.json'
    reuse, policy = read(path), read(here / 'CALIBRATION_POLICY.json')
    for name in ('CALIBRATION_REUSE_ADMISSION.json', 'CALIBRATION_POLICY.json'):
        require(helper.freeze_hash(freeze, PILOT + '/' + name) == sha(here / name), 'Reuse and equal law frozen before acquisition')
    require(reference_admission['calibration_reuse_admission'] == reuse
            and reuse['admitted'] is True and reuse['reused_banks'] == 15
            and reuse['reused_scalar_endpoints'] == 75 and reuse['new_scalar_endpoints'] == 105
            and reuse['total_endpoints'] == 180 and reuse['new_scalar_steps'] == 52500
            and reuse['TEST_access'] is False and reuse['no_native_forward'] is True and reuse['no_fit_replay'] is True,
            'Exact root75-endpoint reuse admission;105 new endpoints only')
    expected_policy = {'temperature_parameter': 'one bank-global logT; positive exp(logT)', 'folds': 5,
        'fold_order': 'CPU randperm seed11709; position modulo5', 'dtype': 'float64 CPU', 'optimizer': 'Adam',
        'learning_rate': .01, 'steps_per_endpoint': 500, 'endpoint': 'final step', 'calibrated_banks': 36,
        'scalar_fits': 180, 'scalar_steps': 90000, 'operations_source': CALIBRATOR,
        'equally_applied_to_all_banks': True, 'TEST_access': False, 'all_VALID_refit': False,
        'scope': 'postselection encountered development', 'risk_rule': 'DECISION.md'}
    prior_policy = read(root / PRIOR / 'CALIBRATION_POLICY.json')
    roster_fields = {'calibrated_banks', 'scalar_fits', 'scalar_steps'}
    require({k: v for k, v in prior_policy.items() if k not in roster_fields}
            == {k: v for k, v in policy.items() if k not in roster_fields}
            and {k: prior_policy[k] for k in roster_fields}
            == {'calibrated_banks': 33, 'scalar_fits': 165, 'scalar_steps': 82500},
            'Same original operator law; only total roster work changes')
    require(policy == expected_policy and reuse['policy_sha256'] == sha(here / 'CALIBRATION_POLICY.json')
            and reuse['prior_policy_sha256'] == sha(root / PRIOR / 'CALIBRATION_POLICY.json')
            and reuse['calibration_operator_sha256'] == inputs['research_files'][CALIBRATOR]
            and reuse['prior_analysis_source_sha256'] == inputs['research_files'][PRIOR_READER], 'Same exact equal calibration law/source')
    admission = read(here / 'SCIENTIFIC_ADMISSION.json')
    require(admission['calibration_policy_sha256'] == reuse['policy_sha256']
            and admission['calibration_reuse_admission_sha256'] == sha(path)
            and admission['reference_admission_sha256'] == sha(here / 'REFERENCE_ADMISSION.json')
            and admission['source_sha256'] == inputs['research_files'][LEARNING + '/run_family.py']
            and admission['original_reference_banks'] == 15 and admission['original_selected_fits'] == 33
            and admission['all_banks_before_interpretation'] == 36
            and admission['fixed_calibration_fits'] == 180 and admission['reused_scalar_endpoints'] == 75
            and admission['new_scalar_endpoints'] == 105,
            'Scientific admission closes the whole36-bank/180-endpoint comparative procedure')
    prior = root / PRIOR
    end_path, summary_path = prior / 'READER_JOB_END_V1.json', prior / 'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json'
    require(sha(end_path) == reuse['prior_reader_end_sha256'] and sha(summary_path) == reuse['prior_complete_summary_sha256'],
            'Prior successful reader and complete summary custody')
    end, summary = read(end_path), read(summary_path)
    require(end['success'] is True and end['exit_code'] == 0 and end['TEST_access'] is False
            and summary['complete'] is True and summary['TEST_access'] is False
            and summary['original_reference_banks'] == 15 and summary['original_selected_fits'] == 33
            and summary['all165_scalar_attempts_terminal_before_interpretation'] is True
            and summary['calibration_folds']['assignment_sha256'] == reuse['fold_assignment_sha256'],
            'Complete prior result, finite original endpoints and fixed folds')
    prior_output = summary_path.parent
    prior_documents = {str(scoped(r['path'], prior_output)): r for r in [summary['full_report']] + summary['partitions']}
    require(len(prior_documents) == 1 + len(summary['partitions']), 'Unique prior output descriptors')
    used = []
    def admitted_document(name):
        document = prior_output / name
        receipt = prior_documents[str(document)]
        require(document.stat().st_size == receipt['bytes'] and sha(document) == receipt['sha256'], 'Prior extracted record bytes: ' + name)
        used.append({'path': str(document), 'bytes': receipt['bytes'], 'sha256': receipt['sha256']})
        return read(document)
    fit_document = admitted_document('CALIBRATION_FIT_DETAILS_part1.json')
    archive_document = admitted_document('ARCHIVE_CUSTODY_part1.json')
    require(fit_document['category'] == 'CALIBRATION_FIT_DETAILS' and archive_document['category'] == 'ARCHIVE_CUSTODY'
            and fit_document['offset'] == archive_document['offset'] == 0
            and fit_document['total_rows'] == len(fit_document['rows']) == 33
            and archive_document['total_rows'] == len(archive_document['rows']) == 33, 'Complete prior endpoint/archive record partitions')
    fit_index = {(r['seed'], r['arm']): r for r in fit_document['rows']}
    archive_index = {(r['seed'], r['arm']): r for r in archive_document['rows']}
    require(len(fit_index) == len(archive_index) == 33, 'Unique prior endpoint/archive bank indexes')
    rows = {(r['seed'], r['arm']): r for r in reuse['records']}
    require(len(rows) == len(reuse['records']) == 15 and set(rows) == {(s, a) for s in SEEDS for a in REFS}, 'All15 unchanged original banks')
    descriptors = []
    for arm in REFS:
        for seed in SEEDS:
            row, raw = rows[seed, arm], expected[seed, arm]
            require(row['raw_archive'] == archive_index[seed, arm]
                    and row['raw_archive']['path'] == raw['path'] and row['raw_archive']['sha256'] == raw['sha256']
                    and row['raw_archive']['members'] == raw['members'] and row['raw_archive']['new'] is False,
                    'Exact original source/config/role/selected-prediction binding')
            record = row['endpoint_record']
            require(record == fit_index[seed, arm] and record['arm'] == arm and record['seed'] == seed
                    and record['status'] == 'finite_complete_five_fold' and record['attempted_scalar_fits'] == 5
                    and len(record['fits']) == 5 and [f['fold'] for f in record['fits']] == list(range(5)), 'Admitted complete original five-fold terminal records')
            for fold, fit in enumerate(record['fits']):
                held_nodes = 1055 if fold < 4 else 1054
                require(fit['status'] == 'finite_fixed_endpoint' and fit['updates'] == 500 and fit['parameters'] == 1
                        and fit['fit_nodes'] == 5274 - held_nodes and fit['held_nodes'] == held_nodes
                        and fit['selected_endpoint'] == 'fixed_final_update_no_heldout_selector'
                        and set(fit['state']) == {'log_T'} and len(fit['state']['log_T']) == 1
                        and math.isfinite(fit['state']['log_T'][0]) and math.isfinite(fit['temperature']) and fit['temperature'] > 0
                        and math.isclose(math.exp(fit['state']['log_T'][0]), fit['temperature'], rel_tol=1e-12)
                        and [r['update'] for r in fit['trace']] == [1, 100, 200, 300, 400, 500]
                        and all(math.isfinite(r['fusion_fit_objective_before_update']) for r in fit['trace'])
                        and math.isfinite(fit['seconds']) and fit['seconds'] >= 0, 'Exact75 fixed scalar endpoints without replay')
            require(record['fit_seconds_sum'] == row['prior_fit_seconds'] == sum(f['seconds'] for f in record['fits']), 'Original scalar costs preserved')
            descriptor = record['OOF_archive']
            archive = scoped(row['calibrated_archive'], prior_output)
            require(archive == prior_output / f'CALIBRATED_{arm}_seed{seed}_OOF.npz'
                    and descriptor['name'] == archive.name and archive.stat().st_size == descriptor['bytes']
                    and sha(archive) == row['calibrated_sha256'] == descriptor['sha256'], 'Exact admitted original OOF prediction bytes')
            descriptors.append({'path': str(archive), 'bytes': descriptor['bytes'], 'sha256': descriptor['sha256'], 'seed': seed, 'arm': arm, 'new': False})
    require(len(descriptors) == len({r['path'] for r in descriptors}) == 15
            and sum(r['endpoint_record']['attempted_scalar_fits'] for r in rows.values()) == 75, 'All75 prior endpoints admitted before arrays')
    return reuse, descriptors, {'reader_end': {'path': str(end_path), 'sha256': sha(end_path)},
        'summary': {'path': str(summary_path), 'sha256': sha(summary_path)}, 'record_partitions': used,
        'policy_sha256': reuse['policy_sha256'], 'fold_assignment_sha256': reuse['fold_assignment_sha256'],
        'original_endpoint_scores_are_preserved': True, 'refitted_endpoints': 0}



def target_arrays(np, context):
    """Saved qualified Q/support/order custody after all72 fits/75 reused endpoints; no target replay."""
    item, meta = context['target_archive'], context['complete']['context_preparation_once']
    require(sha(item['path']) == item['sha256'], 'Admitted target archive unchanged')
    with np.load(item['path'], allow_pickle=False) as archive:
        require(set(archive.files) == TARGET_KEYS, 'Exact saved continuous-affinity target/order schema')
        arrays = {k: archive[k].copy() for k in archive.files}
    require(sha(item['path']) == item['sha256'], 'Target archive unchanged during read')
    ids, labels, order = [arrays[k] for k in ('train_ids', 'train_y', 'single_order')]
    require(ids.dtype == labels.dtype == order.dtype == np.int64
            and ids.shape == labels.shape == order.shape == (580,)
            and len(np.unique(ids)) == 580 and ids.min() >= 0 and ids.max() < 11701
            and set(labels.tolist()) == set(range(10))
            and np.array_equal(order, np.argsort(ids, kind='stable')), 'Native bank/single TRAIN order custody')
    for name, values in (('ordered_TRAIN_ids', ids), ('ordered_TRAIN_y', labels), ('single_order', order)):
        require(hashlib.sha256(values.tobytes(order='C')).hexdigest()
                == meta[name + '_int64_bytes_sha256'], 'Pinned ordered TRAIN bytes: ' + name)
    require(set(meta['array_hashes']) == TARGET_KEYS - {'train_ids', 'train_y', 'single_order'}, 'Complete canonical target hash keys')
    for name, expected in meta['array_hashes'].items():
        value = arrays[name]
        require({'shape': list(value.shape), 'dtype': str(value.dtype),
                 'C_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()} == expected,
                'Qualified canonical target/support bytes: ' + name)
    allowed = ~np.eye(580, dtype=bool)
    positive = arrays['positive']
    require(positive.dtype == np.bool_ and positive.shape == (580, 580)
            and np.array_equal(positive, (labels[:, None] == labels[None, :]) & allowed),
            'Saved positive support is every nonself TRAIN classmate')
    require(arrays['full_allowed'].dtype == np.bool_ and arrays['full_allowed'].shape == (580, 580)
            and np.array_equal(arrays['full_allowed'], allowed), 'Identical all-nonself TRAIN denominator')
    weights, uniform = arrays['weights'], arrays['uniform_weights']
    require(weights.dtype == uniform.dtype == np.float64 and weights.shape == (4, 580, 580)
            and uniform.shape == (580, 580) and np.isfinite(weights).all() and np.isfinite(uniform).all()
            and (weights[:, positive] > 0).all() and (uniform[positive] > 0).all()
            and not weights[:, ~positive].any() and not uniform[~positive].any()
            and np.allclose(weights.sum(-1), 1, rtol=0, atol=1e-12)
            and np.allclose(uniform.sum(-1), 1, rtol=0, atol=1e-12), 'Qualified finite normalized saved affinity/uniform support')
    # No signatures, exponentiated affinities or Q are rebuilt. Exact canonical
    # hashes above bind the qualified weighting law. Cast/reorder saved tensors only.
    require(positive.sum(-1).tolist() == meta['positive_counts_per_anchor']
            and [int(positive.sum())] * 4 == meta['positive_counts_per_context']
            and int(allowed.sum()) == meta['full_denominator_entries_per_route']
            and np.unique(labels, return_counts=True)[1].tolist() == meta['class_counts'], 'Saved full-class support counts')
    for order_name, indices in (('bank', np.arange(580)), ('single', order)):
        tensor_hashes = {}
        for key in ('weights', 'uniform_weights', 'positive', 'full_allowed'):
            original = arrays[key]
            values = (original[:, indices[:, None], indices[None, :]] if original.ndim == 3
                      else original[indices[:, None], indices[None, :]]).copy()
            is_weight = 'weights' in key
            values = values.astype(np.float32 if is_weight else np.bool_, copy=False)
            tensor_hashes[key] = {'shape': list(values.shape), 'dtype': 'torch.float32' if is_weight else 'torch.bool',
                'C_bytes_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest()}
        require(tensor_hashes == meta['actual_loss_tensor_hashes'][order_name], 'Joint actual native-order tensor custody: ' + order_name)
    return {'target_archive_checked': True, 'TRAIN_rows': 580, 'contexts': list(CONTEXT_NAMES),
        'target_policy': OBJECTIVE_CONTRACT['target_policy'], 'objective_contract': OBJECTIVE_CONTRACT,
        'positive_counts_per_context': meta['positive_counts_per_context'],
        'full_denominator_entries_per_route': meta['full_denominator_entries_per_route'],
        'signature_zero_rows_per_context': meta['signature_zero_rows_per_context'],
        'nonuniform_target_rows_per_context_exact_diagnostic': meta['nonuniform_target_rows_per_context_exact_diagnostic'],
        'bank_and_single_joint_Q_support_order_bound': True,
        'signature_affinity_or_target_construction_replayed': False,
        'native_forward_or_loss_replayed': False,
        'scope': 'Saved qualified target/order/support bytes and schema only; no hidden geometry, collapse or unused-quality conclusion.'}



def preflight(root, helper, inputs):
    """All72 selected fits, safe roles and75 exact old endpoints close before arrays."""
    for name, digest in inputs['research_files'].items():
        require(sha(root / name) == digest, 'Pinned prospective bytes changed: ' + name)
    require(sha(root / REFERENCE) == REFERENCE_SHA, 'Original SAGE binding unchanged')
    here = root / PILOT
    cfg, freeze = read(here / 'CONFIG.json'), read(here / 'FREEZE.json')
    protocol = read(Path(__file__).with_name('PROTOCOL.json'))
    require(protocol == EXPECTED_PROTOCOL
            and protocol['production_protocol'] == read(here / 'PROSPECTIVE_PROTOCOL.json')
            and protocol['calibration_policy'] == read(here / 'CALIBRATION_POLICY.json'), 'Exact prospectively fixed context criteria/calibration protocol')
    start = read(here / 'OWNER_START.json')
    require(start['freeze_sha256'] == sha(here / 'FREEZE.json')
            and start['hostname'] == 'anogena-2-0' and start['gpu_uuid'] == UUID,
            'Actual launch freeze/host/UUID custody')
    require(freeze.get('qualification_passed') is True and freeze.get('TEST_access') is False,
            'Qualified no-TEST launch freeze')
    for row in freeze['bound_files']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Frozen file changed')
    for name in inputs['acquisition_files'] + [PILOT + '/' + n for n in
                                              ('CONFIG.json', 'DECISION.md', 'PROSPECTIVE_PROTOCOL.json', 'CALIBRATION_POLICY.json')]:
        require(helper.freeze_hash(freeze, name) == inputs['research_files'][name], 'Prospective launch binding: ' + name)
    require(helper.freeze_hash(freeze, CALIBRATOR) == inputs['research_files'][CALIBRATOR], 'Fixed calibration operator frozen before fitting')
    admission_path, reference_path = here / 'SCIENTIFIC_ADMISSION.json', here / 'REFERENCE_ADMISSION.json'
    admission, reference_admission = read(admission_path), read(reference_path)
    for path in (admission_path, reference_path, here / 'ACTUAL_QUALIFICATION_V1.json',
                 here / 'QUALIFICATION_FREEZE.json', here / 'CALIBRATION_REUSE_ADMISSION.json'):
        require(helper.freeze_hash(freeze, PILOT + '/' + path.name) == sha(path), 'Actual root receipt frozen at launch')
    require(admission['helper_sha256'] == inputs['research_files'][LEARNING + '/context_affinity_objective.py']
            and admission['qualification_sha256'] == sha(here / 'ACTUAL_QUALIFICATION_V1.json'),
            'Actual qualified context helper scientific admission')
    require(admission['admitted'] is True and admission['qualification_passed'] is True
            and admission['scientific_fits'] == admission['native_bodies'] == 39 and admission['banks'] == 21
            and admission['new_outcomes_seen'] is False and admission['TEST_access'] is False
            and admission['source_manifest_sha256'] == inputs['research_files'][LEARNING + '/MANIFEST.json'],
            'Complete prospective context scientific admission')
    require(reference_admission['admitted'] is True and reference_admission['banks'] == 15
            and reference_admission['selected_fits'] == 33
            and reference_admission['reference_bindings_sha256'] == REFERENCE_SHA
            and reference_admission['exact_source_config_role_seed_selector_custody_compatible'] is True
            and reference_admission['TEST_access'] is False, 'Exact root original own-only reference admission')
    qualification = read(here / 'ACTUAL_QUALIFICATION_V1.json')
    require(qualification['qualified'] is True and qualification['failure'] is None
            and qualification['VALID_quality_scored'] is False and qualification['TEST_access'] is False
            and qualification['scientific_fits'] == 0 and qualification['scientific_quality_evidence'] is False
            and qualification['auxiliary_only_VJPs'] == qualification['extra_native_replay_forwards'] == 0,
            'Actual finite TRAIN-only qualification, not predictive evidence')
    require(qualification['config_sha256'] == sha(here / 'CONFIG.json')
            and qualification['source_sha256'] == inputs['research_files'][LEARNING + '/run_family.py']
            and qualification['helper_sha256'] == inputs['research_files'][LEARNING + '/context_affinity_objective.py']
            and qualification['qualifier_sha256'] == inputs['research_files'][PILOT + '/qualify.py']
            and qualification['qualification_freeze_sha256'] == sha(here / 'QUALIFICATION_FREEZE.json'), 'Actual qualification source/config custody')
    require(qualification['counts'] == {'native_forward_units': 13, 'native_member_trajectories': 22,
        'attempted_combined_backwards': 13, 'full_parameter_backwards': 13,
        'attempted_Adam_steps': 13, 'discarded_optimizer_updates': 13}, 'Exact13 disposable TRAIN units/22 trajectories')
    cases = qualification['cases']
    expected_cases = {(arm, m) for arm in NEW for m in range(4 if arm in (F4C, F4A) else 1)}
    require(len(cases) == 13 and {(r['arm'], r['member']) for r in cases} == expected_cases
            and all(r['seed'] == SEEDS[0] and r['mode'] == CONTEXT_MODES[r['arm']]
                    and r['routes'] == (4 if r['arm'].startswith('shared4_') else 1)
                    and ((r['arm'] == F4A and type(r.get('context_index')) is int and r['context_index'] == r['member'])
                         or (r['arm'] != F4A and r.get('context_index') is None))
                    and all(r[k] is True for k in ('combined_gradients_finite', 'live_H_and_auxiliary_graph',
                        'original_parameter_identity_and_storage_retained', 'model_optimizer_streams_restored',
                        'targets_unchanged', 'checkpoint_target_mode_order_bound')) for r in cases), 'All actual qualified arm/body modes and custody')
    for row in read(here / 'QUALIFICATION_FREEZE.json')['bound_files']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Qualified dependency/role bytes changed')
    binding = read(root / REFERENCE)
    native_cfg = {k: v for k, v in cfg.items() if k != 'context_teacher_temperature'}
    require(cfg.get('context_teacher_temperature') == .2
            and reference_admission['configuration_sha256'] == sha(here / 'CONFIG.json')
            and reference_admission['original_configuration_sha256'] == binding['config_sha256']
            and reference_admission['auxiliary_only_config_fields'] == {'context_teacher_temperature': .2}
            and reference_admission['native_config_subset_exclusion'] == ['context_teacher_temperature'],
            'Exact separately bound new/original config; only declared teacher-temperature key excluded')
    require(native_cfg == binding['configuration'] and cfg['seeds'] == list(SEEDS)
            and cfg['hidden'] == 128 and cfg['depth'] == 2 and cfg['dropout'] == .2
            and cfg['learning_rate'] == .001 and cfg['max_updates'] == 1000 and cfg['patience_updates'] == 300,
            'Exact original config and fixed native recipe')
    output = here / 'actual_family_v1'
    old_output = scoped(binding['family_root'], root)
    require(old_output == root / 'common_wrapper_SAGE_root_20261010_v1/actual_family_v1', 'Original parent identity')
    owner, old_owner = helper.closed_owner(here / 'OWNER_END.json'), helper.closed_owner(old_output.parent / 'OWNER_END.json')
    require(sha(old_output.parent / 'OWNER_END.json') == binding['owner_end_sha256'], 'Original owner custody')
    header, old_header = helper.completion_header(output / 'COMPLETE_FAMILY.json'), helper.completion_header(old_output / 'COMPLETE_FAMILY.json')
    for head, groups, fits in ((header, 21, 39), (old_header, 21, 39)):
        require(head['complete'] is True and head['groups'] == head['expected_groups'] == groups
                and head['fit_units'] == head['expected_fit_units'] == fits and head['TEST_access'] is False,
                'Complete declared fresh/original rosters before outcomes')
    require(header['completion_scope'] == 'fresh seven-arm continuous affinity acquisition only'
            and header['full_comparative_family_complete'] is False
            and header['requires_root_admitted_own_only_anchors'] == {'banks': 15, 'selected_fit_records': 33}
            and header['joined_expected_groups'] == 36 and header['joined_expected_selected_fit_records'] == 72
            and header['context_objective_contract'] == OBJECTIVE_CONTRACT,
            'Fresh context completion cannot stand in for complete reference admission')
    complete_sha = sha(output / 'COMPLETE_FAMILY.json')
    require(complete_sha == owner['complete_sha256'] and sha(old_output / 'COMPLETE_FAMILY.json')
            == binding['complete_sha256'] == old_owner['complete_sha256'], 'Owner-bound complete records')
    complete, old = read(output / 'COMPLETE_FAMILY.json'), read(old_output / 'COMPLETE_FAMILY.json')
    require({k: v for k, v in complete.items() if k not in ('results', 'comparisons', 'cost_scope')} == header
            and {k: v for k, v in old.items() if k not in ('results', 'comparisons', 'cost_scope')} == old_header,
            'Complete header/records boundary')
    require(complete['config_sha256'] == sha(here / 'CONFIG.json') and read(output / 'CONFIG.json') == cfg, 'Fresh config authority')
    for row in binding['frozen_inputs']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Original frozen source/config/role bytes')
    old_freeze = read(old_output.parent / 'FREEZE.json')
    require(sha(scoped(binding['config'], root)) == binding['config_sha256'] == old['config_sha256']
            and read(binding['config']) == native_cfg and read(old_output / 'CONFIG.json') == native_cfg, 'Original exact native config subset')
    for suffix, digest in inputs['native_suffixes'].items():
        require(helper.source_hash(complete, suffix) == helper.source_hash(old, suffix) == digest
                and helper.freeze_hash(freeze, suffix) == helper.freeze_hash(old_freeze, suffix) == digest, 'Native source/init/update correspondence: ' + suffix)
    for name in ('run_family.py', 'context_affinity_objective.py'):
        require(helper.source_hash(complete, LEARNING + '/' + name) == inputs['research_files'][LEARNING + '/' + name], 'Exact acquired context source: ' + name)
    rows, old_rows = {(r['seed'], r['arm']): r for r in complete['results']}, {(r['seed'], r['arm']): r for r in old['results']}
    require(len(rows) == len(complete['results']) == 21 and set(rows) == {(s, a) for s in SEEDS for a in NEW}
            and sum(len(r['fits']) for r in rows.values()) == 39
            and helper.sum_operations(complete['results']) == complete['operation_counts'], 'All39 fresh unique context fits')
    require(len(old_rows) == len(old['results']) == 21 and set(old_rows) == {(s, a) for s in SEEDS for a in OLD}
            and sum(len(r['fits']) for r in old_rows.values()) == 39
            and helper.sum_operations(old['results']) == old['operation_counts'], 'Original full parent21/39 closure')
    expected, checkpoints, records = {}, [], {}
    admitted_rows = reference_admission['checkpoint_records']
    require(len(admitted_rows) == 33 and all(set(('path', 'sha256', 'bytes', 'arm', 'seed', 'member', 'selected_step')).issubset(r) for r in admitted_rows), 'All33 frozen original checkpoint attestations')
    admitted_checkpoints = {str(scoped(r['path'], old_output)): r for r in admitted_rows}
    descriptor_paths = {str(scoped(p, old_output)) for bank in binding['banks'] for p in bank['checkpoints']}
    require(len(admitted_checkpoints) == len(admitted_rows) == len(descriptor_paths) == 33
            and set(admitted_checkpoints) == descriptor_paths, 'Exact unique frozen original checkpoint roster')

    def checkpoint(unit, arm, seed, folder, fresh):
        path = scoped(unit['selected_state'], folder)
        require(path == folder / f'{arm}_seed{seed}/member{unit["member"]}/selected.pt'
                and path.stat().st_size == unit['costs']['selected_checkpoint_bytes']
                and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000, 'Canonical selected fit path/size/horizon')
        digest = sha(path)
        if not fresh:
            row = admitted_checkpoints[str(path)]
            require(str(scoped(row['path'], old_output)) == str(path) and row['sha256'] == digest
                    and row['bytes'] == path.stat().st_size and row['arm'] == arm and row['seed'] == seed
                    and row['member'] == unit['member'] and row['selected_step'] == unit['selected_step'], 'Frozen original checkpoint path/hash/size/arm/seed/member/step')
        checkpoints.append({'path': str(path), 'sha256': digest, 'new': fresh, 'arm': arm, 'seed': seed,
            'fit_member': unit['member'], 'context_index': unit.get('context_index'),
            'selected_step': unit['selected_step'], 'bytes': path.stat().st_size})

    preparation = complete['context_preparation_once']
    require(read(output / 'CONTEXT_TARGETS.json') == preparation, 'Target metadata bound by complete closure')
    qualified_preparation = qualification['context_preparation']
    excluded = {'archive_sha256', 'archive_bytes', 'inclusive_preparation_seconds',
                'process_peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'}
    require({k: v for k, v in preparation.items() if k not in excluded}
            == {k: v for k, v in qualified_preparation.items() if k not in excluded}, 'Actual qualified canonical/order/transferred target bytes and law')
    target_path = output / 'CONTEXT_TARGETS.npz'
    require(target_path.stat().st_size == preparation['archive_bytes']
            and sha(target_path) == preparation['archive_sha256'], 'Complete target archive custody; no assumption of identical qualifier ZIP metadata')
    require(preparation['target_policy'] == OBJECTIVE_CONTRACT['target_policy']
            and preparation['objective_contract'] == complete['context_objective_contract'] == OBJECTIVE_CONTRACT
            and preparation['contexts'] == list(CONTEXT_NAMES)
            and preparation['teacher_temperature'] == preparation['student_temperature'] == .2
            and preparation['TRAIN_rows'] == 580 and preparation['TRAIN_panel'] == 'all ordered TRAIN'
            and preparation['signature_dtype'] == 'float64'
            and preparation['native_features_changed'] is False and preparation['native_predictor_graph_changed'] is False
            and preparation['positive_support'] == 'every other TRAIN classmate, identical weighted/uniform support'
            and preparation['denominator_support'] == 'every other TRAIN node, identical in every arm/context'
            and preparation['zero_signature_cosine'] == 0
            and preparation['singleton_class_policy'] == 'fail without omission or repair'
            and preparation['targets_use_VALID_or_predictions'] is False
            and preparation['native_RNG_draws'] == 0 and preparation['target_or_coefficient_grid'] is False
            and preparation['same_target_bank_shared_across_all_arms_and_seeds'] is True,
            'Actual qualified fixed continuous affinities; matched all-classmate support/full denominator')
    require(preparation['teacher_exponential_entries'] == sum(preparation['positive_counts_per_context'])
            and preparation['teacher_row_normalizations'] == 4 * 580
            and preparation['teacher_dense_row_sum_entries'] == 4 * 580 * 580
            and preparation['uniform_row_normalizations'] == 580
            and preparation['uniform_dense_target_entries'] == 580 * 580, 'Actual one-time affinity/uniform preparation work')
    unique_edges = preparation['signature_unique_edges_with_exact_self_loops']
    require(preparation['feature_aggregate_calls'] == 2 and unique_edges >= 11701
            and preparation['signature_edge_dedup_sort_entries'] == preparation['directed_original_edge_entries'] + 11701
            and preparation['feature_aggregate_edge_visits'] == 2 * unique_edges
            and preparation['feature_aggregate_coordinates'] == 2 * unique_edges * 300
            and preparation['signature_gram_calls'] == 4
            and preparation['signature_gram_entries'] == 4 * 580 * 580
            and preparation['signature_gram_multiply_accumulates'] == 4 * 580 * 580 * 300, 'Actual one-time signature/preparation work')
    require(all(math.isfinite(preparation[k]) and preparation[k] >= 0 for k in (
        'inclusive_preparation_seconds', 'process_peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'archive_bytes')), 'Finite retained one-time preparation costs')
    for seed in SEEDS:
        for arm in NEW:
            row = rows[seed, arm]
            bank = arm.startswith('shared4_')
            units, members, routes = (4 if arm in (F4C, F4A) else 1), (4 if bank or arm in (F4C, F4A) else 1), (4 if bank else 1)
            mode = CONTEXT_MODES[arm]
            require(row['serving'] == 'probability_mean' and [u['member'] for u in row['fits']] == list(range(units))
                    and read(output / f'{arm}_seed{seed}/RESULT.json') == row, 'Exact fresh bank/body record identity')
            for unit in row['fits']:
                starts = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[x, x + cfg['factor_seed_offset'], x + cfg['dropout_seed_offset']] for x in starts]
                        and unit['learning_objective'] == 'own_CE_plus_fixed_continuous_context_affinity'
                        and unit['context_mode'] == mode and unit['context_coefficient'] == .05
                        and ((arm == F4A and type(unit['context_index']) is int and unit['context_index'] == unit['member'])
                             or (arm != F4A and unit['context_index'] is None))
                        and unit['context_temperature'] == unit['context_teacher_temperature'] == .2
                        and unit['context_objective_contract'] == OBJECTIVE_CONTRACT
                        and unit['context_target_tensor_hashes'] == preparation['actual_loss_tensor_hashes']['bank' if bank else 'single']
                        and unit['source_sha256'] == complete['source_sha256']
                        and unit['preclassifier_width'] == 128
                        and unit['TRAIN_H_order'] == ('train_ids' if bank else 'native_boolean_mask')
                        and unit['context_target_archive_sha256'] == preparation['archive_sha256'], 'Unchanged starts/live H order and selected checkpoint context-mode/target binding')
                selected_scalars = unit['selected_update_training_scalars']
                require(set(selected_scalars) == {'scope', 'step', 'context_index', 'own_ce', 'auxiliary', 'weighted_auxiliary', 'objective'}
                        and selected_scalars['scope'] == 'pre_AdamW_training_scalar_at_update_whose_post_update_VALID_selects_checkpoint'
                        and selected_scalars['step'] == unit['selected_step']
                        and selected_scalars['context_index'] == unit['context_index']
                        and all(isinstance(selected_scalars[k], (int, float)) and not isinstance(selected_scalars[k], bool)
                                and math.isfinite(selected_scalars[k]) for k in ('own_ce', 'auxiliary', 'weighted_auxiliary', 'objective')),
                        'Finite selected-update scalar provenance; no selected-state objective replay')
                updates = unit['completed_updates']
                reductions = routes * (4 if mode == 'weighted_mixture' else 1)
                target_matrices = 1 if mode in ('uniform_allclassmate', 'weighted_assigned') else 4
                expected_work = {
                    'context_batched_gram_calls': updates, 'context_route_grams': routes * updates,
                    'context_normalized_coordinates': routes * 580 * 128 * updates,
                    'context_cosine_gram_entries': routes * 580 * 580 * updates,
                    'context_gram_multiply_accumulates': routes * 580 * 580 * 128 * updates,
                    'context_score_anchor_rows': routes * 580 * updates,
                    'context_target_support_pair_reductions': reductions * updates,
                    'context_logsumexp_anchor_reductions': routes * 580 * updates,
                    'context_dense_masked_logsumexp_entries': routes * 580 * 580 * updates,
                    'context_dense_weighted_score_entries': reductions * 580 * 580 * updates,
                    'context_target_rowmass_reductions': target_matrices * 580 * updates,
                    'context_target_rowmass_entries': target_matrices * 580 * 580 * updates,
                    'context_extra_native_forwards': 0, 'context_serving_calls': 0,
                    'updates': updates, 'backwards': updates, 'adam_steps': updates,
                    'train_probability_pool_loss_evaluations': updates if bank else 0,
                    'train_probability_pool_backward_evaluations': 0,
                    'train_fullgraph_native_trajectories': routes * updates,
                    'valid_selection_fullgraph_native_trajectories': routes * updates,
                    'selected_valid_fullgraph_native_trajectories': routes,
                    'native_graph_block_calls': 2 * routes * (2 * updates + 1)}
                require(unit['operation_counts'] == expected_work, 'Full actual native/Gram/pair/backward/selector work')
                checkpoint(unit, arm, seed, output, True)
                require(read(scoped(unit['selected_state'], output).parent / 'RESULT.json') == unit, 'Selected fresh unit equals complete record')
            check_group_costs(row)
            archive = output / f'{arm}_seed{seed}/selected_VALID.npz'
            expected[seed, arm] = {'path': str(archive), 'sha256': sha(archive), 'members': members, 'new': True}
            records[seed, arm] = row
    refs = {(r['seed'], r['arm']): r for r in binding['banks']}
    require(len(refs) == len(binding['banks']) == 15 and set(refs) == {(s, a) for s in SEEDS for a in REFS}, 'All15 own-only original reference descriptors')
    for seed in SEEDS:
        for arm in REFS:
            bound, row = refs[seed, arm], old_rows[seed, arm]
            require(read(old_output / f'{arm}_seed{seed}/RESULT.json') == row, 'Original bank record unchanged')
            check_group_costs(row)
            units, members, routes = (4 if 'genuine_I4' in arm else 1), (1 if arm in (O1, F1) else 4), (4 if arm == BASE else 1)
            require(bound['members'] == members and row['serving'] == 'probability_mean'
                    and [u['member'] for u in row['fits']] == list(range(units))
                    and [u['selected_state'] for u in row['fits']] == bound['checkpoints'], 'All33 selected original fit identities')
            for unit in row['fits']:
                starts = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[x, x + cfg['factor_seed_offset'], x + cfg['dropout_seed_offset']] for x in starts], 'Original factor/dropout constructor starts')
                checkpoint(unit, arm, seed, old_output, False)
                require(read(scoped(unit['selected_state'], old_output).parent / 'RESULT.json') == unit, 'Original selected unit unchanged')
            archive = scoped(bound['archive'], old_output)
            require(archive == old_output / f'{arm}_seed{seed}/selected_VALID.npz' and sha(archive) == bound['archive_sha256'], 'Pinned unchanged original raw archive')
            expected[seed, arm] = {'path': str(archive), 'sha256': bound['archive_sha256'], 'members': members, 'new': False}
            records[seed, arm] = row
    require(len(expected) == len(records) == 36 and len(checkpoints) == len({r['path'] for r in checkpoints}) == 72, 'All21fresh/15old banks and39fresh/33old selected fits before arrays')
    reuse, reused_archives, prior_custody = admit_reuse(root, here, freeze, inputs, helper, expected, reference_admission)
    context = {'cfg': cfg, 'freeze': freeze, 'admission': admission, 'reference_admission': reference_admission,
        'qualification': qualification, 'owner_start': start, 'owner': owner, 'old_owner': old_owner,
        'binding': binding, 'complete': complete, 'old_complete': old,
        'new_config_sha256': sha(here / 'CONFIG.json'), 'original_config_sha256': binding['config_sha256'],
        'native_config_subset': native_cfg, 'auxiliary_only_config_fields': {'context_teacher_temperature': .2},
        'complete_sha256': complete_sha, 'freeze_sha256': sha(here / 'FREEZE.json'), 'owner_end_sha256': sha(here / 'OWNER_END.json'),
        'target_archive': {'path': str(target_path), 'sha256': preparation['archive_sha256'], 'bytes': preparation['archive_bytes']},
        'calibration_reuse_admission': reuse, 'reused_OOF_archives': reused_archives, 'prior_calibration_custody': prior_custody}
    return context, expected, checkpoints, records



def payloads(np, expected):
    arrays, ids, labels = {}, None, None
    for arm in ARMS:
        for seed in SEEDS:
            item = expected[seed, arm]
            require(sha(item['path']) == item['sha256'], 'Admitted archive changed')
            with np.load(item['path'], allow_pickle=False) as archive:
                require(set(archive.files) == BASE_KEYS, 'Exact native six-key selected schema')
                a = {k: archive[k].copy() for k in archive.files}
            require(sha(item['path']) == item['sha256'], 'Archive changed during read')
            require(a['ids'].shape == a['y'].shape == (5274,) and a['ids'].dtype == a['y'].dtype == np.int64, 'Ordered VALID roles')
            if ids is None:
                ids, labels = a['ids'], a['y']
                require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10))
                        and {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in [('ids', ids), ('labels', labels)]} == VALID_HASHES,
                        'Frozen full VALID identity')
            require(np.array_equal(ids, a['ids']) and np.array_equal(labels, a['y']), 'Identical rows/labels for all36 banks')
            raw, pool, member, pooled = [a[k] for k in ('raw_logits', 'probability_mean', 'member_errors', 'pooled_errors')]
            require(raw.dtype == pool.dtype == np.float32 and raw.shape == (item['members'], 5274, 10) and pool.shape == (5274, 10)
                    and np.isfinite(raw).all() and np.isfinite(pool).all() and (pool >= 0).all() and (pool <= 1).all()
                    and member.dtype == pooled.dtype == np.bool_ and member.shape == (item['members'], 5274) and pooled.shape == (5274,)
                    and np.array_equal(pool.argmax(1) != labels, pooled), 'Archived float32 prediction/error authority')
            arrays[seed, arm] = a
    require(len(arrays) == 36, 'All36 schemas before scoring/calibration')
    return arrays, ids, labels



def score(helper, np, archive):
    value = helper.score(archive)
    truth = np.eye(10, dtype=np.float64)[archive['y']]
    raw = archive['raw_logits'].astype(np.float64)
    log_probs = raw - helper.logsumexp(raw, -1)[..., None]
    member_brier = ((np.exp(log_probs) - truth[None]) ** 2).sum(-1)
    pool_brier = ((archive['probability_mean'].astype(np.float64) - truth) ** 2).sum(-1)
    quality = value['quality']
    quality.update(pooled_brier=float(pool_brier.mean()), member_brier=member_brier.mean(1).tolist(),
                   mean_member_brier=float(member_brier.mean()), worst_member_brier=float(member_brier.mean(1).max()),
                   worst_member_nll=max(quality['member_nll']))
    for row in value['classes']:
        mask = archive['y'] == row['class']
        row.update(pooled_brier=float(pool_brier[mask].mean()), member_brier=member_brier[:, mask].mean(1).tolist(),
                   mean_member_brier=float(member_brier[:, mask].mean()),
                   worst_member_brier=float(member_brier[:, mask].mean(1).max()), worst_member_nll=max(row['member_nll']))
    return value


def contrast(helper, values, candidate, reference, labels, scope='full_learning_procedures'):
    value = helper.contrast(values, candidate, reference, labels)
    for metric in ('pooled_brier', 'mean_member_brier', 'worst_member_brier', 'worst_member_nll'):
        value['quality_deltas'][metric] = helper.paired(values[s, candidate]['quality'][metric] - values[s, reference]['quality'][metric] for s in SEEDS)
    for row in value['classes']:
        k = row['class']
        for metric in ('pooled_brier', 'mean_member_brier', 'worst_member_brier', 'worst_member_nll'):
            row[metric] = helper.paired(values[s, candidate]['classes'][k][metric] - values[s, reference]['classes'][k][metric] for s in SEEDS)
    if value['members'] is not None:
        for row in value['members']:
            m = row['member']
            row['brier'] = helper.paired(values[s, candidate]['quality']['member_brier'][m] - values[s, reference]['quality']['member_brier'][m] for s in SEEDS)
    value['comparison_scope'] = scope
    value['error_categories_are_descriptive_not_causal'] = True
    value['scope_explanation'] = ('The same native selected checkpoint/member bank is fixed; only the equal-policy OOF temperature readout changes. No new member alternative is acquired.'
                                  if scope == 'same_selected_state_calibration'
                                  else 'Different arms are separately fitted, stopped and selected. Repairs and coverage changes compare full procedures, not an isolated same-state gradient or serving mechanism.')
    value['coverage_category_note'] = 'Existing-member-alternative-served means both bank coverages are true; cross-procedure counts do not certify the same correct route/vector or exclusive pooling causality.'
    return value


def score_raw(helper, np, arrays, records, context, labels):
    raw, diagnostics = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            a, row = arrays[seed, arm], records[seed, arm]
            value = score(helper, np, a)
            raw[seed, arm] = value
            require(value['counts']['pooled_correct'] == round(row['valid']['accuracy'] * 5274)
                    and value['counts']['coverage'] == row['valid']['correct_alternative_count']
                    and value['counts']['lost_correct_alternatives'] == row['valid']['coverage_lost_in_pooling']
                    and value['counts']['member_correct'] == [round(v * 5274) for v in row['valid']['member_accuracy']],
                    'Archived integer decisions agree with complete selected records')
            raw64 = a['raw_logits'].astype(np.float64)
            probability64 = np.exp(raw64 - helper.logsumexp(raw64, -1)[..., None])
            diagnostics[seed, arm] = {
                'current_bank_common_strict_rival': strict_counts(np, strict_mask(np, a['raw_logits'], labels), value['P'], labels),
                'fixed_original_strict_rival_cohorts': {ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS},
                'FP64_raw_pool_argmax_disagreements_with_archived_errors': int(((probability64.mean(0).argmax(1) != labels) != a['pooled_errors']).sum()),
                'raw_logit_argmax_disagreements_with_archived_member_errors': int(((a['raw_logits'].argmax(-1) != labels) != a['member_errors']).sum()),
                'stable_NLL_minus_original_reported': value['quality']['pooled_nll'] - row['valid']['nll'],
                'scope': 'Selected full-procedure output; original raw float32 error flags remain integer authority.'}
    saved = {(r['seed'], r['arm']): r['versus'] for r in context['complete']['comparisons']}
    require(len(saved) == len(context['complete']['comparisons']) == 21 and set(saved) == {(s, a) for s in SEEDS for a in NEW},
            'Complete fresh saved error roster')
    for (seed, arm), versus in saved.items():
        require(set(versus) == set(NEW), 'All fresh comparisons retained')
        for ref, counts in versus.items():
            a, b = raw[seed, arm]['P'], raw[seed, ref]['P']
            require(counts == {'repairs': int((~b & a).sum()), 'harms': int((b & ~a).sum())}, 'Raw saved repair/harm authority')
    return raw, diagnostics




def calibrate(np, torch, operators, arrays, ids, labels, context):
    """Reuse exact75 admitted endpoints; all105 new attempts terminal before scores."""
    start = time.perf_counter()
    generator = torch.Generator(device='cpu').manual_seed(11709)
    permutation = torch.randperm(len(labels), generator=generator)
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    fold_array = folds.numpy()
    fold_meta = {'seed': 11709, 'folds': 5, 'counts': [(folds == k).sum().item() for k in range(5)],
        'assignment_sha256': hashlib.sha256(fold_array.tobytes()).hexdigest(),
        'ordered_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'scope': 'Label-free fixed folds after native selection; encountered development, not whole-pipeline confirmation.'}
    require(fold_meta['assignment_sha256'] == context['calibration_reuse_admission']['fold_assignment_sha256'], 'Exact original fold prediction binding')
    payloads, record_index, binaries = {}, {}, []
    reuse_index = {(r['seed'], r['arm']): r for r in context['calibration_reuse_admission']['records']}
    for arm in REFS:
        for seed in SEEDS:
            row = reuse_index[seed, arm]
            item = next(r for r in context['reused_OOF_archives'] if r['seed'] == seed and r['arm'] == arm)
            require(sha(item['path']) == item['sha256'], 'Admitted original OOF unchanged')
            with np.load(item['path'], allow_pickle=False) as archive:
                require(set(archive.files) == {'ids', 'folds', 'member_log_probability', 'pool_log_probability'}, 'Exact original OOF schema')
                reused = {k: archive[k].copy() for k in archive.files}
            require(sha(item['path']) == item['sha256'], 'Original OOF unchanged during read')
            members = arrays[seed, arm]['raw_logits'].shape[0]
            require(reused['ids'].dtype == reused['folds'].dtype == np.int64
                    and np.array_equal(reused['ids'], ids) and np.array_equal(reused['folds'], fold_array)
                    and reused['member_log_probability'].dtype == reused['pool_log_probability'].dtype == np.float64
                    and reused['member_log_probability'].shape == (members, 5274, 10)
                    and reused['pool_log_probability'].shape == (5274, 10)
                    and np.isfinite(reused['member_log_probability']).all() and np.isfinite(reused['pool_log_probability']).all(),
                    'Exact admitted original roles/folds/member/pool prediction schema')
            payloads[seed, arm] = reused
            record = copy.deepcopy(row['endpoint_record'])
            record['endpoint_origin'] = 'reused_exact_admitted_original_no_fit_replay'
            record['new_scalar_fit_attempts'] = 0
            record['reused_archive'] = item
            record_index[seed, arm] = record
    y, attempted = torch.from_numpy(labels), 0
    for arm in NEW:
        for seed in SEEDS:
            a = arrays[seed, arm]
            log_p = torch.from_numpy(a['raw_logits']).to(torch.float64).log_softmax(-1)
            probability = log_p.exp()
            empty_context = torch.empty((len(log_p), len(labels), 0), dtype=torch.float64)
            log_pool = torch.full((5274, 10), float('nan'), dtype=torch.float64)
            log_members = torch.full(log_p.shape, float('nan'), dtype=torch.float64)
            fits = []
            for fold in range(5):
                fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                attempted += 1
                before, returned_record = time.perf_counter(), None
                try:
                    held, record = operators.fit_fold(torch, 'temperature_global', log_p, probability, empty_context, y, fit_ids, held_ids)
                    returned_record = record
                    log_T = torch.tensor(record['state']['log_T'], dtype=torch.float64)
                    temperature = log_T.exp()
                    require(log_T.shape == (1,) and record['updates'] == 500 and record['parameters'] == 1
                            and record['fit_nodes'] == len(fit_ids) and record['held_nodes'] == len(held_ids)
                            and record['selected_endpoint'] == 'fixed_final_update_no_heldout_selector'
                            and torch.isfinite(temperature).all().item() and (temperature > 0).all().item(), 'One fixed final positive global temperature')
                    scaled = (log_p[:, held_ids] / temperature.reshape(-1, 1, 1)).log_softmax(-1)
                    require(torch.isfinite(scaled).all().item() and torch.isfinite(held).all().item(), 'Finite fixed new endpoint')
                    log_pool[held_ids], log_members[:, held_ids] = held, scaled
                    record.update(status='finite_fixed_endpoint', fold=fold, temperature=temperature.item(), seconds=time.perf_counter() - before)
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    record = {'status': 'failed_retained', 'fold': fold, 'error': str(error), 'fixed_budget_updates': 500,
                        'actual_updates_before_failure': returned_record['updates'] if returned_record is not None else None,
                        'returned_endpoint_record': retained_json(returned_record), 'seconds': time.perf_counter() - before}
                fits.append(record)
            finite = torch.isfinite(log_pool).all().item() and torch.isfinite(log_members).all().item()
            record = {'arm': arm, 'seed': seed, 'status': 'finite_complete_five_fold' if finite else 'failed_retained',
                'fits': fits, 'fit_seconds_sum': sum(f['seconds'] for f in fits), 'attempted_scalar_fits': 5,
                'endpoint_origin': 'new_fixed_endpoint', 'new_scalar_fit_attempts': 5}
            value = {'ids': ids, 'folds': fold_array, 'member_log_probability': log_members.numpy(), 'pool_log_probability': log_pool.numpy()}
            payloads[seed, arm] = value
            stream = io.BytesIO()
            np.savez_compressed(stream, **value)
            name, data = f'CALIBRATED_{arm}_seed{seed}_OOF.npz', stream.getvalue()
            record['OOF_archive'] = {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            binaries.append((name, data))
            record_index[seed, arm] = record
    records = [record_index[seed, arm] for arm in ARMS for seed in SEEDS]
    require(attempted == 105 and len(records) == len(payloads) == 36 and len(binaries) == 21
            and all(len(r['fits']) == 5 for r in records)
            and sum(r['attempted_scalar_fits'] for r in records) == 180
            and sum(r['new_scalar_fit_attempts'] for r in records) == 105, 'All180 terminal endpoints, exactly105 newly attempted')
    new_records = [r for r in records if r['arm'] in NEW]
    successful = sum(f['status'] == 'finite_fixed_endpoint' for r in new_records for f in r['fits'])
    reused_seconds = sum(r['prior_fit_seconds'] for r in reuse_index.values())
    cost = {'reused_scalar_endpoints': 75, 'reused_successful_scalar_updates_already_paid': 37500,
        'reused_prior_fit_seconds_sum_already_paid': reused_seconds, 'reused_scalar_fit_replays': 0,
        'new_attempted_scalar_fits': attempted, 'new_requested_updates_per_fit': 500,
        'new_requested_fullbatch_scalar_updates': 52500, 'new_successful_scalar_fits': successful,
        'new_failed_scalar_fits': 105 - successful, 'new_known_successful_fullbatch_scalar_updates': successful * 500,
        'new_known_completed_fullbatch_scalar_updates_lower_bound': successful * 500 + sum(f['actual_updates_before_failure'] or 0 for r in new_records for f in r['fits'] if f['status'] == 'failed_retained'),
        'new_failed_actual_update_counts_known': all(f.get('actual_updates_before_failure') is not None for r in new_records for f in r['fits'] if f['status'] == 'failed_retained'),
        'new_fit_seconds_sum': sum(r['fit_seconds_sum'] for r in new_records),
        'current_readout_wall_seconds_including_reuse_IO': time.perf_counter() - start,
        'total_comparative_endpoint_records': 180, 'total_comparative_requested_scalar_updates': 90000,
        'native_models_or_forwards_added': 0, 'no_retry_or_final_refit': True,
        'scope': '75 original endpoint costs are retained separately and not charged as new work.105 new attempts only; failure does not imply500 updates. Unknown failed work stays unknown. Fit seconds are nested in current wall time.'}
    return payloads, records, binaries, fold_meta, cost



def calibrated_scores(helper, np, payloads, records, arrays, labels):
    """Scoring begins only after every new and reused endpoint is terminal."""
    scores, index = {}, {(r['seed'], r['arm']): r for r in records}
    for arm in ARMS:
        for seed in SEEDS:
            record, a, p = index[seed, arm], arrays[seed, arm], payloads[seed, arm]
            if record['status'] != 'finite_complete_five_fold':
                scores[seed, arm] = None
                continue
            pool, members = p['pool_log_probability'], p['member_log_probability']
            require(np.allclose(helper.logsumexp(members, -1), 0, rtol=0, atol=1e-12)
                    and np.allclose(helper.logsumexp(pool, -1), 0, rtol=0, atol=1e-12)
                    and np.allclose(pool, helper.logsumexp(members, 0) - math.log(len(members)), rtol=1e-12, atol=1e-12), 'Complete normalized equal-policy OOF predictions')
            compatible = {'raw_logits': members, 'y': labels, 'member_errors': a['member_errors'],
                'pooled_errors': pool.argmax(1) != labels, 'probability_mean': np.exp(pool)}
            value = score(helper, np, compatible)
            disagreements = int(((members.argmax(-1) != labels) != a['member_errors']).sum())
            if arm in REFS:
                require(value['counts'] == record['counts'], 'Original calibrated integer scores unchanged')
                for key, stored in record['quality'].items():
                    require(np.allclose(value['quality'][key], stored, rtol=1e-12, atol=1e-12), 'Original admitted calibrated score/prediction binding: ' + key)
                require(disagreements == record['FP64_calibrated_member_argmax_disagreements_with_raw_archived_errors'], 'Original member precision diagnostics unchanged')
                # Preserve original per-seed scores exactly, including their float values.
                value.update(quality=copy.deepcopy(record['quality']), counts=copy.deepcopy(record['counts']), classes=copy.deepcopy(record['classes']))
            else:
                record.update(quality=value['quality'], counts=value['counts'], classes=value['classes'],
                    FP64_calibrated_member_argmax_disagreements_with_raw_archived_errors=disagreements,
                    fixed_original_strict_rival_cohorts={ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS},
                    member_top_class_authority='Raw archived member flags remain fixed. Positive temperature acquires no alternatives; precision/tie changes are separately diagnosed.')
            record['fixed_complete_roster_strict_rival_cohorts'] = {ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in ARMS}
            scores[seed, arm] = value
    return scores


def aggregate(helper, values, rows):
    metrics = ('pooled_accuracy_pct', 'pooled_nll', 'pooled_brier', 'mean_member_accuracy_pct',
               'mean_member_nll', 'worst_member_accuracy_pct', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier')
    return {'mean_quality': {k: statistics.mean(v['quality'][k] for v in values) for k in metrics},
            'summed_three_seed_readout_counts': {k: sum(v['counts'][k] for v in values) for k in values[0]['counts'] if k != 'member_correct'},
            'acquisition_costs': helper.cost_summary(rows),
            'count_scope': 'Three dependent readouts of the same5274 development nodes; not independent node samples.'}



def native_cost_summary(helper, rows):
    value = helper.cost_summary(rows)
    restoration_recorded = all('restoration_seconds' in r['costs'] for r in rows)
    value.update(restoration_seconds_sum=sum(r['costs']['restoration_seconds'] for r in rows) if restoration_recorded else None,
        restoration_seconds_recorded_for_all_groups=restoration_recorded,
        original_unrecorded_restoration_cost_is_not_zero=True,
        selected_forward_and_metrics_seconds_sum=sum(r['costs']['selected_forward_and_metrics_seconds'] for r in rows),
        selected_pool_seconds_sum=sum(r['costs']['selected_pool_seconds'] for r in rows),
        parameters_sum=sum(r['costs']['parameters'] for r in rows),
        inference_parameter_bytes_sum=sum(r['costs']['inference_parameter_bytes'] for r in rows),
        selected_checkpoint_bytes_sum=sum(r['costs']['selected_checkpoint_bytes'] for r in rows),
        process_lifetime_peak_RSS_bytes_max=max(r['costs']['process_peak_rss_bytes'] for r in rows),
        cuda_peak_allocated_bytes_max=max(r['costs']['cuda_peak_allocated_bytes'] for r in rows),
        cuda_peak_reserved_bytes_max=max(r['costs']['cuda_peak_reserved_bytes'] for r in rows),
        scope='Observed costs; acquisition, restoration, selected serving and one-time support preparation have separate scopes. Peak memory is a maximum, not a sum. Process RSS is lifetime peak.')
    return value



def transport_error_partitions(helper, np, values, labels, readout):
    controls = (BASE, PRIVATE, MIX, UNIFORM, F4C, F4A)
    if any(values[s, arm] is None for s in SEEDS for arm in controls):
        return {'readout': readout, 'control_order': list(controls), 'failed_retained': True,
                'scope': 'Incomplete calibrated required references retain endpoint failures; no subset interpretation.'}
    rows = []
    for seed in SEEDS:
        decisions = [values[seed, arm]['P'] for arm in controls]
        code = sum((decision.astype(np.uint8) << bit) for bit, decision in enumerate(decisions))
        patterns = []
        for value in range(64):
            mask = code == value
            patterns.append({'bits': {arm: bool(value & (1 << bit)) for bit, arm in enumerate(controls)},
                'nodes': int(mask.sum()), 'classes': [{'class': k, 'nodes': int((mask & (labels == k)).sum())} for k in range(10)]})
        base, private, mixture, uniform, independent_mixture, independent_assigned = decisions
        repaired = ~base & private
        other_correct = mixture & uniform & independent_mixture & independent_assigned
        unique_repair = repaired & ~mixture & ~uniform & ~independent_mixture & ~independent_assigned
        unanimous_harm = base & ~private & other_correct
        rows.append({'seed': seed, 'joint_correctness_patterns': patterns,
            'private_repairs_of_BASE': int(repaired.sum()),
            'private_repairs_not_served_by_weighted_mixture': int((repaired & ~mixture).sum()),
            'private_repairs_not_served_by_uniform_allclassmate': int((repaired & ~uniform).sum()),
            'private_repairs_not_served_by_independent_mixture': int((repaired & ~independent_mixture).sum()),
            'private_repairs_not_served_by_independent_assigned': int((repaired & ~independent_assigned).sum()),
            'private_repairs_not_served_by_any_control': int(unique_repair.sum()),
            'private_harms_where_BASE_and_all_four_controls_correct': int(unanimous_harm.sum()),
            'classes': [{'class': k, 'private_repairs_of_BASE': int((repaired & (labels == k)).sum()),
                'private_repairs_not_served_by_any_control': int((unique_repair & (labels == k)).sum()),
                'private_harms_where_BASE_and_all_four_controls_correct': int((unanimous_harm & (labels == k)).sum())} for k in range(10)]})
    return {'readout': readout, 'control_order': list(controls), 'per_seed': rows,
        'scope': 'Full separately fitted/stopped/selected procedures. Complete64-pattern intersections include all five required references and do not isolate a same-state cause or replace five fixed raw/NLL gates.'}



def analyze(helper, np, raw, calibrated, labels, records):
    pairs = [(a, b) for a in NEW for b in ARMS if a != b]
    raw_contrasts = {a + '_minus_' + b: contrast(helper, raw, a, b, labels) for a, b in pairs}
    calibrated_contrasts = {a + '_minus_' + b: contrast(helper, calibrated, a, b, labels)
        if all(calibrated[s, a] is not None and calibrated[s, b] is not None for s in SEEDS)
        else {'failed_retained': True, 'candidate': a, 'reference': b} for a, b in pairs}
    for a, b in pairs:
        fields = set(records[SEEDS[0], a]['costs']) & set(records[SEEDS[0], b]['costs'])
        cost_deltas = {key: helper.paired(records[s, a]['costs'][key] - records[s, b]['costs'][key] for s in SEEDS) for key in sorted(fields)}
        for contrasts in (raw_contrasts, calibrated_contrasts):
            contrasts[a + '_minus_' + b]['native_cost_deltas'] = cost_deltas
            contrasts[a + '_minus_' + b]['native_cost_fields_unavailable_for_delta'] = sorted(set(records[SEEDS[0], a]['costs']) ^ set(records[SEEDS[0], b]['costs']))
            contrasts[a + '_minus_' + b]['native_cost_scope'] = 'Observed full-procedure costs; original work already paid. One-time shared support preparation is reported once outside arm-specific fit/restore/serving deltas.'
    effects = {}
    for arm in ARMS:
        if all(calibrated[s, arm] is not None for s in SEEDS):
            values = {(s, 'raw'): raw[s, arm] for s in SEEDS}
            values.update({(s, 'calibrated'): calibrated[s, arm] for s in SEEDS})
            effects[arm] = contrast(helper, values, 'calibrated', 'raw', labels, 'same_selected_state_calibration')
        else:
            effects[arm] = {'failed_retained': True}
    required = {BASE: .2, F4C: .2, F4A: .2, MIX: .1, UNIFORM: .1}
    accuracy, protection = {}, {}
    for ref, threshold in required.items():
        stat = raw_contrasts[PRIVATE + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        flags = {'required_mean_gain_pp': threshold, 'paired_raw_accuracy_pp': stat,
            'mean_at_least_threshold': stat['mean'] >= threshold,
            'all_seeds_nonnegative': stat['nonnegative_seed_count'] == 3,
            'at_least_two_positive': stat['positive_seed_count'] >= 2}
        flags['pass'] = all(flags[k] for k in ('mean_at_least_threshold', 'all_seeds_nonnegative', 'at_least_two_positive'))
        accuracy[ref] = flags
        value = calibrated_contrasts[PRIVATE + '_minus_' + ref]
        if value.get('failed_retained'):
            protection[ref] = {'finite': False, 'pass': False, 'failure_retained': True}
        else:
            nll, acc = [value['quality_deltas'][k] for k in ('pooled_nll', 'pooled_accuracy_pct')]
            flags = {'finite': True, 'paired_calibrated_NLL_deterioration': nll, 'paired_calibrated_accuracy_pp': acc,
                'mean_NLL_deterioration_at_most_0_02': nll['mean'] <= .02,
                'each_seed_NLL_deterioration_at_most_0_05': nll['max'] <= .05,
                'calibrated_accuracy_reported_without_extra_veto': True}
            flags['pass'] = flags['mean_NLL_deterioration_at_most_0_02'] and flags['each_seed_NLL_deterioration_at_most_0_05']
            protection[ref] = flags
    raw_pass, protected = all(v['pass'] for v in accuracy.values()), all(v['pass'] for v in protection.values())
    capable = {}
    for ref in (O1C, F1C, F4C, F4A, O1, F1, O4, F4):
        stat = raw_contrasts[PRIVATE + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        value = calibrated_contrasts[PRIVATE + '_minus_' + ref]
        capable[ref] = {'paired_raw_pooled_accuracy_pp': stat,
            'adverse_stronger_reference_in_mean_raw_accuracy': stat['mean'] < 0,
            'any_negative_raw_seed': stat['nonnegative_seed_count'] < 3,
            'calibrated_quality_deltas': value.get('quality_deltas'),
            'calibration_failure_retained': bool(value.get('failed_retained')),
            'role': 'capable reference; adverse stronger reference limits a broad superiority statement; no extra private attribution gate'}
    generic = {arm: {'raw_quality_deltas_vs_BASE': raw_contrasts[arm + '_minus_' + BASE]['quality_deltas'],
        'calibrated_quality_deltas_vs_BASE': calibrated_contrasts[arm + '_minus_' + BASE].get('quality_deltas'),
        'calibration_failure_retained': bool(calibrated_contrasts[arm + '_minus_' + BASE].get('failed_retained')),
        'scope': 'Training target/support control contrast; descriptive. This control cannot replace the declared private candidate gate.'} for arm in (MIX, UNIFORM)}
    policy = {'candidate': PRIVATE, 'raw_accuracy_criteria': accuracy, 'raw_private_attribution_screen_pass': raw_pass,
        'equal_policy_calibrated_confidence_criteria': protection, 'calibrated_confidence_protection_pass': protected,
        'joint_accuracy_confidence_development_clue': raw_pass and protected,
        'accuracy_only_clue_with_confidence_tradeoff': raw_pass and not protected,
        'capable_reference_diagnostics': capable,
        'broad_superiority_statement_blocked_by_stronger_equipped_single': any(capable[a]['adverse_stronger_reference_in_mean_raw_accuracy'] for a in (O1C, F1C)),
        'broad_superiority_statement_blocked_by_stronger_capable_reference': any(v['adverse_stronger_reference_in_mean_raw_accuracy'] for v in capable.values()),
        'target_support_control_findings': generic, 'member_quality_is_diagnostic_not_veto': True,
        'calibrated_accuracy_reported_without_extra_veto': True, 'raw_NLL_and_Brier_tradeoffs_preserved': True,
        'historical_A_B_AB_not_candidate_or_K': True, 'automatic_confirmation_launch': False,
        'scope': 'One native SAGE backbone on encountered WikiCS development. Private attribution has five fixed references. Capable singles and original I4s remain visible; control gains do not replace the private gate. No generic novelty, unused confirmation, universal transfer or manuscript verdict.'}
    results = {'raw_full_procedure_contrasts': raw_contrasts, 'calibrated_full_procedure_contrasts': calibrated_contrasts,
        'calibration_minus_raw_same_selected_state': effects,
        'raw_transport_error_partitions': transport_error_partitions(helper, np, raw, labels, 'raw'),
        'calibrated_transport_error_partitions': transport_error_partitions(helper, np, calibrated, labels, 'OOF_calibrated')}
    return results, policy




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0', 'Literal authorized scientific hostname')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID], 'Sole authorized GPU UUID')
    root, report_path = args.research_root.resolve(), args.report.resolve()
    require(root == PHASE and Path(__file__).resolve().is_relative_to(root), 'Authorized research root')
    require(report_path.is_relative_to(root) and not report_path.exists(), 'Fresh in-scope report path')
    source = read(Path(__file__).with_name('SOURCE.json'))
    for name, digest in source['files'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Sealed reader source changed')
    inputs = read(Path(__file__).with_name('INPUT_BINDINGS.json'))
    helper = load_source('SAGE_immutable_count_math', root / MATH, MATH_SHA)
    context, expected, checkpoints, records = preflight(root, helper, inputs)
    # All36 banks/72 fit identities/75 original endpoint records close before NumPy or arrays.
    import numpy as np
    helper.np = np
    target_diagnostics = target_arrays(np, context)
    arrays, ids, labels = payloads(np, expected)
    import torch
    torch.set_num_threads(2)
    operators = load_source('SAGE_fixed_temperature', root / CALIBRATOR, inputs['research_files'][CALIBRATOR])
    calibrated_payloads, calibration_records, binaries, folds, calibration_costs = calibrate(np, torch, operators, arrays, ids, labels, context)
    # No score, contrast, gate or error partition before all180 endpoints are terminal.
    raw, diagnostics = score_raw(helper, np, arrays, records, context, labels)
    for arm in ARMS:
        for seed in SEEDS:
            diagnostics[seed, arm]['fixed_complete_roster_strict_rival_cohorts'] = {
                ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), raw[seed, arm]['P'], labels) for ref in ARMS}
    calibrated = calibrated_scores(helper, np, calibrated_payloads, calibration_records, arrays, labels)
    contrasts, policy = analyze(helper, np, raw, calibrated, labels, records)
    aggregates = {'raw': {a: aggregate(helper, [raw[s, a] for s in SEEDS], [records[s, a] for s in SEEDS]) for a in ARMS},
        'calibrated': {a: aggregate(helper, [calibrated[s, a] for s in SEEDS], [records[s, a] for s in SEEDS])
            if all(calibrated[s, a] is not None for s in SEEDS) else {'failed_retained': True} for a in ARMS}}
    for values in aggregates.values():
        for arm, value in values.items():
            if not value.get('failed_retained'):
                value['acquisition_costs'] = native_cost_summary(helper, [records[s, arm] for s in SEEDS])
    groups = [{'arm': arm, 'seed': seed, 'raw': {k: raw[seed, arm][k] for k in ('quality', 'counts', 'classes')},
        'diagnostics': diagnostics[seed, arm], 'original_group_record': records[seed, arm],
        'acquisition_origin': 'new_continuous_context_affinity' if arm in NEW else 'unchanged_original_reference'} for arm in ARMS for seed in SEEDS]
    context_keys = [key for key in context['complete']['operation_counts'] if key.startswith('context_')]
    context_work_by_arm = {arm: {key: sum(u['operation_counts'][key] for s in SEEDS for u in records[s, arm]['fits'])
        for key in context_keys} for arm in NEW}
    costs = {'new39_native_optimizer_fits': native_cost_summary(helper, context['complete']['results']),
        'reused33_selected_reference_fits_already_paid': native_cost_summary(helper, [records[s, a] for s in SEEDS for a in REFS]),
        'original39_parent_context_not_additive': native_cost_summary(helper, context['old_complete']['results']),
        'one_shared_context_target_preparation': context['complete']['context_preparation_once'],
        'context_operation_counts': {key: context['complete']['operation_counts'][key] for key in context_keys},
        'context_operation_counts_by_arm': context_work_by_arm,
        'calibration_scalar_work': calibration_costs, 'new_owner': context['owner'], 'original_owner': context['old_owner'],
        'scope': 'Target/signature construction, affinity normalization, archive and transfer are charged once and nested in owner wall time. Native acquisition, recorded restoration, selected forward/metrics/pool and scalar work remain separate. Original unrecorded restoration is unavailable, never zero. Original native/scalar work remains already paid; observed timings are not isolated benchmarks.'}
    limits = [
        'All21fresh/15old banks and39fresh/33old selected fit records close before arrays. Exact75 admitted original OOF endpoints are not fit or temperature replayed;105new attempts are terminal before any scores/contrasts/gates.',
        'The target archive is source/role/order bound and checked against actual qualified canonical/transfer hashes. Signature propagation, teacher cosine/affinity construction, native outputs and auxiliary gradients are not replayed.',
        'Cross-arm repairs compare complete separately fitted/stopped/selected procedures. Weighted-mixture/uniform-control intersections describe specificity, not an isolated same-state cause.',
        'Calibration-minus-raw fixes each selected native bank. Positive temperature acquires no member alternatives; raw float32 member flags and coverage remain authority, with precision/tie discrepancies reported.',
        'Private raw gains require0.2pp versus BASE/full-mixture genuineI4/same-assigned genuineI4 and0.1pp versus common weighted mixture/uniform all-classmate, all3seeds nonnegative and at least2positive for every reference. Same-reference calibrated NLL harm is bounded by0.02mean/0.05each seed; calibrated accuracy adds no veto.',
        'Both equipped singles and original ordinary/factorized singles/I4s remain capable references; a stronger comparator limits broad superiority language. Member weakness alone is not a veto.',
        'Mixture-equipped singles, the mixture-equipped I4 and common shared routes receive the full four-weighted-objective mixture; each assigned I4 body m instead receives Q_m, matching shared route m. All use the same all-nonself denominator. The source and actual qualifier establish learning/selector/stream custody, not utility or conditional-ratio realizability.',
        'OOF calibration follows VALID native selection on encountered development. Three seeds on the same graph/nodes are not independent graph or node replications. Future confirmation requires unused whole-pipeline evidence.',
        'Historical original references are exact admitted source/config/role/seed/selector outputs. Their prior endpoint bytes and scores remain fixed; historical and newly qualified runtime metadata are retained rather than presumed identical.',
        'MLCL, weighted InfoNCE, y-Aware/X-CLR, graph-filter contexts and shared/private ensembles are credited ingredients. No latent-mode recovery, collapse-prevention guarantee, observed collapse diagnosis, primitive novelty or acceptance verdict is added.',
        'One native SAGE backbone suffices for the declared comparison. No TEST, model/checkpoint deserialization, native loader, native forward or optional target/math parity occurs in this reader.']
    report = {'complete': True, 'backbone': 'SAGE', 'new_banks': 21, 'new_native_fits': 39,
        'new_native_trajectories_per_complete_roster_forward': 66, 'reused_reference_banks': 15, 'reused_reference_fits': 33,
        'joined_banks': 36, 'joined_selected_fit_records': 72, 'seeds': SEEDS, 'roster': ARMS, 'TEST_access': False,
        'calibration_reused_endpoints': 75, 'calibration_new_attempts': 105, 'calibration_complete_terminal_endpoints': 180,
        'frozen_policy': policy, 'aggregates': aggregates, 'groups_detail': groups, 'contrasts': contrasts,
        'calibration_folds': folds, 'calibration_records': calibration_records, 'costs': costs,
        'target_support_diagnostics': target_diagnostics, 'source_custody': context,
        'selected_archive_custody': [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS],
        'checkpoint_custody': checkpoints, 'reused_calibrated_archive_custody': context['reused_OOF_archives'],
        'ordered_VALID_hashes': VALID_HASHES, 'analysis_source_sha256': sha(Path(__file__)),
        'immutable_count_math_sha256': MATH_SHA, 'reviewed_kernel_reader_parent_sha256': inputs['kernel_reader_parent_sha256'],
        'reviewed_context_reader_parent_sha256': inputs['context_reader_parent_sha256'],
        'new_config_sha256': context['new_config_sha256'], 'original_config_sha256': context['original_config_sha256'],
        'native_config_subset_exclusion': ['context_teacher_temperature'],
        'reviewed_original_calibration_reader_sha256': inputs['research_files'][PRIOR_READER],
        'numerical_reader_runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_threads': torch.get_num_threads(), 'calibration_device': 'cpu', 'calibration_dtype': 'float64'},
        'input_bindings': inputs,
        'authority': 'Raw integer decisions use archived native float32 error flags. Stable float64 NLL/Brier and calibrated OOF decisions remain explicit. Original admitted calibrated per-seed scores are retained exactly.',
        'interpretation_limits': limits}
    documents = [('SAGE_POLICY_SUMMARY.json', {'complete': True, 'backbone': 'SAGE', 'new_banks': 21, 'new_native_fits': 39,
        'reused_banks': 15, 'reused_selected_fits': 33, 'joined_banks': 36, 'joined_selected_fit_records': 72,
        'frozen_policy': policy, 'all_terminal_endpoints': 180, 'reused_endpoints': 75, 'new_endpoint_attempts': 105, 'calibration_costs': calibration_costs})]
    documents += [('CONTEXT_PREPARATION_AND_WORK_COSTS.json', {'preparation': context['complete']['context_preparation_once'],
        'target_support_diagnostics': target_diagnostics, 'context_operations': costs['context_operation_counts'],
        'context_operations_by_arm': context_work_by_arm, 'costs': {k: v for k, v in costs.items() if k not in ('new_owner', 'original_owner')}})]
    documents += chunk_documents('SAGE_ARM_AGGREGATES', [{'readout': kind, 'arm': arm, 'aggregate': value}
        for kind, bank in aggregates.items() for arm, value in bank.items()])
    documents += chunk_documents('SAGE_ARM_DETAILS', groups)
    compact_groups = []
    calibrated_index = {(r['seed'], r['arm']): r for r in calibration_records}
    for row in groups:
        arm, seed = row['arm'], row['seed']
        calibrated_value, endpoint_record = calibrated[seed, arm], calibrated_index[seed, arm]
        compact_groups.append({k: row[k] for k in ('arm', 'seed', 'raw', 'diagnostics', 'acquisition_origin')})
        compact_groups[-1]['calibrated'] = ({k: calibrated_value[k] for k in ('quality', 'counts', 'classes')}
            if calibrated_value is not None else {'failed_retained': True})
        compact_groups[-1]['calibrated_fixed_complete_roster_strict_rival_cohorts'] = endpoint_record.get('fixed_complete_roster_strict_rival_cohorts')
    documents += chunk_documents('SAGE_COMPACT_SCORE_ERROR_DETAILS', compact_groups)
    documents += chunk_documents('CALIBRATION_FIT_DETAILS', calibration_records)
    for category, values in contrasts.items():
        if category.endswith('_error_partitions'):
            documents.append(('SAGE_' + category + '.json', values))
        else:
            documents += chunk_documents('SAGE_' + category, [{'key': name, 'contrast': value} for name, value in values.items()])
    documents += chunk_documents('CHECKPOINT_CUSTODY', checkpoints)
    documents += chunk_documents('ARCHIVE_CUSTODY', [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS])
    documents += chunk_documents('REUSED_OOF_ARCHIVE_CUSTODY', context['reused_OOF_archives'])
    prepared = [(report_path, encode(report))]
    prepared += [(report_path.parent / name, encode(value)) for name, value in documents]
    prepared += [(report_path.parent / name, data) for name, data in binaries]
    require(len({path for path, _ in prepared}) == len(prepared), 'Distinct deterministic output names')
    for path, data in prepared:
        require(not path.exists(), 'Every output path must be fresh')
        if path != report_path and path.suffix == '.json':
            require(len(data) < LIMIT, 'Bounded JSON export required')
    # Every artifact and input pin is rehashed before any report/OOF output is written.
    for item in list(expected.values()) + checkpoints + [context['target_archive']] + context['reused_OOF_archives'] + context['prior_calibration_custody']['record_partitions']:
        require(sha(item['path']) == item['sha256'], 'Admitted artifact changed before output')
    for name, digest in inputs['research_files'].items():
        require(sha(root / name) == digest, 'Pinned source/receipt changed during readout')
    for name, digest in source['files'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Reader seal changed during readout')
    require(sha(root / PILOT / 'actual_family_v1/COMPLETE_FAMILY.json') == context['complete_sha256']
            and sha(root / PILOT / 'OWNER_END.json') == context['owner_end_sha256']
            and sha(root / PILOT / 'FREEZE.json') == context['freeze_sha256']
            and sha(Path(context['binding']['family_root']) / 'COMPLETE_FAMILY.json') == context['binding']['complete_sha256']
            and sha(Path(context['binding']['family_root']).parent / 'OWNER_END.json') == context['binding']['owner_end_sha256']
            and sha(root / PRIOR / 'READER_JOB_END_V1.json') == context['calibration_reuse_admission']['prior_reader_end_sha256']
            and sha(root / PRIOR / 'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json') == context['calibration_reuse_admission']['prior_complete_summary_sha256'], 'Whole current/original/reused closure unchanged')
    receipts = [{'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in prepared]
    summary = {'complete': True, 'backbone': 'SAGE', 'new_banks': 21, 'new_native_fits': 39,
        'original_reference_banks': 15, 'original_selected_fits': 33, 'joined_banks': 36, 'joined_selected_fit_records': 72,
        'TEST_access': False, 'frozen_policy': policy, 'calibration_folds': folds, 'calibration_costs': calibration_costs,
        'raw_authority_preserved': True, 'original_raw_and_calibrated_scores_preserved': True,
        'all180_endpoint_records_terminal_before_interpretation': True, 'reused_endpoints': 75, 'new_scalar_fit_attempts': 105,
        'post_selection_OOF_scope': 'Encountered development, not unused whole-pipeline confirmation.',
        'repair_scope': 'Cross-arm full procedures; calibration-only effects at fixed native selected state.',
        'target_mode_order_custody_preserved': True, 'target_construction_replayed': False,
        'selected_scalar_target_source_contract_bound': True,
        'new_config_sha256': context['new_config_sha256'], 'original_config_sha256': context['original_config_sha256'],
        'native_config_subset_exclusion': ['context_teacher_temperature'],
        'full_report': receipts[0], 'partitions': receipts[1:], 'reused_OOF_archives': context['reused_OOF_archives'],
        'input_binding_sha256': sha(Path(__file__).with_name('INPUT_BINDINGS.json')),
        'new_complete_sha256': context['complete_sha256'], 'new_freeze_sha256': context['freeze_sha256'],
        'new_owner_end_sha256': context['owner_end_sha256'], 'original_complete_sha256': context['binding']['complete_sha256'],
        'analysis_source_sha256': sha(Path(__file__))}
    summary_path = report_path.parent / 'COMPLETE_ANALYSIS_SUMMARY.json'
    require(summary_path not in {p for p, _ in prepared} and not summary_path.exists(), 'Fresh summary path')
    summary_bytes = encode(summary)
    require(len(summary_bytes) < LIMIT, 'Small complete summary')
    prepared.append((summary_path, summary_bytes))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for path, data in prepared:
        with path.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'complete': True, 'summary': str(summary_path),
        'raw_private_attribution_screen_pass': policy['raw_private_attribution_screen_pass'],
        'calibrated_confidence_protection_pass': policy['calibrated_confidence_protection_pass'],
        'joint_accuracy_confidence_development_clue': policy['joint_accuracy_confidence_development_clue'],
        'reused_scalar_endpoints': 75, 'new_scalar_fit_attempts': 105,
        'all_terminal_endpoints': 180, 'partitions': len(receipts) - 1}))



if __name__ == '__main__':
    main()
