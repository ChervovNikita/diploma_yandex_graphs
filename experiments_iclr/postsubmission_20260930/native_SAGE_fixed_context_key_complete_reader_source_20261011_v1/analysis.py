"""Complete native SAGE fixed context key reader; stdlib until full custody.

No native model, loader, checkpoint deserialization, forward, selector or TEST.
Exactly75 admitted original calibration endpoints are reused without fitting;
150 fresh fixed endpoint attempts finish before any score or contrast.
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
PILOT = 'native_SAGE_fixed_context_key_pilot_root_20261011_v1'
LEARNING = 'native_SAGE_fixed_context_key_source_20261011_v1'
PRIOR = 'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
PRIOR_READER = 'SAGE_GNCL_SupCon_2x2_complete_reader_source_20261010_v2/analysis.py'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
REFERENCE = 'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
REFERENCE_SHA = 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
CALIBRATOR = 'common_wrapper_graph_reliability_source_20261010_v2/operators.py'
SEEDS = (7301, 7403, 7507)
NEW = ('ordinaryM1_key_mixture', 'factorizedM1_key_mixture',
       'genuinefactorizedI4_key_mixture', 'genuinefactorizedI4_key_assigned',
       'shared4_key_private', 'shared4_key_mixture', 'shared4_key_uniform',
       'shared4_projected_peer_private', 'shared4_key_moment_private', 'shared4_key_shuffled_private')
REFS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
OLD = REFS + ('separable_equal_size', 'exchange')
ARMS = NEW + REFS
O1C, F1C, F4C, F4A, PRIVATE, MIX, UNIFORM, PEER, MOMENT, SHUFFLE = NEW
O1, O4, F1, F4, BASE = REFS
CONTEXT_MODES = dict(zip(NEW, ('key_mixture', 'key_mixture', 'key_mixture', 'key_assigned',
    'key_private', 'key_mixture', 'key_uniform', 'projected_peer_private', 'key_moment_private', 'key_shuffled_private')))
CONTEXT_NAMES = ('X', 'X-PX', 'PX', 'P^2X')
TARGET_KEYS = {'train_ids', 'train_y', 'single_order', 'weights', 'uniform_weights', 'positive',
               'full_allowed', 'keys', 'shuffled_weights', 'shuffle_permutation'}
OBJECTIVE_CONTRACT = {'target_policy': 'continuous_all_classmate_full_denominator_v1',
 'coefficient': 0.05,
 'student_temperature': 0.2,
 'teacher_temperature': 0.2,
 'objective': 'unit_logZ_fixed_keys_minus_actual_Q_key_moment_dot_over_tau',
 'key_policy': 'fixed_normalized_graph_context_keys_unit_logZ_v1',
 'query_map': 'one shared bare128to300_no_bias_training_only_per_bank; own_map_per_independent_body',
 'query_initialization': 'separateCPU_float64_QR_sign_fixed_float32_semiorthogonal; '
                         'same_per_seed_bytes_all_arms_bodies',
 'query_seed_offset': 5000011,
 'target_mean': 'actual_float32_Q_times_actual_float32_keys; not_normalized',
 'lognormalizer_coefficient': 'literal1; inherited_cast_Q_rowmass_diagnostic_only',
 'moment_control': '(1_minus_query_dot_actual_target_key_mean)/student_temperature',
 'peer_control': 'same_query_map_on_live_query_and_live_same_route_peer_keys; both_gradient_paths',
 'shuffle': 'one_class_preserving_permutation_on_both_Q_axes; fixed_keys_unchanged',
 'shuffle_seed': 11711,
 'mixture': 'mean_of_four_complete_key_Q_denominator_objectives; never_average_keys',
 'assigned_independent_body': 'body_m_uses_Q_m_with_explicit_context_index',
 'native_factual_forwards_per_route_update': 1}
ARM_MAPPING = {'ordinary_M1_key_mixture': 'ordinaryM1_key_mixture',
 'factorized_M1_key_mixture': 'factorizedM1_key_mixture',
 'genuine_I4_key_mixture': 'genuinefactorizedI4_key_mixture',
 'genuine_I4_assigned_keys': 'genuinefactorizedI4_key_assigned',
 'shared4_private_fixed_keys': 'shared4_key_private',
 'shared4_key_mixture': 'shared4_key_mixture',
 'shared4_uniform_classmates': 'shared4_key_uniform',
 'shared4_projected_live_peer_private': 'shared4_projected_peer_private',
 'shared4_weighted_mean_attraction_private': 'shared4_key_moment_private',
 'shared4_shuffled_Q_incidence_private': 'shared4_key_shuffled_private'}
VALID_HASHES = {'ids': '48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801',
                'labels': '2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'}
EXPECTED_PROTOCOL = {'schema': 'native_SAGE_fixed_context_key_complete_reader_protocol_v1',
 'production_protocol': {'schema': 'gnnm-fixed-context-key-prospective-utility-v1',
                         'UTC': '2026-10-10T23:48:09.070691+00:00',
                         'status': 'preparation_only_no_scientific_fit',
                         'task': 'encountered WikiCS development; no TEST or unused-confirmation claim',
                         'backbone': 'native SAGE',
                         'members': 4,
                         'seeds': [7301, 7403, 7507],
                         'hidden_width': 128,
                         'depth': 2,
                         'dropout': 0.2,
                         'learning_rate': 0.001,
                         'maximum_updates': 1000,
                         'patience': 300,
                         'objective': 'own member CE plus 0.05 specified auxiliary',
                         'query_temperature': 0.2,
                         'target_temperature': 0.2,
                         'serving': 'unchanged mean member softmax probabilities; auxiliary head removed',
                         'candidate': 'shared4_private_fixed_keys',
                         'arms': ['ordinary_M1_key_mixture',
                                  'factorized_M1_key_mixture',
                                  'genuine_I4_key_mixture',
                                  'genuine_I4_assigned_keys',
                                  'shared4_private_fixed_keys',
                                  'shared4_key_mixture',
                                  'shared4_uniform_classmates',
                                  'shared4_projected_live_peer_private',
                                  'shared4_weighted_mean_attraction_private',
                                  'shared4_shuffled_Q_incidence_private'],
                         'counts': {'fresh_banks': 30,
                                    'fresh_optimizer_fits': 48,
                                    'original_banks': 15,
                                    'original_native_records': 33,
                                    'all_banks': 45,
                                    'all_native_records': 81,
                                    'reused_scalar_endpoints': 75,
                                    'new_scalar_endpoints': 150,
                                    'all_scalar_endpoints': 225},
                         'primary_accuracy_screen': {'references': ['unchanged_shared_BASE',
                                                                    'genuine_I4_key_mixture',
                                                                    'genuine_I4_assigned_keys',
                                                                    'ordinary_M1_key_mixture',
                                                                    'factorized_M1_key_mixture'],
                                                     'each_reference_mean_gain_pp_at_least': 0.2,
                                                     'each_reference_every_paired_seed_gain_pp_at_least': 0.0,
                                                     'each_reference_positive_paired_seeds_at_least': 2,
                                                     'scope': 'exploratory progression rule, not a '
                                                              'significance test or a broad-superiority '
                                                              'claim'},
                         'attribution_checks': {'route_assignment': ['shared4_key_mixture'],
                                                'fixed_key_source': ['shared4_projected_live_peer_private'],
                                                'dictionary_partition': ['shared4_weighted_mean_attraction_private'],
                                                'nonuniform_target_need': ['shared4_uniform_classmates'],
                                                'target_key_incidence': ['shared4_shuffled_Q_incidence_private'],
                                                'for_each_attribution_mean_gain_pp_at_least': 0.1,
                                                'for_each_attribution_every_paired_seed_gain_pp_at_least': 0.0,
                                                'for_each_attribution_positive_paired_seeds_at_least': 2,
                                                'interpretation': 'Each claim requires its own comparator. '
                                                                  'Primary accuracy success alone cannot '
                                                                  'establish target allocation, dictionary '
                                                                  'necessity, graph-specific novelty or a '
                                                                  'full relation-distribution transfer.'},
                         'secondary_confidence_screen': {'metric': 'equally applied fixed five-fold '
                                                                   'calibrated NLL',
                                                         'references': ['unchanged_shared_BASE',
                                                                        'genuine_I4_key_mixture',
                                                                        'genuine_I4_assigned_keys'],
                                                         'each_mean_harm_cap': 0.02,
                                                         'each_seed_harm_cap': 0.05,
                                                         'separate_from_primary_accuracy': True,
                                                         'all_raw_NLL_and_Brier_visible': True,
                                                         'old_failed_conjunctions_unchanged': True,
                                                         'folds_after_checkpoint_selection_not_unused': True},
                         'member_competence': {'report_mean_worst_and_per_seed': True,
                                               'not_an_automatic_accuracy_veto': True,
                                               'claim_of_improved_individual_members_requires_positive_member_evidence': True},
                         'required_error_analysis': ['coverage additions and removals',
                                                     'pooling losses and aggregation-only corrections',
                                                     'paired repairs and harms against BASE and capable I4',
                                                     'fixed BASE strict wrong rivals repaired and new '
                                                     'strict rivals created',
                                                     'class/seed breakdown and descriptive uncertainty',
                                                     'parameter, training, auxiliary and serving costs'],
                         'closure_rule': 'All 48 fresh fits, 45 banks and 225 reused/new scalar endpoints '
                                         'must be terminal before comparative interpretation. Preserve any '
                                         'failure and do not rescue by adjusting frozen settings.',
                         'promotion': 'A worthwhile accuracy signal takes matched original-GNNM comparison '
                                      'and unused whole-pipeline confirmation priority. No primitive '
                                      'novelty follows from established projection, contrastive, '
                                      'graph-signature or ensemble ingredients.',
                         'qualification_scope': 'Independent source audit and finite actual TRAIN '
                                                'execution resolving gradient/restore/RNG/ownership risks; '
                                                'no optional parity expansion and no VALID/TEST quality '
                                                'reads.'},
 'arm_mapping': {'ordinary_M1_key_mixture': 'ordinaryM1_key_mixture',
                 'factorized_M1_key_mixture': 'factorizedM1_key_mixture',
                 'genuine_I4_key_mixture': 'genuinefactorizedI4_key_mixture',
                 'genuine_I4_assigned_keys': 'genuinefactorizedI4_key_assigned',
                 'shared4_private_fixed_keys': 'shared4_key_private',
                 'shared4_key_mixture': 'shared4_key_mixture',
                 'shared4_uniform_classmates': 'shared4_key_uniform',
                 'shared4_projected_live_peer_private': 'shared4_projected_peer_private',
                 'shared4_weighted_mean_attraction_private': 'shared4_key_moment_private',
                 'shared4_shuffled_Q_incidence_private': 'shared4_key_shuffled_private'},
 'calibration_policy': {'temperature_parameter': 'one bank-global logT; positive exp(logT)',
                        'folds': 5,
                        'fold_order': 'CPU randperm seed11709; position modulo5',
                        'dtype': 'float64 CPU',
                        'optimizer': 'Adam',
                        'learning_rate': 0.01,
                        'steps_per_endpoint': 500,
                        'endpoint': 'final step',
                        'calibrated_banks': 45,
                        'scalar_fits': 225,
                        'scalar_steps': 112500,
                        'operations_source': 'common_wrapper_graph_reliability_source_20261010_v2/operators.py',
                        'equally_applied_to_all_banks': True,
                        'TEST_access': False,
                        'all_VALID_refit': False,
                        'scope': 'postselection encountered development',
                        'risk_rule': 'DECISION.md',
                        'reused_scalar_endpoints': 75,
                        'new_scalar_endpoints': 150},
 'calibration': {'folds': 5,
                 'fold_seed': 11709,
                 'global_log_temperature_initial': 0,
                 'optimizer': 'CPU float64 Adam',
                 'learning_rate': 0.01,
                 'betas': [0.9, 0.999],
                 'epsilon': 1e-08,
                 'weight_decay': 0,
                 'updates': 500,
                 'endpoint': 'fixed final update',
                 'reused_endpoints': 75,
                 'new_fit_attempts': 150,
                 'all_terminal_records': 225,
                 'new_requested_scalar_updates': 75000,
                 'original_scalar_updates_already_paid': 37500,
                 'total_comparative_requested_scalar_updates': 112500,
                 'retries': 0,
                 'grid': False,
                 'whole_VALID_refit': False},
 'primary_accuracy_separate_from_attribution_and_confidence': True,
 'calibrated_accuracy_diagnostic_only': True,
 'member_competence_diagnostic_not_extra_veto': True,
 'both_equipped_singles_primary_accuracy_references': True,
 'raw_NLL_and_Brier_tradeoffs_visible': True,
 'target_preparation_replayed': False,
 'query_initialization_replayed': False,
 'moment_matrix_products_replayed': False,
 'native_model_or_checkpoint_deserialization': False,
 'TEST_access': False,
 'automatic_launch': False,
 'scope': 'One native SAGE backbone on encountered WikiCS development; separately fitted/stopped/selected '
          'procedures, not same-state causal isolation or unused confirmation.',
 'fixed_target_policy': 'continuous_all_classmate_full_denominator_v1',
 'fixed_key_policy': 'fixed_normalized_graph_context_keys_unit_logZ_v1',
 'objective_contract': {'target_policy': 'continuous_all_classmate_full_denominator_v1',
                        'coefficient': 0.05,
                        'student_temperature': 0.2,
                        'teacher_temperature': 0.2,
                        'objective': 'unit_logZ_fixed_keys_minus_actual_Q_key_moment_dot_over_tau',
                        'key_policy': 'fixed_normalized_graph_context_keys_unit_logZ_v1',
                        'query_map': 'one shared bare128to300_no_bias_training_only_per_bank; '
                                     'own_map_per_independent_body',
                        'query_initialization': 'separateCPU_float64_QR_sign_fixed_float32_semiorthogonal; '
                                                'same_per_seed_bytes_all_arms_bodies',
                        'query_seed_offset': 5000011,
                        'target_mean': 'actual_float32_Q_times_actual_float32_keys; not_normalized',
                        'lognormalizer_coefficient': 'literal1; inherited_cast_Q_rowmass_diagnostic_only',
                        'moment_control': '(1_minus_query_dot_actual_target_key_mean)/student_temperature',
                        'peer_control': 'same_query_map_on_live_query_and_live_same_route_peer_keys; '
                                        'both_gradient_paths',
                        'shuffle': 'one_class_preserving_permutation_on_both_Q_axes; fixed_keys_unchanged',
                        'shuffle_seed': 11711,
                        'mixture': 'mean_of_four_complete_key_Q_denominator_objectives; never_average_keys',
                        'assigned_independent_body': 'body_m_uses_Q_m_with_explicit_context_index',
                        'native_factual_forwards_per_route_update': 1},
 'teacher_temperature': 0.2,
 'collapse_prevention_or_realizability_guarantee': False,
 'assigned_independent_target_index_bound': True}


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

def strict_transitions(np, reference, current, served, labels):
    cohorts = {'fixed_reference_strict_rivals': reference,
        'fixed_reference_strict_rivals_repaired_by_current_pool': reference & served,
        'reference_strict_rival_condition_resolved': reference & ~current,
        'new_current_strict_rivals_created': ~reference & current,
        'persistent_strict_rivals': reference & current,
        'current_strict_rivals_corrected_by_pool': current & served}
    return {'counts': {key: int(mask.sum()) for key, mask in cohorts.items()},
        'classes': [{'class': k, **{key: int((mask & (labels == k)).sum()) for key, mask in cohorts.items()}} for k in range(10)],
        'authority': 'Strict wrong rivals use archived raw float32 member logits; served correctness uses this readout. Cross-arm transitions compare separately fitted procedures.'}


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
            and reuse['reused_scalar_endpoints'] == 75 and reuse['new_scalar_endpoints'] == 150
            and reuse['total_endpoints'] == 225 and reuse['new_scalar_steps'] == 75000
            and reuse['TEST_access'] is False and reuse['no_native_forward'] is True and reuse['no_fit_replay'] is True,
            'Exact root75-endpoint reuse admission;150 new endpoints only')
    expected_policy = {'temperature_parameter': 'one bank-global logT; positive exp(logT)', 'folds': 5,
        'fold_order': 'CPU randperm seed11709; position modulo5', 'dtype': 'float64 CPU', 'optimizer': 'Adam',
        'learning_rate': .01, 'steps_per_endpoint': 500, 'endpoint': 'final step', 'calibrated_banks': 45,
        'scalar_fits': 225, 'scalar_steps': 112500, 'operations_source': CALIBRATOR,
        'equally_applied_to_all_banks': True, 'TEST_access': False, 'all_VALID_refit': False,
        'scope': 'postselection encountered development', 'risk_rule': 'DECISION.md',
        'reused_scalar_endpoints': 75, 'new_scalar_endpoints': 150}
    prior_policy = read(root / PRIOR / 'CALIBRATION_POLICY.json')
    roster_fields = {'calibrated_banks', 'scalar_fits', 'scalar_steps'}
    operator_exclusions = roster_fields | {'reused_scalar_endpoints', 'new_scalar_endpoints'}
    require({k: v for k, v in prior_policy.items() if k not in operator_exclusions}
            == {k: v for k, v in policy.items() if k not in operator_exclusions}
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
            and admission['all_banks_before_interpretation'] == 45
            and admission['all_native_records_before_interpretation'] == 81
            and admission['fixed_calibration_endpoints'] == 225 and admission['reused_endpoints'] == 75
            and admission['new_endpoints'] == 150
            and admission['comparison_blocked_until_complete_225'] is True,
            'Scientific admission closes the whole45-bank/225-endpoint comparative procedure')
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
    """Saved qualified K/Q/support/order custody after45banks/81fits/75 old endpoints; no construction replay."""
    item, meta = context['target_archive'], context['complete']['context_preparation_once']
    require(sha(item['path']) == item['sha256'], 'Admitted target archive unchanged')
    with np.load(item['path'], allow_pickle=False) as archive:
        require(set(archive.files) == TARGET_KEYS, 'Exact saved fixed-key target/order schema')
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
    require(set(meta['array_hashes']) == TARGET_KEYS - {'single_order'}, 'Complete canonical target/key hash keys')
    for name, expected in meta['array_hashes'].items():
        value = arrays[name]
        require({'shape': list(value.shape), 'dtype': str(value.dtype),
                 'C_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()} == expected,
                'Qualified canonical target/key/support bytes: ' + name)
    allowed = ~np.eye(580, dtype=bool)
    positive = arrays['positive']
    require(positive.dtype == np.bool_ and positive.shape == (580, 580)
            and np.array_equal(positive, (labels[:, None] == labels[None, :]) & allowed),
            'Saved positive support is every nonself TRAIN classmate')
    require(arrays['full_allowed'].dtype == np.bool_ and arrays['full_allowed'].shape == (580, 580)
            and np.array_equal(arrays['full_allowed'], allowed), 'Identical all-nonself TRAIN denominator')
    weights, uniform, shuffled, keys, permutation = [arrays[k] for k in
        ('weights', 'uniform_weights', 'shuffled_weights', 'keys', 'shuffle_permutation')]
    require(weights.dtype == uniform.dtype == shuffled.dtype == keys.dtype == np.float64
            and weights.shape == shuffled.shape == (4, 580, 580) and uniform.shape == (580, 580)
            and keys.shape == (4, 580, 300) and np.isfinite(keys).all()
            and all(np.isfinite(v).all() for v in (weights, uniform, shuffled))
            and all((v[..., positive] > 0).all() and not v[..., ~positive].any()
                    and np.allclose(v.sum(-1), 1, rtol=0, atol=1e-12) for v in (weights, uniform, shuffled)),
            'Qualified finite saved fixed keys and normalized target support')
    require(permutation.dtype == np.int64 and permutation.shape == (580,)
            and np.array_equal(np.sort(permutation), np.arange(580))
            and np.array_equal(labels[permutation], labels)
            and np.array_equal(shuffled, weights[:, permutation[:, None], permutation[None, :]]),
            'Fixed class-preserving permutation moves both Q axes with factual keys unchanged')
    require(positive.sum(-1).tolist() == meta['positive_counts_per_anchor']
            and [int(positive.sum())] * 4 == meta['positive_counts_per_context']
            and int(allowed.sum()) == meta['full_denominator_entries_per_route']
            and np.unique(labels, return_counts=True)[1].tolist() == meta['class_counts'], 'Saved full-class support counts')
    moment_names = {'moments', 'shuffled_moments', 'uniform_moments'}
    for order_name, indices in (('bank', np.arange(580)), ('single', order)):
        tensor_hashes = {}
        def descriptor(value, dtype):
            return {'shape': list(value.shape), 'dtype': dtype,
                'C_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
        for key in ('weights', 'shuffled_weights', 'uniform_weights', 'positive', 'full_allowed'):
            original = arrays[key]
            values = (original[:, indices[:, None], indices[None, :]] if original.ndim == 3
                      else original[indices[:, None], indices[None, :]]).copy()
            is_weight = 'weights' in key
            values = values.astype(np.float32 if is_weight else np.bool_, copy=False)
            tensor_hashes[key] = descriptor(values, 'torch.float32' if is_weight else 'torch.bool')
        tensor_hashes['keys'] = descriptor(keys[:, indices].astype(np.float32), 'torch.float32')
        for key in ('train_ids', 'train_y'):
            tensor_hashes[key] = descriptor(arrays[key][indices].copy(), 'torch.int64')
        inverse = permutation.copy()
        inverse[indices] = np.arange(580)
        tensor_hashes['shuffle_permutation'] = descriptor(inverse[permutation[indices]].copy(), 'torch.int64')
        actual = meta['actual_loss_tensor_hashes'][order_name]
        require(set(actual) == set(tensor_hashes) | moment_names
                and tensor_hashes == {k: actual[k] for k in tensor_hashes}, 'Actual native-order archive-backed tensor custody: ' + order_name)
        for key in moment_names:
            require(actual[key]['shape'] == [4, 580, 300] and actual[key]['dtype'] == 'torch.float32'
                    and len(actual[key]['C_bytes_sha256']) == 64,
                    'Qualified actual device moment descriptor: ' + order_name + '/' + key)
        # Actual float32 GPU reductions are bound by qualification and unit records.
        # Do not recompute moments on CPU and demand cross-device byte equality.
        require(actual == context['qualification']['target_preparation']['actual_loss_tensor_hashes'][order_name],
                'Moment descriptors bind actual qualified device tensors without reduction replay')
    return {'target_archive_checked': True, 'TRAIN_rows': 580, 'contexts': list(CONTEXT_NAMES),
        'target_policy': OBJECTIVE_CONTRACT['target_policy'], 'key_policy': OBJECTIVE_CONTRACT['key_policy'],
        'objective_contract': OBJECTIVE_CONTRACT, 'positive_counts_per_context': meta['positive_counts_per_context'],
        'full_denominator_entries_per_route': meta['full_denominator_entries_per_route'],
        'signature_zero_rows_per_context': meta['signature_zero_rows_per_context'],
        'nonuniform_target_rows_per_context_exact_diagnostic': meta['nonuniform_target_rows_per_context_exact_diagnostic'],
        'actual_Q_rowmass_diagnostics': meta['actual_Q_rowmass_diagnostics'],
        'actual_key_moment_preparation': meta['actual_key_moment_preparation'],
        'actual_moment_descriptors_bound_without_CPU_recomputation': True,
        'bank_and_single_joint_K_Q_support_order_bound': True,
        'shuffle_both_Q_axes_and_class_support_bound': True,
        'signature_affinity_target_key_or_moment_construction_replayed': False,
        'query_initialization_replayed': False, 'native_forward_or_loss_replayed': False,
        'scope': 'Saved qualified target/key/order/support bytes and schema; no hidden geometry, collapse or unused-quality conclusion.'}


def context_work(mode, routes, updates):
    """Literal sealed helper operation law; no native call or numerical replay."""
    matrices = 0 if mode == 'key_moment_private' else routes * (4 if mode == 'key_mixture' else 1)
    moment_rows = 0 if mode == 'projected_peer_private' else routes * (4 if mode == 'key_mixture' else 1)
    return {key: value * updates for key, value in {
        'context_query_projection_calls': 1, 'context_query_projection_rows': routes * 580,
        'context_query_projection_MACs': routes * 580 * 128 * 300,
        'context_normalized_query_coordinates': routes * 580 * 300,
        'context_dictionary_score_calls': 4 if mode == 'key_mixture' else int(matrices > 0),
        'context_dictionary_score_matrices': matrices, 'context_dictionary_score_entries': matrices * 580 * 580,
        'context_dictionary_score_MACs': matrices * 580 * 580 * 300,
        'context_logsumexp_anchor_reductions': matrices * 580,
        'context_dense_masked_logsumexp_entries': matrices * 580 * 580,
        'context_fixed_moment_dot_entries': moment_rows * 580 * 300,
        'context_live_peer_target_score_entries': routes * 580 * 580 if mode == 'projected_peer_private' else 0,
        'context_complete_objective_anchor_rows': routes * (4 if mode == 'key_mixture' else 1) * 580,
        'context_extra_native_forwards': 0, 'context_teacher_GNN_forwards': 0, 'context_serving_calls': 0}.items()}


def preflight(root, helper, inputs):
    """All81 selected fits, safe roles and75 exact old endpoints close before arrays."""
    for name, digest in inputs['research_files'].items():
        require(sha(root / name) == digest, 'Pinned prospective bytes changed: ' + name)
    require(sha(root / REFERENCE) == REFERENCE_SHA, 'Original SAGE binding unchanged')
    here = root / PILOT
    cfg, freeze = read(here / 'CONFIG.json'), read(here / 'FREEZE.json')
    protocol = read(Path(__file__).with_name('PROTOCOL.json'))
    require(protocol == EXPECTED_PROTOCOL
            and protocol['production_protocol'] == read(here / 'PROSPECTIVE_PROTOCOL.json')
            and protocol['calibration_policy'] == read(here / 'CALIBRATION_POLICY.json')
            and read(root / LEARNING / 'MANIFEST.json')['arm_mapping'] == ARM_MAPPING
            and [ARM_MAPPING[a] for a in protocol['production_protocol']['arms']] == list(NEW), 'Exact prospectively fixed context criteria/calibration protocol')
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
    require(admission['helper_sha256'] == inputs['research_files'][LEARNING + '/fixed_context_key_objective.py']
            and admission['qualification_sha256'] == sha(here / 'ACTUAL_QUALIFICATION_V1.json'),
            'Actual qualified context helper scientific admission')
    require(admission['admitted'] is True and admission['qualification_passed'] is True
            and admission['scientific_fits'] == 48 and admission['banks'] == 30
            and admission['reader_source_preparation_only'] is True
            and admission['new_outcomes_seen'] is False and admission['TEST_access'] is False
            and admission['source_manifest_sha256'] == inputs['research_files'][LEARNING + '/MANIFEST.json'],
            'Complete prospective context scientific admission')
    require(reference_admission['admitted'] is True and reference_admission['banks'] == 15
            and reference_admission['selected_fits'] == 33
            and reference_admission['reference_bindings_sha256'] == REFERENCE_SHA
            and reference_admission['exact_source_config_role_seed_selector_custody_compatible'] is True
            and reference_admission['TEST_access'] is False, 'Exact root original own-only reference admission')
    qualification = read(here / 'ACTUAL_QUALIFICATION_V1.json')
    require(qualification['qualified'] is True and qualification['status'] == 'PASS'
            and qualification['VALID_quality_scored'] is False and qualification['TEST_access'] is False
            and qualification['scientific_fits'] == 0 and qualification['VALID_quality_calls'] == 0
            and qualification['scientific_launch_admitted'] is False,
            'Actual finite TRAIN-only disposable qualification, not predictive evidence')
    require(qualification['configuration_sha256'] == sha(here / 'CONFIG.json')
            and helper.source_hash(qualification, LEARNING + '/run_family.py') == inputs['research_files'][LEARNING + '/run_family.py']
            and helper.source_hash(qualification, LEARNING + '/fixed_context_key_objective.py') == inputs['research_files'][LEARNING + '/fixed_context_key_objective.py']
            and qualification['target_objective_contract'] == OBJECTIVE_CONTRACT,
            'Actual qualification source/config and complete fixed-key law custody')
    require(qualification['diagnostic_autograd_VJPs'] == 32 and qualification['combined_backwards'] == 16
            and qualification['optimizer_steps'] == 16 and qualification['native_TRAIN_trajectories'] == 34,
            'Exact16 disposable TRAIN units/34 trajectories and32 diagnostic VJPs')
    cases = qualification['units']
    expected_cases = {(arm, m) for arm in NEW for m in range(4 if arm in (F4C, F4A) else 1)}
    require(len(cases) == 16 and {(r['arm'], r['member']) for r in cases} == expected_cases
            and all(r['seed'] == SEEDS[0] and r['mode'] == CONTEXT_MODES[r['arm']]
                    and r['TRAIN_H_order'] == ('bank' if r['arm'].startswith('shared4_') else 'single')
                    and ((r['arm'] == F4A and type(r['context_index']) is int and r['context_index'] == r['member'])
                         or (r['arm'] != F4A and r['context_index'] is None))
                    and r['training_query_parameters'] == 38400 and r['serving_query_parameters'] == 0
                    and r['initial_query_sha256'] == qualification['query_map_initial_sha256_same_all_units']
                    and r['query_and_hidden_aux_gradient_nonzero'] is True and r['slow_aux_gradient_nonzero'] is True
                    and (r['fast_aux_gradient_nonzero'] is None if r['arm'] == O1C else r['fast_aux_gradient_nonzero'] is True)
                    and len(r['native_parameter_aux_gradient_present']) > 0 and r['full_state_restore_equal'] is True
                    and r['operations'] == context_work(r['mode'], 4 if r['arm'].startswith('shared4_') else 1, 1)
                    for r in cases), 'All actual qualified arm/body modes, A ownership, gradients and restore custody')


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
    for head, groups, fits in ((header, 30, 48), (old_header, 21, 39)):
        require(head['complete'] is True and head['groups'] == head['expected_groups'] == groups
                and head['fit_units'] == head['expected_fit_units'] == fits and head['TEST_access'] is False,
                'Complete declared fresh/original rosters before outcomes')
    require(header['completion_scope'] == 'fresh ten-arm fixed-context-key acquisition only'
            and header['full_comparative_family_complete'] is False
            and header['requires_root_admitted_own_only_anchors'] == {'banks': 15, 'selected_fit_records': 33}
            and header['joined_expected_groups'] == 45 and header['joined_expected_selected_fit_records'] == 81
            and header['quality_interpretation_eligible'] is False
            and header['requires_total_terminal_scalar_endpoints'] == 225
            and header['cached_scalar_endpoints'] == 75 and header['new_scalar_endpoints'] == 150
            and header['requires_root_frozen_quality_attribution_confirmation_criteria_before_fit'] is True
            and header['context_objective_contract'] == OBJECTIVE_CONTRACT,
            'Fresh context completion cannot stand in for complete reference admission')
    complete_sha = sha(output / 'COMPLETE_FAMILY.json')
    require(complete_sha == owner['complete_sha256'] and sha(old_output / 'COMPLETE_FAMILY.json')
            == binding['complete_sha256'] == old_owner['complete_sha256'], 'Owner-bound complete records')
    complete, old = read(output / 'COMPLETE_FAMILY.json'), read(old_output / 'COMPLETE_FAMILY.json')
    require({k: v for k, v in complete.items() if k not in ('results', 'comparisons', 'cost_scope')} == header
            and {k: v for k, v in old.items() if k not in ('results', 'comparisons', 'cost_scope')} == old_header,
            'Complete header/records boundary')
    require(complete['config_sha256'] == sha(here / 'CONFIG.json') and read(output / 'CONFIG.json') == cfg
            and complete['source_sha256'] == qualification['source_sha256'], 'Fresh config and qualified full source authority')
    for row in binding['frozen_inputs']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Original frozen source/config/role bytes')
    old_freeze = read(old_output.parent / 'FREEZE.json')
    require(sha(scoped(binding['config'], root)) == binding['config_sha256'] == old['config_sha256']
            and read(binding['config']) == native_cfg and read(old_output / 'CONFIG.json') == native_cfg, 'Original exact native config subset')
    for suffix, digest in inputs['native_suffixes'].items():
        require(helper.source_hash(complete, suffix) == helper.source_hash(old, suffix) == digest
                and helper.freeze_hash(freeze, suffix) == helper.freeze_hash(old_freeze, suffix) == digest, 'Native source/init/update correspondence: ' + suffix)
    for name in ('run_family.py', 'fixed_context_key_objective.py'):
        require(helper.source_hash(complete, LEARNING + '/' + name) == inputs['research_files'][LEARNING + '/' + name], 'Exact acquired context source: ' + name)
    rows, old_rows = {(r['seed'], r['arm']): r for r in complete['results']}, {(r['seed'], r['arm']): r for r in old['results']}
    require(len(rows) == len(complete['results']) == 30 and set(rows) == {(s, a) for s in SEEDS for a in NEW}
            and sum(len(r['fits']) for r in rows.values()) == 48
            and helper.sum_operations(complete['results']) == complete['operation_counts'], 'All48 fresh unique fixed-key fits')
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
    qualified_preparation = qualification['target_preparation']
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
            and preparation['same_target_bank_shared_across_all_arms_and_seeds'] is True
            and preparation['key_policy'] == OBJECTIVE_CONTRACT['key_policy'] and preparation['key_width'] == 300
            and preparation['shuffle_seed'] == 11711 and preparation['key_or_teacher_GNN_fits'] == 0
            and preparation['shuffle_moves'] == 'both Q axes only; factual fixed keys unchanged'
            and preparation['shuffle_support_self_zero_preserved'] is True
            and preparation['shuffle_preserves_rowmass_multiset_and_class_support'] is True
            and preparation['target_means'] == 'computed once per actual float32 bank/single order from actual Q/K; no normalization'
            and preparation['target_loss_dtype'] == 'float32; unit_logZ_coefficient; actual_Q_key_moments_not_normalized',
            'Actual qualified fixed keys and continuous targets; matched all-classmate support/full denominator')
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
    require(preparation['fixed_key_coordinates'] == preparation['key_normalization_elements'] == 4 * 580 * 300
            and preparation['shuffle_permutation_entries'] == 580
            and preparation['shuffle_target_copy_entries'] == 4 * 580 * 580, 'Actual one-time fixed-key/shuffle preparation work')
    for order_name in ('bank', 'single'):
        moment_work = preparation['actual_key_moment_preparation'][order_name]
        require(moment_work['bmm_calls'] == 3 and moment_work['matrix_products'] == 12
                and moment_work['multiply_accumulates'] == 12 * 580 * 580 * 300
                and moment_work['output_coordinates'] == 12 * 580 * 300
                and moment_work['moment_normalization'] is False, 'Actual device moment preparation work separately by order')
    initial_query_hashes = {seed: set() for seed in SEEDS}
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
                        and unit['learning_objective'] == 'own_CE_plus_fixed_context_key_or_declared_control'
                        and unit['context_mode'] == mode and unit['context_coefficient'] == .05
                        and ((arm == F4A and type(unit['context_index']) is int and unit['context_index'] == unit['member'])
                             or (arm != F4A and unit['context_index'] is None))
                        and unit['context_temperature'] == unit['context_teacher_temperature'] == .2
                        and unit['context_objective_contract'] == OBJECTIVE_CONTRACT
                        and unit['context_target_tensor_hashes'] == preparation['actual_loss_tensor_hashes']['bank' if bank else 'single']
                        and unit['source_sha256'] == complete['source_sha256']
                        and unit['preclassifier_width'] == 128
                        and unit['TRAIN_H_order'] == ('train_ids' if bank else 'native_boolean_mask')
                        and unit['context_target_archive_sha256'] == preparation['archive_sha256'], 'Unchanged starts/live H order and selected checkpoint context-mode/key/target binding')
                require(unit['query_map_seed'] == seed + 5000011
                        and unit['query_map_ownership'] == 'one_auxiliary_map_per_optimizer_body; genuinely_shared_across_shared_routes'
                        and unit['query_map_served'] is False
                        and all(isinstance(unit[k], str) and len(unit[k]) == 64 for k in
                            ('query_map_initial_sha256', 'selected_query_map_sha256')), 'Selected and initial A hashes/seed/ownership bound')
                initial_query_hashes[seed].add(unit['query_map_initial_sha256'])
                c = unit['costs']
                require(c['auxiliary_training_parameters'] == 38400
                        and c['training_parameters'] == c['parameters'] + 38400
                        and c['auxiliary_training_parameter_bytes'] == 38400 * 4
                        and c['training_parameter_bytes'] == c['inference_parameter_bytes'] + 38400 * 4
                        and c['auxiliary_serving_parameters'] == c['auxiliary_serving_forwards'] == 0
                        and c['auxiliary_optimizer_tensor_state_bytes'] > 0
                        and c['optimizer_tensor_state_bytes'] >= c['auxiliary_optimizer_tensor_state_bytes']
                        and c['query_map_creation_seconds_nested_in_acquisition'] <= c['acquisition_seconds'],
                        'A training parameters/state are paid; native serving excludes A; map time is nested')
                selected_scalars = unit['selected_update_training_scalars']
                require(set(selected_scalars) == {'scope', 'step', 'context_index', 'own_ce', 'auxiliary', 'weighted_auxiliary', 'objective'}
                        and selected_scalars['scope'] == 'pre_AdamW_training_scalar_at_update_whose_post_update_VALID_selects_checkpoint'
                        and selected_scalars['step'] == unit['selected_step']
                        and selected_scalars['context_index'] == unit['context_index']
                        and all(isinstance(selected_scalars[k], (int, float)) and not isinstance(selected_scalars[k], bool)
                                and math.isfinite(selected_scalars[k]) for k in ('own_ce', 'auxiliary', 'weighted_auxiliary', 'objective')),
                        'Finite selected-update scalar provenance; no selected-state objective replay')
                updates = unit['completed_updates']
                expected_work = dict(context_work(mode, routes, updates),
                    updates=updates, backwards=updates, adam_steps=updates,
                    train_probability_pool_loss_evaluations=updates if bank else 0,
                    train_probability_pool_backward_evaluations=0,
                    train_fullgraph_native_trajectories=routes * updates,
                    valid_selection_fullgraph_native_trajectories=routes * updates,
                    selected_valid_fullgraph_native_trajectories=routes,
                    native_graph_block_calls=2 * routes * (2 * updates + 1))


                require(unit['operation_counts'] == expected_work, 'Full actual native/query/key/moment/peer/backward/selector work')
                checkpoint(unit, arm, seed, output, True)
                require(read(scoped(unit['selected_state'], output).parent / 'RESULT.json') == unit, 'Selected fresh unit equals complete record')
            check_group_costs(row)
            archive = output / f'{arm}_seed{seed}/selected_VALID.npz'
            expected[seed, arm] = {'path': str(archive), 'sha256': sha(archive), 'members': members, 'new': True}
            records[seed, arm] = row
    require(all(len(hashes) == 1 for hashes in initial_query_hashes.values())
            and next(iter(initial_query_hashes[SEEDS[0]])) == qualification['query_map_initial_sha256_same_all_units'],
            'Identical initial A bytes across every new arm/body per paired seed; seed7301 binds qualification without QR replay')
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
    require(len(expected) == len(records) == 45 and len(checkpoints) == len({r['path'] for r in checkpoints}) == 81, 'All30fresh/15old banks and48fresh/33old selected fits before arrays')
    reuse, reused_archives, prior_custody = admit_reuse(root, here, freeze, inputs, helper, expected, reference_admission)
    context = {'cfg': cfg, 'freeze': freeze, 'admission': admission, 'reference_admission': reference_admission,
        'qualification': qualification, 'owner_start': start, 'owner': owner, 'old_owner': old_owner,
        'binding': binding, 'complete': complete, 'old_complete': old,
        'new_config_sha256': sha(here / 'CONFIG.json'), 'original_config_sha256': binding['config_sha256'],
        'native_config_subset': native_cfg, 'auxiliary_only_config_fields': {'context_teacher_temperature': .2},
        'complete_sha256': complete_sha, 'freeze_sha256': sha(here / 'FREEZE.json'), 'owner_end_sha256': sha(here / 'OWNER_END.json'),
        'query_map_initial_hashes_by_seed': {str(seed): next(iter(hashes)) for seed, hashes in initial_query_hashes.items()},
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
            require(np.array_equal(ids, a['ids']) and np.array_equal(labels, a['y']), 'Identical rows/labels for all45 banks')
            raw, pool, member, pooled = [a[k] for k in ('raw_logits', 'probability_mean', 'member_errors', 'pooled_errors')]
            require(raw.dtype == pool.dtype == np.float32 and raw.shape == (item['members'], 5274, 10) and pool.shape == (5274, 10)
                    and np.isfinite(raw).all() and np.isfinite(pool).all() and (pool >= 0).all() and (pool <= 1).all()
                    and member.dtype == pooled.dtype == np.bool_ and member.shape == (item['members'], 5274) and pooled.shape == (5274,)
                    and np.array_equal(pool.argmax(1) != labels, pooled), 'Archived float32 prediction/error authority')
            arrays[seed, arm] = a
    require(len(arrays) == 45, 'All45 schemas before scoring/calibration')
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
    value['S'] = strict_mask(np, archive['raw_logits'], archive['y'])
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
    value['strict_rival_transitions'] = [{'seed': seed, **strict_transitions(helper.np, values[seed, reference]['S'],
        values[seed, candidate]['S'], values[seed, candidate]['P'], labels)} for seed in SEEDS]
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
                'current_bank_common_strict_rival': strict_counts(np, value['S'], value['P'], labels),
                'BASE_to_current_strict_rival_transitions': strict_transitions(np, strict_mask(np, arrays[seed, BASE]['raw_logits'], labels), value['S'], value['P'], labels),
                'fixed_original_strict_rival_cohorts': {ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS},
                'FP64_raw_pool_argmax_disagreements_with_archived_errors': int(((probability64.mean(0).argmax(1) != labels) != a['pooled_errors']).sum()),
                'raw_logit_argmax_disagreements_with_archived_member_errors': int(((a['raw_logits'].argmax(-1) != labels) != a['member_errors']).sum()),
                'stable_NLL_minus_original_reported': value['quality']['pooled_nll'] - row['valid']['nll'],
                'scope': 'Selected full-procedure output; original raw float32 error flags remain integer authority.'}
    require(context['complete']['comparisons'] == [], 'Acquisition performs no cross-arm comparison before225 endpoint closure')


    return raw, diagnostics




def calibrate(np, torch, operators, arrays, ids, labels, context):
    """Reuse exact75 admitted endpoints; all150 new attempts terminal before scores."""
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
    require(attempted == 150 and len(records) == len(payloads) == 45 and len(binaries) == 30
            and all(len(r['fits']) == 5 for r in records)
            and sum(r['attempted_scalar_fits'] for r in records) == 225
            and sum(r['new_scalar_fit_attempts'] for r in records) == 150, 'All225 terminal endpoints, exactly150 newly attempted')
    new_records = [r for r in records if r['arm'] in NEW]
    successful = sum(f['status'] == 'finite_fixed_endpoint' for r in new_records for f in r['fits'])
    reused_seconds = sum(r['prior_fit_seconds'] for r in reuse_index.values())
    cost = {'reused_scalar_endpoints': 75, 'reused_successful_scalar_updates_already_paid': 37500,
        'reused_prior_fit_seconds_sum_already_paid': reused_seconds, 'reused_scalar_fit_replays': 0,
        'new_attempted_scalar_fits': attempted, 'new_requested_updates_per_fit': 500,
        'new_requested_fullbatch_scalar_updates': 75000, 'new_successful_scalar_fits': successful,
        'new_failed_scalar_fits': 150 - successful, 'new_known_successful_fullbatch_scalar_updates': successful * 500,
        'new_known_completed_fullbatch_scalar_updates_lower_bound': successful * 500 + sum(f['actual_updates_before_failure'] or 0 for r in new_records for f in r['fits'] if f['status'] == 'failed_retained'),
        'new_failed_actual_update_counts_known': all(f.get('actual_updates_before_failure') is not None for r in new_records for f in r['fits'] if f['status'] == 'failed_retained'),
        'new_fit_seconds_sum': sum(r['fit_seconds_sum'] for r in new_records),
        'current_readout_wall_seconds_including_reuse_IO': time.perf_counter() - start,
        'total_comparative_endpoint_records': 225, 'total_comparative_requested_scalar_updates': 112500,
        'native_models_or_forwards_added': 0, 'no_retry_or_final_refit': True,
        'scope': '75 original endpoint costs are retained separately and not charged as new work.150 new attempts only; failure does not imply500 updates. Unknown failed work stays unknown. Fit seconds are nested in current wall time.'}
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
            value['S'] = strict_mask(np, a['raw_logits'], labels)
            record['BASE_to_current_strict_rival_transitions'] = strict_transitions(np, strict_mask(np, arrays[seed, BASE]['raw_logits'], labels), value['S'], value['P'], labels)
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
    extra_fields = ('training_parameters', 'auxiliary_training_parameters', 'training_parameter_bytes',
        'auxiliary_training_parameter_bytes', 'optimizer_tensor_state_bytes', 'auxiliary_optimizer_tensor_state_bytes',
        'auxiliary_serving_parameters', 'auxiliary_serving_forwards', 'query_map_creation_seconds_nested_in_acquisition')
    value['training_auxiliary_serving_fields'] = {key: {'sum': sum(r['costs'][key] for r in rows)
        if all(key in r['costs'] for r in rows) else None, 'recorded_for_all_groups': all(key in r['costs'] for r in rows)} for key in extra_fields}
    value['query_map_creation_time_is_nested_in_acquisition_not_additive'] = True
    value['native_parameters_are_serving_parameters'] = True
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
        scope='Observed costs; acquisition, restoration, selected serving and one-time key/target preparation have separate scopes. Peak memory is a maximum, not a sum. Process RSS is lifetime peak.')
    return value



def transport_error_partitions(helper, np, values, labels, readout):
    controls = (BASE, PRIVATE, MIX, PEER, MOMENT, UNIFORM, SHUFFLE, F4C, F4A)
    if any(values[s, arm] is None for s in SEEDS for arm in controls):
        return {'readout': readout, 'control_order': list(controls), 'failed_retained': True,
                'scope': 'Incomplete calibrated control cohort retains endpoint failures; no subset interpretation.'}
    rows = []
    other_controls = controls[2:]
    for seed in SEEDS:
        decisions = [values[seed, arm]['P'] for arm in controls]
        code = sum((decision.astype(np.uint16) << bit) for bit, decision in enumerate(decisions))
        patterns = []
        for value in range(512):
            mask = code == value
            patterns.append({'pattern': value, 'bits': {arm: bool(value & (1 << bit)) for bit, arm in enumerate(controls)},
                'nodes': int(mask.sum()), 'classes': [{'class': k, 'nodes': int((mask & (labels == k)).sum())} for k in range(10)]})
        base, private = decisions[:2]
        repaired = ~base & private
        none_control = np.logical_and.reduce([~v for v in decisions[2:]])
        all_control = np.logical_and.reduce(decisions[2:])
        unique_repair, unanimous_harm = repaired & none_control, base & ~private & all_control
        rows.append({'seed': seed, 'joint_correctness_patterns': patterns,
            'private_repairs_of_BASE': int(repaired.sum()),
            'private_repairs_not_served_by_each_control': {arm: int((repaired & ~values[seed, arm]['P']).sum()) for arm in other_controls},
            'private_repairs_not_served_by_any_of_seven_controls': int(unique_repair.sum()),
            'private_harms_where_BASE_and_all_seven_controls_correct': int(unanimous_harm.sum()),
            'classes': [{'class': k, 'private_repairs_of_BASE': int((repaired & (labels == k)).sum()),
                'private_repairs_not_served_by_each_control': {arm: int((repaired & ~values[seed, arm]['P'] & (labels == k)).sum()) for arm in other_controls},
                'private_repairs_not_served_by_any_of_seven_controls': int((unique_repair & (labels == k)).sum()),
                'private_harms_where_BASE_and_all_seven_controls_correct': int((unanimous_harm & (labels == k)).sum())} for k in range(10)]})
    return {'readout': readout, 'control_order': list(controls), 'patterns_per_seed': 512, 'integer_pattern_dtype': 'uint16',
        'per_seed': rows, 'equipped_singles': 'Required pairwise contrasts report both singles separately; this exact intersection cohort is BASE/private/five attribution/two equipped I4 procedures.',
        'scope': 'Full separately fitted/stopped/selected procedures. Complete512-pattern intersections describe specificity and do not isolate a same-state cause or replace separate fixed criteria.'}


def partition_export_rows(value):
    """No monolithic512-pattern export assumption; deterministic small rows."""
    header = {k: v for k, v in value.items() if k != 'per_seed'}
    rows = [{'kind': 'cohort_metadata', **header}]
    for seed in value.get('per_seed', []):
        rows.append({'kind': 'seed_summary', **{k: v for k, v in seed.items() if k != 'joint_correctness_patterns'}})
        rows.extend({'kind': 'correctness_pattern', 'seed': seed['seed'], **pattern} for pattern in seed['joint_correctness_patterns'])
    return rows


def analyze(helper, np, raw, calibrated, labels, records):
    pairs = [(a, b) for a in NEW for b in ARMS if a != b]
    require(len(pairs) == len(set(pairs)) == 140, 'Exactly ten fresh candidates versus fourteen other procedures each')
    required_primary = {(PRIVATE, ref) for ref in (BASE, F4C, F4A, O1C, F1C)}
    required_attribution = {(PRIVATE, ref) for ref in (MIX, PEER, MOMENT, UNIFORM, SHUFFLE)}
    required_confidence = {(PRIVATE, ref) for ref in (BASE, F4C, F4A)}
    require(required_primary | required_attribution | required_confidence <= set(pairs),
            'Every primary5/attribution5/confidence3 declared contrast is included')
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
            contrasts[a + '_minus_' + b]['native_cost_scope'] = 'Observed full-procedure costs; original work already paid. One-time shared key/target preparation is reported once outside arm-specific fit/restore/serving deltas.'
    effects = {}
    for arm in ARMS:
        if all(calibrated[s, arm] is not None for s in SEEDS):
            values = {(s, 'raw'): raw[s, arm] for s in SEEDS}
            values.update({(s, 'calibrated'): calibrated[s, arm] for s in SEEDS})
            effects[arm] = contrast(helper, values, 'calibrated', 'raw', labels, 'same_selected_state_calibration')
        else:
            effects[arm] = {'failed_retained': True}
    def accuracy_criterion(reference, threshold):
        stat = raw_contrasts[PRIVATE + '_minus_' + reference]['quality_deltas']['pooled_accuracy_pct']
        flags = {'required_mean_gain_pp': threshold, 'paired_raw_accuracy_pp': stat,
            'mean_at_least_threshold': stat['mean'] >= threshold,
            'all_seeds_nonnegative': stat['nonnegative_seed_count'] == 3,
            'at_least_two_positive': stat['positive_seed_count'] >= 2}
        flags['pass'] = all(flags[k] for k in ('mean_at_least_threshold', 'all_seeds_nonnegative', 'at_least_two_positive'))
        return flags
    accuracy = {ref: accuracy_criterion(ref, .2) for ref in (BASE, F4C, F4A, O1C, F1C)}
    attribution_refs = {'route_assignment': MIX, 'fixed_key_source': PEER, 'dictionary_partition': MOMENT,
        'nonuniform_target_need': UNIFORM, 'target_key_incidence': SHUFFLE}
    attribution = {claim: {'reference': ref, **accuracy_criterion(ref, .1),
        'scope': 'This claim requires its own declared full-procedure comparator; primary accuracy success does not establish it.'}
        for claim, ref in attribution_refs.items()}
    protection = {}
    for ref in (BASE, F4C, F4A):
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
    accuracy_pass, confidence_pass = all(v['pass'] for v in accuracy.values()), all(v['pass'] for v in protection.values())
    capable = {}
    for ref in (O1C, F1C, F4C, F4A, O1, F1, O4, F4):
        stat = raw_contrasts[PRIVATE + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        value = calibrated_contrasts[PRIVATE + '_minus_' + ref]
        capable[ref] = {'paired_raw_pooled_accuracy_pp': stat,
            'adverse_stronger_reference_in_mean_raw_accuracy': stat['mean'] < 0,
            'any_negative_raw_seed': stat['nonnegative_seed_count'] < 3,
            'calibrated_quality_deltas': value.get('quality_deltas'),
            'calibration_failure_retained': bool(value.get('failed_retained')),
            'role': 'Both equipped singles are primary references; historical capable references limit broad-superiority claims. Member diagnostics add no veto.'}
    generic = {arm: {'raw_quality_deltas_vs_BASE': raw_contrasts[arm + '_minus_' + BASE]['quality_deltas'],
        'calibrated_quality_deltas_vs_BASE': calibrated_contrasts[arm + '_minus_' + BASE].get('quality_deltas'),
        'calibration_failure_retained': bool(calibrated_contrasts[arm + '_minus_' + BASE].get('failed_retained')),
        'scope': 'Full-procedure training control contrast; retain each result and its own attribution criterion.'}
        for arm in (MIX, PEER, MOMENT, UNIFORM, SHUFFLE)}
    policy = {'candidate': PRIVATE, 'semantic_candidate': 'shared4_private_fixed_keys', 'arm_mapping': ARM_MAPPING,
        'primary_accuracy_criteria': accuracy, 'primary_accuracy_screen_pass': accuracy_pass,
        'attribution_checks': attribution, 'all_five_attribution_checks_pass': all(v['pass'] for v in attribution.values()),
        'secondary_confidence_criteria': protection, 'secondary_confidence_screen_pass': confidence_pass,
        'secondary_confidence_does_not_veto_primary_accuracy': True,
        'primary_accuracy_with_secondary_confidence_tradeoff': accuracy_pass and not confidence_pass,
        'capable_reference_diagnostics': capable,
        'broad_superiority_statement_blocked_by_stronger_capable_reference': any(v['adverse_stronger_reference_in_mean_raw_accuracy'] for v in capable.values()),
        'target_key_control_findings': generic, 'member_quality_is_diagnostic_not_veto': True,
        'improved_individual_member_claim_requires_positive_member_evidence': True,
        'calibrated_accuracy_reported_without_extra_veto': True, 'raw_NLL_and_Brier_tradeoffs_preserved': True,
        'historical_A_B_AB_not_candidate_or_K': True, 'automatic_confirmation_launch': False,
        'matched_original_projector_only_GNNM_comparison_required_before_worthwhile_signal_promotion': True,
        'unused_whole_pipeline_confirmation_required': True, 'primitive_novelty_claim': False,
        'scope': 'One native SAGE backbone on encountered WikiCS development. Primary accuracy, each attribution claim and secondary confidence are reported separately. Three seed repeats are dependent development evidence; no primitive novelty, universal transfer, unused confirmation or manuscript verdict.'}


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
    # All45 banks/81 fit identities/75 original endpoint records close before NumPy or arrays.
    import numpy as np
    helper.np = np
    target_diagnostics = target_arrays(np, context)
    arrays, ids, labels = payloads(np, expected)
    import torch
    torch.set_num_threads(2)
    operators = load_source('SAGE_fixed_temperature', root / CALIBRATOR, inputs['research_files'][CALIBRATOR])
    calibrated_payloads, calibration_records, binaries, folds, calibration_costs = calibrate(np, torch, operators, arrays, ids, labels, context)
    # No score, contrast, gate or error partition before all225 endpoints are terminal.
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
        'acquisition_origin': 'new_fixed_context_key_or_declared_control' if arm in NEW else 'unchanged_original_reference'} for arm in ARMS for seed in SEEDS]
    context_keys = [key for key in context['complete']['operation_counts'] if key.startswith('context_')]
    context_work_by_arm = {arm: {key: sum(u['operation_counts'][key] for s in SEEDS for u in records[s, arm]['fits'])
        for key in context_keys} for arm in NEW}
    costs = {'new48_native_optimizer_fits': native_cost_summary(helper, context['complete']['results']),
        'reused33_selected_reference_fits_already_paid': native_cost_summary(helper, [records[s, a] for s in SEEDS for a in REFS]),
        'original39_parent_context_not_additive': native_cost_summary(helper, context['old_complete']['results']),
        'one_shared_context_target_preparation': context['complete']['context_preparation_once'],
        'context_operation_counts': {key: context['complete']['operation_counts'][key] for key in context_keys},
        'context_operation_counts_by_arm': context_work_by_arm,
        'calibration_scalar_work': calibration_costs, 'new_owner': context['owner'], 'original_owner': context['old_owner'],
        'scope': 'Target/signature/key construction, affinity normalization, both-order device moments, archive and transfer are charged once and nested in owner wall time. Native acquisition, recorded restoration, selected forward/metrics/pool and scalar work remain separate. Original unrecorded restoration is unavailable, never zero. Original native/scalar work remains already paid; observed timings are not isolated benchmarks.'}
    limits = [
        'All30fresh/15old banks and48fresh/33old selected fit records close before arrays. Exactly75 admitted original OOF endpoints are not refit or temperature replayed;150new attempts are terminal before scores/contrasts/criteria.',
        'Saved fixed K/Q/shuffle/order/support bytes bind the actual qualified tensors and selected source contracts. Signature/teacher/Q/key construction, QR initialization and device moment reductions are not replayed.',
        'One bare no-bias A128to300 is shared by all routes of a bank; independent bodies own separate A parameters. All new arms/bodies at a paired seed bind identical initial A bytes. A and its optimizer state are training costs; serving uses only the native predictor.',
        'The fixed-key objective has literal unit logZ coefficient and actual float32 Q-times-K moments without normalization. Mixture averages four complete dictionary objectives and never averages keys. The projected-peer control keeps both query and peer-key gradients live.',
        'Cross-arm repairs, strict-rival transitions and512-pattern intersections compare separately fitted/stopped/selected procedures and describe specificity without an isolated same-state cause.',
        'Calibration-minus-raw fixes each selected native bank. Positive temperature acquires no member alternatives; raw float32 member flags and coverage remain authority, with precision/tie discrepancies reported.',
        'Primary raw accuracy requires0.2pp versus BASE, both equipped genuineI4s and both equipped singles; every paired seed nonnegative and at least2positive for each reference. Each of five attribution claims separately requires0.1pp with the same seed conditions.',
        'Secondary calibrated NLL versus BASE and both equipped I4s has0.02mean/0.05each-seed harm caps. Secondary failure does not erase primary accuracy success; calibrated accuracy and member competence add no veto.',
        'Stable raw NLL/Brier, member mean/worst/per-seed performance, coverage additions/removals, pooling losses, repairs/harms, current/fixed strict rivals and class uncertainty are retained.',
        'OOF calibration follows VALID native selection on encountered development. Three seeds on one graph/nodes are dependent readouts; a worthwhile signal requires matched original projector-only GNNM comparison and unused whole-pipeline confirmation.',
        'Original admitted raw archives, checkpoint hashes, calibrated endpoints, scores, failures and costs remain unchanged. Missing original restoration or auxiliary accounting stays unavailable, never zero.',
        'Established projection, contrastive objectives, graph signatures and ensemble ingredients confer no primitive novelty, latent-mode recovery, collapse-prevention guarantee, observed collapse diagnosis or acceptance verdict.',
        'One native SAGE backbone; no TEST, checkpoint deserialization, native loader/forward or optional target/math parity suite in this reader.']


    report = {'complete': True, 'backbone': 'SAGE', 'new_banks': 30, 'new_native_fits': 48,
        'new_native_trajectories_per_complete_roster_forward': 102, 'reused_reference_banks': 15, 'reused_reference_fits': 33,
        'joined_banks': 45, 'joined_selected_fit_records': 81, 'seeds': SEEDS, 'roster': ARMS, 'TEST_access': False,
        'calibration_reused_endpoints': 75, 'calibration_new_attempts': 150, 'calibration_complete_terminal_endpoints': 225,
        'comparison_roster': {'candidate_arms': list(NEW), 'reference_arms_for_each_candidate': {a: [b for b in ARMS if b != a] for a in NEW},
            'directional_pairs_per_readout': 140, 'all_fifteen_arm_ordered_pairs_would_be': 210,
            'scope': 'Fresh candidate versus every other joined procedure; all declared primary, attribution and secondary confidence pairs included.'},
        'frozen_policy': policy, 'aggregates': aggregates, 'groups_detail': groups, 'contrasts': contrasts,
        'calibration_folds': folds, 'calibration_records': calibration_records, 'costs': costs,
        'target_support_diagnostics': target_diagnostics, 'source_custody': context,
        'selected_archive_custody': [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS],
        'checkpoint_custody': checkpoints, 'reused_calibrated_archive_custody': context['reused_OOF_archives'],
        'ordered_VALID_hashes': VALID_HASHES, 'analysis_source_sha256': sha(Path(__file__)),
        'immutable_count_math_sha256': MATH_SHA, 'reviewed_kernel_reader_parent_sha256': inputs['kernel_reader_parent_sha256'],
        'reviewed_context_reader_parent_sha256': inputs['context_reader_parent_sha256'],
        'reviewed_affinity_v3_reader_parent_sha256': inputs['affinity_v3_reader_parent_sha256'],
        'query_map_initial_hashes_by_seed': context['query_map_initial_hashes_by_seed'],
        'new_config_sha256': context['new_config_sha256'], 'original_config_sha256': context['original_config_sha256'],
        'native_config_subset_exclusion': ['context_teacher_temperature'],
        'reviewed_original_calibration_reader_sha256': inputs['research_files'][PRIOR_READER],
        'numerical_reader_runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_threads': torch.get_num_threads(), 'calibration_device': 'cpu', 'calibration_dtype': 'float64'},
        'input_bindings': inputs,
        'authority': 'Raw integer decisions use archived native float32 error flags. Stable float64 NLL/Brier and calibrated OOF decisions remain explicit. Original admitted calibrated per-seed scores are retained exactly.',
        'interpretation_limits': limits}
    documents = [('SAGE_POLICY_SUMMARY.json', {'complete': True, 'backbone': 'SAGE', 'new_banks': 30, 'new_native_fits': 48,
        'reused_banks': 15, 'reused_selected_fits': 33, 'joined_banks': 45, 'joined_selected_fit_records': 81,
        'frozen_policy': policy, 'all_terminal_endpoints': 225, 'reused_endpoints': 75, 'new_endpoint_attempts': 150, 'calibration_costs': calibration_costs})]
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
            documents += chunk_documents('SAGE_' + category, partition_export_rows(values))
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
    summary = {'complete': True, 'backbone': 'SAGE', 'new_banks': 30, 'new_native_fits': 48,
        'original_reference_banks': 15, 'original_selected_fits': 33, 'joined_banks': 45, 'joined_selected_fit_records': 81,
        'TEST_access': False, 'frozen_policy': policy, 'calibration_folds': folds, 'calibration_costs': calibration_costs,
        'raw_authority_preserved': True, 'original_raw_and_calibrated_scores_preserved': True,
        'all225_endpoint_records_terminal_before_interpretation': True, 'reused_endpoints': 75, 'new_scalar_fit_attempts': 150,
        'post_selection_OOF_scope': 'Encountered development, not unused whole-pipeline confirmation.',
        'repair_scope': 'Cross-arm full procedures; calibration-only effects at fixed native selected state.',
        'target_mode_order_custody_preserved': True, 'target_construction_replayed': False,
        'selected_scalar_target_source_contract_bound': True,
        'primary_accuracy_separate_from_attribution_and_secondary_confidence': True,
        'directional_fresh_candidate_vs_other_procedure_contrasts_per_readout': 140,
        'contrast_roster_scope': {'candidates': list(NEW), 'references_for_each_candidate': {a: [b for b in ARMS if b != a] for a in NEW},
            'count': 140, 'scope': 'Ten fresh candidates each versus fourteen other joined procedures; original-candidate pairs are outside this declared subset. This is not the full210 ordered fifteen-arm pair set.'},
        'all_declared_primary_attribution_and_confidence_pairs_included': True,
        'exact_error_cohort_patterns_per_seed': 512,
        'query_map_training_and_serving_costs_separate': True,
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
        'primary_accuracy_screen_pass': policy['primary_accuracy_screen_pass'],
        'attribution_pass_by_claim': {claim: row['pass'] for claim, row in policy['attribution_checks'].items()},
        'secondary_confidence_screen_pass': policy['secondary_confidence_screen_pass'],
        'reused_scalar_endpoints': 75, 'new_scalar_fit_attempts': 150,
        'all_terminal_endpoints': 225, 'partitions': len(receipts) - 1}))



if __name__ == '__main__':
    main()
