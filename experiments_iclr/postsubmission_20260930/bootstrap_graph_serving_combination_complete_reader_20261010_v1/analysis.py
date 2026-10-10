"""Post-closure complete66-bank serving reader; no fit, forward or TEST."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT_DIR = 'bootstrap_graph_serving_combination_root_20261010_v1'
GRAPH_READER = 'graph_reliability_complete_reader_20261010_v1/analysis.py'
GRAPH_SHA = 'b91ea46a92a0084079a271c58ee70b4540df54302f6890c6a6eac67e1715c66f'
MATH_READER = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
FREEZE_SHA = '8f4076c5a433dbe2ea85fdf2b3caaa59df83ea93bcde030277b12b7cbc745f9b'
SUPPORT_SHA = '23eddb7ab73107034f2e04db1e07969f912546d52ff35d843a13a28ed20262c3'
CHECKPOINT_SHA = '6a172d98695de173712d3b2204d3dd4d0c1a5fdd752be965ecaa8f1fde763aa5'
DECISION_SHA = 'f98fbffe734c3b2583f41620210bae5f11bbd8884c74fa50b97e6e1e109649b7'
BOOTSTRAP_REPORT = 'private_class_balanced_bootstrap_pilot_decision_20261010_v1/JOINED_ANALYSIS.json'
BOOTSTRAP_REPORT_SHA = '027e15420a46f193362405fc2666b73ff51182da4a36d5c38438e479414eb80d'
BACKBONES, SEEDS = ('GAT', 'SAGE'), (7301, 7403, 7507)
UNWEIGHTED = ('shared4_coherent', 'shared4_paired_graph', 'shared4_rank1_lora_graph')
BOOTSTRAP = tuple(a + '_bootstrap' for a in UNWEIGHTED)
REFERENCES = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
              'factorized_allmap_genuine_I4', 'shared4_unchanged')
ARMS = REFERENCES + UNWEIGHTED + BOOTSTRAP
RULES = ('temperature_global', 'temperature_member', 'reliability_self', 'linear_stacking', 'reliability_graph')
NATIVE, UNIFORM, GRAPH, SELF = 'native_archived', 'uniform_FP64', 'reliability_graph', 'reliability_self'
PREDICTORS = (NATIVE, UNIFORM) + RULES


def sha(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
        return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def load(root, relative, digest, name):
    path = root / relative
    require(sha(path) == digest, 'Immutable readout dependency required: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admitted(root, helper, graph):
    here = root / ROOT_DIR
    require(sha(here / 'FREEZE.json') == FREEZE_SHA and sha(here / 'FROZEN_SUPPORT.json') == SUPPORT_SHA
            and sha(here / 'CHECKPOINT_BINDINGS.json') == CHECKPOINT_SHA and sha(here / 'DECISION.md') == DECISION_SHA,
            'Frozen combination decision/support/source identity required')
    freeze, support = read(here / 'FREEZE.json'), read(here / 'FROZEN_SUPPORT.json')
    for bound in freeze['bound_files']:
        require(sha(graph.scoped(root / bound['path'], root)) == bound['sha256'], 'Execution binding changed')
    require(freeze['new_banks'] == 36 and freeze['reused_reference_banks'] == 30 and freeze['fit_calls'] == 900
            and freeze['native_forward_calls'] == 36 and freeze['member_trajectories'] == 144
            and freeze['new_GNN_training'] == 0 and freeze['TEST_access'] is False,
            'Fixed36/900 closure policy required')
    require(support['schema'] == 'bootstrap-graph-serving-combination-support-v1'
            and support['backbones'] == list(BACKBONES) and support['seeds'] == list(SEEDS)
            and support['unweighted_arms'] == list(UNWEIGHTED) and support['bootstrap_arms'] == list(BOOTSTRAP)
            and support['new_banks'] == 36 and support['nonduplicate_fusion_fit_calls'] == 900
            and support['fusion']['rules'] == list(RULES) and support['fusion']['folds'] == 5
            and support['fusion']['updates'] == 500 and support['fusion']['fold_seed'] == 11709
            and support['TEST_access'] is False and support['new_GNN_training'] is False
            and support['original_model_predictions_changed'] is False,
            'Exact native/fusion roster and policy required')
    owner = helper.closed_owner(here / 'OWNER_END.json')
    study_dir = here / 'actual_study_v1'
    study_path, export_path = study_dir / 'COMPLETE_STUDY.json', study_dir / 'COMPLETE_EXPORT.json'
    require(sha(study_path) == owner['complete_sha256'], 'Closed owner binds full study before metrics')
    # The frozen owner admits only complete36/900. No array or numerical score
    # is read until all new and original closure/hash/endpoint checks finish.
    study, export, folds = read(study_path), read(export_path), read(study_dir / 'FOLDS.json')
    require(study['schema'] == 'bootstrap-graph-serving-development-v1' and study['complete_roster']
            and len(study['banks']) == 36 and len(study['reused_reference_banks']) == 30
            and study['fit_calls'] == 900 and study['maximum_small_head_updates'] == 450000
            and study['support_sha256'] == SUPPORT_SHA and study['checkpoint_bindings_sha256'] == CHECKPOINT_SHA
            and study['export_sha256'] == sha(export_path) and study['TEST_access'] is False
            and study['new_base_fits'] == study['new_reference_refits'] == 0
            and study['changed_base_member_predictions'] is False
            and study['no_final_refit_or_held_fold_checkpoint_selection'] is True,
            'Full fixed complete descriptor required')
    require(export['schema'] == 'bootstrap-graph-serving-neighbors-v1' and export['complete']
            and len(export['banks']) == 36 and export['support_sha256'] == SUPPORT_SHA
            and export['checkpoint_bindings_sha256'] == CHECKPOINT_SHA
            and export['selected_model_forward_calls'] == study['new_selected_model_forward_calls'] == 36
            and export['native_fullgraph_member_trajectories'] == study['new_native_member_trajectories'] == 144
            and export['new_base_fits'] == 0 and export['TEST_access'] is False
            and export['labels_exported'] is False and export['fullnode_probabilities_persisted'] is False,
            'Complete one-pass export accounting required')
    require(folds['seed'] == 11709 and folds['count'] == 5 and folds['counts'] == [1055, 1055, 1055, 1055, 1054]
            and folds['ids_sha256'] == support['ordered_VALID_hashes']['ids']
            and study['ordered_VALID_hashes'] == support['ordered_VALID_hashes']
            and folds['fold_assignment_sha256'] == study['fold_assignment_sha256']
            == support['reuse_original_graph_study']['fold_assignment_sha256'], 'Exact original common folds/roles')
    require([(c['backbone'], c['kind']) for c in support['contexts']]
            == [(b, k) for b in BACKBONES for k in ('unweighted', 'bootstrap')], 'All four closed native contexts')
    expected = {b['key']: b for c in support['contexts'] for b in c['banks']}
    index, exports = {r['key']: r for r in study['banks']}, {r['key']: r for r in export['banks']}
    roster = {f'{b}_{a}_seed{s}' for b in BACKBONES for a in UNWEIGHTED + BOOTSTRAP for s in SEEDS}
    require(len(expected) == len(index) == len(exports) == 36
            and set(expected) == set(index) == set(exports) == roster, 'Exact36 unique bank identities')
    for context in support['contexts']:
        folder = graph.scoped(context['family_root'], root)
        for bound in context['frozen_inputs']:
            artifact = (root.parent.parent / bound['path']).resolve()
            require(artifact.is_relative_to(root.parent.parent) and sha(artifact) == bound['sha256'],
                    'Original frozen native input changed')
        require(sha(folder / 'COMPLETE_FAMILY.json') == context['complete_sha256']
                and sha(context['owner_end_path']) == context['owner_end_sha256']
                and sha(context['config']) == context['config_sha256']
                and sha(context['family_source']) == context['family_source_sha256'], 'Original selected native family identity')
        helper.closed_owner(context['owner_end_path'])
        header = helper.completion_header(folder / 'COMPLETE_FAMILY.json')
        count = 12 if context['kind'] == 'unweighted' else 9
        require(header['complete'] and header['groups'] == header['expected_groups'] == count
                and header['fit_units'] == header['expected_fit_units'] == count
                and header['TEST_access'] is False and header['backbone'] == context['backbone'], 'Full12/9 parent completeness')
    checkpoints = read(here / 'CHECKPOINT_BINDINGS.json')
    bindings = {r['path']: r for r in checkpoints['checkpoints']}
    require(checkpoints['support_sha256'] == SUPPORT_SHA and len(bindings) == len(checkpoints['checkpoints']) == 36
            and set(bindings) == {b['selected_state'] for b in expected.values()}, 'Complete selected-state custody')
    fits, failed = 0, []
    for key in sorted(roster):
        bank, record, cached = expected[key], index[key], exports[key]
        require((record['backbone'], record['arm'], record['seed'], record['members'])
                == (bank['backbone'], bank['arm'], bank['seed'], 4)
                and record['changed_base_member_predictions'] is False, 'Native bank identity')
        require(sha(graph.scoped(bank['archive'], root)) == bank['archive_sha256']
                == record['archive_sha256'] == cached['archive_sha256'] and cached['archive'] == bank['archive']
                and sha(graph.scoped(cached['cache'], root)) == cached['cache_sha256'] == record['neighbor_cache_sha256']
                and cached['discrepancies']['within_fixed_tolerance'] is True
                and cached['discrepancies']['max_abs_logit_difference'] <= 1e-4, 'Original archive/cache custody')
        state = bindings[bank['selected_state']]
        require(state['key'] == key and state['selected_step'] == bank['selected_step']
                and len(cached['selected_states']) == 1
                and cached['selected_states'][0]['path'] == state['path']
                and cached['selected_states'][0]['sha256'] == state['sha256'], 'Selected export identity unchanged')
        oof_path = study_dir / (key + '_OOF_VALID.npz')
        require(graph.scoped(record['VALID_OOF_path'], root) == oof_path
                and sha(oof_path) == record['VALID_OOF_sha256'] and set(record['rules']) == set(RULES), 'OOF/rule hash identity')
        for rule in RULES:
            endpoints = record['rules'][rule]['fits']
            require(len(endpoints) == 5 and [e['fold'] for e in endpoints] == list(range(5)), 'Every fixed endpoint retained')
            fits += len(endpoints)
            for endpoint in endpoints:
                if endpoint['status'] == 'failed_retained':
                    failed.append({'key': key, 'rule': rule, **endpoint})
                else:
                    fold = endpoint['fold']
                    require(endpoint['status'] == 'finite_fixed_endpoint' and endpoint['updates'] == 500
                            and endpoint['fit_nodes'] == 5274 - folds['counts'][fold]
                            and endpoint['held_nodes'] == folds['counts'][fold]
                            and endpoint['selected_endpoint'] == 'fixed_final_update_no_heldout_selector', 'Fixed final endpoint')
    failure_index = lambda values: {(v['key'], v['rule'], v['fold']): v for v in values}
    require(fits == 900 and len(failed) == len(study['failures']) == len(failure_index(failed))
            and failure_index(failed) == failure_index(study['failures'])
            and study['all_fixed_endpoints_finite'] is (not failed), 'Full900 retained failure accounting')
    prior = graph.admitted(root, helper)  # Reuse immutable original45/855 admission and historical costs.
    reuse = support['reuse_original_graph_study']
    require(prior['study_sha256'] == reuse['sha256'] and prior['export_sha256'] == reuse['export_sha256']
            and prior['study']['ordered_VALID_hashes'] == study['ordered_VALID_hashes']
            and prior['study']['fold_assignment_sha256'] == study['fold_assignment_sha256']
            and prior['study']['all_fixed_endpoints_finite'] and reuse['new_refits'] == 0, 'Exact prior study reuse')
    reference_keys = {f'{b}_{a}_seed{s}' for b in BACKBONES for a in REFERENCES for s in SEEDS}
    reference_index = {r['key']: r for r in study['reused_reference_banks']}
    require(len(reference_index) == 30 and set(reference_index) == reference_keys
            and all(reference_index[k] == prior['index'][k] for k in reference_keys), 'Original30 OOF records unchanged')
    bootstrap_path = root / BOOTSTRAP_REPORT
    require(sha(bootstrap_path) == BOOTSTRAP_REPORT_SHA, 'Complete preexisting bootstrap report identity')
    bootstrap = read(bootstrap_path)
    require(bootstrap['complete'] and bootstrap['joined_groups'] == 66 and bootstrap['new_acquisition_units'] == 18
            and bootstrap['TEST_access'] is False, 'Whole closed bootstrap readout required')
    new_context = {'study_dir': study_dir, 'support': support, 'study': study, 'folds': folds,
                   'expected': expected, 'index': index}
    reference_context = {**prior, 'expected': {k: prior['expected'][k] for k in reference_keys}, 'index': reference_index}
    return {'here': here, 'support': support, 'owner': owner, 'study': study, 'export': export, 'folds': folds,
            'new_context': new_context, 'reference_context': reference_context, 'prior': prior, 'bootstrap': bootstrap,
            'study_sha256': sha(study_path), 'export_sha256': sha(export_path)}


def contrast(helper, graph, scores, backbone, candidate, reference, labels):
    value = graph.contrast(helper, scores, backbone, candidate, reference, labels)
    if value['available']:
        local = {(seed, 'candidate'): scores[backbone, candidate[0], seed, candidate[1]] for seed in SEEDS}
        local.update({(seed, 'reference'): scores[backbone, reference[0], seed, reference[1]] for seed in SEEDS})
        member = helper.contrast(local, 'candidate', 'reference', labels)
        value.update(member_quality_deltas={k: v for k, v in member['quality_deltas'].items() if 'member' in k},
                     members=member['members'], member_pairing=member['member_pairing'],
                     class_member_deltas=[{k: v for k, v in row.items() if k == 'class' or 'member' in k} for row in member['classes']],
                     summed_three_seed_readout_diagnostics=member['summed_three_seed_readout_diagnostics'])
    return value


def combination(helper, np, scores, strict, backbone, unweighted, bootstrap, labels):
    roles = {'C': (unweighted, NATIVE), 'A': (unweighted, GRAPH),
             'B': (bootstrap, NATIVE), 'A+B': (bootstrap, GRAPH)}
    missing = [seed for seed in SEEDS if any(scores[backbone, arm, seed, rule] is None for arm, rule in roles.values())]
    if missing:
        return {'available': False, 'roles': roles, 'failed_seeds_retained': missing,
                'scope': 'No partial interaction, repair-survival cohort or surviving-seed estimate.'}
    per_seed = []
    for seed in SEEDS:
        values = {name: scores[backbone, arm, seed, rule] for name, (arm, rule) in roles.items()}
        p = {name: v['P'] for name, v in values.items()}
        ra, rb, rab = (~p['C'] & p[name] for name in ('A', 'B', 'A+B'))
        ha, hb, hab = (p['C'] & ~p[name] for name in ('A', 'B', 'A+B'))
        new = values['B']['V'] & ~values['C']['V']
        removed = values['C']['V'] & ~values['B']['V']
        require(np.array_equal(values['A']['V'], values['C']['V'])
                and np.array_equal(values['A+B']['V'], values['B']['V']), 'Serving cannot acquire member alternatives')
        sc, sb = strict[backbone, unweighted, seed], strict[backbone, bootstrap, seed]
        cohorts = {'A_repairs': ra, 'B_repairs': rb, 'AB_repairs': rab, 'A_harms': ha, 'B_harms': hb, 'AB_harms': hab,
                   'A_only_repairs': ra & ~rb, 'B_only_repairs': rb & ~ra, 'shared_A_B_repairs': ra & rb,
                   'A_repairs_survive_AB': ra & p['A+B'], 'A_repairs_become_AB_harms_versus_A': ra & ~p['A+B'],
                   'B_repairs_survive_AB': rb & p['A+B'], 'B_repairs_become_AB_harms_versus_B': rb & ~p['A+B'],
                   'A_only_repairs_survive_AB': ra & ~rb & p['A+B'], 'B_only_repairs_survive_AB': rb & ~ra & p['A+B'],
                   'shared_A_B_repairs_survive_AB': ra & rb & p['A+B'], 'new_AB_repairs': rab & ~ra & ~rb,
                   'A_harms_survive_AB': ha & ~p['A+B'], 'B_harms_survive_AB': hb & ~p['A+B'],
                   'A_harms_recovered_AB': ha & p['A+B'], 'B_harms_recovered_AB': hb & p['A+B'],
                   'new_AB_harms_where_A_and_B_both_correct': hab & ~ha & ~hb,
                   'bootstrap_new_coverage': new, 'bootstrap_removed_coverage': removed,
                   'new_coverage_served_B_native': new & p['B'], 'new_coverage_lost_B_native': new & ~p['B'],
                   'new_coverage_served_AB_graph': new & p['A+B'], 'new_coverage_lost_AB_graph': new & ~p['A+B'],
                   'new_coverage_lost_B_native_rescued_AB': new & ~p['B'] & p['A+B'],
                   'new_coverage_served_B_native_lost_AB': new & p['B'] & ~p['A+B'],
                   'previously_covered_B_pooling_loss_rescued_AB': values['B']['V'] & ~new & ~p['B'] & p['A+B'],
                   'B_served_alternative_lost_AB': values['B']['V'] & p['B'] & ~p['A+B'],
                   'C_common_strict_rival': sc, 'B_common_strict_rival': sb,
                   'C_common_strict_rival_acquires_B_correct_member': sc & new,
                   'C_common_strict_rival_acquired_alternative_served_AB': sc & new & p['A+B'],
                   'B_common_strict_rival_corrected_AB': sb & p['A+B'],
                   'AB_aggregation_only_correct': ~values['B']['V'] & p['A+B']}
        counts = {key: int(mask.sum()) for key, mask in cohorts.items()}
        interaction = sum(sign * values[name]['counts']['pooled_correct'] for name, sign in [('A+B', 1), ('A', -1), ('B', -1), ('C', 1)])
        require(interaction == counts['AB_repairs'] - counts['AB_harms'] - counts['A_repairs'] + counts['A_harms'] - counts['B_repairs'] + counts['B_harms'],
                'Interaction repair/harm identity')
        require(counts['AB_repairs'] == sum(counts[k] for k in ('A_only_repairs_survive_AB', 'B_only_repairs_survive_AB', 'shared_A_B_repairs_survive_AB', 'new_AB_repairs')),
                'Exact surviving repair partition')
        per_seed.append({'seed': seed, 'interaction_correct_count': interaction, 'cohorts': counts,
                         'classes': [{'class': k, 'cohorts': {key: int((mask & (labels == k)).sum()) for key, mask in cohorts.items()}} for k in range(10)]})
    return {'available': True, 'roles': roles,
            'accuracy_interaction_AplusB_minus_A_minus_B_plus_C_pp': helper.paired(100 * row['interaction_correct_count'] / 5274 for row in per_seed),
            'per_seed_repair_survival_and_acquisition_serving': per_seed,
            'summed_three_seed_readout_cohorts': {key: sum(row['cohorts'][key] for row in per_seed) for key in per_seed[0]['cohorts']},
            'scope': 'Exact same-node development cohorts; a lost ingredient repair is a harm versus that ingredient. Positive interaction cannot rescue worse final pooled quality.'}


def block_report(helper, graph, np, scores, strict, backbone, unweighted, bootstrap, labels, old_flags):
    pairs = {'A_minus_C': ((unweighted, GRAPH), (unweighted, NATIVE)),
             'B_minus_C': ((bootstrap, NATIVE), (unweighted, NATIVE)),
             'AB_minus_C': ((bootstrap, GRAPH), (unweighted, NATIVE)),
             'AB_minus_A': ((bootstrap, GRAPH), (unweighted, GRAPH)),
             'AB_minus_B': ((bootstrap, GRAPH), (bootstrap, NATIVE))}
    cells = {name: contrast(helper, graph, scores, backbone, a, b, labels) for name, (a, b) in pairs.items()}
    controls = [(bootstrap, NATIVE)] + [(bootstrap, r) for r in RULES if r != GRAPH]
    controls += [(unweighted, GRAPH), (REFERENCES[1], GRAPH), (REFERENCES[3], GRAPH)]
    primary = {arm + '/' + rule: contrast(helper, graph, scores, backbone, (bootstrap, GRAPH), (arm, rule), labels) for arm, rule in controls}
    accuracy, protection = {}, {}
    for name, value in primary.items():
        threshold = .2 if name == bootstrap + '/' + NATIVE else .1
        accuracy[name] = {'available_all_three_seeds': value['available'], 'mean_required_accuracy_pp': threshold,
                          'positive_mean_and_threshold': value['available'] and value['accuracy_pp']['mean'] >= threshold,
                          'all_seed_nonnegative': value['available'] and value['accuracy_pp']['nonnegative_seed_count'] == 3,
                          'at_least_two_positive_seeds': value['available'] and value['accuracy_pp']['positive_seed_count'] >= 2}
        if name != bootstrap + '/' + NATIVE:
            protection[name] = {'available_all_three_seeds': value['available'],
                                'mean_nll_deterioration_at_most_0_02': value['available'] and value['nll']['mean'] <= .02,
                                'each_seed_nll_deterioration_at_most_0_05': value['available'] and value['nll']['max'] <= .05}
    native = cells['B_minus_C']
    member, nll = native['member_quality_deltas']['mean_member_accuracy_pct'], native['nll']
    original_protection = {'mean_member_vs_own_counterpart_nonnegative': native['summed_three_seed_readout_diagnostics']['member_correct_delta'] >= 0,
                           'each_seed_member_delta_at_least_minus_0_1pp': member['min'] >= -.1,
                           'mean_nll_deterioration_at_most_0_02': nll['mean'] <= .02,
                           'each_seed_nll_deterioration_at_most_0_05': nll['max'] <= .05}
    require(original_protection == old_flags['quality_protection_criteria'], 'Static original bootstrap member/proper-risk flags changed')
    return {'unweighted_arm': unweighted, 'bootstrap_arm': bootstrap,
            'C_A_B_AB_cell_contrasts': cells, 'primary_AB_graph_contrasts': primary,
            'capable_reference_context_contrasts': {a + '/' + r: contrast(helper, graph, scores, backbone, (bootstrap, GRAPH), (a, r), labels)
                                                    for a in REFERENCES for r in (NATIVE, GRAPH)},
            'accuracy_transfer_criteria': accuracy, 'accuracy_transfer_within_backbone': all(all(v[k] for k in ('available_all_three_seeds', 'positive_mean_and_threshold', 'all_seed_nonnegative', 'at_least_two_positive_seeds')) for v in accuracy.values()),
            'proper_risk_protection_criteria': protection, 'proper_risk_protection_within_backbone': all(all(v.values()) for v in protection.values()),
            'original_bootstrap_frozen_flags_unchanged': old_flags, 'original_protection_recomputed_unchanged': original_protection,
            'member_flag_is_separate_from_final_pooled_quality': True,
            'combination': combination(helper, np, scores, strict, backbone, unweighted, bootstrap, labels)}


def write(path, value, compact=False, limit=None):
    encoded = json.dumps(value, separators=(',', ':') if compact else None,
                         indent=None if compact else 2, allow_nan=False).encode() + b'\n'
    require(limit is None or len(encoded) < limit, 'Compact partition exceeds declared transfer size')
    with Path(path).open('xb') as handle:
        handle.write(encoded)
    return {'path': str(path), 'bytes': len(encoded), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root, path = args.research_root.resolve(), args.report.resolve()
    require(path.is_relative_to(root) and not path.exists(), 'Fresh in-scope report path')
    manifest = read(Path(__file__).with_name('SOURCE.json'))
    require(manifest['files']['analysis.py'] == sha(Path(__file__)), 'Sealed reader source identity')
    helper = load(root, MATH_READER, MATH_SHA, 'bootstrap_serving_paired_math')
    graph = load(root, GRAPH_READER, GRAPH_SHA, 'bootstrap_serving_graph_readout')
    context = admitted(root, helper, graph)
    import numpy as np
    helper.np = np
    fresh, fresh_oof, ids, labels = graph.payloads(context['new_context'], np)
    old, old_oof, old_ids, old_labels = graph.payloads(context['reference_context'], np)
    require(np.array_equal(ids, old_ids) and np.array_equal(labels, old_labels), 'All66 common role identities before scores')
    arrays, oof = {**fresh, **old}, {**fresh_oof, **old_oof}
    require(len(arrays) == len(oof) == 66, 'All66 selected arrays before scoring')
    all_index = {**context['new_context']['index'], **context['reference_context']['index']}
    scores, detail = graph.bank_scores(helper, np, {'index': all_index}, arrays, oof, labels)
    strict = {}
    for key, a in arrays.items():
        raw = a['raw_logits'].astype(np.float64)
        probability = np.exp(raw - helper.logsumexp(raw, -1)[..., None])
        truth = np.take_along_axis(probability, labels[None, :, None], 2)
        row = all_index[key]
        strict[row['backbone'], row['arm'], row['seed']] = (probability > truth).all(0).any(1)
    families = {}
    for backbone in BACKBONES:
        blocks = {u: block_report(helper, graph, np, scores, strict, backbone, u, b, labels,
                                  context['bootstrap']['families'][backbone]['frozen_flags'][b]) for u, b in zip(UNWEIGHTED, BOOTSTRAP)}
        families[backbone] = {'blocks': blocks,
            'graph_minus_identical_P_equals_I_every_bank': {a: contrast(helper, graph, scores, backbone, (a, GRAPH), (a, SELF), labels) for a in ARMS},
            'each_rule_minus_original_native_every_bank': {a + '/' + r: contrast(helper, graph, scores, backbone, (a, r), (a, NATIVE), labels) for a in ARMS for r in RULES + (UNIFORM,)},
            'AB_graph_matched_capacity_pair_minus_LoRA': contrast(helper, graph, scores, backbone, (BOOTSTRAP[1], GRAPH), (BOOTSTRAP[2], GRAPH), labels),
            'all_block_accuracy_transfer': all(b['accuracy_transfer_within_backbone'] for b in blocks.values()),
            'all_block_proper_risk_protection': all(b['proper_risk_protection_within_backbone'] for b in blocks.values())}
    accuracy = all(families[b]['all_block_accuracy_transfer'] for b in BACKBONES)
    protection = all(families[b]['all_block_proper_risk_protection'] for b in BACKBONES)
    finite = context['study']['all_fixed_endpoints_finite']
    report = {'complete': True, 'banks': 66, 'new_banks': 36, 'reused_reference_banks': 30,
              'new_fit_calls': 900, 'prior_fit_calls_with_original_history_retained': 855,
              'backbones': BACKBONES, 'seeds': SEEDS, 'arms': ARMS, 'rules': PREDICTORS, 'TEST_access': False,
              'analysis_source_sha256': sha(Path(__file__)), 'reader_manifest_sha256': sha(Path(__file__).with_name('SOURCE.json')),
              'immutable_math_reader_sha256': MATH_SHA, 'immutable_graph_reader_sha256': GRAPH_SHA,
              'decision_sha256': DECISION_SHA, 'freeze_sha256': FREEZE_SHA, 'support_sha256': SUPPORT_SHA,
              'checkpoint_bindings_sha256': CHECKPOINT_SHA, 'owner_end_sha256': sha(context['here'] / 'OWNER_END.json'),
              'complete_study_sha256': context['study_sha256'], 'complete_export_sha256': context['export_sha256'],
              'original_bootstrap_report_sha256': BOOTSTRAP_REPORT_SHA,
              'folds': context['folds'], 'ordered_VALID_hashes': context['support']['ordered_VALID_hashes'],
              'banks_detail': detail, 'families': families, 'all_fixed_endpoints_finite': finite,
              'failed_endpoints_retained': context['study']['failures'],
              'frozen_decision': {'whole_roster_accuracy_transfer': accuracy, 'whole_roster_pooled_NLL_protection': protection,
                                  'whole_roster_pooled_quality_development_clue': finite and accuracy and protection,
                                  'original_member_protection_retained_separately': True},
              'costs': {'owner': context['owner'], 'new_head_fitting_seconds': context['study']['head_fitting_seconds'],
                        'new_fit_calls_including_failures': 900, 'new_declared_updates': 450000,
                        'new_export_seconds': context['export']['seconds'], 'new_selected_forwards': 36,
                        'new_native_member_trajectories': 144, 'new_GNN_acquisitions': 0, 'reference_refits': 0,
                        'original_complete855_study_seconds': context['prior']['study']['seconds'],
                        'original_complete855_fit_calls': 855, 'original_declared_updates': 427500,
                        'original_complete855_fit_and_rule_records': [graph.public_record(r) for r in context['prior']['study']['banks']],
                        'original_export_records_and_costs': context['prior']['export'],
                        'original_failed_prefix_and_costs': context['prior']['support']['prefix_reuse'],
                        'new_export_records_and_costs': context['export'], 'all_fit_and_rule_costs_preserved_in_bank_records': True},
              'native_authority': 'Original float32 archived decisions/probabilities; stable raw-logit float64 NLL; uniform_FP64 separate.',
              'strict_rival_definition': 'A wrong class strictly outranks truth in every original member. No epsilon; graph/self scalar convex mixtures cannot reverse it in exact arithmetic. All-member-wrong is insufficient; floating/tie caveats remain.',
              'coverage_identity': 'pooled correct = any-member coverage - lost correct alternatives + aggregation-only correct',
              'interpretation_limits': ['Encountered development after full-VALID base checkpoint selection; aggregator OOF is not whole-pipeline cross-fitting or unused confirmation.',
                  'Three seed intervals describe optimization repeats on one connected graph; node readouts are not independent graph replications.',
                  'A failed endpoint has no partial metric or surviving-seed interval; whole-roster promotion requires every declared endpoint finite.',
                  'Member protection failure stays recorded and does not forbid a true pooled accuracy/NLL win; graph serving cannot retroactively change the original flag.',
                  'Positive interaction alone cannot rescue worse final accuracy or proper risk. Both equally graph-processed genuine I4 references remain decisive.',
                  'No source tolerance, engineering check, observed overlap cost, novelty assertion or TEST result establishes accuracy utility.',
                  'Close the fixed diagnostic after complete readout; a broader paper claim requires a frozen whole pipeline and unused confirmation.']}
    path.parent.mkdir(parents=True, exist_ok=True)
    full = write(path, report)
    partitions = []
    for backbone in BACKBONES:
        for arm in UNWEIGHTED:
            selected_arms = (arm, arm + '_bootstrap')
            value = {'backbone': backbone, 'block': families[backbone]['blocks'][arm],
                     'banks_detail': [r for r in detail if r['backbone'] == backbone and r['arm'] in selected_arms],
                     'graph_minus_self': {a: families[backbone]['graph_minus_identical_P_equals_I_every_bank'][a] for a in selected_arms},
                     'each_rule_minus_native': {k: v for k, v in families[backbone]['each_rule_minus_original_native_every_bank'].items() if k.split('/')[0] in selected_arms}}
            partitions.append(write(path.parent / (backbone + '_' + arm + '_COMPLETE_RESULTS.json'), value, True, 2000000))
        for arm in REFERENCES:
            value = {'backbone': backbone, 'reference_arm': arm,
                     'banks_detail': [r for r in detail if r['backbone'] == backbone and r['arm'] == arm],
                     'graph_minus_self': families[backbone]['graph_minus_identical_P_equals_I_every_bank'][arm],
                     'each_rule_minus_native': {k: v for k, v in families[backbone]['each_rule_minus_original_native_every_bank'].items() if k.split('/')[0] == arm}}
            partitions.append(write(path.parent / (backbone + '_REFERENCE_' + arm + '_COMPLETE_RESULTS.json'), value, True, 2000000))
    summary = {k: v for k, v in report.items() if k not in ('banks_detail', 'families', 'costs')}
    summary.update(full_report=full, partitions=partitions,
                   scalar_bank_index=[{'key': r['key'], 'backbone': r['backbone'], 'arm': r['arm'], 'seed': r['seed'],
                                       'predictions': {rule: {'available': value['available'],
                                                              **({k: value[k] for k in ('quality', 'counts')} if value['available'] else {'status': value['status']})}
                                                       for rule, value in r['predictions'].items()}} for r in detail],
                   per_block_frozen_flags={b: {a: {k: v for k, v in block.items() if k in ('accuracy_transfer_criteria', 'accuracy_transfer_within_backbone', 'proper_risk_protection_criteria', 'proper_risk_protection_within_backbone', 'original_bootstrap_frozen_flags_unchanged')}
                                               for a, block in families[b]['blocks'].items()} for b in BACKBONES},
                   matched_capacity_contrast={b: families[b]['AB_graph_matched_capacity_pair_minus_LoRA'] for b in BACKBONES},
                   cost_scalars={k: v for k, v in report['costs'].items() if k not in ('original_export_records_and_costs', 'original_failed_prefix_and_costs', 'new_export_records_and_costs', 'original_complete855_fit_and_rule_records')})
    summary_record = write(path.parent / 'COMPLETE_ANALYSIS_SUMMARY.json', summary, True, 2000000)
    print(json.dumps({'complete': True, 'summary': summary_record, 'full_report': full,
                      'partitions': partitions, 'frozen_decision': report['frozen_decision']}))


if __name__ == '__main__':
    main()
