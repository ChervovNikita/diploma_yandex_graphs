"""Independent aggregate arithmetic only; never follow execution descriptors."""
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RESULT = BASE / 'amazon_polynormer_logits_graph_moment_cpu_root_preparation_20261005_v2/development_observation_20261005T174207Z/amazon_polynormer_logits_graph_moment_retrospective_cpu_execution_root_20261005_v2/RESULT.json'
PROTO = BASE / 'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2'
SOURCE = BASE / 'amazon_polynormer_logits_graph_moment_source_preparation_20261005_v3'
ROOT1 = BASE / 'amazon_polynormer_logits_graph_moment_complete_outcome_root_20261005_v1'
ROOT2 = BASE / 'amazon_polynormer_logits_graph_moment_complete_outcome_root_20261005_v2'
BANKS = ['gnnm_boundary_4', 'independent_author_4_same_width', 'single_author']
METRICS = ['Brier', 'NLL', 'accuracy']
TOL = 1e-12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def emit(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


inputs = [RESULT, PROTO / 'PROTOCOL.json', PROTO / 'REFERENCE_CORRECTION.json']
inputs += [SOURCE / n for n in ['MANIFEST.json', 'SEAL.json', 'study.py', 'numerical.py', 'custody.py']]
inputs += [folder / n for folder in [ROOT1, ROOT2] for n in ['MANIFEST.json', 'SEAL.json', 'adopt.py', 'REPORT.md', 'SUMMARY.json', 'SOURCE_BINDINGS.json']]
inputs += [ROOT2 / 'REPAIR.json']
bindings = [{"path": str(p.relative_to(BASE)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in inputs]
initial_hashes = {str(p): digest(p) for p in inputs}
expected_hashes = {
    RESULT: '67db49a7a9604d0cabcf024b65d1d5a5188bc79dbc043ef3e133a9248ba6b5cb',
    PROTO / 'PROTOCOL.json': '5d667f15640102995fa998ca6932c53c5b71554273c84d4938a331c0a4eae7ba',
    PROTO / 'REFERENCE_CORRECTION.json': 'd5ea58d79f5183c36fc199bbcb895c731091b1c9f2b75385dfb1a551bf1e61c9',
    SOURCE / 'MANIFEST.json': '14d2a075bf48ef54e045f580aaf05822e2f88aceb49248c6d57d483f0e46e4ac',
    SOURCE / 'SEAL.json': 'c4b45cf05e3ef2b22de003cc0df57695d56564badc1b0f2ab5a9872451984298',
    SOURCE / 'numerical.py': '8bbf79221b7a397a6a49c899d0d67bb07c7a13c0e91c9f75a7f47f120677ec4d',
    ROOT1 / 'MANIFEST.json': 'ae374856dd2f30628597c081ca610c8c70e06ae9b43e72c71f06b4e88d5f5dc3',
    ROOT2 / 'MANIFEST.json': '681c3614e12ddc1aa0d0602ba0f6e131dd92801adf42639bf75982bb93499d73',
}
for path, expected in expected_hashes.items():
    assert digest(path) == expected, str(path)
for folder in [ROOT1, ROOT2]:
    manifest = read(folder / 'MANIFEST.json')
    assert read(folder / 'SEAL.json')['manifest_sha256'] == digest(folder / 'MANIFEST.json')
    for entry in manifest['files']:
        assert entry['path'] in ['REPORT.md', 'SOURCE_BINDINGS.json', 'SUMMARY.json', 'adopt.py', 'REPAIR.json']
        p = folder / entry['path']
        assert digest(p) == entry['sha256'] and p.stat().st_size == entry['bytes']

r = read(RESULT)
p = read(PROTO / 'PROTOCOL.json')
groups = {(g['bank'], g['split']): g for g in r['groups']}
assert len(groups) == len(r['groups']) == 9
assert set(groups) == {(bank, split) for bank in BANKS for split in range(3)}
assert r['status'] == 'complete_retrospective_development'
assert r['final_TEST_labels_used'] is False and r['refit_requires_separate_admission'] is True
selected = {}
finalists = {}
group_checks = []
class_supports = {}
metric_panels = 0
max_accuracy_closure_error = 0.0
for key, g in groups.items():
    bank, split = key
    configs = g['all_configurations']
    expected = [(op['id'], j) for op in p['operators'] for j in range(op['settings'])]
    if bank == 'single_author':
        expected = [(0, 0), (10, 0), (10, 1)]
    assert len(configs) == len(expected)
    assert [(c['operator'], c['setting']) for c in configs] == expected
    by_op = defaultdict(list)
    for c in configs:
        by_op[c['operator']].append(c)
        for mode in ['raw', 'corrected']:
            m = c[mode]
            assert m['count'] == 6123
            assert all(math.isfinite(m[k]) for k in METRICS)
            assert 0 <= m['Brier'] <= 2 and m['NLL'] >= 0 and 0 <= m['accuracy'] <= 1
            ca = m['class_accuracy']
            assert [q['class'] for q in ca] == list(range(5))
            support = [q['count'] for q in ca]
            assert sum(support) == m['count']
            assert support == ([1640, 2252, 1419, 546, 266] if split == 1 else [1640, 2252, 1420, 546, 265])
            class_supports[split] = support
            for q in ca:
                assert 0 <= q['accuracy'] <= 1
                assert abs(q['accuracy'] * q['count'] - round(q['accuracy'] * q['count'])) < 1e-9
            weighted = math.fsum(q['count'] * q['accuracy'] for q in ca) / m['count']
            error = abs(weighted - m['accuracy'])
            max_accuracy_closure_error = max(max_accuracy_closure_error, error)
            assert error < TOL
            metric_panels += 1
    own_selected = [min(by_op[op], key=lambda c: (c['corrected']['Brier'], c['setting'])) for op in sorted(by_op)]
    assert own_selected == g['selected_settings']
    finalist = min(own_selected, key=lambda c: (c['corrected']['Brier'], c['operator']))
    assert finalist == g['finalist']
    selected[key] = {c['operator']: c for c in own_selected}
    finalists[key] = finalist
    native_op = 0 if bank == 'single_author' else 1
    assert g['native_uncorrected'] == selected[key][native_op]['raw']
    assert g['development_biased'] is True
    assert g['base_checkpoint_not_cross_fitted'] is True
    assert g['folds_are_not_independent_replicates'] is True
    assert g['final_refits_performed'] == 0
    expected_mlp, expected_cal = (6, 0) if bank == 'single_author' else (12, 6)
    assert g['cost']['MLP_fits'] == expected_mlp and g['cost']['calibration_fits'] == expected_cal
    assert g['cost']['optimizer_updates'] == 150 * (expected_mlp + expected_cal)
    graph = g['graph']
    assert graph['bank'] == bank and graph['split'] == split
    assert graph['source_edges'] == p['graph']['source_edge_shape'][1]
    assert graph['CSR']['indptr']['shape'] == [24493]
    assert graph['CSR']['indices']['shape'] == graph['CSR']['data']['shape'] == [graph['retained_edges']]
    assert abs(graph['retained_fraction'] - graph['retained_edges'] / graph['source_edges']) < TOL
    assert abs(graph['degree']['mean'] - graph['retained_edges'] / 24492) < TOL
    assert len(g['provenance']) == (4 if bank == BANKS[1] else 1)
    group_checks.append({'bank': bank, 'split': split, 'configs': len(configs), 'selected_settings': [[c['operator'], c['setting']] for c in own_selected], 'finalist': [finalist['operator'], finalist['setting']], 'MLP_fits': expected_mlp, 'calibration_fits': expected_cal, 'optimizer_updates': g['cost']['optimizer_updates']})
for split in range(3):
    assert groups[BANKS[2], split]['provenance'] == [groups[BANKS[1], split]['provenance'][0]]
    assert len({groups[b, split]['graph']['retained_edge_logical_sha256'] for b in BANKS}) == 3
assert metric_panels == 198

comparison_leaves = Counter()
max_numeric_error = 0.0


def reconcile(actual, expected, path):
    global max_numeric_error
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        comparison_leaves[type(expected).__name__] += 1
        assert actual == expected and type(actual) == type(expected), path
    elif isinstance(expected, (int, float)):
        comparison_leaves['number'] += 1
        error = abs(actual - expected)
        max_numeric_error = max(max_numeric_error, error)
        assert math.isfinite(actual) and error <= TOL, (path, actual, expected)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, item in enumerate(expected):
            reconcile(actual[i], item, f'{path}/{i}')
    else:
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for k, item in expected.items():
            reconcile(actual[k], item, f'{path}/{k}')


def average(values):
    return math.fsum(values) / len(values)


def diffs(candidate, reference):
    return {m: candidate[m] - reference[m] for m in METRICS}


def pair(rows):
    out = {}
    assert len(rows) == 3
    for m in METRICS:
        values = [z[m] for z in rows]
        center = average(values)
        half = 4.303 * math.sqrt(math.fsum((v - center) ** 2 for v in values) / 2) / math.sqrt(3)
        out[m] = {'split_differences': values, 'mean': center, 'range': [min(values), max(values)], 'descriptive_t_df2_interval': [center - half, center + half]}
    return out


def predicates(rows):
    t = p['go_no_go']['bank_route_vs_best_cheap']
    return {
        'mean_Brier_improvement': -average([v['Brier'] for v in rows]) >= t['mean_Brier_improvement_min'],
        'positive_Brier_splits': sum(v['Brier'] < 0 for v in rows) >= t['positive_Brier_split_count_min'],
        'mean_accuracy_gain': 100 * average([v['accuracy'] for v in rows]) >= t['mean_accuracy_gain_pp_min'],
        'maximum_split_accuracy_loss': -100 * min(v['accuracy'] for v in rows) <= t['max_split_accuracy_loss_pp'],
        'mean_NLL_harm': average([v['NLL'] for v in rows]) <= t['mean_NLL_harm_max_nats'],
    }


def screen(rows):
    s = pair(rows)
    return {'passed': all(predicates(rows).values()), 'mean_Brier_improvement': -s['Brier']['mean'], 'mean_accuracy_gain_pp': 100 * s['accuracy']['mean'], 'mean_NLL_harm': s['NLL']['mean'], 'paired_descriptive_summary': s}


assert p['go_no_go']['bank_route_vs_best_cheap'] == p['go_no_go']['shared_vs_processed_references']['requirements_against_each']
chosen = []
screens = {}
failed_predicates = {}
for bank in BANKS[:2]:
    native_rows, cheap_rows = [], []
    for split in range(3):
        own = list(selected[bank, split].values())
        single = list(selected[BANKS[2], split].values())
        tags = [(c, 0) for c in own if c['operator'] <= 5] + [(c, 1) for c in single]
        cheap = min(tags, key=lambda v: (v[0]['corrected']['Brier'], v[1], v[0]['operator'], v[0]['setting']))[0]
        candidate = finalists[bank, split]
        chosen.append({'bank': bank, 'split': split, 'operator': candidate['operator'], 'setting': candidate['setting'], 'strongest_cheap': cheap})
        native_rows.append(diffs(candidate['corrected'], groups[bank, split]['native_uncorrected']))
        cheap_rows.append(diffs(candidate['corrected'], cheap['corrected']))
    screens[bank] = {'against_native': screen(native_rows), 'against_strongest_cheap': screen(cheap_rows)}
    for name, rows in [('against_native', native_rows), ('against_strongest_cheap', cheap_rows)]:
        failed_predicates[f'{bank}/{name}'] = {'checks': predicates(rows), 'failed': [k for k, v in predicates(rows).items() if not v]}
cross = {'attribution': 'whole per-split processing pipelines; prediction graphs differ; no pure causal sharing claim'}
cross_rows = {}
for name, bank in [('shared_vs_best_processed_independent', BANKS[1]), ('shared_vs_best_processed_single', BANKS[2])]:
    rows = [diffs(finalists[BANKS[0], s]['corrected'], finalists[bank, s]['corrected']) for s in range(3)]
    cross[name] = screen(rows)
    cross_rows[name] = rows
    failed_predicates[name] = {'checks': predicates(rows), 'failed': [k for k, v in predicates(rows).items() if not v]}
moments = {}
moment_gate_checks = {}
for bank in BANKS[:2]:
    contrasts = {str(op): pair([diffs(selected[bank, s][6]['corrected'], selected[bank, s][op]['corrected']) for s in range(3)]) for op in [4, 5, 8, 9]}
    matched = {}
    for setting in range(2):
        for control in [4, 5]:
            rows = []
            for split in range(3):
                configs = {(c['operator'], c['setting']): c for c in groups[bank, split]['all_configurations']}
                rows.append(diffs(configs[6, setting]['corrected'], configs[control, setting]['corrected']))
            matched[f'rho_setting{setting}_control{control}'] = pair(rows)
    thresholds = p['go_no_go']['moment_attribution_vs_global_full_and_local_diagonal']
    checks = {str(op): {'mean_improvement': -contrasts[str(op)]['Brier']['mean'] >= thresholds['mean_Brier_improvement_min'], 'max_split_harm': max(contrasts[str(op)]['Brier']['split_differences']) <= thresholds['max_split_Brier_harm']} for op in [4, 5]}
    moment_gate_checks[bank] = checks
    match_tolerance = p['go_no_go']['stacker_or_posterior_match_Brier_tolerance']
    moments[bank] = {'contrasts_candidate_minus_control': contrasts, 'matched_rho_contrasts': matched, 'global_and_diagonal_gate_passed': all(v for q in checks.values() for v in q.values()), 'same_info_matches_or_wins': contrasts['8']['Brier']['mean'] >= -match_tolerance, 'posterior_projection_matches_or_wins': contrasts['9']['Brier']['mean'] >= -match_tolerance}
go = screens[BANKS[0]]['against_native']['passed'] and screens[BANKS[0]]['against_strongest_cheap']['passed'] and all(cross[k]['passed'] for k in cross_rows)
synthesis = {'selected_per_split_pipelines': chosen, 'quality_screens': screens, 'complete_cross_bank_comparison': cross, 'moment_attribution': moments, 'shared_pipeline_confirmation_screen': 'GO_PROPOSAL_ONLY' if go else 'NO_GO', 'execution_or_TEST_authorization': False, 'finalist_selection_is_development_biased': True, 'uncertainty': 'descriptive only; common graph/overlapping blocks and tuning are not iid confirmation'}
reconcile(r['synthesis'], synthesis, 'RESULT/synthesis')
assert not go and not any(s['passed'] for z in screens.values() for s in z.values())

cost = r['cost']
for key, expected in [('MLP_fits', 90), ('calibration_fits', 36), ('optimizer_updates', 18900)]:
    assert cost[key] == sum(g['cost'][key] for g in groups.values()) == expected
assert cost['final_refits'] == cost['base_model_fits_or_replays'] == 0
assert cost['peak_rss_bytes'] == max(g['cost']['peak_rss_bytes'] for g in groups.values())
h = cost['H_and_QP_counters']
calls = h['calls']
assert len(calls) == h['H_calls'] == p['budget']['maximum_batched_H_calls'] == 144
assert sum(c['sparse_steps'] for c in calls) == h['sparse_steps'] == p['budget']['maximum_batched_sparse_steps'] == 2880
assert sum(c['logical_H_applications'] for c in calls) == h['logical_H_applications'] == p['budget']['total_logical_H_applications'] == 684
assert sum(c['logical_H_applications'] * c['sparse_steps'] for c in calls) == p['budget']['logical_sparse_steps'] == 13680
assert all(c['sparse_steps'] == p['graph']['updates'] == 20 for c in calls)
assert max(c['field_width'] for c in calls) == h['maximum_field_width'] == 75
assert abs(math.fsum(c['wall_seconds'] for c in calls) - h['H_wall_seconds']) < TOL
purpose = defaultdict(lambda: {'calls': 0, 'logical': 0, 'sparse_steps': 0})
for c in calls:
    for dst, src in [('logical', 'logical_H_applications'), ('sparse_steps', 'sparse_steps')]:
        purpose[c['purpose']][dst] += c[src]
    purpose[c['purpose']]['calls'] += 1
assert dict(purpose) == {'label_free_neighbour': {'calls': 9, 'logical': 9, 'sparse_steps': 180}, 'seed_context': {'calls': 54, 'logical': 54, 'sparse_steps': 1080}, 'single_mass_context': {'calls': 27, 'logical': 27, 'sparse_steps': 540}, 'CS_correction': {'calls': 27, 'logical': 297, 'sparse_steps': 540}, 'CS_smoothing': {'calls': 27, 'logical': 297, 'sparse_steps': 540}}
assert sum(len(g['all_configurations']) * 3 for g in groups.values()) == p['budget']['total_outer_config_predictions'] == 297
# Frozen source: each of 18 bank outer folds has 4 inner full-moment calls,
# 2 global one-row calls, 4 N-row local calls, and one N-row posterior call.
qp_calls = 6 * 3 * (4 + 2 + 4 + 1)
qp_solutions = 6 * 3 * (4 * 2041 + 2 + 5 * 24492)
assert h['QP_calls'] == qp_calls == 198
assert h['QP_solutions'] == qp_solutions == 2351268

means = {bank: {m: average([finalists[bank, s]['corrected'][m] for s in range(3)]) for m in METRICS} for bank in BANKS}
operator_rows = []
names = {v['id']: v['name'] for v in p['operators']}
names.update({0: 'single_native', 10: 'single_score_context_residual_MLP'})
for bank in BANKS:
    for op in sorted(selected[bank, 0]):
        operator_rows.append({'bank': bank, 'operator': op, 'name': names[op], 'settings_by_split': [selected[bank, s][op]['setting'] for s in range(3)], 'raw': {m: average([selected[bank, s][op]['raw'][m] for s in range(3)]) for m in METRICS}, 'corrected': {m: average([selected[bank, s][op]['corrected'][m] for s in range(3)]) for m in METRICS}})
supplementary = {name: pair([diffs(selected[BANKS[0], s][6]['corrected'], finalists[bank, s]['corrected']) for s in range(3)]) for name, bank in [('shared_analytic_moment_vs_processed_independent', BANKS[1]), ('shared_analytic_moment_vs_processed_single', BANKS[2])]}

root_summary = read(ROOT2 / 'SUMMARY.json')
reconcile(root_summary['finalist_means'], means, 'ROOT2/finalist_means')
reconcile(root_summary['retained_quality_screens'], screens, 'ROOT2/quality_screens')
reconcile(root_summary['retained_moment_attribution'], moments, 'ROOT2/moment_attribution')
reconcile(root_summary['cost'], cost, 'ROOT2/cost')
reconcile(root_summary['selected_operators'], {bank: [[finalists[bank, s]['operator'], finalists[bank, s]['setting']] for s in range(3)] for bank in BANKS}, 'ROOT2/selected_operators')
for name, rows in cross_rows.items():
    expected = {'candidate_minus_reference': {m: [v[m] for v in rows] for m in METRICS}, 'mean_Brier_improvement': cross[name]['mean_Brier_improvement'], 'mean_accuracy_gain_pp': cross[name]['mean_accuracy_gain_pp'], 'mean_NLL_harm': cross[name]['mean_NLL_harm'], 'checks': predicates(rows), 'passed': cross[name]['passed']}
    reconcile(root_summary['complete_processed_comparisons'][name], expected, 'ROOT2/' + name)
assert (root_summary['panels'], root_summary['configurations'], root_summary['raw_and_corrected_metric_panels']) == (9, 99, 198)
assert root_summary['status'] == 'COMPLETE_NEGATIVE_DEVELOPMENT_SCREEN'
assert root_summary['confirmation_admitted'] is False and root_summary['final_TEST_labels_used'] is False
old_summary = read(ROOT1 / 'SUMMARY.json')
new_without_time = dict(root_summary)
old_without_time = dict(old_summary)
new_without_time.pop('UTC')
old_without_time.pop('UTC')
assert new_without_time == old_without_time
old_phrase = 'new model inference'
new_phrase = 'new base-model inference'
for filename in ['adopt.py', 'REPORT.md']:
    old_text = (ROOT1 / filename).read_text()
    assert old_text.count(old_phrase) == 1
    assert old_text.replace(old_phrase, new_phrase) == (ROOT2 / filename).read_text()
assert (ROOT1 / 'SOURCE_BINDINGS.json').read_bytes() == (ROOT2 / 'SOURCE_BINDINGS.json').read_bytes()
repair = read(ROOT2 / 'REPAIR.json')
assert repair['predecessor_manifest_sha256'] == digest(ROOT1 / 'MANIFEST.json')
assert repair['scientific_result_changed'] is False and repair['original_v1_preserved'] is True
for path in inputs:
    assert digest(path) == initial_hashes[str(path)], str(path)

finding = {'id': 'R1', 'severity': 'minor wording', 'status': 'repaired in root V2; V1 preserved', 'file': str((ROOT1 / 'REPORT.md').relative_to(BASE)), 'finding': 'The V1 phrase "No ... new model inference" is broader than the frozen source, which fits and serves small MLP/calibration heads. V2 correctly says "new base-model inference".', 'effect_on_metrics_or_NO_GO': 'none'}
review = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_AGGREGATE_RECONCILIATION_NO_GO_PRESERVED', 'scope': 'RESULT aggregate arithmetic and frozen protocol/source/root adoption text only; no inspected-module imports or prediction/label/graph/checkpoint/model/fold payload access', 'scientific_execution': False, 'concrete_findings': [finding], 'unresolved_numerical_or_gate_errors': [], 'finalist_means': means, 'operator_mean_rows': operator_rows, 'shared_analytic_moment_supplementary_contrasts': supplementary, 'decision': 'NO_GO under the existing fixed protocol; no special analytic local/full moment benefit supported', 'limits': ['Fit/update counts reconcile reported metadata and source schedules; model diagnostics, optimizer trajectories and fold data were not independently decoded.', 'Graph, single alias and provenance checks compare recorded descriptors; underlying tensors and weights were not rehashed or examined.', 'Base checkpoints and processor/settings selection use VALID; overlapping splits on one graph provide descriptive development observations, not fresh confirmation.', 'The root physical-child duration and preserved failed-attempt history are not newly revalidated in this aggregate-only review.', 'This is a bounded result audit, not a paper acceptance, novelty or causal representation finding.']}
checks = {'tolerance': TOL, 'group_checks': group_checks, 'configuration_count': 99, 'raw_and_corrected_metric_panels': metric_panels, 'outer_config_prediction_count': 297, 'class_supports_by_split': class_supports, 'max_weighted_accuracy_closure_error': max_accuracy_closure_error, 'reconciled_synthesis_and_root_leaf_counts': dict(comparison_leaves), 'max_numeric_reconciliation_error': max_numeric_error, 'reported_cost_closure': {'MLP_fits': 90, 'calibration_fits': 36, 'total_fits': 126, 'optimizer_updates': 18900, 'H_calls': 144, 'sparse_steps': 2880, 'logical_H_applications': 684, 'logical_sparse_steps': 13680, 'H_purpose_breakdown': dict(purpose), 'QP_calls': qp_calls, 'QP_solutions': qp_solutions, 'final_refits': 0, 'base_model_fits_or_replays': 0}, 'practical_screen_predicates': failed_predicates, 'moment_gate_predicates': moment_gate_checks, 'reconstructed_synthesis': synthesis, 'root_V2_bounded_repair': {'source_and_report_only_phrase_replacement': True, 'source_bindings_unchanged': True, 'summary_unchanged_except_UTC': True, 'V1_preserved_and_bound': True}, 'all_bound_inputs_unchanged': True}
emit('REVIEW.json', review)
emit('CHECKS.json', checks)
emit('SOURCE_BINDINGS.json', {'inputs': bindings, 'descriptor_targets_opened': False})

lines = ['# Independent audit of the complete Amazon aggregation result', '', 'Status: **PASS for aggregate reconciliation; the fixed NO_GO decision is preserved.** One minor wording finding in root V1 was repaired in a separately sealed V2. No unresolved numerical, selection or gate error was found.', '', 'This audit read RESULT.json and frozen protocol/source/adoption text. It used only Python standard-library arithmetic. It did not open prediction, label, graph, model, checkpoint or fold payloads, import the inspected source, fit models, run inference, use SSH, score TEST, or edit canonical/manuscript files.', '', '## Closure and selection', '', 'All nine exact bank/split groups are complete: six four-member bank groups with 15 configurations each and three single groups with three each. All 99 configurations retain raw and corrected metrics (198 panels), count 6,123, complete class support and weighted accuracy closure. All setting choices and finalists reproduce the corrected OOF Brier selector and prescribed ties.', '', 'Every bank finalist is operator 7, score-context residual MLP, settings 1,0,1. The processed single selects operator 10, settings 1,0,0. The strongest cheap reference for every bank/split is that processed single. Operator 6, the analytic local-full moment proposal, is never selected.', '', '| Selected corrected pipeline | Accuracy | Brier | NLL |', '| --- | ---: | ---: | ---: |']
for bank, label in zip(BANKS, ['Shared four', 'Independent four', 'Single']):
    m = means[bank]
    lines.append(f"| {label} | {100*m['accuracy']:.7f}% | {m['Brier']:.9f} | {m['NLL']:.9f} |")
lines += ['', 'Reported fit counts close by group and total: 90 MLP + 36 calibration = 126 fits; the fixed 150-update schedule gives 18,900 reported updates. Per-call propagation records close to 144 batched H calls, 2,880 sparse steps, 684 logical H applications and 13,680 logical sparse steps. The C&S total is 594 logical applications for 297 outer configuration predictions. Frozen call geometry also gives 198 QP calls and 2,351,268 QP solutions, exactly reported. No final refit or base-model fit/replay is reported.', '', 'These are aggregate/counter checks. RESULT does not expose the per-fit trajectories/competence diagnostics or whole-fold rows; this review does not certify their numerical contents. Source schedules support the reported arithmetic.', '', '## Every failed practical screen', '', 'Differences are candidate minus reference. Brier improvement is minus the mean difference; accuracy gains are percentage points. The prescribed thresholds are Brier improvement >=0.002, at least two improved Brier splits, mean accuracy gain >=0.25 pp, worst split accuracy loss <=0.5 pp, and mean NLL harm <=0.01.', '', '| Candidate / reference | Brier improvement | Accuracy gain pp | NLL harm | Failed predicates |', '| --- | ---: | ---: | ---: | --- |']
labels = {f'{BANKS[0]}/against_native': ('Shared / native', screens[BANKS[0]]['against_native']), f'{BANKS[0]}/against_strongest_cheap': ('Shared / cheap single', screens[BANKS[0]]['against_strongest_cheap']), f'{BANKS[1]}/against_native': ('Independent / native', screens[BANKS[1]]['against_native']), f'{BANKS[1]}/against_strongest_cheap': ('Independent / cheap single', screens[BANKS[1]]['against_strongest_cheap']), 'shared_vs_best_processed_independent': ('Shared / processed independent', cross['shared_vs_best_processed_independent']), 'shared_vs_best_processed_single': ('Shared / processed single', cross['shared_vs_best_processed_single'])}
short = {'mean_Brier_improvement': 'mean Brier', 'positive_Brier_splits': 'Brier split count', 'mean_accuracy_gain': 'mean accuracy', 'maximum_split_accuracy_loss': 'worst accuracy split', 'mean_NLL_harm': 'mean NLL'}
for key, (label, s) in labels.items():
    failures = ', '.join(short[z] for z in failed_predicates[key]['failed'])
    lines.append(f"| {label} | {s['mean_Brier_improvement']:.9f} | {s['mean_accuracy_gain_pp']:.7f} | {s['mean_NLL_harm']:.9f} | {failures} |")
lines += ['', 'All six screens fail. Shared versus native also loses 0.9309162 pp on split 2. Shared versus independent has Brier gains in all three splits but loses 1.1105667 and 1.5188633 pp on splits 1 and 2 and worsens mean NLL. Both full bank routes fail against their strongest cheap references. The complete shared confirmation screen therefore remains NO_GO; none of the failed predicates has been dropped.', '', '## Analytic moment attribution', '', '| Bank | Local-full minus global-full Brier | Minus local-diagonal | Minus full-information stacker | Minus posterior projection |', '| --- | ---: | ---: | ---: | ---: |']
for bank, label in zip(BANKS[:2], ['Shared four', 'Independent four']):
    v = moments[bank]['contrasts_candidate_minus_control']
    lines.append('| ' + label + ' | ' + ' | '.join(f"{v[str(op)]['Brier']['mean']:+.9f}" for op in [4, 5, 8, 9]) + ' |')
lines += ['', 'The fixed mean improvement >=0.001 requirement fails against both global-full and local-diagonal controls in both banks. The selected-route maximum harm <=0.002 predicates pass, which does not rescue the conjunction. The stacker/posterior match flags are true because IDs 8 and 9 match or beat ID 6 within 0.001; they are not flags of analytic superiority. Analytic moments are substantially worse in Brier than the full-information stacker and posterior projection.', '', 'All eight already-available matched-rho comparisons were also recomputed:', '', '| Bank | rho setting | Local-full minus global-full Brier | Local-full minus diagonal Brier |', '| --- | ---: | ---: | ---: |']
for bank, label in zip(BANKS[:2], ['Shared four', 'Independent four']):
    matched = moments[bank]['matched_rho_contrasts']
    for j in range(2):
        lines.append(f"| {label} | {j} | {matched[f'rho_setting{j}_control4']['Brier']['mean']:+.9f} | {matched[f'rho_setting{j}_control5']['Brier']['mean']:+.9f} |")
lines += ['', 'No matched-rho mean reaches the 0.001 required improvement. On independent setting 0, the maximum split Brier harms exceed 0.002 (0.002755074 versus global and 0.002718910 versus diagonal). These are reported contrasts using the existing settings, not new gates or tuning.', '', 'For completeness, shared analytic moments minus the independently processed independent/single finalists are:']
for name, c in supplementary.items():
    lines.append(f"- {name}: mean Brier {c['Brier']['mean']:+.9f}, accuracy {100*c['accuracy']['mean']:+.7f} pp, NLL {c['NLL']['mean']:+.9f}.")
lines += ['', 'Gains over badly calibrated native pools do not identify a special local/off-diagonal mechanism. The processed single has lower mean Brier and NLL than either selected bank pipeline. The result supports preserving the negative development outcome; it does not establish causal sharing, representation defects, independent confirmation or paper acceptance.', '', '## Root adoption and repaired wording', '', 'Root V1/V2 means, signs, split differences, all practical predicates, retained moment contrasts and NO_GO interpretation reconcile. V1 said "No ... new model inference" although frozen study.py fits and serves the small heads. This minor scope overstatement has no metric/gate effect. V2 changes it to "new base-model inference". The V2 source and report are exact phrase replacements, its source bindings are byte-identical, and its summary differs only in UTC. Both sealed versions remain intact; REPAIR.json binds V1.', '', 'The physical child duration and prior failed-attempt history are outside the new aggregate arithmetic check. Historical acquisition remains a separately recorded cost; current RESULT only reports the successful study cost. The existing retrospective VALID and overlapping-split limitations are preserved.', '', '## Artifacts', '', '- REVIEW.json: concise conclusions, raw/corrected all-operator means and supplementary comparisons.', '- CHECKS.json: complete reconstructed synthesis, every gate/interval, counter closure and V2 repair checks.', '- SOURCE_BINDINGS.json: exact authorized input hashes; descriptor targets were not opened.', '- audit.py: standalone stdlib audit, with no inspected-source imports or execution-payload reads.', '']
with (HERE / 'REPORT.md').open('x') as handle:
    handle.write('\n'.join(lines))
manifest_files = [{'path': q.name, 'bytes': q.stat().st_size, 'sha256': digest(q)} for q in sorted(HERE.iterdir()) if q.is_file()]
emit('MANIFEST.json', {'files': manifest_files, 'status': review['status'], 'scientific_execution': False, 'input_result_sha256': digest(RESULT)})
emit('SEAL.json', {'manifest_sha256': digest(HERE / 'MANIFEST.json'), 'status': 'SEALED_INDEPENDENT_AGGREGATE_REVIEW'})
for q in HERE.iterdir():
    if q.is_file():
        os.chmod(q, 0o444)
print(json.dumps({'status': review['status'], 'folder': str(HERE), 'manifest_sha256': digest(HERE / 'MANIFEST.json'), 'seal_sha256': digest(HERE / 'SEAL.json'), 'max_numeric_reconciliation_error': max_numeric_error, 'leaves': dict(comparison_leaves), 'unresolved_numerical_or_gate_errors': 0, 'wording_finding': finding['status']}))
