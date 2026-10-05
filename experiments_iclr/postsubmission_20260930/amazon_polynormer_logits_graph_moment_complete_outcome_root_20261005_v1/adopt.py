"""Describe the entire completed, fixed VALID screen without new fits or selection."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OBS = PHASE / 'amazon_polynormer_logits_graph_moment_cpu_root_preparation_20261005_v2/development_observation_20261005T174207Z'
RESULT = OBS / 'amazon_polynormer_logits_graph_moment_retrospective_cpu_execution_root_20261005_v2/RESULT.json'
PROTOCOL = PHASE / 'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/PROTOCOL.json'
EXPECTED_RESULT = '67db49a7a9604d0cabcf024b65d1d5a5188bc79dbc043ef3e133a9248ba6b5cb'
EXPECTED_PROTOCOL = '5d667f15640102995fa998ca6932c53c5b71554273c84d4938a331c0a4eae7ba'
BANKS = ['gnnm_boundary_4', 'independent_author_4_same_width', 'single_author']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def contrast(candidate, reference, thresholds):
    differences = {k: [candidate[i][k] - reference[i][k] for i in range(3)]
                   for k in ('accuracy', 'Brier', 'NLL')}
    checks = {
        'mean_Brier_improvement': -mean(differences['Brier']) >= thresholds['mean_Brier_improvement_min'],
        'positive_Brier_splits': sum(x < 0 for x in differences['Brier']) >= thresholds['positive_Brier_split_count_min'],
        'mean_accuracy_gain': 100 * mean(differences['accuracy']) >= thresholds['mean_accuracy_gain_pp_min'],
        'maximum_split_accuracy_loss': -100 * min(differences['accuracy']) <= thresholds['max_split_accuracy_loss_pp'],
        'mean_NLL_harm': mean(differences['NLL']) <= thresholds['mean_NLL_harm_max_nats'],
    }
    return dict(candidate_minus_reference=differences,
                mean_Brier_improvement=-mean(differences['Brier']),
                mean_accuracy_gain_pp=100 * mean(differences['accuracy']),
                mean_NLL_harm=mean(differences['NLL']), checks=checks,
                passed=all(checks.values()))


assert sha(RESULT) == EXPECTED_RESULT and sha(PROTOCOL) == EXPECTED_PROTOCOL
r = json.loads(RESULT.read_text())
p = json.loads(PROTOCOL.read_text())
assert r['status'] == 'complete_retrospective_development'
assert not r['final_TEST_labels_used']
assert len(r['groups']) == 9
groups = {(g['bank'], g['split']): g for g in r['groups']}
assert set(groups) == {(bank, split) for bank in BANKS for split in range(3)}
for (bank, split), g in groups.items():
    expected = {(o['id'], setting) for o in p['operators'] for setting in range(o['settings'])}
    if bank == 'single_author':
        expected = {(0, 0), (10, 0), (10, 1)}
    configs = g['all_configurations']
    assert len(configs) == len(expected)
    assert {(c['operator'], c['setting']) for c in configs} == expected
    assert g['finalist'] == min(g['selected_settings'], key=lambda c: (c['corrected']['Brier'], c['operator']))
    for c in configs:
        for mode in ('raw', 'corrected'):
            assert c[mode]['count'] == 6123
            assert sum(row['count'] for row in c[mode]['class_accuracy']) == 6123
    assert not g['final_refits_performed'] and g['development_biased']
cost = r['cost']
for key, expected in [('MLP_fits', 90), ('calibration_fits', 36), ('optimizer_updates', 18900)]:
    assert cost[key] == sum(g['cost'][key] for g in groups.values()) == expected
assert cost['final_refits'] == cost['base_model_fits_or_replays'] == 0
assert cost['H_and_QP_counters']['H_calls'] == 144
assert cost['H_and_QP_counters']['sparse_steps'] == 2880
assert cost['H_and_QP_counters']['logical_H_applications'] == 684
finalists = {bank: [groups[bank, split]['finalist']['corrected'] for split in range(3)] for bank in BANKS}
means = {bank: {k: mean(v[k] for v in rows) for k in ('accuracy', 'Brier', 'NLL')} for bank, rows in finalists.items()}
thresholds = p['go_no_go']['shared_vs_processed_references']['requirements_against_each']
comparisons = {name: contrast(finalists[BANKS[0]], finalists[bank], thresholds)
               for name, bank in [('shared_vs_best_processed_independent', BANKS[1]),
                                  ('shared_vs_best_processed_single', BANKS[2])]}
for name, c in comparisons.items():
    stored = r['synthesis']['complete_cross_bank_comparison'][name]
    assert c['passed'] == stored['passed']
    for key in ('mean_Brier_improvement', 'mean_accuracy_gain_pp', 'mean_NLL_harm'):
        assert abs(c[key] - stored[key]) < 1e-14
assert not any(c['passed'] for c in comparisons.values())
assert r['synthesis']['shared_pipeline_confirmation_screen'] == 'NO_GO'
summary = dict(UTC=datetime.now(timezone.utc).isoformat(),
               status='COMPLETE_NEGATIVE_DEVELOPMENT_SCREEN',
               result_sha256=sha(RESULT), protocol_sha256=sha(PROTOCOL),
               panels=9, configurations=99, raw_and_corrected_metric_panels=198,
               finalist_means=means, complete_processed_comparisons=comparisons,
               retained_quality_screens=r['synthesis']['quality_screens'],
               retained_moment_attribution=r['synthesis']['moment_attribution'],
               cost=cost, selected_operators={bank: [(groups[bank, s]['finalist']['operator'], groups[bank, s]['finalist']['setting']) for s in range(3)] for bank in BANKS},
               all_configuration_metrics_preserved=True, no_new_fits=True,
               confirmation_admitted=False, original_paper_scores_changed=False,
               final_TEST_labels_used=False, interpretation='Retrospective VALID development; no novelty or superiority claim.')
emit('SUMMARY.json', summary)
emit('SOURCE_BINDINGS.json', dict(result=str(RESULT.relative_to(PHASE)), result_sha256=sha(RESULT),
                                  protocol=str(PROTOCOL.relative_to(PHASE)), protocol_sha256=sha(PROTOCOL),
                                  physical_terminal=str((OBS / 'DEVELOPMENT_TERMINAL.json').relative_to(PHASE)),
                                  terminal_sha256=sha(OBS / 'DEVELOPMENT_TERMINAL.json')))
report = '''# Complete saved-prediction aggregation outcome

The fixed Amazon/Polynormer screen completed all three splits for the shared four-member ensemble, the independent four-member ensemble and a comparable processed single. It used existing selected predictions and the permitted VALID folds. The result retains all 99 configurations, both raw and graph-corrected metrics, fitted-control diagnostics and costs. No backbone training, new model inference, final refit or TEST scoring occurred.

## What was tested

The proposed operation combines predictions using local, graph-transported moments of member errors. Its controls include ordinary pools, global temperature/weight calibration, global and diagonal error moments, capable nonlinear score stacking, and projection toward a graph-diffused label posterior. The single has a comparable nonlinear processor. All configurations and three splits were fixed before execution. Setting selection uses the prescribed corrected out-of-fold Brier loss; no accuracy-based replacement was made after inspection.

## Complete processed comparisons

| Prescribed Brier-selected pipeline, mean over three splits | Accuracy | Brier | NLL |
| --- | ---: | ---: | ---: |
'''
for bank, name in zip(BANKS, ['Shared four', 'Independent four', 'Single']):
    m = means[bank]
    report += f"| {name} | {100*m['accuracy']:.4f}% | {m['Brier']:.6f} | {m['NLL']:.6f} |\n"
report += '''
The capable score-context MLP (operator 7) was the selected processor in every ensemble split, with settings 1, 0, 1 for each ensemble family. The processed single selected operator 10, with settings 1, 0, 0. The graph-moment proposal was not selected.

Shared four trails processed independent four by 0.7077 accuracy points, despite a 0.005092 mean Brier advantage; its mean NLL is 0.048940 worse. Shared four trails the processed single by 0.5825 accuracy points, with both worse Brier (+0.023703) and NLL (+0.340901). It fails the fixed quality requirements against both references. Its accuracy differences against independent are +0.5063, -1.1106, -1.5189 points; against the single they are +0.1633, -1.3065, -0.6043 points. No favorable split or metric is substituted for the complete result.

The selected shared processor improves calibration losses over its native pool, but mean accuracy improves by only 0.1415 points and one split loses 0.9309 points. The selected independent processor adds only 0.0762 accuracy points. Both fail their prescribed full quality screens. Improved calibration alone does not establish improved classification or a shared-ensemble advantage.

## Ingredient evidence and research decision

For shared predictions, tuned local-full moments have 0.000714 worse Brier than global-full moments, approximately 0.000020 better Brier than local-diagonal moments, and 0.011447 worse Brier than posterior projection. The complete result also retains same-rho contrasts for both moment controls. There is no supported useful off-diagonal/local-moment contribution under this screen. Stacking and posterior controls must remain in the interpretation.

**Decision: do not launch confirmation or expand the hyperparameter grid for this proposal.** Preserve the complete failure and use it to guide training research. Together with the independent prediction-error diagnosis, the result motivates seeking competent complementary members rather than assuming a new combination rule repairs their repeated errors. This is a hypothesis choice, not causal proof of a representation defect or clearance of methodological novelty. Ordinary NCL/GNCL error-coupled training is an attributed control; its known own/pool loss equivalence prevents treating it as a new method.

The base checkpoints were selected on VALID, and the out-of-fold processor results also choose settings. These are biased retrospective development observations on one graph with overlapping splits. The retained descriptive intervals are not independent confirmation intervals. Final TEST labels remain unopened; original manuscript results are unchanged.

## Costs and preserved failures

The completed screen used 90 small MLP fits, 36 calibration fits and 18,900 optimizer updates. Numerical study time was 165.664 seconds; the physical child including serialization took 169.628 seconds. Peak child RSS was 719,511,552 bytes. There were 144 batched graph propagations, 2,880 sparse steps and 684 logical field propagations. Historical base-model acquisition is an additional separately retained cost.

The first execution failed before fitting/scoring because of projection roundoff. Its original failure, actual row witness, source versions, independent repair review and affected qualification remain preserved. The corrected run changed projection arithmetic, with an explicit numerical rank cutoff; it did not alter scientific settings or spare an unfavorable arm. Neither attempt is omitted from costs or provenance.

- [Recomputed means, gates and retained complete contrasts](SUMMARY.json)
- [Immutable input bindings](SOURCE_BINDINGS.json)
- [All configurations and original terminal result](../amazon_polynormer_logits_graph_moment_cpu_root_preparation_20261005_v2/development_observation_20261005T174207Z/amazon_polynormer_logits_graph_moment_retrospective_cpu_execution_root_20261005_v2/RESULT.json)
- [Independent prediction-error interpretation review](../amazon_polynormer_valid_error_analysis_independent_interpretation_review_20261005_v1/REPORT.md)
'''
with (HERE / 'REPORT.md').open('x') as handle:
    handle.write(report)
files = [dict(path=f.name, bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(HERE.iterdir()) if f.is_file()]
emit('MANIFEST.json', dict(files=files, scientific_execution=False, original_paper_scores_changed=False))
emit('SEAL.json', dict(manifest_sha256=sha(HERE / 'MANIFEST.json'), status='SEALED_COMPLETE_NEGATIVE_SCREEN'))
print(json.dumps(dict(status=summary['status'], panels=9, means=means, full_screen='NO_GO', manifest_sha256=sha(HERE / 'MANIFEST.json'))))
