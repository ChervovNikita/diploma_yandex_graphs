"""Apply the original practical gates to the complete, unchanged Mixed40 evaluation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

PHASE = Path(__file__).resolve().parents[1]
EVALUATION = 'graph_mixed40_complete_evaluation_execution_root_20261003_v3'
DESIGN = 'graph_mixed_block_objectives_execution_root_v1/FROZEN_SCIENTIFIC_DESIGN.json'
OUTPUT = 'graph_mixed40_frozen_scientific_decision_20261003_v1'


def read(path):
    return json.loads((PHASE / path).read_text())


def descriptor(path):
    data = (PHASE / path).read_bytes()
    return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def contrast(rows, graph, control, seeds, recorded, gate):
    by_cell = {(r['dataset'], r['arm'], r['seed']): r for r in rows}
    nll = []
    f1 = []
    for seed in seeds:
        a = by_cell[graph, 'pool/own', seed]['metrics']['raw']
        b = by_cell[graph, control, seed]['metrics']['raw']
        nll.append(a['NLL_FP32'] - b['NLL_FP32'])
        f1.append(a['macro_F1'] - b['macro_F1'])
    assert all(math.isfinite(v) for v in nll + f1)
    assert nll == recorded['raw.NLL_FP32']['vector']
    assert f1 == recorded['raw.macro_F1']['vector']
    mean_nll, mean_f1 = sum(nll) / len(nll), sum(f1) / len(f1)
    assert math.isclose(mean_nll, recorded['raw.NLL_FP32']['mean'], abs_tol=1e-15)
    assert math.isclose(mean_f1, recorded['raw.macro_F1']['mean'], abs_tol=1e-15)
    wins = sum(v < 0 for v in nll)
    checks = dict(
        mean_NLL_margin=mean_nll <= gate['candidate_minus_control_NLL_mean_at_most'],
        strict_NLL_wins=wins >= gate['strict_paired_NLL_wins_at_least'],
        macro_F1_noninferiority=mean_f1 >= gate['mean_macro_F1_delta_at_least'])
    return dict(graph=graph, control=control, paired_seed_order=seeds,
                candidate_minus_control_NLL=nll, mean_NLL_delta=mean_nll,
                strict_NLL_wins=wins, candidate_minus_control_macro_F1=f1,
                mean_macro_F1_delta=mean_f1, checks=checks, passes=all(checks.values()))


def main():
    design, evaluation = read(DESIGN), read(EVALUATION + '/EVALUATION_run01.json')
    closure, physical = read(EVALUATION + '/AUDIT_CLOSURE.json'), read(EVALUATION + '/EVALUATION_PROCESS_EXIT.json')
    assert descriptor(DESIGN)['sha256'] == '739160acd4ffb0380348f79496e88624db6163f4a3a93902b043d58eaacc18c5'
    assert descriptor(EVALUATION + '/EVALUATION_run01.json')['sha256'] == closure['evaluation']['sha256'] == 'a977ebcf4b2507130fe3c00d3123b294be690358681a132087d60c41715d5fe2'
    assert closure['status'] == 'complete40_and_required_native15'
    assert physical['exit_code'] == 0
    assert not evaluation['heldout_labels_opened'] and closure['heldout_labels_closed']
    assert evaluation['status'] == 'complete' and not evaluation['successful_subset_scored']
    assert evaluation['full40_comparison_denominator'] and evaluation['originals_preserved']
    rows = evaluation['rows']
    seeds = design['seeds']
    expected = {(g, p, s) for g in ('HGB-ACM', 'HGB-DBLP') for p in ('own/own', 'pool/pool', 'pool/own', 'own/pool') for s in seeds}
    assert len(rows) == len(expected) == 40
    assert {(r['dataset'], r['arm'], r['seed']) for r in rows} == expected
    assert all(r['status'] == 'complete' and r['checkpoint_replay']['selected_inference_replay']
               and r['checkpoint_replay']['full_state_restored']
               and r['checkpoint_replay']['checkpoint_source_binding_verified'] for r in rows)
    assert len(evaluation['native_secondary']['native_rows']) == closure['native_secondary_rows'] == 15
    assert closure['native_secondary_failure'] is None
    primary_gate = design['scientific_success_gate']
    primary = []
    role = []
    role_gate = dict(candidate_minus_control_NLL_mean_at_most=design['role_assignment_gate']['same_mean_NLL_margin'],
                     strict_paired_NLL_wins_at_least=design['role_assignment_gate']['strict_paired_NLL_wins_at_least'],
                     mean_macro_F1_delta_at_least=design['role_assignment_gate']['mean_macro_F1_delta_at_least'])
    for graph in ('HGB-ACM', 'HGB-DBLP'):
        for control in ('own/own', 'pool/pool'):
            record = {metric: values[control] for metric, values in evaluation['frozen_primary_selected_raw_contrasts'][graph].items()}
            primary.append(contrast(rows, graph, control, seeds, record, primary_gate))
        role.append(contrast(rows, graph, 'own/pool', seeds,
                             evaluation['frozen_role_assignment_selected_raw_contrasts'][graph], role_gate))
    passed = all(c['passes'] for c in primary)
    result = dict(schema='complete_mixed40_frozen_practical_decision_v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  inputs=[descriptor(DESIGN), descriptor(EVALUATION + '/EVALUATION_run01.json'),
                          descriptor(EVALUATION + '/AUDIT_CLOSURE.json'), descriptor(EVALUATION + '/EVALUATION_PROCESS_EXIT.json')],
                  all40_and_native15_verified=True, primary_gate=primary_gate,
                  primary_contrasts=primary, primary_pass=passed,
                  role_assignment_gate=design['role_assignment_gate'], role_assignment_contrasts=role,
                  role_assignment_pass=all(c['passes'] for c in role),
                  status='CONTINUE_DECLARED_HYPOTHESIS' if passed else 'CLOSED_WITHOUT_PROMOTION',
                  practical_gate_is_significance_or_acceptance=False,
                  uncertainty_scope=evaluation['uncertainty_scope'],
                  thresholds_changed=False, calibration_or_final_endpoint_rescue=False,
                  heldout_labels_opened=False, heldout_promotion=False,
                  original_paper_scores_changed=False, predictive_superiority_claim=False,
                  methodological_novelty_established=False, manuscript_acceptance_claim=False)
    output = PHASE / OUTPUT
    output.mkdir(exist_ok=False)
    (output / 'DECISION.json').write_text(json.dumps(result, indent=2) + '\n')
    table = '\n'.join(f"| {c['graph']} | {c['control']} | {c['mean_NLL_delta']:+.7f} | {c['strict_NLL_wins']}/5 | {c['mean_macro_F1_delta']:+.7f} | {'pass' if c['passes'] else 'fail'} |" for c in primary + role)
    (output / 'RESULTS_SUMMARY.md').write_text('''# Complete shared/private objective comparison

The declared extension did not meet its predefined practical improvement criteria on either graph. It is closed without tuning or heldout promotion. This result concerns the fixed HGT setting; it does not establish that every possible block objective is ineffective.

All 40 cases and the 15 required native secondary rows are retained. The evaluator exited 0 in 208.063467 seconds. Selected checkpoints were restored, and the paired vectors below were recomputed from the served raw scores and checked against the complete evaluator. TEST remains closed.

The candidate sends pooled-prediction gradients to shared parameters and member-specific gradients to private factors. Each graph/control comparison required an average NLL reduction of at least 0.005 nats, at least four strict wins out of five paired blocks, and no reduction in mean macro-F1. These are practical continuation criteria, not significance or conference-acceptance thresholds.

| Graph | Control | Mean NLL delta | NLL wins | Mean macro-F1 delta | Gate |
| --- | --- | ---: | ---: | ---: | --- |
''' + table + '''

Deltas are candidate minus control; smaller NLL is better and larger macro-F1 is better. The reverse role control is own/pool. All vectors, descriptive intervals and leave-one-block-out means remain in the original complete evaluation. Overlapping graph splits do not support independent population-level inference. Calibration uses the same source validation labels and cannot rescue the failed raw primary gate. No selected subset, final checkpoint, member-level metric or changed threshold replaces the declared endpoint.

This study supplies no new predictive winner, established novelty or manuscript acceptance verdict. Original paper scores are unchanged. Earlier failed evaluation attempts and costs remain preserved.
''')
    print(json.dumps(dict(decision=descriptor(OUTPUT + '/DECISION.json'), status=result['status'], primary_pass=passed, role_assignment_pass=result['role_assignment_pass'])))


if __name__ == '__main__':
    main()
