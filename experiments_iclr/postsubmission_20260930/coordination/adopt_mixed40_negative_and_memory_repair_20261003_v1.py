"""Preserve the complete negative decision and current source/runtime evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

PHASE = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((PHASE / name).read_text())


def desc(name):
    data = (PHASE / name).read_bytes()
    return dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    now = datetime.now(timezone.utc).isoformat()
    decision_path = 'graph_mixed40_frozen_scientific_decision_20261003_v1/DECISION.json'
    decision = read(decision_path)
    assert decision['status'] == 'CLOSED_WITHOUT_PROMOTION' and not decision['primary_pass']
    assert not decision['role_assignment_pass'] and decision['all40_and_native15_verified']
    prior = 'graph_ncNC_structural_pattern_full_graph_execution_root_20261003_v2/remote_receipts_observation01'
    terminal = read(prior + '/supervision/run01/SUPERVISOR_TERMINAL.json')
    attempts = read(prior + '/full_graph/run01/ATTEMPTS.json')
    assert terminal['child_exit_code'] == 88 and terminal['qualification_receipt'] is None
    assert attempts['attempts'][0]['work']['completed_batches'] == 0
    amazon_root = 'amazon_polynormer_paired_family_execution_root_20261003_v3'
    runtimes = {}
    for name in ('v6_runtime_cpu_v1', 'v6_runtime_cuda0_v1'):
        freeze = read(amazon_root + '/' + name + '/FREEZE.json')
        physical = read(amazon_root + '/' + name + '/TERMINAL.json')
        assert freeze['status'] == physical['status'] == 'success'
        assert physical['physical_exit_code'] == 0
        for record in freeze['files']:
            assert desc(record['descriptor']['path']) == record['descriptor']
        runtimes[name] = dict(freeze=desc(amazon_root + '/' + name + '/FREEZE.json'),
                              terminal=desc(amazon_root + '/' + name + '/TERMINAL.json'),
                              wall_seconds=physical['whole_process_wall_seconds'],
                              peak_RSS_bytes=physical['aggregate_peak_observed_rss_bytes'])
    ledger = read('research_ledger.json')
    prior_keys = set(ledger)
    event = dict(UTC=now, kind='closed_negative_mixed40_and_checkpoint_repair_preparation',
                 mixed_decision=desc(decision_path), second_pattern_cap_failure=desc(prior + '/supervision/run01/SUPERVISOR_TERMINAL.json'),
                 NCNC_V4_source=desc('graph_ncNC_structural_pattern_pilot_preparation_20261003_v4/MANIFEST.json'),
                 Amazon_V6_runtime=runtimes,
                 new_literature=desc('ncnc_coherent_completion_recent_literature_scout_20261003_v1/MANIFEST.json'),
                 predictive_winner=False, established_novelty=False, manuscript_acceptance=False,
                 original_paper_scores_changed=False)
    ledger.setdefault('postsubmission_event_log', []).append(event)
    ledger['mixed40_frozen_complete_negative_decision_20261003_v1'] = dict(
        decision=event['mixed_decision'], status=decision['status'], primary_pass=False,
        role_assignment_pass=False, all40_and_native15_verified=True,
        hypotheses_closed_without_tuning_or_heldout_promotion=True)
    ledger['NCNC_pattern_full_graph_cap_failure_20261003_v2'] = dict(
        terminal=event['second_pattern_cap_failure'], attempts=desc(prior + '/full_graph/run01/ATTEMPTS.json'),
        child_exit_code=88, inclusive_wall_seconds=terminal['supervisor_inclusive_wall_seconds'],
        peak_allocated_bytes=terminal['CUDA_peak_observation']['cuda_peak_allocated_bytes'],
        peak_reserved_bytes=terminal['CUDA_peak_observation']['cuda_peak_reserved_bytes'],
        first_J_batches_completed=0, qualification_pass=False, predictive_fits_started=0,
        other_jobs_signaled=False, predecessor_preserved=True)
    ledger['NCNC_pattern_V4_activation_checkpoint_source_20261003_v1'] = dict(
        manifest=event['NCNC_V4_source'], source_only=True, independent_review_pending=True,
        numerical_and_full_graph_parity_pending=True, scientific_budget_shortened=False,
        caps_raised_again=False, predictive_fits_started=0, more_GPU_need_established=False)
    ledger['Amazon_polynormer_V6_actual_runtime_capture_20261003_v1'] = dict(
        runtimes=runtimes,
        closure=desc(amazon_root + '/v6_runtime_execution_receipts_20261003_v1/EXECUTION_CLOSURE.json'),
        fresh_qualification_required=True, original15_cohort_preserved=True,
        numerical_or_predictive_success_claim=False)
    ledger['NCNC_recent_coherent_completion_literature_20261003_v1'] = dict(
        manifest=event['new_literature'],
        conclusions=desc('ncnc_coherent_completion_recent_literature_scout_20261003_v1/PAPER_CONCLUSIONS.json'),
        accounting=desc('ncnc_coherent_completion_recent_literature_scout_20261003_v1/READ_ACCOUNTING.json'),
        six_new_scoped_method_reads=True, global_novelty_clearance=False,
        full_paper_read_certification=False, canonical_index_integration_pending=True)
    ledger['NCNC_cardinality_single_comparator_implemented_source_20261003_v1'] = dict(
        manifest=desc('ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1/MANIFEST.json'),
        source_only=True, independent_review_and_runtime_pending=True,
        complete_support_resource_cost_unmeasured=True)
    ledger['current_priority'] = (
        'Mixed40 failed both frozen practical gates and is closed without promotion. Independently review '
        'NCNC V4 recomputation, qualify exact numerical/gradient/RNG/update parity and full-graph feasibility, '
        'then the frozen J/F predictive pair if feasible. Run fresh bounded Amazon V6 qualification and '
        'admit the original15fits using actual resource evidence. Score complete BUDDY/NCNC families only. '
        'A contribution still needs capable controls, paired replication, heldout confirmation and fresh manuscript review.')
    ledger['last_updated_utc'] = ledger['current_status_update_UTC'] = now
    assert prior_keys <= set(ledger)
    snap = PHASE / 'coordination_snapshots/20261003_mixed40_negative_and_memory_repair_adoption_v1'
    snap.mkdir(exist_ok=False)
    for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
        (snap / ('BEFORE_' + name)).write_bytes((PHASE / name).read_bytes())
    (PHASE / 'research_ledger.json').write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
    status = (PHASE / 'PUBLIC_STATUS.md').read_text()
    a, b = status.index('Updated: '), status.index('. Goal active')
    status = status[:a] + 'Updated: ' + now + status[b:]
    a, b = status.index('The next fixed comparison uses authored Amazon'), status.index('\nThe completed DBLP comparison')
    status = status[:a] + '''The next fixed comparison uses authored Amazon Polynormer-r with raw features, 200 local and 2500 global epochs. V5 passed complete real-graph numerical qualification and ten complete-state next-update replays. V6 retains the selected local transition and final states, every epoch/decision record and explicit retirement records; it preserves the neural recipe and original three-block, 15-fit cohort. Independent source/forecast review passed. Fresh V6 CPU/GPU runtime captures exited 0 in 22.5874 and 21.6142 seconds. Fresh five-form numerical qualification is separately admitted; predictive fits have not started. The 53.369-hour / 80.054-hour-margin and 18.39-GB estimates remain planning evidence pending actual resource admission. Implementation success is not a predictive gain.
''' + status[b:]
    a, b = status.index('At 19:08:32 UTC, all 40'), status.index('\nAt 19:21:39 UTC, the NCNC')
    status = status[:a] + '''All 40 mixed shared/private HGT cases and 15 required native secondary rows are now evaluated. The one-line metadata comparison repair preserved the trained states and native replay schema; the evaluator exited 0 in 208.063467 seconds. Both failed transports and the earlier schema failure remain recorded. Recomputed candidate-minus-control contrasts fail every frozen primary gate on ACM and DBLP. The reverse-role gate also fails. This extension is closed without calibration rescue, threshold changes, extra tuning or heldout promotion. See [complete decision and results](graph_mixed40_frozen_scientific_decision_20261003_v1/RESULTS_SUMMARY.md).
''' + status[b:]
    a, b = status.index('The same read-only observation found GPU0'), status.index('\n## Comparator qualification')
    status = status[:a] + '''That read-only resource observation was used for a second, prospectively admitted full-graph pattern qualification. Both paid attempts ended before a completed J update; no other jobs were signaled. Current availability must be checked before further execution.
''' + status[b:]
    a, b = status.index('The frozen NCNC pilot compares'), status.index('\nA capable count-aware single-model comparator')
    status = status[:a] + '''The frozen NCNC pilot compares reconstruction of whole TRAIN connection patterns with equally supervised reconstruction of individual incidences. V3 passed numerical qualification. Two full-graph attempts failed before the first completed J update: 26.045865 and 29.334816 seconds, physical exit 88. The second exceeded its 75-GiB reserved-memory cap at 83,332,431,872 bytes; peak allocated memory was 73,043,022,336 bytes. All failures, identities and costs are retained. No predictive J/F fit has begun.

V4 changes only auxiliary activation retention through nonreentrant checkpoint recomputation. It preserves RNG, complete support and native batches, target/inference paths, losses and serving. Added fixed V3/V4 probes check forward values, gradients, exact RNG, flags and two real Adam updates; the original 34 native plus six replay updates and two complete VALID traversals remain. Independent source review and fresh numerical/full-graph qualification are pending. No further cap increase or need for more GPUs is established. GRAN supplies the likelihood ancestry; source zeros mean unobserved TRAIN incidences. The [scientific path assessment](active_graph_hypotheses_scientific_acceptance_path_assessment_20261003_v1/MEMO.md) still requires capable controls, mechanism evidence, paired replication and heldout confirmation.
''' + status[b:]
    a, b = status.index('A capable count-aware single-model comparator'), status.index('\nLatest verified pushed')
    status = status[:a] + '''A count-aware single-model comparator is now implemented in a sealed source-only packet. It models dependence through total residual count and retains that distribution across four paid decoder draws. Matched actual-marginal diagnostics remove dependence from the same fixed bank. Independent review, numerical qualification and complete-support resource measurement remain pending. The analytical witnesses establish information possibilities, not usefulness on Collab.

Six additional scoped primary method reads cover TGSBM, SDG, FLEX, PALP, SAGMM and a heuristic-informed multilayer MoE. Their conclusions and exact read boundaries are saved in [the new literature report](ncnc_coherent_completion_recent_literature_scout_20261003_v1/REPORT.md); canonical index integration remains pending. These reads add relevant structured-single and expert-routing controls, not global proof of novelty.
''' + status[b:]
    a, b = status.index('Latest verified pushed and synchronized head:'), status.index('\nNormal execution stays')
    status = status[:a] + '''Latest verified pushed and synchronized head: `4281dfb44b9db992232fa44f4119693efbb43bd9` on `codex/postsubmission-research-20260930`. New complete negative results, exact gate decision, memory failures, runtime captures, source repair and literature records await publication.
''' + status[b:]
    (PHASE / 'PUBLIC_STATUS.md').write_text(status)
    (PHASE / 'RESEARCH_STATE.md').write_text(f'''# GNNM post-submission research state

Updated: {now}. Goal active and incomplete. Original scores are unchanged. No new audited predictive winner, established novelty, revised manuscript or independent acceptance verdict.

The complete 40-case shared/private HGT comparison failed the original primary and reverse-role practical gates on both graphs. All 40 cases and 15 native secondary rows are retained; the declared hypothesis is closed without tuning or heldout promotion. See [complete results and decision](graph_mixed40_frozen_scientific_decision_20261003_v1/RESULTS_SUMMARY.md).

Graph observation-pattern supervision remains a plausible research hypothesis. V3 numerical qualification passed, but two full-graph attempts failed their memory caps before completing one J update. V4 activation recomputation is sealed for independent review and exact value/gradient/RNG/update parity. The complete scientific work is preserved, with added audit probes. No predictive J/F result or need for more GPUs is established; GRAN ancestry is credited.

Amazon V6 source review and both physical runtime captures succeeded. Fresh numerical qualification is separately admitted before resource admission of the original 15 fits. BUDDY and NCNC base families require complete closure before scoring. Heldout labels remain closed. The count-aware structural single has a source implementation, with review and runtime still pending. Six new scoped method reads are saved separately from the unchanged canonical index.

See [status](PUBLIC_STATUS.md), [scientific path](active_graph_hypotheses_scientific_acceptance_path_assessment_20261003_v1/MEMO.md), [new literature](ncnc_coherent_completion_recent_literature_scout_20261003_v1/REPORT.md) and [ledger](research_ledger.json). Latest verified pushed/synchronized head: `4281dfb44b9db992232fa44f4119693efbb43bd9`. Newer concrete records await publication.
''')
    (snap / 'ADOPTION_EVENT.json').write_text(json.dumps(event, indent=2) + '\n')
    print(json.dumps(dict(UTC=now, preserved_ledger_keys=len(prior_keys), event=desc(str((snap / 'ADOPTION_EVENT.json').relative_to(PHASE))))))


if __name__ == '__main__':
    main()
