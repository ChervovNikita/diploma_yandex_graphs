"""Preserve previous status and record completed evidence without changing scores."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
UTC = datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def seal(folder):
    rows = [dict(path=str(p.relative_to(folder)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(folder.rglob('*')) if p.is_file()]
    save(folder/'MANIFEST.json', dict(UTC=UTC, files=rows, original_scores_changed=False))
    save(folder/'SEAL.json', dict(manifest_sha256=sha(folder/'MANIFEST.json'), sealed=True))


snapshot = P/'coordination_snapshots/20261004_actual_count_and_pubmed_results_before_state_v1'
snapshot.mkdir()
for name in ['PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json']:
    shutil.copy2(P/name, snapshot/name)
seal(snapshot)

execution = P/'pubmed_shared4_owned_continuation_diagnostic_execution_root_20261004_v2'
summary = json.loads((execution/'compact_monitor01/SUMMARY.json').read_text())
d = summary['diagnostic']
assert summary['terminal_status']=='COMPLETE' and summary['terminal_exit_code']==0
assert d['progress']['Adam_completed']==72 and d['native_streams_exact'] is True
assert d['saved_tree_mutated_after_first_epoch'] is False and d['saved_tree_mutated_after_second_epoch'] is False
assert d['observed_final_continuation_parity_within_fixed_rule'] is False
folder = P/'pubmed_isolated_step_divergence_root_adoption_20261004_v1'
folder.mkdir()
diag = dict(UTC=UTC,status='ADOPTED_COMPLETE_ENGINEERING_DIAGNOSTIC_FIXED_RULE_FAILED',
    actual_summary=dict(path=str((execution/'compact_monitor01/SUMMARY.json').relative_to(P)),sha256=sha(execution/'compact_monitor01/SUMMARY.json')),
    source_manifest_sha256=d['source_manifest_sha256'],engineering_updates=72,
    VALID_serves=0,TEST_reads=0,saved_tree_preserved=True,native_streams_exact=True,
    first_observed_gradient_exact_difference_step=1,first_gradient_fixed_rule_failure_step=2,
    first_post_Adam_predictor_fixed_rule_failure_step=2,loss=d['loss'],
    hook_RNG_neutral_checks=d['hook_RNG_neutral_checks'],
    engineering_qualification_PASS=False,scientific_fit_admitted=False,
    kernel_cause_established=False,original_tolerance_changed=False,
    raw_diagnostic_retained_server_side=True,
    limitation='A completed diagnostic is not a numerical qualification PASS; native-only repeat control remains unexecuted.')
save(folder/'ROOT_ADOPTION.json',diag)
(folder/'RESULTS_SUMMARY.md').write_text('''# Isolated continuation diagnostic

Two complete reconstructed Pubmed shared4 continuations completed 72 updates. The saved state remained unchanged; prestates, native streams and RNG observations matched. The first exact gradient difference arose at step 1 before Adam; the fixed numerical rule first failed for gradients and the post-Adam predictor at step 2.

Final losses were 0.34525973598162335 and 0.3452373643716176. Their absolute difference, 0.000022371610005733622, exceeds the unchanged allowance of 0.000020527034545618033. The continuation qualification remains failed. No kernel cause is established, no tolerance was widened and no engineering state may initialize a scientific fit.

The next control repeats unmodified native training from two isolated restored copies. It has a different fresh warm-state history and cannot alone establish a matched causal explanation. The large diagnostic remains pinned on the authorized server; local compact evidence preserves its descriptor.
''')
seal(folder)

census = P/'graph_count_conditioned_train_support_census_execution_root_20261004_v2'
aggregate = json.loads((census/'FULL_CENSUS_AGGREGATE.json').read_text())
assert aggregate['status']=='FULL_CENSUS_PINNED_INTEGER_PROJECTION'
positive, negative = aggregate['populations']
assert positive['population']=='positive' and negative['population']=='negative'
assert positive['query_occurrences']==negative['query_occurrences']==3342336
folder = P/'graph_count_conditioned_train_support_census_root_adoption_20261004_v1'
folder.mkdir()
support = dict(UTC=UTC,status='ADOPTED_FULL_TRAIN_SUPPORT_METADATA_WITH_PINNED_PROJECTION',
    source_manifest_sha256='e6c7f6bc64e9d3454f2a7881acb16f493ec00b2126de0397b62c7b64a8b064e2',
    actual_terminal_sha256=sha(census/'owned_monitor01/supervision/run01/TERMINAL.json'),
    projection=dict(path=str((census/'FULL_CENSUS_COMPACT_PROJECTION.json').relative_to(P)),sha256=sha(census/'FULL_CENSUS_COMPACT_PROJECTION.json')),
    aggregate=aggregate,complete_seed_mask_streams=3,full_batches_per_seed=17,
    optimizer_updates=0,features_models_VALID_TEST_read=False,
    root_remote_histogram_arithmetic_and_pin_checks=True,fresh_independent_projection_audit='pending',
    raw_histograms_retained_server_side=True,predictive_gain=False,novelty_established=False,
    methodological_implications=[
        'Cross-side member association can act only on both-nonconstant patterns: 5.8242% of positive query occurrences.',
        'Both genuine subset choices occur in 1.3599% of positives; categorical/complement cases are classical and require attribution.',
        'Almost all native sampled negatives have a constant side; their count-conditioned pattern objective has no association signal.',
        'Exact arithmetic cells are modest, but up to 86629 per-group slot loops per positive side could make the current implementation impractical.',
        'Native full-batch forward/backward time and memory, representative predictive fits, replication and prior comparison remain required.'],
    interpretation='Repeated TRAIN queries/masks are not independent graphs, predictive outcomes, or confirmation of methodological novelty.')
save(folder/'ROOT_ADOPTION.json',support)
(folder/'RESULTS_SUMMARY.md').write_text('''# Complete TRAIN support census

All three seed mask streams completed 17 native batches, with 65,536 positive and 65,536 sampled negative queries per batch. The exact original source, data and runtime custody passed. Full histogram JSON remains on 18.77; a pinned compact projection rechecks every histogram scalar and joint/marginal count using standard-library integer arithmetic.

Across 3,342,336 positive query occurrences, 194,664 (5.8242%) had nonconstant patterns on both sides, 45,453 (1.3599%) had genuine subset choices on both sides and 306,577 (9.1725%) had a genuine subset on either side. Of 3,342,336 negative occurrences, one had both sides nonconstant, one had either side genuine and none had both sides genuine.

This supports testing the hypothesis on real positive patterns. It also reveals that the current auxiliary provides essentially no cross-side association signal on native sampled negatives. These are repeated TRAIN observations, not independent graphs or accuracy evidence.

One positive side can have 958 genuine exact groups, 86,629 Python group-slot iterations, 749,572 dynamic-program transition cells and 1,661,851 total residual slots. Mathematical work counts do not establish practical GPU time or backward memory. The sequential single-model control also needs exact vectorization before native feasibility is assumed.

No loss, model, feature, optimizer, VALID or TEST was evaluated. A fresh audit of compact projection, source and custody is pending; it will explicitly distinguish its checks from an independent raw histogram reread.
''')
seal(folder)

queue = P/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0021_RESULT.json'
q = json.loads(queue.read_text())
completed = [r for r in q['registered_fit_progress'] if r.get('physical_terminal',{} ) and r['physical_terminal'].get('status')=='success']
running = [r for r in q['registered_fit_progress'] if r['output_exists'] and not r.get('physical_terminal')]
assert q['status']=='running' and len(completed)==4 and len(running)==1 and not q['failures']
run = running[0]
ledger=json.loads((P/'research_ledger.json').read_text())
key='actual_count_cpu_pubmed_and_census_results_20261004_v1'
assert key not in ledger
ledger[key]=dict(UTC=UTC,diagnostic=diag,census=support,
    count_cpu_root_adoption=dict(path='graph_count_conditioned_pattern_cpu_qualification_root_adoption_20261004_v1/ROOT_ADOPTION.json',sha256=sha(P/'graph_count_conditioned_pattern_cpu_qualification_root_adoption_20261004_v1/ROOT_ADOPTION.json')),
    Amazon_monitor21=dict(UTC=q['UTC'],completed_fits=4,registered_fits=15,running_fit=run['fit_id'],updates=run['trace_progress']['actual_update'],source_sha256=sha(queue),failures=0,partial_quality_or_TEST_read=False),
    literature_offload=dict(path='literature_raw_asset_offload_20261004_v1',local_raw_files_removed=38,bytes_freed=22655883,conclusions_and_original_manifests_preserved=True),
    theory_packet='graph_count_conditioned_joint_identifiability_assessment_preparation_20261004_v1',
    core_v2_manifest_sha256='1cb23d832832bbec6323ca391c02c77b91c96f6e91925a867d9ed903256d00d5',core_v2_executed=False,
    native_control_manifest_sha256='28c9cd5232123655ba3588ee6ca8b358133d2afa189c07a1fef4505ef998e7a3',native_control_executed=False,
    confirmed_methodological_advantage=False,fresh_manuscript_acceptance=False,original_scores_changed=False)
ledger['active_parallel_work_current']=[
    'Root: monitor exact original queues, adopt actual diagnostic/census/CPU evidence, publish complete history and admit reviewed bounded controls.',
    'NCNC agent: independent review of grouped exact pattern-core v2 from another author.',
    'Pubmed agent: independent native-only repeat-control review; then exact vectorized single-control implementation.',
    'Pooled agent: independent audit of completed census compact projection, custody and scientific support.']
ledger['updated_UTC']=ledger['updated_utc']=UTC
(P/'research_ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True,allow_nan=False)+'\n')
(P/'PUBLIC_STATUS.md').write_text(f'''# Current GNNM research status

Updated: {UTC}. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The five-seed official ogbl-collab TEST comparison measured GNNM private completion at **67.2909% Hits@50**, native single64 at **66.4426%** and an independent four-model ensemble at **67.6298%**. The exploratory **+0.8483-point** single-model gain is positive in all five seeds. The independent ensemble remains higher.

The frozen private-minus-pooled primary contrast is **+0.2236 points**, paired 95% seed interval **[-0.7775,+1.2247]**, exact sign-flip p=0.6875. It is inconclusive. TEST is consumed for this family. [Full result and uncertainty](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed new methodological advantage or fresh manuscript acceptance exists.**

## Concrete current direction

The new auxiliary trains which candidate neighbours belong together after conditioning on their observed TRAIN count. It compares cross-side member association with independently mixed sides and a capable structured single model. Classical conditional Bernoulli, mixture, categorical/InfoNCE and fixed-size DPP ingredients are attributed; novelty and predictive value remain unproven.

Actual fabricated law/gradient QA passed **14,071 checks**. This verifies mathematical implementation on declared fixtures, not predictive improvement. A successor closes endpoint/float32/ragged gaps; it is sealed, disabled and under independent review.

The full TRAIN census completed three17-batch streams. Both sides are nonconstant for **5.8242% of positive queries**, with genuine subset choices on both sides for **1.3599%**. Almost all sampled negatives have no cross-side pattern signal. Group dispatch and the sequential single-control implementation need practical optimization before full-batch feasibility is assumed. [Support and limits](graph_count_conditioned_train_support_census_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## Runs and engineering

At {q['UTC']}, Amazon training had **4/15 completed fits**; the fifth was at **{run['trace_progress']['actual_update']}/2700 updates**, with no failures or restarts. No partial quality or TEST decisions were made.

Six native Pubmed baselines are complete; no GNNM predictive result exists. A completed72-update diagnostic located divergence before Adam with exact prestates/streams and preserved saved state. It still fails the unchanged numeric rule. A fresh native-only repeat control is under review; it has a different warm-state history and cannot alone establish cause. [Diagnostic](pubmed_isolated_step_divergence_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v44 retains196 scoped conclusions across147 paper identities and2 software identities, not full-paper read totals. Inactive raw literature assets were offloaded with full hash verification, freeing22.7MB locally; conclusions and history remain.

Latest verified GitHub head before this update: 9bfcdb509d4750221d4f096626c5dd9fcb822d09. New complete updates await publication. Science uses the authorized anogena-2 and18.77 repositories only. Seven-GPU access is forwarding only. No sudo, PDF compilation, GENLINK, server configuration or unrelated-job changes.
''')
(P/'RESEARCH_STATE.md').write_text(f'''# GNNM post-submission research state

Updated: {UTC}. Goal active and incomplete. [Current evidence](PUBLIC_STATUS.md). Prior status preserved in coordination_snapshots/20261004_actual_count_and_pubmed_results_before_state_v1.

## Decisions and next actions

1. Preserve frozen NCNC TEST evidence and uncertainty. Single-model gain is exploratory; primary private-vs-pooled evidence is inconclusive and independent ensemble mean is higher. Successor designs must disclose consumed-TEST development.
2. Maintain original Amazon queue. Monitor21:4/15fits complete, fifth224/2700, no failures or partial score decisions.
3. Preserve completed isolated Pubmed diagnostic as a failed fixed-rule continuation comparison. Exact streams and saved-state preservation do not erase numerical divergence. Review and execute the fresh252-update native-only control; disclose different warm-state context and infer no automatic cause/qualification or tolerance amendment.
4. Preserve actual count-law CPU PASS. Independently review grouped core-v2 endpoint/float32/ragged assurance and run actual successor QA after separate admission.
5. Audit full completed TRAIN census. Both-informative positive exposure5.8242%; both-genuine1.3599%; negatives essentially zero. Do not claim accuracy, novelty or independent-graph sample size from this census.
6. Resolve native batch performance with exact vectorized teacher forcing for the structured single and practical grouped conditional-law evaluation. Arithmetic cells alone cannot prove GPU feasibility; never weaken/drop controls to make a pilot cheaper.
7. Then conduct representative prospectively frozen J_K/J_K_sep/W_K/P0/capable-single paired scientific fits with same-state coupling diagnostics, relevant prior comparisons, and a second benchmark. Do not attribute separate-arm fitted differences solely to coupling or claim global novelty clearance.
8. Publish complete explicit inventories including failures/reviews; retain large raw assets on authorized servers. Revise manuscript from supported outcomes, then fresh skill-based reviewers without a requested verdict.

## Boundaries

Original paper scores unchanged. No confirmed methodological advantage or fresh acceptance. Authorized anogena-2 one-GPU and18.77 project repositories only for science; seven-GPU route forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes. Small incidental caches allowed. Preserve all unsuccessful results, decisions, reviews, and paper conclusions.
''')
print(json.dumps(dict(status='ACTUAL_RESULTS_ADOPTED_AND_HISTORY_PRESERVED',UTC=UTC,original_scores_changed=False,predictive_advantage_claimed=False)))
