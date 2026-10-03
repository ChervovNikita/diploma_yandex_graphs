"""Record completed checks and current research; do not promote a quality claim."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

PHASE = Path(__file__).resolve().parents[1]
SNAP = PHASE / 'coordination_snapshots/20261003_numerical_repair_scientific_path_adoption_v1'


def read(path):
    return json.loads((PHASE / path).read_text())


def descriptor(path):
    data = (PHASE / path).read_bytes()
    return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify_packet(path):
    manifest = read(path + '/MANIFEST.json')
    for row in manifest.get('files', manifest.get('payload', [])):
        actual = descriptor(path + '/' + row['path'])
        assert actual['bytes'] == row.get('bytes', row.get('size'))
        assert actual['sha256'] == row['sha256']
    return descriptor(path + '/MANIFEST.json')


def main():
    now = datetime.now(timezone.utc).isoformat()
    index = verify_packet('literature_memory/index_v40')
    verify_packet('graph_structure_conditioned_specialization_scout_20261003_v1')
    integration = read('literature_memory/index_v40/INTEGRATION.json')
    old = read('literature_memory/index_v39/LITERATURE_INDEX.json')
    new = read('literature_memory/index_v40/LITERATURE_INDEX.json')
    assert new['paper_records'][:len(old['paper_records'])] == old['paper_records']
    assert integration['successor_conclusion_records'] == 168
    assert integration['normalized_paper_identifiers'] == 119
    assert not integration['methodological_novelty_established']
    memo = verify_packet('active_graph_hypotheses_scientific_acceptance_path_assessment_20261003_v1')
    nroot = 'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v2'
    nqual = nroot + '/remote_receipts_run01/numerical/run01/QUALIFICATION.json'
    nterm = nroot + '/remote_receipts_run01/supervision/run01/SUPERVISOR_TERMINAL.json'
    q, t = read(nqual), read(nterm)
    assert q['status'] == 'PASS' and q['fabricated_inputs_only']
    assert q['identity']['driver_manifest_sha256'] == 'fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89'
    assert t['status'] == 'COMPLETE' and t['child_exit_code'] == 0 and t['cap_violation'] is None
    assert descriptor(nqual)['sha256'] == t['qualification_receipt']['sha256']
    aroot = 'amazon_polynormer_paired_family_execution_root_20261003_v3'
    aq = read(aroot + '/qualification_block0_cuda0/RESULT.json')
    assert aq['status'] == 'passed' and not aq['report_eligible']
    at = read(aroot + '/qualification_block0_cuda0/TERMINAL.json')
    assert at['physical_exit_code'] == 0
    resources = read(aroot + '/QUALIFICATION_REMOTE_CUSTODY_AND_RESOURCE_v1.json')
    mixed_path = 'coordination_snapshots/20261003_numerical_v3_and_current_families_v1/MIXED_OBSERVATION.json'
    mixed = read(mixed_path)
    assert mixed['terminal_status_counts'] == {'selected': 39, 'unterminated': 1}
    assert all(p['exact_handle_matches'] for p in mixed['exact_processes'])
    nobs_path = 'gpu77_connection_recovery_v1/commands/ncnc77_current_family_after_v3_numerical_20261003_v1/RECEIPT.json'
    nt = read(nobs_path)
    assert nt['exit_code'] == 0
    nobs = json.loads(nt['stdout'])
    assert not nobs['predictive_metrics_or_checkpoint_payloads_opened']
    buddy_path = 'gpu77_connection_recovery_v1/commands/gpu77_accountability_user_20261003_v8/RECEIPT.json'
    bt = read(buddy_path)
    assert bt['exit_code'] == 0
    buddy = json.loads(bt['stdout'])
    assert buddy['read_only_observation'] and not buddy['configuration_mount_driver_or_job_mutations']
    SNAP.mkdir(exist_ok=False)
    for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
        (SNAP / ('BEFORE_' + name)).write_bytes((PHASE / name).read_bytes())
    ledger = read('research_ledger.json')
    before = set(ledger)
    event = dict(UTC=now, kind='numerical_repairs_and_scientific_evidence_path',
                 NCNC_qualification=descriptor(nqual), NCNC_terminal=descriptor(nterm),
                 Amazon_qualification=descriptor(aroot + '/qualification_block0_cuda0/RESULT.json'),
                 Amazon_terminal=descriptor(aroot + '/qualification_block0_cuda0/TERMINAL.json'),
                 Amazon_resources=descriptor(aroot + '/QUALIFICATION_REMOTE_CUSTODY_AND_RESOURCE_v1.json'),
                 literature=index, scientific_path=memo, mixed=descriptor(mixed_path),
                 NCNC_family=descriptor(nobs_path), BUDDY=descriptor(buddy_path),
                 predictive_winner=False, methodological_novelty_established=False,
                 manuscript_acceptance=False, original_paper_scores_changed=False)
    ledger.setdefault('postsubmission_event_log', []).append(event)
    ledger['NCNC_pattern_V3_numerical_qualification_20261003_v1'] = dict(
        qualification=descriptor(nqual), terminal=descriptor(nterm),
        actual_child_exit_code=0, numerical_pass=True, full_graph_pass=False,
        predictive_fits_started=0, predecessor_failed_attempt_preserved=True,
        cross_version_failure_costs=read(nroot + '/ROOT_NUMERICAL_ADMISSION.json')['predecessor_costs'])
    ledger['Amazon_polynormer_V5_numerical_qualification_20261003_v1'] = dict(
        qualification=event['Amazon_qualification'], terminal=event['Amazon_terminal'],
        full_graph_numerical_pass=True, predictive_fits_started=0,
        retention_forecast_bytes=5947364930400, observed_filesystem_available_bytes=2323333513216,
        user_quota_certified=False, prospectively_bounded_retention_successor_required=True,
        resources=event['Amazon_resources'], new_GPU_requirement_established=False)
    ledger['scientific_acceptance_path_20261003_v1'] = dict(
        memo=memo, priority='NCNC joint versus matched marginal observation-pattern supervision',
        contribution_unproven=True, joint_Bernoulli_ancestry='GRAN',
        requirements=['repeatable served quality gain', 'capable structured single and independent ensemble controls',
                      'mechanism diagnostics', 'prospectively frozen heldout confirmation',
                      'fresh independent manuscript review'], acceptance_prediction=False)
    ledger['literature_memory_current'] = dict(
        index=descriptor('literature_memory/index_v40/LITERATURE_INDEX.json'),
        conclusion_records=168, normalized_paper_identities=119, software_identities=2,
        full_paper_read_count_certified=False, newly_scoped_primary_method_reads=5,
        methodological_novelty_established=False)
    ledger['mixed_full40_current_observation_20261003_v5'] = dict(
        UTC=mixed['UTC'], selected=39, total=40, exact_handles=mixed['exact_processes'],
        partial_scores_opened=False, study_closed=False, evidence=descriptor(mixed_path))
    ledger['NCNC_current_observation_20261003_v5'] = dict(
        UTC=nobs['UTC'], evidence=descriptor(nobs_path), live_processes=nobs['live_processes'],
        progress=nobs['progress_metadata'], partial_scores_opened=False, study_closed=False)
    ledger['BUDDY_current_observation_20261003_v5'] = dict(
        UTC=buddy['UTC'], exact_processes=buddy['our_BUDDY_processes'], progress=buddy['progress_only'],
        partial_scores_opened=False, evidence=descriptor(buddy_path))
    ledger['goal_turn_classification'] = dict(classification='concrete_progress',
        reason='Repaired NCNC numerical stage launched and passed; Amazon complete qualification audited; scientific path and literature successor preserved.')
    ledger['current_priority'] = 'Run complete-graph J/F qualification, then the frozen pair if feasible. Evaluate complete current families. Implement bounded Amazon retention before its unchanged paired family. Add capable controls and heldout confirmation only under prospectively fixed protocols.'
    ledger['last_updated_utc'] = ledger['current_status_update_UTC'] = now
    assert before <= set(ledger)
    (PHASE / 'research_ledger.json').write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
    status = (PHASE / 'PUBLIC_STATUS.md').read_text()
    start, end = status.index('Updated: '), status.index('. Goal active')
    status = status[:start] + 'Updated: ' + now + status[end:]
    start, end = status.index('The next fixed comparison uses authored Amazon'), status.index('\nThe completed DBLP')
    status = status[:start] + '''The next fixed comparison uses authored Amazon Polynormer-r with raw features, 200 local and 2500 global epochs. Its V5 implementation passed complete real-graph numerical qualification, all ten complete-state next-update replays, and all declared local/global parity checks. The qualifier exited 0 in 213.68 seconds. The scientific design remains three paired blocks, 15 fits, GNNM4 versus four independent native members, with native single aliasing member 0. No predictive fit has started. Retaining every improved checkpoint could consume about 5.95 TB, exceeding the observed 2.32 TB available. A prospective retention successor will keep the local transition and final selected states, all scores/decisions and honest retirement records. Existing V4 failure costs remain preserved. Numerical success establishes implementation behavior, not predictive utility.
''' + status[end:]
    start, end = status.index('At 17:56:40 UTC, the mixed'), status.index('\n## Comparator qualification')
    status = status[:start] + '''At 18:43:56 UTC, the mixed shared/private objective family had 39 of 40 selected cases. The original handles matched; one ACM case remained. No partial quality comparison was performed.

At 18:48:22 UTC, the NCNC GPU1 queue was COMPLETE. The original GPU0 queue remained live, with factor-private seed 4 at epoch 76; its remaining units still had to finish. The family requires five seeds, 35 fits and 25 served cells. TEST remains locked.

At 18:38:42 UTC, BUDDY's original supervisors remained live. Factorized4 and independent4 seed 2 reached epochs 45 and 18. The complete family requires 15 cells and 24 fits. No partial quality comparison was performed. The read-only host check found uptime of 199 days and continuing training; the earlier isolation and restart mistakes remain recorded.
''' + status[end:]
    start, end = status.index('Literature memory [index_v39]'), status.index('\nA capable count-aware single-model comparator')
    status = status[:start] + '''Literature memory [index_v40](literature_memory/index_v40/LITERATURE_INDEX.json) retains 168 scoped conclusion records across 119 normalized paper identities and two software identities. Five additional method scopes were read. These are bounded reads, not whole-paper certifications or proof of prior-work absence. Earlier unresolved primary leads remain preserved.

The frozen NCNC pilot compares joint reconstruction of whole TRAIN observation patterns with equally supervised reconstruction of individual incidences. Repaired V3 passed its numerical stage at 18:43:40 UTC and the physical child exited 0. Complete-graph qualification and its two fresh 100-epoch predictive fits remain pending. GRAN supplies the mixture likelihood ancestry. Observation absence is not verified latent nonlink truth. The [scientific path assessment](active_graph_hypotheses_scientific_acceptance_path_assessment_20261003_v1/MEMO.md) ranks this as the clearest mechanism test, while requiring capable structured singles, independent ensembles, replicated quality gains and heldout confirmation before stronger claims.
''' + status[end:]
    (PHASE / 'PUBLIC_STATUS.md').write_text(status)
    (PHASE / 'RESEARCH_STATE.md').write_text(f'''# GNNM post-submission research state

Updated: {now}. Goal active and incomplete. Original paper scores are unchanged.

Repaired NCNC V3 passed numerical qualification. Complete-graph qualification and the frozen J/F pair remain next. The likelihood is credited to GRAN; the prospective contribution is its graph completion supervision under restricted sharing. No predictive gain or novelty is established. Capable structured singles, independent ensembles, mechanism analysis, paired replication and heldout confirmation remain required.

Amazon Polynormer V5 passed complete real-graph numerical qualification. Its 15-fit study has not started. Worst-case checkpoint retention exceeds observed disk availability; a prospective bounded retention successor is being prepared without changing scientific selection or recipe. Mixed 39/40, BUDDY and the NCNC GPU0 queue are advancing. Partial results remain unscored.

Literature index_v40 retains 168 scoped conclusions across 119 paper identities, with earlier unresolved leads preserved. See [status](PUBLIC_STATUS.md), [scientific path](active_graph_hypotheses_scientific_acceptance_path_assessment_20261003_v1/MEMO.md), and [full ledger](research_ledger.json). No new audited winner, revised manuscript or acceptance verdict. New records await publication; latest verified synchronized head is `e2bcb4a61f9e5d8667f4823a102ea69bb70360fb`.
''')
    (SNAP / 'ADOPTION_EVENT.json').write_text(json.dumps(event, indent=2) + '\n')
    print(json.dumps(dict(UTC=now, preserved_ledger_keys=len(before), event=descriptor(str((SNAP / 'ADOPTION_EVENT.json').relative_to(PHASE))))))


if __name__ == '__main__':
    main()
