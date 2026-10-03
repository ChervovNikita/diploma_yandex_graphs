"""Adopt verified source/literature work without promoting unmeasured methods."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

PHASE = Path(__file__).resolve().parent.parent
SNAP = PHASE / 'coordination_snapshots/20261003_pattern_literature_source_adoption_v1'


def read(relative):
    return json.loads((PHASE / relative).read_text())


def desc(relative):
    path = PHASE / relative
    data = path.read_bytes()
    return dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify_packet(relative):
    packet = PHASE / relative
    manifest = json.loads((packet / 'MANIFEST.json').read_text())
    rows = manifest.get('payload', manifest.get('files'))
    assert isinstance(rows, list)
    for row in rows:
        data = (packet / row['path']).read_bytes()
        assert len(data) == row.get('bytes', row.get('size'))
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    return desc(relative + '/MANIFEST.json')


def main():
    now = datetime.now(timezone.utc).isoformat()
    review_path = 'amazon_polynormer_v5_adam_image_ownership_source_review_20261003_v1/REVIEW.json'
    review = read(review_path)
    assert review['status'] == 'passed' and not review['execution_authorized']
    source = verify_packet('amazon_polynormer_paired_family_source_preparation_20261003_v5')
    assert review['source']['manifest'] == source
    assert not review['numerical_equivalence_of_repair_verified']
    ncnc_review_path = 'graph_ncNC_structural_pattern_v2_closure_repair_independent_source_review_20261003_v1/REVIEW.json'
    ncnc_review = read(ncnc_review_path)
    assert ncnc_review['status'] == 'passed'
    pattern = verify_packet('graph_ncNC_structural_pattern_pilot_preparation_20261003_v2')
    index = verify_packet('literature_memory/index_v39')
    integration = read('literature_memory/index_v39/INTEGRATION.json')
    assert integration['successor_conclusion_records'] == 163
    assert integration['normalized_paper_identifiers'] == 114
    assert integration['software_documentation_identifiers'] == 2
    assert integration['verification']['old_paper_records_prefix_equal']
    assert not integration['methodological_novelty_established']
    comparator = verify_packet('ncnc_cardinality_single_comparator_source_plan_20261003_v1')
    mixed_path = 'coordination_snapshots/20261003_post_accountability_resume_v1/MIXED_OBSERVATION.json'
    mixed = read(mixed_path)
    assert all(p['exact_handle_matches'] for p in mixed['exact_processes'])
    assert mixed['terminal_status_counts'] == {'selected': 37, 'unterminated': 3}
    assert not mixed['study']['exists']
    ncnc_path = 'gpu77_connection_recovery_v1/commands/ncnc77_post_accountability_resume_20261003_v1/RECEIPT.json'
    transport = read(ncnc_path)
    assert transport['exit_code'] == 0
    ncnc = json.loads(transport['stdout'])
    queues = [q for q in ncnc['live_processes'] if q['PID'] in (3136924, 3136925)]
    assert len(queues) == 2 and all(q['starttime_ticks'] == '1717764805' for q in queues)
    assert not ncnc['predictive_metrics_or_checkpoint_payloads_opened']
    buddy_path = 'gpu77_connection_recovery_v1/commands/gpu77_accountability_user_20261003_v7/RECEIPT.json'
    buddy_transport = read(buddy_path)
    assert buddy_transport['exit_code'] == 0
    buddy = json.loads(buddy_transport['stdout'])
    assert buddy['read_only_observation'] and not buddy['configuration_mount_driver_or_job_mutations']
    failure_path = 'amazon_polynormer_paired_family_execution_root_20261003_v2/qualification_block0_cuda0/TERMINAL.json'
    failure = read(failure_path)
    assert failure['physical_exit_code'] == 1
    SNAP.mkdir(exist_ok=False)
    for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
        (SNAP / ('BEFORE_' + name)).write_bytes((PHASE / name).read_bytes())
    ledger = read('research_ledger.json')
    before_keys = set(ledger)
    event = dict(UTC=now, kind='V5_source_review_and_pattern_literature_adoption',
                 Amazon_source=source, Amazon_review=desc(review_path),
                 preserved_V4_numerical_failure=desc(failure_path),
                 pattern_source=pattern, pattern_review=desc(ncnc_review_path),
                 literature=index, comparator_plan=comparator,
                 mixed=desc(mixed_path), NCNC=desc(ncnc_path), BUDDY=desc(buddy_path),
                 predictive_winner=False, methodological_novelty_established=False,
                 manuscript_acceptance=False, original_paper_scores_changed=False)
    ledger.setdefault('postsubmission_event_log', []).append(event)
    ledger['previous_goal_turn_classification'] = dict(classification='verified_wait',
        reason='Read-only17:54UTC observation confirmed original BUDDY supervisors and advancing seed2 workers. No scientific result or server repair was claimed.')
    ledger['Amazon_polynormer_V5_source_review_20261003_v1'] = dict(
        source=source, review=desc(review_path), preserved_V4_failure=desc(failure_path),
        numerical_qualification_passed=False, predictive_fits_started=0,
        exact_saved_optimizer_image_alias_defect_identified=True,
        repair_success_or_sole_numerical_cause_unproven=True,
        original_tolerances_and_scientific_recipe_unchanged=True)
    ledger['NCNC_pattern_V2_source_and_comparator_plan_20261003_v1'] = dict(
        source=pattern, review=desc(ncnc_review_path), comparator_plan=comparator,
        numerical_qualification_passed=False, predictive_fits_started=0,
        mechanism='Joint versus factorial reconstruction of TRAIN observation-incidence patterns in four tied NCNC scorers.',
        known_mixture_ancestry='GRAN', missing_link_posterior_claimed=False,
        strong_single_control_required=True)
    ledger['literature_memory_current'] = dict(index=desc('literature_memory/index_v39/LITERATURE_INDEX.json'),
        conclusion_records=163, normalized_paper_identities=114, software_identities=2,
        full_paper_read_count_certified=False, newly_scoped_primary_method_reads=5,
        unresolved_primary_leads=integration['blocked_metadata_leads'],
        predictive_evidence=False, methodological_novelty_established=False)
    ledger['mixed_full40_current_observation_20261003_v4'] = dict(UTC=mixed['UTC'],
        selected=37, total=40, exact_handles=mixed['exact_processes'],
        partial_scores_opened=False, study_closed=False, evidence=desc(mixed_path))
    ledger['NCNC_current_observation_20261003_v4'] = dict(UTC=ncnc['UTC'],
        exact_queue_handles=queues, predictive_scores_opened=False, study_closed=False, evidence=desc(ncnc_path))
    ledger['BUDDY_current_observation_20261003_v4'] = dict(UTC=buddy['UTC'],
        exact_processes=buddy['our_BUDDY_processes'], progress=buddy['progress_only'],
        predictive_scores_opened=False, evidence=desc(buddy_path))
    ledger['current_priority'] = 'Run the exact V5 Amazon runtime/numerical qualification, then admit the full15-fit family from measured costs. Qualify the frozen NCNC joint/factorial pilot, retain a capable structured-single comparator, and evaluate only complete closed families.'
    ledger['last_updated_utc'] = ledger['current_status_update_UTC'] = now
    assert before_keys <= set(ledger)
    (PHASE / 'research_ledger.json').write_text(json.dumps(ledger, indent=2, sort_keys=True) + '\n')
    status = (PHASE / 'PUBLIC_STATUS.md').read_text()
    first = status.index('Updated: ')
    end = status.index('. Goal active', first)
    status = status[:first] + 'Updated: ' + now + status[end:]
    start = status.index('The next fixed comparison uses source-authored Amazon')
    end = status.index('\nThe completed DBLP', start)
    status = status[:start] + '''The next fixed comparison uses authored Amazon Polynormer-r with raw features, 200 local epochs, 2500 global epochs and its native state transition. Three blocks compare GNNM4 with four genuinely independent native members. A native single aliases the first independent member, giving 15 distinct fits. V4 failed its complete next-update replay gate with exit 1. The installed Adam implementation allowed its saved step tensor to be aliased between restores. V5 gives each optimizer an owned saved-state copy and checks image immutability. Independent source review passed, but repaired numerical equivalence remains unverified. The original gates, scientific recipe, and all failed costs are preserved. No predictive fit has started.
''' + status[end:]
    start = status.index('At17:03:53UTC,')
    end = status.index('\n## Comparator qualification', start)
    status = status[:start] + '''At 17:56:40 UTC, the mixed shared/private objective family had 37 of 40 selected cases. The original supervisor and child matched their recorded scripts and start times. Three ACM cases remained, with no study closure and no partial quality comparison.

At 17:56:40 UTC, both original NCNC queues on 18.77 remained live with exact recorded start times. Native bank seed 2 finished 100 epochs. Native70 seed 2 reached epoch 14 and native bank seed 3 reached epoch 71. The complete family requires five seeds, 35 fits and 25 served cells. TEST remains locked.

At 17:54:08 UTC, the original BUDDY supervisors on 18.77 were live. Native1024 and single256 seed 2 finished 100 epochs. Factorized4 and independent4 seed 2 reached epochs 29 and 3. The complete family requires 15 cells and 24 fits. No partial quality comparisons were performed.
''' + status[end:]
    start = status.index('Literature memory [index_v38]')
    end = status.index('\nLatest verified pushed', start)
    status = status[:start] + '''Literature memory [index_v39](literature_memory/index_v39/LITERATURE_INDEX.json) retains 163 scoped conclusion records across 114 normalized paper identities and two software identities. Five new primary method scopes were read. These counts do not certify whole-paper reads. Two close primary sources remain inaccessible and unresolved.

The new NCNC pilot compares joint reconstruction of TRAIN observation patterns with equally supervised reconstruction of their marginal incidences. Its two fresh 100-epoch fits use the same masking, completion bank and main target loss. GRAN supplies the mixture-of-Bernoulli ancestry. Source V2 passed independent review after a preserved closure-custody rejection. Numerical and complete-graph qualification remain required before training. Observation absence is not treated as verified absence of a latent link. No novelty or predictive value is established.

A capable count-aware single-model comparator is specified prospectively. It can model dependence through the total residual count and keeps that distribution through four paid decoder draws. A matched-marginal diagnostic removes dependence from the same selected bank. This is a source plan, not an implemented or qualified method. Its complete-support dynamic programming may be expensive. The analytical higher-order witnesses do not establish that such patterns occur usefully in the real graph.
''' + status[end:]
    (PHASE / 'PUBLIC_STATUS.md').write_text(status)
    (PHASE / 'RESEARCH_STATE.md').write_text(f'''# GNNM post-submission research state

Updated: {now}. Goal remains active and incomplete. Original paper scores remain unchanged.

All 30 initializer checkpoints passed replay. Its scientific no-promotion decision is unchanged. Amazon V4 failed next-update replay. V5 repairs optimizer-image ownership and passed source review, with numerical qualification still required before its fixed 15-fit family. NCNC joint/factorial observation-pattern source V2 passed review. Its numerical pilot and a capable structured single control remain unmeasured. Mixed 37/40, BUDDY and the original NCNC family remain live. Partial outcomes stay unscored.

Literature index_v39 retains 163 scoped conclusions, with two unresolved primary leads. See [current status](PUBLIC_STATUS.md), [full ledger](research_ledger.json), and immutable evidence packets. Latest verified pushed and synchronized head is `e2bcb4a61f9e5d8667f4823a102ea69bb70360fb`. New follow-up records await publication. No new audited predictive winner, established novelty, revised manuscript or independent acceptance verdict.
''')
    (SNAP / 'ADOPTION_EVENT.json').write_text(json.dumps(event, indent=2) + '\n')
    print(json.dumps(dict(UTC=now, preserved_ledger_keys=len(before_keys), adopted_event=desc(str((SNAP / 'ADOPTION_EVENT.json').relative_to(PHASE))))))


if __name__ == '__main__':
    main()
