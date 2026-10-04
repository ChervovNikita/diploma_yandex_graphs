"""Preserve completed research findings and publish an explicit text inventory."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

P = Path(__file__).resolve().parents[1]
IDENTITY = 'native_success_ddi_and_loader_failure_20261004_v1'
PUB = P / 'publication' / IDENTITY
SNAP = P / 'coordination_snapshots' / ('20261004_' + IDENTITY + '_before_state')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((P / relative).read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    previous_path = P / 'publication/completed_qa_and_stop_decision_20261004_v1/PUSH_RECEIPT.json'
    previous = json.loads(previous_path.read_text())
    assert previous['verified'] and previous['remote_commit'] == '81241158e31bf2ee087fca25974468cd5e421c13'
    native = read('exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1/ROOT_ADOPTION.json')
    assert native['decision'] == 'PASS_FIXED_ORIGINAL_RULE'
    assert native['physical_completion']['physical_session_closed'] and native['physical_completion']['physical_exit_code'] == 0
    ddi = read('ddi_selective_train_acquisition_execution_root_20261004_v2/MONITOR_01_RESULT.json')['files']['ACQUISITION_RESULT.json']['data']
    assert ddi['status'] == 'COMPLETE_OFFICIAL_TRAIN_ACQUISITION_ONLY' and ddi['heldout_split_payloads_decoded'] is False
    pencil = read('pencil_collab_resource_qualifier_execution_root_20261004_v2/owned_monitor02/OBSERVATION.json')
    assert pencil['terminal']['status'] == 'FAILED_NO_RESOURCE_ADOPTION'
    assert pencil['physical_terminal']['physical_session_closed'] and pencil['physical_terminal']['physical_exit_code'] == 1
    stderr = (P / 'pencil_collab_resource_qualifier_execution_root_20261004_v2/owned_monitor02/supervision/run01/STDERR.txt').read_text()
    assert 'Caught RuntimeError in pin memory thread' in stderr and 'CUDA error: invalid argument' in stderr
    monitor_path = P / 'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0029_RESULT.json'
    amazon = json.loads(monitor_path.read_text())
    progress = [dict(fit_id=x['fit_id'], updates=x['trace_progress']['actual_update'],
                    terminal=(x.get('physical_terminal') or {}).get('status'))
                for x in amazon['registered_fit_progress'] if x.get('trace_progress')]
    assert sum(x['terminal'] == 'success' for x in progress) == 5 and not amazon['failures']
    now = datetime.now(timezone.utc).isoformat()
    PUB.mkdir(); SNAP.mkdir()
    for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
        shutil.copy2(P / name, SNAP / name)
    event = dict(UTC=now, goal='active_incomplete', original_paper_scores_unchanged=True,
                 predictive_success_new=False, fresh_manuscript_acceptance=False,
                 native_bucket=dict(decision=native['decision'], comparisons=96, both_unused=16,
                    optimizer_updates=0, heldout_reads=False,
                    loss_forward_seconds=dict(direct=8.704110708087683, bucket=.8715233653783798),
                    joint_reverse_seconds=dict(direct=29.064976133406162, bucket=4.559713277965784),
                    timing_limit='Single fixed-order retained-graph diagnostic; no fit speed or memory claim.',
                    adoption='exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1/ROOT_ADOPTION.json'),
                 paired_predictive=dict(source='exact_cb_support_bucket_paired_predictive_preparation_20261004_v1',
                    manifest_sha256='0d0899d94f0a20d317f7e30491f3ed5b35c293d462f28a0baf3f4ea350e9d229',
                    fixed_fits=9, total_updates=15300, planning_GPU_hours=[24,30],
                    execution=False, independent_source_review='Fresh review in progress; no result adopted by this event.'),
                 PENCIL=dict(status='FAILED_NO_RESOURCE_ADOPTION', TRAIN_batches=7, optimizer_updates=0,
                    VALID_queries=0, TEST_reads=False, physical_session_closed=True,
                    cause='RuntimeError in DataLoader pin-memory thread: CUDA error: invalid argument.',
                    server_configuration_modified=False, automatic_retry=False,
                    next_action='Preserve failed v2; prepare a disclosed loader-only transfer repair without changing scientific budget/model.'),
                 DDI=dict(status=ddi['status'], TRAIN_edges=ddi['TRAIN_edges'], nodes=ddi['nodes'],
                    support_census=False, predictive_fit=False, heldout_decoded=False,
                    failed_HTTPS_attempt_preserved=True,
                    result='ddi_selective_train_acquisition_execution_root_20261004_v2/MONITOR_01_RESULT.json'),
                 Amazon=dict(observation_UTC=amazon['UTC'], completed=5, total=15,
                    progress=progress, failures=[], quality_or_TEST_selection=False),
                 literature=dict(index='literature_memory/index_v46/LITERATURE_INDEX.json',
                    scoped_conclusions=200, normalized_paper_identities=149,
                    full_paper_read_total_certified=False,
                    note='Two scoped primary method reads; generic reconstruction/mixture credit is prior. Mathematical overlap analysis does not guarantee diversity.'))
    save(PUB / 'RESEARCH_EVENT.json', event)
    ledger_path = P / 'research_ledger.json'
    ledger = json.loads(ledger_path.read_text()); original_keys = set(ledger)
    assert IDENTITY not in ledger
    ledger[IDENTITY] = event
    ledger['updated_UTC'] = ledger['updated_utc'] = now
    ledger['publication_history'].append(dict(event='completed_qa_and_stop_decision_20261004_v1',
        commit=previous['remote_commit'], receipt=str(previous_path.relative_to(P)), exact_ref_verified=True))
    ledger['latest_verified_publication'] = dict(commit=previous['remote_commit'], branch=previous['branch'],
        receipt=str(previous_path.relative_to(P)), receipt_sha256=sha(previous_path), GitHub_exact_ref_verified=True,
        before_current_publication=True)
    ledger['publication_state'] = dict(branch=previous['branch'], latest_pushed_commit=previous['remote_commit'],
        pending_changes='Native bucket diagnostic PASS, DDI TRAIN acquisition, PENCIL loader failure, precise literature and Amazon29 metadata.')
    assert original_keys <= set(ledger)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + '\n')
    status = f'''# Current GNNM research status

Updated: {now}. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified new result remains five-seed official ogbl-collab TEST: GNNM private completion 67.2909% Hits@50, native single64 66.4426%, independent ensemble4 67.6298%. The exploratory single-model gain is +0.8483 percentage points, positive in all five seeds. The independent ensemble remains higher by 0.3389 points.

The frozen private-minus-pooled contrast is +0.2236 points, paired descriptive 95% seed interval [-0.7775,+1.2247], exact sign-flip p=.6875. It is inconclusive. Collab TEST is consumed; successor development must disclose this history. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed methodological advantage or fresh manuscript acceptance exists.**

## Current methodological test

The TRAIN-only pattern objective asks whether one member can explain residual-neighbour subsets at both endpoints of an edge. The served link ranker stays count-free. A fixed comparison of target-only, shared-responsibility and independently mixed endpoints uses three paired seeds and nine fresh full 100-epoch fits. Source preparation is complete; fresh technical review and root execution preparation are in progress. No fit has launched in this event. Planned serial runtime is 24–30 GPU-hours, rather than a measured duration.

The joint-versus-separate contrast measures endpoint responsibility overlap. Identical members give a zero contrast, so it does not guarantee diversity or prevent collapse. [Algebra and limits](pattern_responsibility_overlap_analysis_20261004_v1/ANALYSIS.md). Conditional Bernoulli laws, mixtures/cardinality inference, MaskGAE, GRAN and neighbourhood reconstruction are prior. [New scoped comparisons](conditional_neighbourhood_generation_prior_check_20261004_v1/CONCLUSIONS.md).

The actual native direct-versus-bucket comparison passed all 96 numerical reports plus 16 both-unused records under the original tolerances. Loss forward took 8.7041s versus .8715s; joint reverse took 29.0650s versus 4.5597s. This one fixed-order diagnostic establishes implementation agreement and practical feasibility, with zero updates or heldout reads. It does not establish end-to-end speed, fit memory, novelty or prediction gains. [Native result](exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The earlier native gradient signal is small: joint norm 2.785% of target and joint-minus-separate 0.0561% of target, with both-genuine supports on 1.3599% of positive queries. Preserve that concern; no numerical non-equivalence or predictive transfer is claimed.

## Active comparisons and failures

Amazon's original queue at {amazon['UTC']} had five of fifteen fits complete. The sixth, `split1_gnnm_boundary_4_seed29`, had 1115/2700 updates. No failure, restart or partial quality/TEST selection was observed.

PENCIL's pinned feature-enabled author model was actually attempted on authorized 18.77 GPU0. It stopped after seven TRAIN batches, before the first optimizer update or VALID traversal, with a host-memory pinning error. The owned processes exited cleanly, and all failure evidence remains preserved. No baseline quality or full-epoch resource result exists. A disclosed transfer-only repair is the next action; no server configuration or unrelated job was changed.

DDI's official TRAIN acquisition succeeded: 1,067,911 unique undirected edges, 4,267 nodes, no self-loops. Only TRAIN and the node-count member were decoded; opaque archive bytes include heldout payloads. The HTTPS failure is preserved, and acquisition succeeded using the exact official HTTP URL. No support census, fit or DDI score exists yet.

Six native Pubmed baselines are complete; no GNNM predictive result exists. The 72-update continuation diagnostic failed the original final parity rule, and the ad hoc engineering branch is closed. [Preserved result](pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v46 contains 200 scoped conclusions across 149 paper identities and two software identities; these are not full-paper read totals. Failed experiments, original scores, all reviews and decisions remain preserved. Latest verified GitHub head before this update: {previous['remote_commit']}; this event awaits publication.

Science uses authorized anogena-2 and 18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.
'''
    (P / 'PUBLIC_STATUS.md').write_text(status)
    (P / 'RESEARCH_STATE.md').write_text(f'''# GNNM research state

Updated: {now}. Goal active and incomplete. [Evidence and limits](PUBLIC_STATUS.md). Prior canonical files preserved in {SNAP.relative_to(P)}; all ledger keys retained.

## Next actions

1. Finish fresh source review and launch the fixed nine-fit target/joint/separate comparison on GPU1. No further qualification ladder or TEST release.
2. Preserve PENCIL v2's cleanly exited pin-memory failure. Prepare a disclosed loader transfer repair, retaining architecture, data draws, batch1024, accumulation8, native schedule and caps.
3. Maintain the original Amazon queue; observation29 has five complete fits and sixth at1115/2700. Wait for the fixed family rather than selecting partial outcomes.
4. Use the newly authenticated DDI TRAIN payload for a prospective native support/cost census before fitting. Finish recent strong-DDI source scouting.
5. Keep Pubmed's ad hoc continuation branch closed. Revise the manuscript only from supported results, then run fresh skill-based independent review without a requested verdict.

Original scores and failures remain preserved. No confirmed methodological advantage or acceptance. Authorized project repositories only, no sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.
''')
    roots = [
        'exact_cb_support_bucket_native_equivalence_resource_execution_root_20261004_v1',
        'exact_cb_support_bucket_native_equivalence_resource_root_adoption_20261004_v1',
        'exact_cb_support_bucket_paired_predictive_preparation_20261004_v1',
        'pencil_collab_resource_qualifier_independent_source_review_20261004_v1',
        'pencil_collab_resource_qualifier_preparation_20261004_v2',
        'pencil_collab_resource_qualifier_independent_source_review_20261004_v2',
        'pencil_collab_resource_qualifier_execution_root_20261004_v2',
        'ddi_selective_train_acquisition_execution_root_20261004_v1',
        'ddi_selective_train_acquisition_execution_root_20261004_v2',
        'dense_graph_auxiliary_transfer_dataset_scout_20261004_v1_followup_metadata',
        'conditional_neighbourhood_generation_prior_check_20261004_v1',
        'pattern_responsibility_overlap_analysis_20261004_v1',
        'literature_memory/index_v46', str(SNAP.relative_to(P))]
    paths = [P / n for n in ['PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json',
        'prepare_ddi_selective_train_acquisition_20261004_v1.py', 'prepare_ddi_selective_train_acquisition_20261004_v2.py',
        'publication/record_neighbourhood_prior_memory_20261004_v1.py']]
    paths += [Path(__file__), PUB / 'RESEARCH_EVENT.json', previous_path, previous_path.with_name('ACKNOWLEDGEMENT.json'), monitor_path]
    allowed = {'.py', '.json', '.jsonl', '.md', '.txt', '.html', '.diff', '.patch', '.log', '.raw', '.sha256', '.csv', '.xml', '.sh', '.yml', '.yaml'}
    for directory in roots:
        paths += [p for p in (P / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and (p.suffix in allowed or p.name in ('.project-root', '.gitignore', 'SHA256SUMS', 'NOTE_SHA256SUMS'))]
    rows = []
    for path in sorted(set(paths)):
        assert not path.is_symlink() and path.resolve().is_relative_to(P) and path.stat().st_size < 2_000_000
        relative = str(path.relative_to(P))
        rows.append(dict(source=relative, target='experiments_iclr/postsubmission_20260930/' + relative,
                         bytes=path.stat().st_size, sha256=sha(path)))
    save(PUB / 'INVENTORY.json', dict(UTC=now, branch=previous['branch'], expected_head=previous['remote_commit'],
        files=rows, remove=[], message='Record native auxiliary implementation result, DDI acquisition and preserved PENCIL loader failure'))
    print(json.dumps(dict(inventory=str((PUB / 'INVENTORY.json').relative_to(P)), files=len(rows), bytes=sum(r['bytes'] for r in rows))))


if __name__ == '__main__':
    main()
