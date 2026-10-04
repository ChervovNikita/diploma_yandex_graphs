"""Preserve history and publish completed evidence without predictive promotion."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parents[1]
PUB = P / 'publication/completed_qa_and_stop_decision_20261004_v1'
SNAP = P / 'coordination_snapshots/20261004_completed_qa_and_stop_before_state_v1'
PUB.mkdir(); SNAP.mkdir()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
for name in ['PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json']:
    shutil.copy2(P / name, SNAP / name)
now = datetime.now(timezone.utc).isoformat()
prior_path = P / 'publication/repeatability_and_baseline_progress_20261004_v1/PUSH_RECEIPT.json'
prior = json.loads(prior_path.read_text())
assert prior['verified'] and prior['remote_commit'] == 'c03ae8df191b5a06816ee928ab9b06db51b4d40e'
cpu = json.loads((P / 'exact_cb_support_bucket_cpu_qa_root_adoption_20261004_v1/ROOT_ADOPTION.json').read_text())
pub = json.loads((P / 'pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/ROOT_ADOPTION.json').read_text())
assert pub['decision'] == 'STOP_ENGINEERING_BRANCH_NO_DEMONSTRATED_REPAIR'
installer = json.loads((P / 'pencil_repo_owned_dependency_install_execution_root_20261004_v1/MONITOR_01_RESULT.json').read_text())['INSTALL_RESULT.json']
assert installer['value']['status'] == 'COMPLETE_REPO_OVERLAY_INSTALL_ONLY'
monitor_path = P / 'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1/MONITOR_0028_RESULT.json'
monitor = json.loads(monitor_path.read_text())
progress = [{'fit_id': x['fit_id'], 'updates': x['trace_progress']['actual_update'], 'terminal': (x.get('physical_terminal') or {}).get('status')} for x in monitor['registered_fit_progress'] if x.get('trace_progress')]
assert sum(x['terminal'] == 'success' for x in progress) == 5
event = dict(UTC=now, goal='active_incomplete',
    bucket_CPU=dict(cases=119, comparisons=3288, actual_PASS=True, adoption='exact_cb_support_bucket_cpu_qa_root_adoption_20261004_v1/ROOT_ADOPTION.json', native_GPU_or_predictive_benefit=False),
    bucket_native=dict(source_review='PASS source only', manifest_sha256='8b8adcbb90366098c2482220612b07868d9fe2bb86e7e879fa928e0c0537d98c', numerical_result=False),
    Pubmed=dict(decision=pub['decision'], updates=72, original_qualification='FAILED', tolerance_relaxed=False, predictive_result=False, adoption='pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/ROOT_ADOPTION.json'),
    PENCIL=dict(overlay_installed=True, new_packages=16, core_versions_unchanged=True, actual_install_result=installer, resource_execution=False, v1_source_ownership_concern='Independent review raised torchrun rank session escape; review completion and minimal direct-rank repair pending.'),
    DDI=dict(prospective_documentation_scout='dense_graph_auxiliary_transfer_dataset_scout_20261004_v1', dataset_acquired=False, TRAIN_support_measured=False, predictive_result=False),
    Amazon=dict(observation_UTC=monitor['UTC'], completed=5, total=15, progress=progress, failures=monitor['failures'], quality_or_TEST_selection=False),
    new_predictive_advantage=False, fresh_manuscript_acceptance=False, original_paper_scores_unchanged=True)
save(PUB / 'RESEARCH_EVENT.json', event)
ledger = json.loads((P / 'research_ledger.json').read_text())
old_keys = set(ledger); key = 'completed_qa_and_stop_decision_20261004_v1'; assert key not in ledger
ledger[key] = event
ledger['updated_UTC'] = ledger['updated_utc'] = now
ledger['publication_history'].append(dict(event='repeatability_and_baseline_progress_20261004_v1', commit=prior['remote_commit'], receipt=str(prior_path.relative_to(P)), exact_ref_verified=True))
ledger['latest_verified_publication'] = dict(commit=prior['remote_commit'], branch=prior['branch'], receipt=str(prior_path.relative_to(P)), receipt_sha256=sha(prior_path), GitHub_exact_ref_verified=True, before_current_publication=True)
ledger['publication_state'] = dict(branch=prior['branch'], latest_pushed_commit=prior['remote_commit'], pending_changes='Actual CPU equivalence, Pubmed stop decision, PENCIL installed overlay and source packets, prospective DDI scout and Amazon28 metadata.')
assert old_keys <= set(ledger)
(P / 'research_ledger.json').write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False)+'\n')
(P / 'PUBLIC_STATUS.md').write_text(f'''# Current GNNM research status

Updated: {now}. Goal active and incomplete. Original paper scores unchanged.

## Predictive evidence

The strongest verified new result remains five-seed official ogbl-collab TEST: GNNM private completion 67.2909% Hits@50, native single64 66.4426%, independent ensemble4 67.6298%. The exploratory single-model gain is +0.8483 percentage points, positive in all five seeds. The independent ensemble remains higher by 0.3389 points.

The frozen private-minus-pooled contrast is +0.2236 points, paired descriptive95% seed interval [-0.7775,+1.2247], exact sign-flip p=.6875. It is inconclusive. Collab TEST is consumed; successor development must disclose this history. [Full results](ncnc_frozen_all25_heldout_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

**No confirmed methodological advantage or fresh manuscript acceptance exists.**

## Active research

The TRAIN-only pattern objective tests whether one latent member can explain residual-neighbour patterns at both endpoints of an edge. Joint and independently mixed endpoint responsibilities are distinct hypotheses. The serving ranker stays count-free. Conditional Bernoulli laws, mixture/cardinality inference, MaskGAE topology/degree supervision and GRAN shared components are prior; generic novelty is not cleared.

The actual fullTRAIN native gradient check connected this objective to shared/member parameters, with zero optimizer updates and no VALID/TEST. Joint gradient norm was2.785% of target; joint-minus-separate was0.0561% of target. Both-genuine supports occur on1.3599% of positive queries, a sparse-transfer concern. The small initial derivatives are not predictive evidence. [Native diagnostic](graph_count_conditioned_pattern_native_gradient_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

The exact bucket implementation now passed all119 declared CPU cases and3,288 comparisons against independent oracles under unchanged tolerances. Its full native GPU equivalence/resource source passed independent review; actual GPU result and speedup are not yet measured. [Bounded CPU result](exact_cb_support_bucket_cpu_qa_root_adoption_20261004_v1/RESULTS_SUMMARY.md). A three-block paired target/joint/separate predictive decision is recorded prospectively; complete paired fits and an independent benchmark remain required.

DDI was selected for a prospective TRAIN-support census from official documentation and pinned native source, before own data acquisition/scoring. No DDI dataset, support census or fit exists yet. Published whole-graph density does not establish actual TRAIN support.

## Predictive queue and baselines

At {monitor['UTC']}, Amazon's original paired queue had5/15 completed fits; the sixth, `split1_gnnm_boundary_4_seed29`, had730/2700 updates. The monitor recorded no failures, new launches, restarts or partial quality/TEST selection.

PENCIL's stronger feature-enabled link-prediction baseline has a pinned author recipe and complete-epoch/fullVALID resource source. Its16 missing packages are now installed inside the18.77 project overlay; core versions are unchanged. No numerical resource run or baseline score exists. Independent source review raised a concrete torchrun-session ownership concern; the original packet stays immutable while a minimal direct-rank repair is prepared.

Six native Pubmed baselines are complete; no GNNM predictive result exists. The latest two36-update continuations completed physically, but final loss, encoder, predictor, Adam and gradient parity fail the original rule. Initial isolation and selected first-two-update checks pass without establishing a causal repair. Adopted decision: STOP_ENGINEERING_BRANCH_NO_DEMONSTRATED_REPAIR. No further ad hoc repeats, relaxed tolerance or automatic continuation/donor admission. [Result](pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1/RESULTS_SUMMARY.md).

## History and boundaries

Literature index_v45 retains198 scoped conclusions across147 paper identities and two software identities; these are not full-paper read totals. New software/documentation scopes are not counted as new full papers. Failed experiments, all reviews, original scores and prior decisions remain preserved.

Latest verified GitHub head before this update: {prior['remote_commit']}. Current changes await explicit publication. Science uses authorized anogena-2 and18.77 project repositories only; the seven-GPU route is forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes.
''')
(P / 'RESEARCH_STATE.md').write_text(f'''# GNNM post-submission research state

Updated: {now}. Goal active and incomplete. [Current evidence](PUBLIC_STATUS.md). Prior state preserved in {SNAP.relative_to(P)}; all ledger keys retained.

## Next actions

1. Execute the independently reviewed full native bucket equivalence/resource comparison. Keep measured numerical agreement and speed separate from predictive benefit. Then run the prospectively recorded representative target/joint/separate fits if evidence supports resource feasibility.
2. Maintain Amazon's original15-fit queue; monitor28 has5 complete fits and sixth at730/2700. No partial outcome selection or TEST decisions.
3. Close Pubmed's ad hoc continuation debugging branch after the72-update diagnostic. Preserve original FAILED and its evidence. Further science requires its own explicit prospective fresh-start protocol, not automatic donor/continuation admission or a loosened old rule.
4. Use the successfully installed PENCIL repo overlay. Complete source review and minimal direct-rank ownership repair before the full native epoch/fullVALID resource check. No stronger-baseline score follows from environment installation.
5. Consider the prospective DDI TRAIN-only support census to test structural applicability before committing to fits; no dataset or score exists yet.
6. Revise the manuscript only from supported predictive outcomes, then use fresh independent skill-based reviewers with immutable evidence and no requested verdict. Acceptance remains unachieved.

## Boundaries

Original paper scores unchanged. Authorized anogena-2 one-GPU and18.77 project repositories only for science; seven-GPU route forwarding only. No sudo, PDF compilation, GENLINK, Desktop writes, server configuration or unrelated-job changes. Small incidental caches allowed.
''')
roots = [
    'exact_cb_support_bucket_cpu_qa_preparation_20261004_v1',
    'exact_cb_support_bucket_cpu_qa_independent_source_review_20261004_v1',
    'exact_cb_support_bucket_cpu_qa_execution_root_20261004_v1',
    'exact_cb_support_bucket_cpu_qa_root_adoption_20261004_v1',
    'exact_cb_support_bucket_native_equivalence_resource_preparation_20261004_v1',
    'exact_cb_support_bucket_native_equivalence_resource_independent_source_review_20261004_v1',
    'pubmed_shared4_update_inclusive_continuation_diagnostic_source_20261004_v1',
    'pubmed_shared4_update_inclusive_continuation_independent_source_review_20261004_v1',
    'pubmed_shared4_update_inclusive_continuation_diagnostic_execution_root_20261004_v1',
    'pubmed_shared4_update_inclusive_continuation_root_adoption_20261004_v1',
    'pencil_repo_owned_dependency_install_plan_20261004_v1',
    'pencil_repo_owned_dependency_install_execution_root_20261004_v1',
    'pencil_collab_resource_qualifier_preparation_20261004_v1',
    'dense_graph_auxiliary_transfer_dataset_scout_20261004_v1', str(SNAP.relative_to(P))]
paths = [P / name for name in ['PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json']]
paths += [Path(__file__), P / 'publication/prepare_pencil_dependency_install_execution_20261004_v1.py', PUB/'RESEARCH_EVENT.json', prior_path, prior_path.with_name('ACKNOWLEDGEMENT.json')]
for root in roots: paths.extend(f for f in (P/root).rglob('*') if f.is_file() and '__pycache__' not in f.parts)
paths += [monitor_path, monitor_path.with_name('MONITOR_0028_STDOUT.txt'), PUB/'LOCAL_INVENTORY_PREPARATION_FAILURE.json', P/'publication/publish_exact_inventory_v13.py']
allowed = {'.py','.json','.jsonl','.md','.txt','.html','.diff','.patch','.log','.raw','.sha256','.csv','.xml','.sh','.yml','.yaml'}
rows = []
for f in sorted(set(paths)):
    assert f.stat().st_size < 2_000_000 and (f.suffix in allowed or f.name in ['.gitignore','SHA256SUMS','NOTE_SHA256SUMS','.project-root'])
    source = str(f.relative_to(P))
    rows.append(dict(source=source,target='experiments_iclr/postsubmission_20260930/'+source,bytes=f.stat().st_size,sha256=sha(f)))
save(PUB/'INVENTORY.json',dict(UTC=now,branch=prior['branch'],expected_head=prior['remote_commit'],files=rows,remove=[],message='Record completed bounded QA, Pubmed stop decision and baseline readiness'))
print(json.dumps(dict(inventory=str((PUB/'INVENTORY.json').relative_to(P)),files=len(rows),bytes=sum(r['bytes'] for r in rows))))
