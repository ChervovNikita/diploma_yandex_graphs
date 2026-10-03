"""Update current records from adopted physical evidence and exact live handles."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
UTC = datetime.now(timezone.utc).isoformat()
SNAP = ROOT / 'coordination_snapshots/20261003_closed_audits_v1'
SNAP.mkdir(exist_ok=False)


def load(p):
    return json.loads(p.read_text())


def binding(p):
    b = p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


for name in ('research_ledger.json', 'PUBLIC_STATUS.md', 'RESEARCH_STATE.md'):
    (SNAP / ('BEFORE_' + name)).write_bytes((ROOT / name).read_bytes())
ledger = load(ROOT / 'research_ledger.json')
old_keys = set(ledger)
init_path = ROOT / 'graph_init_analysis_companion_v10_cpu_release_v1/NUMERICAL_AUDIT_ADOPTION_v1.json'
q_path = ROOT / 'forde_graph_small_real_B2_oracle_execution_root_20261003_v1/Q03_RESULTS_ADOPTION_v1.json'
init, q = load(init_path), load(q_path)
mixed = load(ROOT / 'graph_mixed40_current_closure_independent_audit_20261003_v1/OBSERVATION.json')
assert mixed['terminal_status_counts'] == {'selected': 32, 'unterminated': 8}
assert all(p['exact_handle_matches'] for p in mixed['exact_processes'])
ncnc_path = ROOT / 'gpu77_connection_recovery_v1/commands/ncnc77_goal_continuation_progress_20261003_v4/RECEIPT.json'
ncnc_outer = load(ncnc_path)
assert ncnc_outer['exit_code'] == 0
ncnc = json.loads(ncnc_outer['stdout'])
buddy_path = ROOT / 'gpu77_connection_recovery_v1/commands/gpu77_user_server_accountability_20261003_v4/RECEIPT.json'
buddy_outer = load(buddy_path)
assert buddy_outer['exit_code'] == 0
buddy = json.loads(buddy_outer['stdout'])
push_path = ROOT / 'publication/complete6_quality_and_audit_progress_20261003_v1/PUSH_RECEIPT.json'
push = load(push_path)
assert push['verified'] and push['remote_commit'] == buddy['git_head'] == '870bfae124e0606c01039bddccbeba015656fbf0'
event = dict(UTC=UTC, kind='complete_numerical_audits_and_current_training',
             initializer_adoption=binding(init_path), Q03_adoption=binding(q_path),
             mixed_observation=binding(ROOT/'graph_mixed40_current_closure_independent_audit_20261003_v1/OBSERVATION.json'),
             ncnc_observation=binding(ncnc_path), BUDDY_observation=binding(buddy_path),
             original_scores_changed=False, new_scientific_winner=False,
             prior_turn_classification='verified_wait',
             prior_turn_evidence='Read-only 77 observation at15:32:37UTC verified original BUDDY supervisor and advancing workers.',
             current_turn_classification='progress',
             next_action='Qualify the fixed authored Polynormer-r comparison, then train all 15 distinct fits after numerical passage. Finish initializer serialized replay. Evaluate only closed mixed/NCNC/BUDDY families.')
ledger.setdefault('postsubmission_event_log', []).append(event)
ledger['previous_goal_turn_classification'] = dict(classification='verified_wait', reason=event['prior_turn_evidence'])
ledger['active_goal_turn_evidence_20261003_v1'] = event
ledger['initializer_V10_complete_saved_array_audit_20261003_v1'] = dict(
    UTC=UTC, adoption=binding(init_path), complete30_cases=True, complete72_phase_bindings=True,
    serialized_checkpoint_inference_completed=False, original_scores_changed=False,
    all_preserved_V8_V9_failures_retained=True, science_decision=init['decision'])
ledger['FoRDE_Q03_complete_full_X_B2_numerical_result_20261003_v1'] = dict(
    UTC=UTC, adoption=binding(q_path), physical_exit_code=0, both_fixed_recipes_passed=True,
    B128_numerical_equivalence=False, scientific_fit_released=False, predictive_success=False)
ledger['mixed_full40_current_observation_20261003_v3'] = dict(
    UTC=mixed['UTC'], selected=32, remaining=8, original_handles=mixed['exact_processes'],
    study_closed=False, partial_outcomes_scored=False, observation=event['mixed_observation'])
ledger['NCNC_current_observation_20261003_v3'] = dict(
    UTC=ncnc['UTC'], live_processes=ncnc['live_processes'],
    journals=[r for r in ncnc['progress_metadata'] if r['kind']=='epoch_journal_metadata_only'],
    complete_family_quality_scored=False, TEST_locked=True, observation=binding(ncnc_path))
ledger['BUDDY_GPU77_progress_20261003_current'] = dict(
    UTC=buddy['UTC'], progress=buddy['progress_only'], live_processes=buddy['our_BUDDY_processes'],
    full_family_quality_scored=False, observation=binding(buddy_path))
ledger['latest_verified_publication'] = dict(commit=push['commit'], branch=push['branch'],
    GitHub_verified=True, gpu77_synchronized=True, one_GPU_repo_committed=True,
    push_receipt=binding(push_path), synchronization_receipt=binding(ROOT/'gpu77_connection_recovery_v1/commands/git77_complete6_quality_audit_safe_sync_20261003_v1/RECEIPT.json'))
ledger['publication_state'] = dict(latest_pushed_commit=push['commit'], gpu77_head=buddy['git_head'],
    current_followup='V9/V10 and Q03 result, literature_v37 and current record publication pending',
    prior_receipt_mismatch_preserved_and_resolved=True)
ledger['literature_memory_current'] = dict(index=binding(ROOT/'literature_memory/index_v37/LITERATURE_INDEX.json'),
    conclusion_records=157, normalized_paper_identities=108, software_identities=2,
    full_paper_read_count_certified=False, latest_scoped_method_read='Dynamic Negative Correlation Learning in Deep Ensemble Learning',
    graph_kernel_NCL_identity_retained=True, newly_promoted_pilots=0)
ledger['latest_literature_memory'] = ledger['literature_memory_current']
ledger['current_priority'] = event['next_action']
ledger['status'] = 'Initializer30_saved_array_audit_complete_no_consistent_gain_Q03_B2_numerical_pass_Amazon_authored_source_preparing_mixed32of40_BUDDY_NCNC_live'
for k in ('updated_UTC','updated_utc','last_updated_utc','current_status_update_UTC'):
    ledger[k] = UTC
ledger['research_outcome_boundary'] = 'No audited new GNNM predictive winner, established methodological extension, revised manuscript or fresh acceptance verdict. Original paper scores remain frozen.'
assert old_keys <= set(ledger)
save(ROOT/'research_ledger.json', ledger)
save(SNAP/'ADOPTION_EVENT.json', event)

status = f'''# Current GNNM research status

Updated: {UTC}. Goal active and incomplete. No new audited GNNM predictive winner, established methodological extension, revised manuscript or independent acceptance verdict. Original paper scores remain unchanged.

## Current scientific decisions

The graph-based initialization variant is closed without more tuning or heldout promotion. All 30 fits and 72 phases finished. The numerical audit recomputed selected-state validation NLL within the original 1e-6 tolerance and the supervisor reaped its child with exit 0. On Photo, graph initialization averages 0.35047 nats NLL versus 0.32173 for unchanged warm copying, with the same mean accuracy. On Squirrel, its differences from random or topology-permuted initialization are very small. These are selected-checkpoint development results on three overlapping split blocks. Serialized-checkpoint inference replay remains outstanding. All failures and costs are retained. See [complete results](graph_init_analysis_companion_v10_cpu_release_v1/RESULTS_SUMMARY_v1.md).

The six native Amazon Ratings fits and independent serialized-checkpoint replay are complete. Defaults average 42.7024% validation accuracy and the Roman transfer 44.0851%, with worse mean NLL and substantial overfit. These scores use the disclosed 80/20 official-TRAIN fit/control adaptation. They establish neither a GNNM gain nor competence under an interchangeable published protocol. Both recipes remain recorded. See [results](amazon_ratings_native_warm_execution_root_20261003_v3/RESULTS_SUMMARY_v1.md).

The next fixed comparison uses source-authored Amazon Polynormer-r, raw features, 200 local epochs followed by 2500 global epochs, and the native state transition. Three blocks compare GNNM4 boundaries with four genuinely independent native members. A native single aliases the immutable first independent member, giving 15 distinct fits. The source is being finished before independent review and real numerical qualification. The one-GPU allocation was verified idle and available at this turn's check. Training has not started.

The completed DBLP comparison includes GAT, Simple-HGN and SeHGNN over five paired split blocks. All 15 native checkpoints passed independent restoration and validation-logit replay. Mean validation accuracy was 93.169% for global BatchEnsemble HGT, 93.909% for GAT, 94.239% for Simple-HGN and 93.992% for SeHGNN. All 30 requested score rows and nine paired comparisons remain available. VALID selects checkpoints and has 243 nodes on each overlapping split. These results do not show GNNM superiority. The completed relation-conditioned HGT study, Squirrel shared-ensemble study and PPI spectral study also supplied no promoted predictive winner.

## Training confirmed live

At 15:37:35 UTC, mixed shared/private objectives had 32 of 40 selected cases, with eight ACM slots remaining. Original supervisor 379192 and child 379193 matched their recorded start times and scripts. No study closure exists and no partial quality comparison was performed.

At 15:40:39 UTC, both original NCNC queues on 77 were live. Seeds 0/1 had completed their private, pooled, native-bank and width70 units. Seed2 private training had reached epoch61 and seed3 private epoch3. The fixed family retains five seeds, 35 distinct fits and 25 served cells. TEST remains locked and no partial outcomes were scored.

At 15:32:37 UTC, original BUDDY supervisors and workers on77 were live. Independent seed1 was at epoch96, matched-single seed1 had finished100 epochs, and native1024 seed2 was at epoch32. The complete family requires15 cells and24 fits. No partial quality comparisons were performed.

## Comparator qualification

FoRDE's Gram backend failed severe float32 cancellation and remains excluded. The explicit streamed backend passed its tiny CPU checks and both full-Amazon M4/B128 derivative resource profiles, with peak allocated memory1.350/2.319 GiB. Q03 then passed every declared value and private-gradient comparison for two fixed full-input M4/B2 CPU-cache/CUDA endpoints. Its console exited0 in279.111 seconds. There were252/860 comparison records, with largest scaled errors0.001808/0.001860 against a required maximum1. This establishes those B2 endpoints. B128 numerical equivalence and predictive benefit remain unproven. The graph adapter epsilon1e-24 differs from the upstream default1e-12. No trained profile state becomes a donor. See [Q03 report](forde_graph_small_real_B2_oracle_execution_root_20261003_v1/RESULTS_SUMMARY_v1.md).

## Literature and provenance

Literature memory [index_v37](literature_memory/index_v37/LITERATURE_INDEX.json) retains157 scoped conclusion records across108 normalized paper identities and2 software identities. These are not whole-paper-read counts. The latest new method read is Dynamic Negative Correlation Learning. Adaptive scalar loss balancing is prior, and fixed graph-frequency/neighbor-error transforms reduce to graph-kernel NCL. This follow-up promoted zero pilots. A separate current scout examines whether a concrete shared-factor operation can represent dependent missing-neighbor configurations beyond marginal uncertainty. It has no adopted method or experiment yet.

Latest verified pushed and synchronized head: `870bfae124e0606c01039bddccbeba015656fbf0` on `codex/postsubmission-research-20260930`. This turn's new audits, reviews and records await the next publication. Earlier status bytes are preserved in `coordination_snapshots/20261003_closed_audits_v1` and Git history.

Normal execution stays inside authorized repositories. Ordinary incidental caches are allowed. The seven-GPU account is forwarding only. No filesystem isolation, sudo, driver changes, PDF compilation, GENLINK or unrelated changes. The earlier unnecessary isolation and restart of our four BUDDY processes were agent mistakes, recorded in [server accountability](gpu77_connection_recovery_v1/SERVER77_ACCOUNTABILITY_20261003.md). Fresh manuscript reviewers receive immutable paper/evidence without author history or a requested verdict. Source approvals and resource checks do not count as acceptance.
'''
(ROOT/'PUBLIC_STATUS.md').write_text(status)
(ROOT/'RESEARCH_STATE.md').write_text('# GNNM post-submission research state\n\n'+
    f'Updated: {UTC}. Goal remains active and incomplete. Original paper scores remain unchanged.\n\n'+
    'The initializer saved-array audit completed all30 cases, but showed no consistent graph-specific gain. Serialized inference replay remains open. FoRDE passed its two fixed full-input B2 numerical endpoints, without a quality result or B128 numerical claim. The fixed authored Amazon Polynormer-r comparison is the next training priority. Mixed32/40, BUDDY and NCNC families remain live and partial outcomes stay unscored.\n\n'+
    'Literature index_v37 records the scoped prior overlap and zero newly promoted pilots. See [current status](PUBLIC_STATUS.md), [full ledger](research_ledger.json), and preserved source/result/review packets. Latest verified pushed/synchronized head is870bfae124e0606c01039bddccbeba015656fbf0.\n')
print(json.dumps(dict(UTC=UTC, prior_turn='verified_wait', current_turn='progress', goal_complete=False)))
