"""Prepare a compact exact inventory; no experiment or score is executed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
D = P / 'publication/label_evidence_scopes_and_required_controls_20261008_v1'
D.mkdir(exist_ok=False)


def bind(relative):
    f = P / relative
    return dict(path=relative, bytes=f.stat().st_size,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest())


def write(f, value):
    f.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


receipt = json.loads((P / 'gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1416_v1/RECEIPT.json').read_text())
assert receipt['exit_code'] == 0
graph = json.loads(receipt['stdout'].strip())
assert graph['parent']['start_ticks'] == 1760762180
assert graph['lanes']['1'][0]['status'] == 'complete'
assert graph['6101_allJ.CHILD_STARTED.json']['actual']['start_ticks'] == 1761859089
monitor_dir = P / 'representative_full_family_progress_root_20261008_1421_v1'
monitor_dir.mkdir(exist_ok=False)
write(monitor_dir / 'GPU77_GRAPH_RELATION_MONITOR.json', graph)
write(monitor_dir / 'ALLOCATION_LATEST_REFERENCE.json', bind('representative_full_family_progress_root_20261008_1410_v1/ALLOCATION_MONITOR.json'))

prior = 'literature_memory/ACTIVE_SUPPLEMENTS_20261008_v24.json'
supplement = dict(
    schema='active_literature_scoped_supplements_v1', UTC=datetime.now(timezone.utc).isoformat(),
    status='ADOPTED_BOUNDED_PRIOR_SCOPES_AND_CONDITIONAL_LABEL_EVIDENCE_HYPOTHESIS',
    prior_supplements=bind(prior),
    new_scoped_method_reads=['arxiv:2609.02638v1', 'arxiv:2605.11987v1', 'arxiv:2009.03509v5'],
    new_primary_scope_identity_credit=3, new_full_papers_read=0,
    new_author_implementation_audits=0, new_proof_audits=0, published_results_adopted=0,
    scope_packets=[bind('recent_shared_graph_ensemble_primary_discovery_20261008_v1/READ_SCOPES.json'),
                  bind('shared_private_label_conditioning_next_pass_20261008_v1/READ_SCOPES.json')],
    conclusions=[bind('recent_shared_graph_ensemble_primary_discovery_20261008_v1/PAPER_CONCLUSIONS.json'),
                 bind('shared_private_label_conditioning_next_pass_20261008_v1/PROPOSAL.md')],
    raw_recent_bodies_custody=bind('recent_shared_graph_ensemble_primary_discovery_20261008_v1/RAW_SERVER_CUSTODY.json'),
    root_label_decision=bind('shared_private_label_conditioning_root_decision_20261008_v1.md'),
    saved_baseline_reuse=bind('efficient_shared_ensemble_baseline_gap_assessment_20261008_v1/CONCLUSIONS.json'),
    masking_operation_assessment=bind('common_label_query_mask_inverse_inclusion_assessment_20261008_v1/NOTE.md'),
    baseline_reuse_new_primary_credit=0,
    decision='Retain recent complementarity and belief-head scopes plus UniMP label/mask ancestry. Label-only private correction is a conditional attributed utility hypothesis, not implemented or novelty-cleared. No running source/gate/outcome change.',
    scientific_fits_or_models=0, scientific_source_mutations=0,
    new_conditional_label_method_implemented=False, novelty_established=False)
current = 'literature_memory/ACTIVE_SUPPLEMENTS_20261008_v25.json'
assert not (P / current).exists(); write(P / current, supplement)
pointer = json.loads((P / 'literature_memory/CURRENT_SUPPLEMENT.json').read_text())
pointer['current'] = bind(current); write(P / 'literature_memory/CURRENT_SUPPLEMENT.json', pointer)

ledger_file = P / 'research_ledger.json'; ledger = json.loads(ledger_file.read_text())
name = 'Label_evidence_prior_scopes_required_controls_and_live_queue_20261008_root_v1'
assert name not in ledger
ledger[name] = dict(
    UTC=datetime.now(timezone.utc).isoformat(), previous_goal_turn_classification='progress',
    previous_progress_evidence='publication/complete15_result_graph_diversity_and_literature_20261008_v1/PUSH_RECEIPT.json',
    new_primary_bounded_scopes=3, new_full_papers_read=0, new_author_code_audits=0,
    conditional_label_hypothesis='shared_private_label_conditioning_next_pass_20261008_v1/PROPOSAL.md',
    root_label_decision='shared_private_label_conditioning_root_decision_20261008_v1.md',
    label_method_implemented=False, novelty_cleared=False, new_accuracy_advantage=False,
    single8_source='internal_be_single8_own_sequential_source_20261008_v1/single8.py',
    single8_runtime_qualified=False, single8_scientific_execution_admitted=False,
    reference_packet='graph_relation_reference_admissibility_20261008_v1/REPORT.md',
    arxiv_blind_WikiCS_recipe_transfer_not_adopted=True,
    stronger_baseline_assessment='efficient_shared_ensemble_baseline_gap_assessment_20261008_v1/REPORT.md',
    quantile_synthesis='contrastive_graph_evidence_next_method_synthesis_20261008_v1/PROPOSAL.md',
    quantile_new_method_or_fit=False,
    label_query_mask_note='common_label_query_mask_inverse_inclusion_assessment_20261008_v1/NOTE.md',
    inverse_inclusion_is_known_sampling_primitive=True,
    graph12_complete_fits=1, graph12_partial_comparative_scores_opened=False,
    graph12_current=dict(seed6101='allJ epoch86/1100', seed6203='alphaF epoch641/1100'),
    graph12_owner=dict(pid=3713404, start_ticks=1760762180),
    new_child_6101_allJ=dict(pid=3738864, start_ticks=1761859089),
    Mol18_last_verified=dict(UTC='2026-10-08T14:06', complete_fits=9, current='I_7203 epoch54/100'),
    sources_or_gates_of_running_families_changed=False, new_scientific_jobs_admitted=0,
    original_scores_changed=False, manuscript_acceptance=False)
write(ledger_file, ledger)

state_file = P / 'RESEARCH_STATE.md'; state = state_file.read_text()
state = state.replace('Updated 8 October 2026, 13:43 UTC.', 'Updated 8 October 2026. Graph progress verified 14:21 UTC; allocation progress verified 14:06 UTC.')
state = state.replace('52e00fb47f59d489bb70f0fa6228626c80e99f77', '0f50f22ec3dd43cad1c10e936054e43029a750e1')
state = state.replace('I_7203 at 34/100 epochs, 13:43 UTC', 'I_7203 at 54/100 epochs, 14:06 UTC')
state = state.replace('Both GPU lanes admitted; 6101_alphaF at 948/1100 and 6203_alphaF at 427/1100, 13:43 UTC', 'Both lanes admitted; 1/12 fits complete; 6101_allJ at 86/1100 and 6203_alphaF at 641/1100, 14:21 UTC')
state = state.replace('Both graph-study children were verified live: 3713425 / 1760762232 and 3727602 / 1761283669.', 'The first6101 control is complete. Current children were verified live: 3738864 / 1761859089 (6101_allJ) and 3727602 / 1761283669 (6203_alphaF).')
state = state.replace('Latest snapshots: `representative_full_family_progress_root_20261008_1345_v1/`.', 'Latest graph snapshot: `representative_full_family_progress_root_20261008_1421_v1/`; allocation snapshot: `representative_full_family_progress_root_20261008_1410_v1/`. Their observation times differ.')
state = state.replace('now supplement v24; v23 contains the two new method scopes. It adds two new bounded method reads:', 'now supplement v25. The prior v23 update added two bounded method reads:')
insert = '''
### Additional scoped evidence and conditional direction

Three bounded primary scopes were added, with zero full-paper, author-code or result-reproduction credit: a2026 link-prediction oracle analysis, a2026 random-set node head, and UniMPv5's label/ masked-label operation. Choosing the best existing rank is not a universal upper bound for score fusion. Random-set heads concern uncertainty and do not establish useful member alternatives. UniMP already establishes label inputs/masked targets; C&S and the staged residual family already establish graph corrections/protected bases.

One conditional hypothesis remains: native feature-only learning with four private label-only-value attention correctors, whose correction gradients stop at the backbone. All query labels must be hidden from all route/context fields. It is not implemented, admitted or novelty-cleared. Attention normalization and correction learning signal require explicit source choices. Same-context capable single/untied, live-gradient, C&S and strong label-aware references are necessary. See `shared_private_label_conditioning_root_decision_20261008_v1.md`.

The minimum reference packet preserves the six completed historical single/ordinary records and distinguishes them from fresh matched controls. Blind transfer of WikiCS's numerical recipe to arxiv is not adopted; new graph baselines need verified dataset-specific competence before unused scoring. A disabled memory-bounded single8 own-CE source now accumulates eight backwards at old parameters then makes one native Adam step. It is source-prepared only, with no runtime or fit claim. A separate saved-source assessment retains PE4 gamma1 as one stronger efficient-ensemble comparator; a naive graph MIMO port is not certified by view counts. These additions do not change any running recipe, gate or score.

'''
state = state.replace('## Data provenance and evidence limits', insert + '## Data provenance and evidence limits')
state_file.write_text(state)
status_file = P / 'PUBLIC_STATUS.md'; status = status_file.read_text()
status = status.replace('At 13:43 UTC the 6101 and 6203 matched controls were running at 948/1100 and 427/1100 epochs.', 'At 14:21 UTC one fit was complete;6101_allJ and6203_alphaF were running at86/1100 and641/1100 epochs.')
status = status.replace('I_7203 was at 34/100 epochs at 13:43 UTC.', 'I_7203 was at54/100 epochs at14:06 UTC.')
status += '\nA label-only graph-correction hypothesis is retained for future source preparation with close UniMP/C&S ancestry and strict common target exclusion. Its utility/novelty are unproven. The required single8 control has a disabled memory-bounded source successor; no fit or runtime claim follows. Stronger packed/MIMO baseline boundaries and exact historical references are saved.\n'
status_file.write_text(status)

old = (P / 'publication/complete15_result_graph_diversity_and_literature_20261008_v1/README_MAIN.md').read_text()
updated = old.replace('At 13:43 UTC its two initial matched controls were at 948 and 427 epochs. Later conditions remain queued;', 'At 14:21 UTC its first control had completed;6101_allJ and6203_alphaF were at86 and641 epochs. Later conditions remain queued;')
updated = updated.replace('the current fit is at 34/100 epochs.', 'the current fit was at54/100 epochs at14:06 UTC.')
paragraph = '''A new conditional direction uses private attention whose message values are permitted training labels around a concurrently learned feature-only backbone. Query labels are excluded from every correction path, and correction gradients stop at the backbone. UniMP, C&S and shared/private residual methods provide close ancestry. It is not implemented, launched or novelty-cleared. [Hypothesis and obligations](experiments_iclr/postsubmission_20260930/shared_private_label_conditioning_next_pass_20261008_v1/PROPOSAL.md). A disabled sequential single8 control prepares the view-opportunity comparison without retaining eight live tapes; full-input runtime remains unqualified. [Source preparation](experiments_iclr/postsubmission_20260930/internal_be_single8_own_sequential_source_20261008_v1/README.md) · [Stronger baseline boundaries](experiments_iclr/postsubmission_20260930/efficient_shared_ensemble_baseline_gap_assessment_20261008_v1/REPORT.md).

'''
updated = updated.replace('Raw datasets, representations and checkpoints remain on authorized servers.', paragraph + 'Raw datasets, representations and checkpoints remain on authorized servers.')
assert old[old.index('## Original method'):] == updated[updated.index('## Original method'):]
(D / 'README_MAIN.md').write_text(updated)

directories = [
    'recent_shared_graph_ensemble_primary_discovery_20261008_v1',
    'graph_relation_reference_admissibility_20261008_v1',
    'contrastive_graph_evidence_next_method_synthesis_20261008_v1',
    'shared_private_label_conditioning_next_pass_20261008_v1',
    'internal_be_single8_own_sequential_source_20261008_v1',
    'efficient_shared_ensemble_baseline_gap_assessment_20261008_v1',
    'common_label_query_mask_inverse_inclusion_assessment_20261008_v1',
    'representative_full_family_progress_root_20261008_1410_v1',
    'representative_full_family_progress_root_20261008_1421_v1',
    'gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1410_v1',
    'gpu77_connection_recovery_v1/commands/graph_relation_full12_same_owner_progress_20261008_1416_v1',
    'gpu77_connection_recovery_v1/commands/complete15_graph_diversity_exact_commit_sync_20261008_v1',
]
files = {'RESEARCH_STATE.md', 'PUBLIC_STATUS.md', 'research_ledger.json', prior, current,
         'literature_memory/CURRENT_SUPPLEMENT.json',
         'graph_relation_reference_admissibility_20261008_root_decision_v1.md',
         'shared_private_label_conditioning_root_decision_20261008_v1.md',
         'internal_be_single8_own_sequential_root_source_review_20261008_v1.json',
         'publication/prepare_label_evidence_and_controls_publication_20261008_v1.py'}
for name in directories:
    files.update(str(f.relative_to(P)) for f in (P / name).rglob('*')
                 if f.is_file() and '__pycache__' not in f.parts)
for f in (P / 'publication/complete15_result_graph_diversity_and_literature_20261008_v1').iterdir():
    if f.is_file() and f.name != 'README_MAIN.md': files.add(str(f.relative_to(P)))
rows = []
for relative in sorted(files):
    b = bind(relative); assert b['bytes'] < 2_000_000
    rows.append(dict(source=relative, target='experiments_iclr/postsubmission_20260930/' + relative,
                     bytes=b['bytes'], sha256=b['sha256']))
relative = str((D / 'README_MAIN.md').relative_to(P)); b = bind(relative)
rows.append(dict(source=relative, target='README.md', bytes=b['bytes'], sha256=b['sha256']))
write(D / 'INVENTORY.json', dict(branch='codex/postsubmission-research-20260930',
    expected_head='0f50f22ec3dd43cad1c10e936054e43029a750e1',
    message='Prepare label-evidence hypothesis and required graph ensemble controls',
    remove=[], files=rows))
print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows),
                     inventory=str((D / 'INVENTORY.json').relative_to(P)))))
