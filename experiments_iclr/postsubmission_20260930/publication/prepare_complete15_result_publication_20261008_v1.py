"""Prepare the explicit publication of completed results and current decisions.

This edits project notes and an inventory only. It does not run training,
score data, alter a scientific protocol, or contact either server.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
D = P / 'publication/complete15_result_graph_diversity_and_literature_20261008_v1'
D.mkdir(exist_ok=False)


def binding(relative):
    f = P / relative
    return dict(path=relative, bytes=f.stat().st_size,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest())


def write_json(f, value):
    f.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


receipt = json.loads((P / 'gpu77_connection_recovery_v1/commands/graph_relation_full12_both_lanes_progress_20261008_1345_v1/RECEIPT.json').read_text())
assert receipt['exit_code'] == 0
graph = json.loads(receipt['stdout'].strip())
assert graph['parent']['start_ticks'] == 1760762180
write_json(P / 'representative_full_family_progress_root_20261008_1345_v1/GPU77_GRAPH_RELATION_MONITOR.json', graph)
mol = json.loads((P / 'representative_full_family_progress_root_20261008_1345_v1/ALLOCATION_MONITOR.json').read_text())
assert mol['exit_code'] == 0 and mol['result']['owner']['start_ticks'] == 6019318952
assert len(mol['result']['terminal']['HANDLES.json']) == 9

prior = 'literature_memory/ACTIVE_SUPPLEMENTS_20261008_v23.json'
supplement = dict(
    schema='active_literature_scoped_supplements_v1',
    UTC=datetime.now(timezone.utc).isoformat(),
    status='ADOPTED_BY_ROOT_PRIOR_ACCESS_MEMORY_ONLY',
    prior_supplements=binding(prior),
    new_scoped_method_reads=[], new_primary_scope_identity_credit=0,
    new_primary_read_credit_for_integration=0, new_full_papers_read=0,
    new_author_implementation_audits=0, new_proof_audits=0,
    published_results_adopted=0, experimental_results_adopted=0,
    conclusions=binding('GENNN_author_copy_gap_check_20261008_v1/CONCLUSIONS.json'),
    report=binding('GENNN_author_copy_gap_check_20261008_v1/REPORT.md'),
    manifest=binding('GENNN_author_copy_gap_check_20261008_v1/MANIFEST.json'),
    candidate_author_preprint_locator='doi:10.2139/ssrn.4535927',
    published_identity='doi:10.1016/j.inffus.2024.102461',
    body_equivalence_verified=False, method_scope_accessible=False,
    decision='Save the new matching author-preprint locator and the access limit. No method read or novelty clearance; do not repeat the blocked routes.',
    scientific_source_mutations=0, scientific_fits_or_models=0)
supplement_path = 'literature_memory/ACTIVE_SUPPLEMENTS_20261008_v24.json'
assert not (P / supplement_path).exists()
write_json(P / supplement_path, supplement)
pointer = json.loads((P / 'literature_memory/CURRENT_SUPPLEMENT.json').read_text())
pointer['current'] = binding(supplement_path)
write_json(P / 'literature_memory/CURRENT_SUPPLEMENT.json', pointer)

ledger_path = P / 'research_ledger.json'
ledger = json.loads(ledger_path.read_text())
entry = 'Complete15_decision_redundancy_publication_and_prior_locator_20261008_root_v1'
assert entry not in ledger
ledger[entry] = dict(
    UTC=datetime.now(timezone.utc).isoformat(),
    completed_result='Wiki12_SupCon15_union_collection_execution_root_20261008_v1/compact/REPORT.md',
    primary_mean_accuracy_change_pp=-0.0695233851609703,
    stable_accuracy_gain_established=False,
    prediction_diagnosis='Wiki15_functional_diversity_analysis_20261008_v1/REPORT.md',
    decision_redundancy_observed=True, hidden_state_collapse_established=False,
    relation12_lanes_live=True, relation12_epochs=dict(seed6101=948, seed6203=427),
    Mol18_complete_fits=9, Mol18_current_fit='I_7203', Mol18_current_epoch=34,
    comparative_new_family_scores_opened=False,
    progress_snapshot='representative_full_family_progress_root_20261008_1345_v1',
    active_literature_supplement='v24', new_paper_or_method_read_credit=0,
    new_author_preprint_locator='doi:10.2139/ssrn.4535927',
    novelty_unresolved=True, original_paper_scores_changed=False,
    scientific_source_or_recipe_changes=0, new_fits_launched=0)
write_json(ledger_path, ledger)

state_path = P / 'RESEARCH_STATE.md'
state = state_path.read_text()
state = state.replace('Updated 8 October 2026, 13:24 UTC.', 'Updated 8 October 2026, 13:43 UTC.')
state = state.replace('Last synchronized Git head:', 'Published base for this update:')
state = state.replace('I_7203 at 18/100 epochs, 13:24 UTC', 'I_7203 at 34/100 epochs, 13:43 UTC')
state = state.replace('6101_alphaF at 749/1100 and 6203_alphaF at 212/1100, 13:08 UTC', '6101_alphaF at 948/1100 and 6203_alphaF at 427/1100, 13:43 UTC')
state = state.replace('representative_full_family_progress_root_20261008_1250_v1/', 'representative_full_family_progress_root_20261008_1345_v1/')
state = state.replace('now supplement v23', 'now supplement v24; v23 contains the two new method scopes')
state = state.replace('The closest-prior access gap for DOI 10.1016/j.inffus.2024.102461 is being investigated through a newly identified author-preprint locator.', 'The closest-prior access pass found matching author-preprint DOI 10.2139/ssrn.4535927 for published DOI 10.1016/j.inffus.2024.102461. Its body remains inaccessible. Shared feature weights, private attention, gradient allocation and accepted-version equivalence remain unknown. The saved access note receives no new paper or method-read credit.')
state_path.write_text(state)
status_path = P / 'PUBLIC_STATUS.md'
status = status_path.read_text().replace('At 13:08 UTC the 6101 and 6203 matched controls were running at 749/1100 and 212/1100 epochs.', 'At 13:43 UTC the 6101 and 6203 matched controls were running at 948/1100 and 427/1100 epochs.')
status = status.replace('I_7203 was at 18/100 epochs at 13:24 UTC.', 'I_7203 was at 34/100 epochs at 13:43 UTC.')
status_path.write_text(status)

old = (P / 'publication/graph_relation_full12_started_source_and_controls_20261008_v1/README_MAIN.md').read_text()
original = old[old.index('## Original method'):]
current = '''# GNNM: shared propagation in graph ensembles

## Current research — 8 October 2026

We are developing an ensemble with shared graph-layer weights and four prediction paths. The goal is better predictions than capable single models and ordinary independent ensembles. Original paper scores are unchanged. No new accuracy advantage or fresh manuscript acceptance is established.

The complete fifteen-state WikiCS contrastive comparison shows no reliable accuracy gain. Combined alignment and repulsion averages −0.0695 percentage points versus its plain control; canonical supervised contrastive loss averages −0.1959 points. Every seed and probability-score harm is retained. Four-member class predictions agree on 98.1–99.6% of development nodes. No state rescues a node through pooling when every member is wrong. This measures redundant decisions; it does not prove identical hidden states. [Complete result](experiments_iclr/postsubmission_20260930/Wiki12_SupCon15_union_collection_execution_root_20261008_v1/compact/REPORT.md) · [Prediction diagnosis](experiments_iclr/postsubmission_20260930/Wiki15_functional_diversity_analysis_20261008_v1/REPORT.md).

**A twelve-fit graph-attention experiment is training on both 18.77 GPUs.** Four matched conditions change which parameters receive feedback from the combined prediction: all parameters, internal BatchEnsemble factors, or private graph-relation parameters, against ordinary member supervision. In the proposed relation rule, shared feature transformations retain ordinary member supervision. The study uses the full WikiCS graph, three paired seeds and 1,100 epochs. At 13:43 UTC its two initial matched controls were at 948 and 427 epochs. Later conditions remain queued; no partial comparative quality has been opened. The owner is detached and continues without the paired Mac. [Fixed protocol](experiments_iclr/postsubmission_20260930/graph_relation_private_credit_source_20261008_v2/PROTOCOL.json).

The allocation's eighteen-fit molecular study compares the placement of ensemble feedback against capable single and ordinary independent ensemble references on all official MolHIV training graphs. Nine fits are complete and the current fit is at 34/100 epochs. Its internal-factor candidate and loss mixture remain fixed. Full rosters, failures and costs precede comparisons. A standard independent attention-scorer initialization control remains a conditional proposal, not an admitted run or a novel initializer.

The prior complete graph-context target study is negative. Recent literature includes graph consistency methods and an ICML 2026 molecular ensemble-consensus method with a distinct low-label protocol. Own/pool loss mixtures, private attention and selective gradients have established ancestry. A matching preprint locator for the closest graph-ensemble prior was found, but its method remains inaccessible. Novelty is unresolved. No result here establishes superiority to ordinary ensembles on unused data.

Raw datasets, representations and checkpoints remain on authorized servers. Existing official acquisition records link the exact WikiCS training and development payloads to split 0; acquisition loaded public test labels before omitting them from the safe trainer payload. No test scoring or test-based selection occurred. Checkpoint selection and current readouts share development data, so optimizer-seed intervals are descriptive. [Independent aggregate assessment](experiments_iclr/postsubmission_20260930/Wiki12_SupCon15_closed_result_independent_assessment_20261008_v1.md) · [Provenance limits](experiments_iclr/postsubmission_20260930/WikiCS_normal77_official_role_provenance_scoped_note_20261008_v1.md).

Source decisions, negative results, failures and costs are preserved. [Current state](experiments_iclr/postsubmission_20260930/RESEARCH_STATE.md) · [Research ledger](experiments_iclr/postsubmission_20260930/research_ledger.json) · [Saved literature conclusions](experiments_iclr/postsubmission_20260930/literature_memory/CURRENT_SUPPLEMENT.json).

'''
(D / 'README_MAIN.md').write_text(current + original)

directories = [
    'Wiki12_SupCon15_closed_result_independent_packet_20261008_v1',
    'Wiki12_SupCon15_collection_owner_execution_root_20261008_v1',
    'Wiki12_SupCon15_external_collection_owner_source_20261008_v1',
    'Wiki12_SupCon15_union_closure_adoption_root_20261008_v1',
    'Wiki12_SupCon15_union_collection_execution_root_20261008_v1',
    'Wiki12_SupCon15_union_collection_source_20261008_v1',
    'Wiki12_SupCon15_whole_result_interpretation_root_20261008_v1',
    'Wiki15_functional_diversity_analysis_20261008_v1',
    'graph_relation_quality_recent_primary_search_20261008_v1',
    'GENNN_author_copy_gap_check_20261008_v1',
    'representative_full_family_progress_root_20261008_1250_v1',
    'representative_full_family_progress_root_20261008_1345_v1',
]
files = {
    'RESEARCH_STATE.md', 'PUBLIC_STATUS.md', 'research_ledger.json',
    'literature_memory/CURRENT_SUPPLEMENT.json',
    prior, supplement_path,
    'WikiCS_normal77_official_role_provenance_scoped_note_20261008_v1.md',
    'Wiki12_SupCon15_closed_result_independent_assessment_20261008_v1.md',
    'Wiki12_SupCon15_root_metadata_adoption_release_template_disabled_20261008_v1.json',
    'Wiki12_SupCon15_root_metadata_adoption_helper_20261008_v1.py',
    'Wiki12_SupCon15_root_metadata_adoption_PREPARATION_20261008_v1.md',
    'Wiki12_SupCon15_root_metadata_adoption_MANIFEST_20261008_v1.json',
    'graph_relation_private_credit_full12_activation_root_20261008_v1/LANE_ADMISSION_INDEX_BEFORE_GPU0_20261008_v1.json',
    'publication/prepare_complete15_result_publication_20261008_v1.py',
}
for name in directories:
    files.update(str(f.relative_to(P)) for f in (P / name).rglob('*')
                 if f.is_file() and '__pycache__' not in f.parts)
fetched = json.loads((P / 'Wiki12_SupCon15_union_closure_adoption_root_20261008_v1/FETCHED_EVIDENCE_INVENTORY.json').read_text())
files.update(row['path'] for row in fetched['files'])
commands = [
    'Wiki12_SupCon15_collection_source_stage_20261008_v1',
    'Wiki12_SupCon15_collection_terminal_probe_20261008_v1',
    'Wiki12_SupCon15_compact_and_owned_evidence_fetch_20261008_v1',
    'Wiki12_SupCon15_metadata_adoption_20261008_v1',
    'Wiki12_SupCon15_metadata_adoption_20261008_v2',
    'Wiki12_SupCon15_owned_collection_start_20261008_v1',
    'graph_relation_full12_both_lanes_progress_20261008_1250_v1',
    'graph_relation_full12_both_lanes_progress_20261008_1345_v1',
]
for name in commands:
    d = P / 'gpu77_connection_recovery_v1/commands' / name
    files.update(str(f.relative_to(P)) for f in d.iterdir() if f.is_file())
prefix = 'experiments_iclr/postsubmission_20260930/'
rows = []
for relative in sorted(files):
    f = P / relative
    assert f.is_file() and not f.is_symlink() and f.stat().st_size < 2_000_000
    rows.append(dict(source=relative, target=prefix + relative,
                     bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
relative = str((D / 'README_MAIN.md').relative_to(P))
row = binding(relative)
rows.append(dict(source=relative, target='README.md', bytes=row['bytes'], sha256=row['sha256']))
write_json(D / 'INVENTORY.json', dict(
    branch='codex/postsubmission-research-20260930',
    expected_head='52e00fb47f59d489bb70f0fa6228626c80e99f77',
    message='Record complete contrastive results and graph decision redundancy',
    remove=[], files=rows))
print(json.dumps(dict(files=len(rows), bytes=sum(row['bytes'] for row in rows),
                      inventory=str((D / 'INVENTORY.json').relative_to(P)))))
