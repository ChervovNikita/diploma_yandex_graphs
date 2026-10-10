"""Integrate complete results and quality-free observations into current notes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import statistics

H=Path(__file__).resolve().parent;P=H.parent
obs=json.loads(sorted(H.glob('OBSERVATION_*.json'))[-1].read_text())
g77=json.loads((H/'GPU77_OBSERVATION.json').read_text())
root=P/'pubmed_strong_reference_complete12_root_20261010_v1/fetched'
comp_path=root/'pubmed_strong_reference_comparison_execution_20261010_v1/COMPARISON.json'
c=json.loads(comp_path.read_text());assert c['complete12'] and not c['TEST_scored']
cost=json.loads((comp_path.parent/'COSTS.json').read_text())
means={}
for condition in ['single_native','single_mean4_dropout','independent4_own','shared4_own']:
 rows=[c['readouts'][f'seed{s}__{condition}']['VALID']['pooled'] for s in [9101,9203,9307]]
 means[condition]=dict(accuracy_percent=statistics.mean(r['accuracy']*100 for r in rows),NLL=statistics.mean(r['NLL'] for r in rows))
delta=[100*(c['readouts'][f'seed{s}__shared4_own']['VALID']['pooled']['accuracy']-c['readouts'][f'seed{s}__independent4_own']['VALID']['pooled']['accuracy']) for s in [9101,9203,9307]]
qualification=[]
for q in (P/'status_refresh_root_20261010_0254_v1/cmcl_v2_metadata').rglob('RAW_OWNER_TERMINAL.json'):
 row=json.loads(q.read_text());assert row['complete'];qualification.append(row['inclusive_seconds'])
assert len(qualification)==6
launch=json.loads((P/'status_refresh_root_20261010_0254_v1/CMCL_V2_SCIENCE_LAUNCH.json').read_text())['science_launched']
old={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['RESEARCH_STATE.md','PUBLIC_STATUS.md','research_ledger.json','RESEARCH_WORKFLOW.md']}
(H/'LOCAL_PREVIOUS_CANONICAL_HASHES.json').write_text(json.dumps(old,indent=2)+'\n')
(H/'REMOTE_PREVIOUS_CANONICAL_HASHES.json').write_text(json.dumps(obs['canonical_sha256'],indent=2)+'\n')
state=(P/'RESEARCH_STATE.md').read_text()
a=state.index('## Running capable-reference science on allocation');b=state.index('## Running science on18.77')
state=state[:a]+'''## Completed capable-reference PubMed comparison

All nine new groups, eighteen complete native trajectories, selected-output comparisons and actual owned process/CUDA closures are complete. Mean selected VALID accuracy/NLL: native single 89.937426%/0.511273; single averaging four factual dropout losses per update 89.869778%/0.456877; genuine independent4 with individually selected members 90.216472%/0.354645; ordinary shared4 90.901404%/0.287697. Shared minus independent4 is +0.887874/+0.532725/+0.634196 percentage points, mean +0.684932. Both accuracy and NLL improve over I4 in every seed. Shared also exceeds both single controls in every seed. Mean member accuracy exceeds I4 by +0.976662pp, so this result concerns member competence as well as pooling.

This weakens the common-checkpoint-selector and simple four-dropout-gradient-averaging explanations. It does not isolate ensemble learning from the additional factor coordinates. Six complete M1 unit-factor controls are being prepared to separate that reparameterization from four-member learning. One encountered graph, one exploratory stratified split and three optimizer seeds do not establish unused confirmation or novelty. PubMed TEST remains closed. Original scores are unchanged.

The full graph has 19717 nodes, 500 features and 88648 directed edges. Split190111 uses11829 TRAIN,3942 VALID and3946 TEST nodes. Native PolyFormer uses monomialK2, width256, two blocks, eight heads, dropout0.5, dprate0.8, max2000/patience250 and its original Adam groups. Each new native trajectory uses its own strict-first maximum raw-logit VALID selector. Paired native single and I4 member0 were separately rerun and charged. Evidence: pubmed_strong_reference_complete12_root_20261010_v1/fetched/pubmed_strong_reference_comparison_execution_20261010_v1/COMPARISON.json and COSTS.json. The finite comparison owner took5.776s; its nested reader timing is not an additional independent cost.

## Running private specialist-learning study on allocation

All six full-input qualifications passed, total owner time48.834s. They accessed complete TRAIN and VALID and establish runnable source, not predictive benefit. The18 full fits launched03:46:16UTC under owner594139/start6041323646 on the verified one-GPU allocation. At04:05:20UTC three conditions of seed9101 had completed and vanilla_cmcl was running; no family failure or partial quality access. Close all18 before comparison.

The shared parameters receive ordinary mean member cross-entropy. Private BatchEnsemble factors additionally receive specialist credit using the attributed CMCL objective, K3 owners, beta0.75 and lambda1. The six fixed conditions are own_floor, private_cmcl, all_block_cmcl, vanilla_cmcl, private_uniform and private_constant_credit, across seeds9101/9203/9307. They test the specialist rule, shared-gradient protection and gradient scaling. Assignment arms require eight full graph forwards per update versus four for ordinary controls; record this cost. Shared protection concerns the current data gradient only, not optimizer moments, decay or later trajectories. Native parameters2069875 plus55772 private factors produce2125647 total parameters.

The first adapter failed before model construction because local queue.py shadowed a standard-library import. Its2.613s cost and failure remain recorded. V2 invokes the unchanged numerical source with python -B -P and uses explicit fresh V2 source/owner reviews; earlier approval digests remain historical. CPU log-mean-softmax selection differs from the earlier direct FP32 probability pool. Selected immutable outputs determine scientific readouts; fresh replay guarantees pooled prediction identity, not member-argmax identity. Source and corrected adapter are committed atba537d93d0fe803ccc86b98bda8e1729fc4170a0. Exact launch: status_refresh_root_20261010_0254_v1/CMCL_V2_SCIENCE_LAUNCH.json.

'''+state[b:]
a=state.index('## Running science on18.77');b=state.index('## Typed label-context source')
state=state[:a]+'''## Running science on18.77

Six WikiCS initializer×feedback full fits run under owners4043992/4043993, start1774146656, boot2de86898-2942-402c-a7a5-29f64e4688d1. At04:05:30UTC the8ced lane closed all four owned fits and its owner/children were absent. Thea998 lane completed6203_alphaF and was training6203_relationJ at277/1100 updates. Neither lane reports failure. Finish all six and actual closure before comparison; partial quality remains unopened. No new77 jobs were launched during this update.

Exact handoff: graph_relation_independent_native_local_scorer_training_owner_preparation_20261010_v1/ACTUAL_LAUNCH_HANDOFF.json. Observation and metadata: status_refresh_root_20261010_three_roles_v1/GPU77_OBSERVATION.json and gpu77_metadata.

'''+state[b:]
a=state.index('## Three research roles');b=state.index('## Publication and execution')
state=state[:a]+'''## Three distinct research roles

The new-mechanisms researcher revisits saved primary-source conclusions, checks recent graph-specific work and proposes one falsifiable competence-focused mechanism with its closest prior collision. The combination researcher prepares the PubMed M1 controls, then combines ingredients only when complete prediction evidence suggests distinct repairs; baseline/A/B/A+B, introduced harms and costs remain required. The history researcher integrates complete12 PubMed and revisits completed families, contradictions and repeated ideas, returning at most three next decisions. Root monitors, selects and publishes. See RESEARCH_WORKFLOW.md. Preparation, increased embedding separation and setup success are not quality improvements.

The completed IMDB24 diagnostic supports member competence as a scoped priority:1059 dependent node×label mistakes lack a correct member and only3 discard an available one. Q/K36, relation18, geometry, Tolokers and molecular negatives remain preserved. A saved-confidence reader and typed-context genuine independent references are sealed but unexecuted. The latest literature assessment retains local uncertainty over graph hop channels as an untested candidate; global structure sampling with shared weights already has prior ancestry. No novelty or predictive gain follows from that proposal.

'''+state[b:]
a=state.index('Latest verified GitHub ref');b=state.index('Use literal',a)
state=state[:a]+'''Latest verified GitHub refc27b5150a6a656329ad7c912e92c30608a79bb4e at03:25:37UTC. The allocation HEADba537d93d0fe803ccc86b98bda8e1729fc4170a0 includes corrected private-credit source. Complete12 findings, actual V2 qualification/launch receipts and updated three-role notes are being published together through an explicit text inventory. All failures and earlier reviews remain preserved. Exact commit/push receipts live in publication; a committed source is not described as pushed before exact remote-ref verification.

'''+state[b:]
# Supersede the old selector limitation in the Stage1 historical paragraph.
state=state.replace('A separately fixed capable-reference study is now running: native single, single averaging four dropout losses per update, and ordinary independent4 with every member individually selected. No partial quality is opened.', 'That selector limitation is now addressed by the separately fixed, complete capable-reference comparison below.')
(P/'RESEARCH_STATE.md').write_text(state)
(P/'PUBLIC_STATUS.md').write_text('''# Current scientific status

10 October 2026. Research active. Original paper scores unchanged. New methodological novelty and independent manuscript acceptance remain unestablished.

## Most useful completed finding

On full PubMed, ordinary shared GNNM achieved90.9014% mean selected validation accuracy. A genuine four-model ensemble with individually selected checkpoints achieved90.2165%; the native single89.9374%; a single averaging four dropout losses per update89.8698%. Shared GNNM beat all three controls in every optimizer seed. Its gain over the independent ensemble averaged0.6849 percentage points, with better NLL and mean member accuracy in all three seeds.

This is promising exploratory evidence from one encountered graph and one split. It weakens two alternative explanations, but extra factor parameters remain a possible cause. Six full single-member factor controls are being prepared. PubMed TEST remains closed, and unused confirmation is still needed.

The separate masked-context/contrastive-reconstruction screen failed its fixed continuation criteria. Its18 additions remain disabled. Failed ideas remain in the record.

## Current experiments and research roles

Eighteen full private-specialist-credit fits are running on the one-GPU allocation. They retain ordinary shared supervision and test whether private factors can learn useful specialist corrections, with attributed CMCL and scaling controls. All six engineering checks passed; no partial predictive results have been used.

Five of six WikiCS initializer×feedback fits have finished on18.77; the final fit was at277/1100 updates at04:05UTC. One lane has fully closed. Compare results after the entire family closes. No new77 experiments were launched in this update.

Three researchers have distinct responsibilities: new graph mechanisms from literature; evidence-based combinations; synthesis of completed experiments and contradictions. The history role updates after every completed family to avoid repeating discarded hypotheses and to identify interactions worth testing. Root selects a small queue and handles execution and publication.

## Evidence

The complete PubMed comparison, per-seed results, saved prediction diagnostics, costs and failures remain in the ledger and repository. Current commit/push receipts are in publication. No PDF compilation, sudo, GENLINK, unrelated-data access or scientific execution on the seven-GPU relay. Acceptance remains an outcome for a fresh reviewer to assess from supported manuscript claims.
''')
workflow=(P/'RESEARCH_WORKFLOW.md').read_text()
workflow=workflow.replace('Bare shared GNNM shows an exploratory advantage over a common-selected independent bank; capable native single, four-dropout single and individually selected independent references are now running and take priority.', 'Bare shared GNNM now beats the complete, individually selected independent bank and both single controls in all three PubMed seeds. M1 unit-factor controls take priority to distinguish four-member learning from added factor reparameterization; unused confirmation remains required.')
workflow+='''
## Current role assignments and review cadence

New mechanisms: adaptive_sharing_novelty_v1. Combinations and capable references: query_value_gate_source_review_20261009. Completed-history synthesis: masked_context_source_audit_20261010. These are research roles, separate from a fresh manuscript reviewer.

After each complete experimental family, update the evidence synthesis before choosing another branch. Revisit the broader synthesis when new results contradict an earlier explanation or reveal complementary repairs. Do not repeatedly reread the same sources without a new scoped question. Save one prospective decision with its predicted benefit, expected harm, closest prior method, decisive control and compute estimate. Confirm promising mechanisms on unused evidence before promoting them to paper claims.
'''
(P/'RESEARCH_WORKFLOW.md').write_text(workflow)
l=json.loads((P/'research_ledger.json').read_text())
stamp=datetime.now(timezone.utc).isoformat()
l['Complete_PubMed_strong_reference12_and_three_distinct_roles_20261010_root_v1']={
 'UTC':stamp,'comparison':str(comp_path.relative_to(P)),'comparison_sha256':hashlib.sha256(comp_path.read_bytes()).hexdigest(),
 'complete12':True,'new_full_trajectories':18,'mean_SELECTED_VALID':means,'paired_shared_minus_individually_selected_I4_pp':delta,
 'TEST_scored':False,'original_paper_scores_unchanged':True,'novelty_or_confirmation_or_acceptance':False,
 'new_engineering_owner_seconds':sum(r['terminal']['inclusive_seconds'] for r in cost['new_engineering'].values()),
 'new_scientific_owner_seconds':sum(r['actual_waited_owner']['inclusive_seconds'] for r in cost['new_scientific'].values()),
 'reader_actual_terminal':'pubmed_strong_reference_complete12_root_20261010_v1/fetched/pubmed_strong_reference_complete12_root_20261010_v1/TERMINAL.json',
 'limitation':'Encountered graph/split, three optimizer seeds, selected validation outputs, extra factor coordinates; M1 controls and unused confirmation remain.',
 'roles':['new mechanisms from primary literature','combinations supported by distinct repairs','completed-history synthesis and contradictions'],
 'next_priority':'M1 factor controls; close private-CMCL18 and WikiCS6; freeze confirmation before TEST.'}
l['Private_CMCL_V2_qualification_and_fixed18_actual_science_20261010_root_v1']={
 'UTC':stamp,'launch':launch,'qualification_all6_passed':True,'qualification_owner_seconds':qualification,
 'qualification_TRAIN_and_VALID_access':True,'qualification_not_accuracy_evidence':True,'partial_quality_opened':False,
 'observation':str(sorted(H.glob('OBSERVATION_*.json'))[-1].relative_to(P)),
 'latest_status':obs['families']['private_cmcl_polyformer_root_preparation_20261010_v2'],
 'source_manifest_sha256':'1a7e7ebff90d4e9dfdec41eb1e6f2038dd4ee6cae205f2fceb162eaa7b69915b',
 'adapter_manifest_sha256':'370f0ac2bf5488bf99e5193042f307fbc12f540bc051bcb1fe5ee696280e4f3c',
 'earlier_import_failure_preserved':True,'learning_source_changed':False,'new_method_superiority_or_acceptance':False,
 'selector_limitation':'CPU log-mean-softmax; differs from preceding direct FP32 probability mean.'}
l['WikiCS6_quality_free_five_complete_one_running_three_roles_20261010_root_v1']={
 'UTC':g77['UTC'],'observation':'status_refresh_root_20261010_three_roles_v1/GPU77_OBSERVATION.json',
 'closed_lanes':[r['lane'] for r in g77['lanes'] if r['lane_closure_present']],
 'progress':{x['cell_id']:x['epoch_progress']['epoch'] for r in g77['lanes'] for x in r['cells']},
 'all6_closed':False,'partial_quality_opened':False,'new_77_jobs_launched':False,
 'owned_owner_failures':[r['lane'] for r in g77['lanes'] if r['owned_owner_failure']]}
l['updated_UTC']=stamp
(P/'research_ledger.json').write_text(json.dumps(l,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(mean_SELECTED_VALID=means,paired_delta_pp=delta,qualification_owner_seconds=sum(qualification),canonical_updated=True)))
