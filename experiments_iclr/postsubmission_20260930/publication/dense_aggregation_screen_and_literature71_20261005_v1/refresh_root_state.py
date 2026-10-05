"""Preserve old state, update current facts, and refresh explicit publication input."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
stamp = datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
snapshot = P / 'coordination_snapshots/20261005_complete_amazon_result_and_fusion_v2_before_state_v1'
snapshot.mkdir()
for name in ['PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json']:
    shutil.copyfile(P / name, snapshot / name)
shutil.copyfile(HERE / 'README_MAIN.md', snapshot / 'README_MAIN.md')

status = f'''# Current GNNM research status

Updated: {stamp}. Goal active and unmet. Original manuscript scores remain unchanged. No new method has established both novelty and superior predictive quality, and no new manuscript acceptance verdict exists.

## Current research

Test aggregation after all four members have run. One candidate transports local uncentered member-error moments across the graph and solves dense probability weights. The second uses fixed private hidden states in a nonlinear readout. Learned weighting, graph correction and feature/depth fusion have close precedents; these are hypotheses with attribution, not cleared novel methods. The relevant experiment asks whether the proposed information improves prediction over competent processing of the same information, a strong processed independent ensemble and a processed single.

The [V2 logits-only development protocol](amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/REPORT.md) addresses the [independent V1 critique](amazon_polynormer_logits_graph_moment_protocol_independent_review_20261005_v1/REPORT.md). It fixes native FP32 graph masks, adds a comparable-budget processed member0 single, retains the best permitted training-objective iterate including the no-op state, pins numerical solvers and charges all processing. It uses only each split's own VALID labels, with whole scored-fold exclusion. Both complete banks/all three splits are required. The base checkpoints already used VALID; outer-fold development and tuning are therefore biased retrospective evidence. Source implementation and V2 review are in progress. No fusion fit, hidden export or TEST access has occurred.

## Completed Amazon/Polynormer comparison

All15 original physical fits and whole-cohort closure completed. The original complete evaluator then succeeded with exact zero maximum selected-checkpoint logit replay differences for every model. On reserved TRAIN-control, mean accuracy is52.7203% for native member0,52.9243% shared4 and54.0533% independent4. Shared-minus-independent accuracy differences are−0.7752,−1.2245,−1.3872points, mean−1.1290points; mean NLL difference is+2.2709nats. This comparison does not support shared superiority. VALID mean accuracies are52.6920%,52.4035%,53.1765%, respectively. The complete [result and scope](amazon_polynormer_complete15_original_comparison_root_adoption_20261005_v1/REPORT.md) are preserved.

The successful evaluator took293.16seconds,45 complete-model evaluation forwards and discarded audit updates; it performed no scientific retraining. The earlier root release erroneously specified CPU while binding the CUDA runtime; its pre-score failure is preserved alongside the manually corrected V2 release. The original schedule took153910.61seconds and closure22.55seconds. TEST remains unopened. TRAIN-control is consumed for the original comparison and cannot silently become fresh fusion confirmation or training.

## Other actual evidence and compute

All three saved independent-four Citeseer-HeaRT banks were screened without inference or fitting: raw averaging27.9204% versus equal Borda28.0921% VALID MRR, mean+0.1717points, two positive/one negative block, descriptive95% interval[−0.7982,+1.1416]points. All681 query ranks replayed exactly; CPU3.81seconds. Keep this cheap comparator; no shared-bank result or TEST promotion follows. [Complete outcome](citeseer_saved_independent_bank_pooling_screen_20261005_v3/OUTCOME.md).

At16:14UTC the owned private-learning allocation block was7/10 complete, b0_U_end_live atcycle31/episode61, with authenticated queue/child identities. Its matched-single block had completed3/3 at15:07UTC. Original30 quality/full39 mechanism requirements remain; withdrawn-server blocks and six unreleased companions are not replaced by favorable subsets. PENCIL was2/3 complete, final seed atnative epoch-index133/532updates. The all-three collector remains disabled until complete custody. No scientific scores were opened by these metadata checks.

Only the explicitly authorized one-GPU allocation is accessed, hostanogena-2-0/UUIDGPU-44039938-fd82-41d2-fefd-de71514e2fac. Access to18.77/MacLink remains withdrawn. No monitoring, fetching, stopping, launching or cleanup occurs there. Earlier wrong-allocation evidence remains excluded.

The complete36-fit Citeseer-HeaRT study remains28.4115% sharedF4,27.9204% independent4,26.7811% native single and28.1795% private frames VALID MRR; all primary descriptive intervals include zero and frames failed the frozen gate. Earlier Collab TEST remains67.2909% private completion,67.6298% independent4,66.4426% single Hits50; it cannot confirm a later design. Original studies and unsuccessful directions remain recorded.

Pubmed TRAIN numerical qualification is not a predictive fit. Original supervision failure and collector repairs remain preserved. Full39 D2 analysis stays disabled until complete custody. No favorable subset is substituted for a complete comparison.

## Literature and theory

Index72 preserves260 scoped records/207paper groups/two software groups. These are not full-paper reading counts. GETS's pinned class-specific positive scales can change argmax; it is an accuracy comparator. A3-GCN consensus training is attributed. GENNN primary method access and A3 published/preprint equivalence remain unresolved. The sealed [member/depth scout](member_depth_late_aggregation_primary_scout_20261005_v1/REPORT.md) separately covers JK,DAGNN,Diverse Peers and VFusion. VFusion directly precedes frozen intermediate-state nonlinear fusion; such fusion principles are not new. A further primary aggregation scout is active. Standalone scouts are not silently counted in index72.

The [Brier geometry assessment](late_pooling_brier_geometry_root_assessment_20261005_v1/REPORT.md) has an independent mathematical review with four clarifications incorporated and the reviewed original retained. Conditional probability weighting projects toward the class posterior within the member hull; transported anchor moments also depend on predictor drift. These are known identities and an elementary estimator-difference bound, not a new ensemble theorem or measured gain.

Private Amazon final states are not cached. Full extraction would need15 model calls/24 trajectories and1.204GB hidden tensors plus overhead, server-side. No hidden export or further GPU fit is justified by preparation alone. Next: qualify and run the fixed representative saved-logit CPU screen; preserve all outcomes, then decide whether any route merits untouched confirmation. Fresh unbiased skill-based paper review follows an evidence-supported revision, without a requested verdict.

## Publication

Latest verified pushed head before this update:716034807b5008aff6fcf16805da7f60229a5f8f. Reviewed source, scoped conclusions and compact outcomes are being committed through the explicit inventory. Historical canonical files are preserved in[the snapshot](coordination_snapshots/20261005_complete_amazon_result_and_fusion_v2_before_state_v1), the append-only ledger and Git. Publication receipts record the exact successor commit/ref; a prepared inventory is not a verified push.
'''
(P / 'PUBLIC_STATUS.md').write_text(status)
(P / 'RESEARCH_STATE.md').write_text(f'''# Current state: graph-ensemble predictive quality

Updated: {stamp}. Goal active and unmet; original paper scores unchanged. [Evidence and limits](PUBLIC_STATUS.md).

1. Access only the authorized one-GPU allocation;18.77/MacLink withdrawn. All wrong-allocation evidence excluded.
2. Amazon original complete15 comparison succeeded with exact logit replay. TRAIN-control shared4 loses to independent4 by1.1290points and2.2709NLL nats on average. No TEST access. Preserve original CPU-release failure and corrected evaluator.
3. Fusion V2 is fixed before outcomes: graph-error moments versus strong cheap/same-information controls, processed independent and comparable processed single;126development fits only, no final refits or GPU forwards. V2 independent review and source implementation in progress. Whole-fold exclusion does not undo base VALID reuse.
4. At16:14UTC private-learning block7/10, matched single3/3, PENCIL2/3(final native epoch133). Full30/full39 requirements remain; no partial quality analysis or missing-server replacement.
5. Citeseer rank screen+0.1717points, mixed/uncertain; Borda remains a control. Original Citeseer/Collab results and failed hypotheses preserved.
6. Hidden/depth fusion has direct VFusion/PCL/FFL/JK/DAGNN ancestry; states not exported. Graph-informed error moments have elementary geometry, not proved novelty or gain.
7. Index72 remains260scoped records/207paper groups/two software groups. Additional sealed scout is standalone. GENNN method unread; further closest primary search active.
8. Publish exact reviewed notes/source/outcomes, verify named-key push/ref and save receipts. Latest prior verified head716034807b5008aff6fcf16805da7f60229a5f8f. Review paper fresh only after supported revision; acceptance remains unmet.

Earlier state is preserved in[the snapshot](coordination_snapshots/20261005_complete_amazon_result_and_fusion_v2_before_state_v1), ledger and Git.
''')

ledger = json.loads((P / 'research_ledger.json').read_text())
key = 'Amazon_complete15_original_comparison_and_fusionV2_20261005_v1'
assert key not in ledger
ledger[key] = {'UTC': stamp, 'original_comparison': 'amazon_polynormer_complete15_original_comparison_root_adoption_20261005_v1',
 'result_summary_sha256': sha(P / 'amazon_polynormer_complete15_original_comparison_root_adoption_20261005_v1/RESULTS_SUMMARY.json'),
 'shared_minus_independent_TRAIN_control_accuracy_pp': [-0.7752,-1.2245,-1.3872],
 'mean_accuracy_gap_pp': -1.1290, 'mean_NLL_gap_nats': 2.2709,
 'all15_replay_max_logit_difference': 0, 'evaluation_seconds': 293.1633,
 'CPU_configuration_failure_preserved': True, 'TEST_access': False,
 'fusion_fit_performed': False, 'protocol_v2': 'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2',
 'protocol_v2_sha256': sha(P / 'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/PROTOCOL.json'),
 'review_v1': 'amazon_polynormer_logits_graph_moment_protocol_independent_review_20261005_v1',
 'V2_review_and_source_implementation': 'in progress', 'new_superiority_or_novelty_or_acceptance': False,
 'private_allocation_progress': {'UTC':'2026-10-05T16:14:07.629696+00:00','complete':7,'total':10,'current':'b0_U_end_live','cycle':31,'episode':61},
 'PENCIL': {'UTC':'2026-10-05T16:14:08.348156+00:00','complete':2,'total':3,'final_epoch_index':133,'final_updates':532},
 'single_block_complete':3, 'missing77_blocks_unobserved':True,
 'member_depth_primary_scout': 'member_depth_late_aggregation_primary_scout_20261005_v1',
 'scout_indexed':False, 'canonical_predecessor': str(snapshot.relative_to(P))}
(P / 'research_ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')

readme = (HERE / 'README_MAIN.md').read_text()
start = readme.index('## Current research')
end = readme.index('## Original method')
newtop = '''## Current research — 5 October 2026

We are testing predictive quality of shared graph ensembles against strong single models and independent ensembles. Original five-dataset paper scores remain unchanged. No new method has established both novelty and confirmed superiority.

The complete fifteen-fit Amazon/Polynormer comparison finished with exact selected-checkpoint logit replay. Reserved TRAIN-control mean accuracy is52.7203% member0 single,52.9243% shared4 and54.0533% independent4. Shared4 loses1.1290percentage points and2.2709NLL nats on average to independent4. This supplies no shared-superiority claim; TEST remains unopened. The root CPU-configuration evaluator failure and successful corrected evaluator are both preserved.

Two aggregation hypotheses remain: transport local member-error moments to weight the existing predictions, and fuse fixed private hidden states with a small nonlinear readout. Learned weighting, graph correction and feature/depth fusion have direct precedents. The fixed V2 CPU screen includes capable same-information stacking, complete processed independent banks and a comparable-budget processed member0 single. It uses each split's own VALID with whole-fold exclusion, but checkpoint/OOF reuse remains retrospective development. No fusion fit, hidden export or new backbone forward has yet occurred.

The completed saved Citeseer independent-four rank screen gives27.9204% raw-pool versus28.0921% equal-Borda VALID MRR, mixed three-block gains with an interval including zero. Keep Borda as a cheap control, without shared-bank or final-score promotion.

Only the authorized one-GPU allocation is accessed;18.77/MacLink remains withdrawn. At16:14UTC the original private-learning block was7/10, its matched-single block3/3 and PENCIL2/3. Full-cohort requirements remain. Index72 preserves260 scoped records/207paper identities/two software identities, not full-paper reading counts. A sealed standalone scout includes VFusion's directly relevant frozen depth/feature fusion. GENNN primary method access remains unresolved.

- [Current evidence and limits](experiments_iclr/postsubmission_20260930/PUBLIC_STATUS.md)
- [Fixed aggregation development protocol](experiments_iclr/postsubmission_20260930/amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2/REPORT.md)
- [Complete Amazon comparison](experiments_iclr/postsubmission_20260930/amazon_polynormer_complete15_original_comparison_root_adoption_20261005_v1/REPORT.md)
- [Complete cheap rank screen](experiments_iclr/postsubmission_20260930/citeseer_saved_independent_bank_pooling_screen_20261005_v3/OUTCOME.md)
- [Literature memory](experiments_iclr/postsubmission_20260930/literature_memory/index_v72/LITERATURE_INDEX.json)
- [Preserved research ledger](experiments_iclr/postsubmission_20260930/research_ledger.json)

'''
readme = readme[:start]+newtop+readme[end:]
obsolete = 'The authored Amazon Polynormer queue was healthy'
if obsolete in readme:
    a = readme.index(obsolete)
    b = readme.index('\n',a)
    readme = readme[:a]+'For the latest completed Amazon comparison, queue state and qualification limits, see [current research status](experiments_iclr/postsubmission_20260930/PUBLIC_STATUS.md). Existing manuscript scores remain unchanged.'+readme[b:]
(HERE / 'README_MAIN.md').write_text(readme)

inv = json.loads((HERE / 'INVENTORY_DRAFT.json').read_text())
sources = {r['source']:r['target'] for r in inv['files'] if (P/r['source']).is_file()}
extra_dirs = [snapshot,
 P/'coordination_snapshots/20261005_completed_pooling_screen_literature71_v1',
 P/'amazon_polynormer_complete15_original_comparison_root_adoption_20261005_v1',
 P/'amazon_polynormer_complete15_evaluation_root_preparation_20261005_v1',
 P/'amazon_polynormer_complete15_evaluation_root_preparation_20261005_v2',
 P/'amazon_polynormer_logits_graph_moment_protocol_independent_review_20261005_v1',
 P/'amazon_polynormer_logits_graph_moment_retrospective_protocol_20261005_v2',
 P/'shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/observation_20261005T161407Z',
 P/'pencil_citeseer_native300_launch_receipts_20261005_v1/metadata_20261005T161408Z']
allowed = {'.py','.json','.jsonl','.md','.txt','.html','.diff','.patch','.log','.raw','.sha256','.csv','.xml','.sh','.yml','.yaml','.in'}
skipped = []
for directory in extra_dirs:
    if not directory.exists():
        skipped.append({'missing':str(directory.relative_to(P))}); continue
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        if path.suffix not in allowed or path.stat().st_size>=2_000_000:
            skipped.append({'excluded':str(path.relative_to(P)),'bytes':path.stat().st_size}); continue
        rel = str(path.relative_to(P))
        sources[rel] = 'experiments_iclr/postsubmission_20260930/'+rel
sources[str(Path(__file__).relative_to(P))] = 'experiments_iclr/postsubmission_20260930/'+str(Path(__file__).relative_to(P))
inv['files'] = [{'source':s,'target':t,'bytes':(P/s).stat().st_size,'sha256':sha(P/s)} for s,t in sorted(sources.items())]
inv['message'] = 'Record complete Amazon comparison, aggregation controls and scoped literature'
out = HERE/'INVENTORY_REFRESHED.json'
with out.open('x') as f: json.dump(inv,f,indent=2);f.write('\n')
with (HERE/'REFRESH_SCOPE.json').open('x') as f:
    json.dump({'UTC':stamp,'files':len(inv['files']),'bytes':sum(r['bytes'] for r in inv['files']),
      'extra_directories':[str(d.relative_to(P)) for d in extra_dirs], 'skipped':skipped,
      'unfinished_source_not_included':True, 'snapshot':str(snapshot.relative_to(P))},f,indent=2);f.write('\n')
print(json.dumps({'files':len(inv['files']),'bytes':sum(r['bytes'] for r in inv['files']), 'skipped':skipped}))
