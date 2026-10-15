# Current continuation state — release 27 preparation

Goal active: a trainable ensemble around one shared native GNN with small
parameter overhead and defensible quality gains over native M1 and genuine I4.
Read ../AGENTS.md. Allocation only; no77/MacLink. Bounded representative pilots
are authorized. The6,460-fit campaign remains unlaunched and awaits user GO.
Original paper scores stay fixed. No original-five reruns, HPO, sudo, PDF,
GENLINK, Desktop or unrelated data. Root owns explicit Git selection.

## Publication

Release26 is confirmed as843bff59dd0b6acefc032e34f5e03eecea23c4ec,
pushed18:12:59UTC15October. COMMIT_RECEIPT_26/PUSH_RECEIPT_26 are authoritative.
Release27 is published only after its own receipts exist. DOC_UPDATE_RECEIPT_26
is the prior mutable-doc authority; DOC_UPDATE_RECEIPT_27 records the next update.
Use tty=true for the existing interactive push helper. Root owns Git.

## Active work — no duplicate launches

Structural MolBACE six-fit owner848485/start6089052668. At18:31:42UTC,
three fits completed; seed202permuted worker850275/start6089664380 is active
at126epochs/4788updates.
Two seed303 fits remain. CURRENT_EXECUTION_27 records the observation. No partial quality was inspected. Source/SPEC/
collection pins and six-fit order remain in CONTINUATION_25 and frozen package
pilot_20261015_structural_private_risk_v1. One new model worker,4hcap/fit,
no retry. Afterall6successful, use frozen selected_VALID.py commands/collect/
analyze with actual terminal pins, reusing33banks/39paidfits:6newbanks,
39totalbanks/45uniquefits. Wait for PubMed engineering to release GPU slot.

PubMed engineering owner849490/start6089402542 is already running and waiting
for those six fits. Then it holds the shared GPU slot and runs3CUDA lifecycle
checks plus4production cost probes. No separate invocation/duplicate queue.
Status qualification/pubmed_cuda_queue_24/QUEUE.json. On failure preserve it;
do not relaunch automatically. No24scientificfits have launched.

OldGRU owner640426/start6055039154 and worker640428/start6055039165 continue.
Coarse29/30banks,44/48trajectories updates after a wholebank, not each update.
GRU_PROGRESS_DETAIL_25 confirmed active body2 after593/282updates in bodies0/1.
Worker was live again18:04. Keep original7daycap and16GiBhost/32GiBGPUgrowth.
Two model-worker slots remain occupied. No third GPU model worker.

## Completed science and next choice

MolBACE residual-only/private-BN:74.518±2.660%VALIDAUC versusI474.847±.500%,
BCE1.952versus.890. Four rank8 input residuals add19,200parameters(1.0183%).
Averaging adds−.043points;95.54%of pooled ranking errors are common toallroutes.
No I4win/equivalence; exposed VALID151 has130positive/21negative. Full33bank/
39fit report remains pilot_20261015_residual_only_private_bn_v1/complete_results_v1.

Default next scientific priority: existing PubMed/Cheb24fit transfer, contingent
on completed MolBACE outcomes and measured cost/resources. Different node task,
native polynomial/attention body,noBN. CPU checks passed;24newfitsneeded for
exactnativeM1/I4/G0/uniform/structured/permuted. It is exposed development,
not protected confirmation. Keep all3seed results, per-class support/harms and
same frozen recipe. See research/next_study_priority_20261015_v1/NOTE.md.
Pairwise MolBACE21fits remain a valid secondary direct-ranking question; PubMed
cannot validate that mechanism. No new manuscript superiority/review warranted.

## Pairwise prepared source — do not repeat CPU work

research/molbace_pairwise_competence_v1: manifest
84a9a2b63244222d9411ce02bdd2035ffe21476465b069acd002f2cd82ffc24b.
TRAIN-only support inspection and all4CPUlifecycles passed:8updates,
12VALID,8constructors,221nativeforwards; all fresh restores bit exact.
PAIR_QUALIFICATION_SUMMARY_25 is authoritative. No repeat.

research/molbace_pairwise_study_v1: frozen21fitstudy,
MANIFEST bcb189fdd847fe50004edb8f29d6883e714a1e637b625816eb8d5b6f8be5ba5e,
SPEC cc4415b651c0564ed130138e473f43e6013027e6dbaa9b07d82e0af08a64abcf.
Rootsource review and actual allocation preflight passed. Reuse18baselinefits;
9shared+12nativeauxiliaryfits produce27rows,69contrasts after all21terminal
and9newcollections. No scientific admission/launch. Preserve every BCEharm.

research/molbace_pairwise_cost_probe_v1: frozen/root-reviewed source,
MANIFEST564eefd7fd441fee9c27199cc48c6dc7c3f33b02da3f1b1e7cf0caa16c82369b.
One constructor/full native1210TRAINepoch(38updates)/2VALID for each of
structured and native_commonbody0seed101. No selector/checkpoint/scientificfit.
All5sourcechecks passed. Not executed. Before actualCUDA:sharedGPUslot,live
resources,finite root supervisor,exactsourceadmission. No automatic owner.

## Actual selected-state scorer repair

NODE_SELECTED_VALID_REPLAY_26 preserves failedv2:7.976supervisorseconds,
1constructor/1restore/0forwards/0VALID. Legitimate _extra_state dict caused
v4tensor-only fingerprint to fail. This is a bug, not numerical mismatch.

final_selected_scoring_v5: manifest
e0b86279593c9f20e49b615c57fce41360babc40201b3172ae76db6eab6f9659.
Recursive typed extra state; existing tensor hashes/serving unchanged.
qualification/final_selected_node_NCNC_VALID_v3: manifest
8fc619d48cc82b1106c1ca6e3534376e661a1dd1c9987f6b335d477fc94f0f97.
Root reviewed/staged; one corrected nodeCPUreplay PASSED8.733supervisorseconds:
1constructor/1restore/4forwards/1wholeVALID665/0TRAIN/0CUDA. Both learnedreadout
and same-bank mean exactly equal saved outputs. Do not repeat. Evidence
NODE_SELECTED_VALID_REPLAY_CORRECTED_26.json and NODE_SCORER_REPAIR_REVIEW_26.json.
NCNCv3 CUDA selectedreplay remains pending32GiBGPU/8GiBRSS/600s with slot.
NoTEST. V1/v2andfailedhistory unchanged.

## Other pending campaign work

OriginalBUDDY3170updates, repairedNCN34updates and composedMolBACE38updates
already passed; no repeats. ComposedBUDDYsource20 handoff remains inactive,
1585updates/3constructors/4VALID; only afterMolBACE6+PubMed7successful andslot.
Fullserving, normalizationmigration, measured3GPUplacement/budget unfinished.
No6,460-fitcampaignGO. Finish reviewed publication, monitor original owners,
collect complete active study, then admit next representative science by cost.

## New completed diagnostic — do not repeat

analysis/completed_probability_failure_v1 is frozen/root-reviewed and actually
executed once on CPU. MANIFEST6e5e2cb6ce2f9b431d29e8297512b59973c5cf8cf553cc8b1063ab9e01fdadb0.
All33banks/11methods/3seeds,39exports,27member collections,zero models/forwards/
updates/checkpoints/TEST. Child3.779s;supervisor4.506s;sampledRSS374,833,152bytes.
PROBABILITY_DIAGNOSTIC_27 records the pass. Full1.73MBRESULT stays on allocation,
hashdf081cd67adb427db714dc66e2aca6d09ecce7efc35ee3ccd05fd747b791c8cf.
CompactSUMMARY/REPORT/INTERPRETATION are local for publication. The9,966graph
contributions remain allocation-only analysis_outputs/completed_probability_failure_v1/
PER_EXAMPLE.jsonl; do not sync/commit. No partial current fit was inspected.

Residual-only vsI4 BCE increases on positives and decreases on negatives in
all3seeds. Fixed0.5threshold false negatives105/95/78 versus67/46/60 of130.
Top5graph contributions explain12.3/17.7/38.2%ofnetexcess; losses are broad.
MeanBrier.473versus.289. TRAINpositive39.67%,VALID86.09%; no pure-label-shift
or causal claim and no fittedcorrection. Ranking recovery is not probability
recovery. Keep current studies unchanged and PubMed default priority.

research/rankexpert_followup_v1 records bounded primary access failure:
ACM403 and checkedauthorpages yieldedno manuscript.19retrievalattempts,
zero newmethod/paperread claims. Rawresponses allocationcacheonly. No change
to pairwise21orPubMed24; stopmetadatachasingthislead without primarytext.
