# Continuation after release 26 — 15 October 2026

Goal active and unachieved. Read ../AGENTS.md and SESSION_STATE.md. Use only
the authorized one-GPU allocation. No77/MacLink or6,460-fit campaign launch.
Representative bounded pilots remain authorized. No original-paper reruns,
score changes, TEST, sudo, PDF or unrelated files. Root owns Git.

## Published work

Release26 was committed and pushed as
843bff59dd0b6acefc032e34f5e03eecea23c4ec, parent
c0173c5b659967bdee8332dcdca76d72dd5b0b27. Push verified at18:12:59UTC.
COMMIT_RECEIPT_26 and PUSH_RECEIPT_26 are authoritative. Remote tracked/staged
changes were empty after push. PUBLICATION_SELECTION_26 selected78files;
80 committed including rootREADME and source manifest. DOC_UPDATE_RECEIPT_26
pins five current mutable docs; ROOT_README_BLOCK_26 is current. The source
selection excludes all raw model/data/prediction files. Push requires an
interactive tty; the second of the user-provided phrases succeeded. No phrase
was recorded in files. Use the existing helper, no new authentication setup.

## Actual improvement this turn

The failed node selected-state replay was caused by scorer v4 treating the
legitimate model _extra_state dictionary as a tensor. Preserved failure:
NODE_SELECTED_VALID_REPLAY_26.json,7.976s,one constructor/restore and zero forwards.
The reviewed scorer v5 recursively fingerprints typed extra state, preserving
tensor fingerprints and serving. Repaired harness v3 imports that actual scorer.
Both frozen sources, their predecessors and the failure are now published.

One corrected node CPU replay PASSED8.733s:one constructor/restore,four native
forwards,one complete665nodeVALID,zeroTRAIN/optimizer/CUDA. Learned readout
and same-bank mean both reproduce saved logits exactly. Evidence:
NODE_SELECTED_VALID_REPLAY_CORRECTED_26.json. Do not repeat this check.
Scorer v5 manifest:e0b86279593c9f20e49b615c57fce41360babc40201b3172ae76db6eab6f9659.
Harness v3 manifest:8fc619d48cc82b1106c1ca6e3534376e661a1dd1c9987f6b335d477fc94f0f97.
NCNC selected CUDA replay remains unexecuted, requiring the shared GPU slot.

Pairwise21fit scientific source and preflight, plus its whole-epoch CUDA cost
probe source, are reviewed and published. No21fit or cost-probe execution.
The probe uses one full1210TRAINepoch/38updates/twoVALID/one constructor for
structured or native_commonbody0seed101. Manifest
564eefd7fd441fee9c27199cc48c6dc7c3f33b02da3f1b1e7cf0caa16c82369b.
No new queue/owner was created. Existing CPU checks already passed; no repeat.

## Scientific choice

Independent synthesis is saved in research/next_study_priority_20261015_v1.
Default priority is the existing PubMed/Cheb24fit transfer after the complete
MolBACE outcomes and actual GPU cost review. It tests node accuracy, another
native backbone and noBN. Repeated decisions on151MolBACEVALID molecules
provide less information about scope. This is exposed development, not clean
confirmation. Keep all original frozen arms, seed outcomes and class harms.
The pairwise-first counterargument is retained:its21fit design directly tests
ranking errors and includes same-loss nativeI4 controls. PubMed cannot validate
or reject that ranking-specific mechanism. Neither study has scientific launch
admission yet. No new manuscript advantage or acceptance review is warranted.
No new/repeated primary paper read was counted this turn. Existing derivations
and conclusions were reused, including pair-risk population compatibility.

## Active workers — do not duplicate

CURRENT_EXECUTION_26_LATE at18:14:12UTC:
- Structural owner848485/start6089052668 live. First3fits complete149epochs/
  5662updates each. Fourth seed202permuted worker850275/start6089664380 live,
  48epochs/1824updates. Both seed303fits unstarted;END absent.
- PubMed engineering owner849490/start6089402542 live, waiting for the six.
  It then acquires the established slot and runs its existing seven checks.
- OldGRU worker640428/start6055039165 live. Coarse bank counter only advances
  after a whole bank; age alone is not a stall. Original caps/envelopes remain.
- GPU snapshot85%,1932MiB. No partial quality inspected or running setting changed.

The two model-worker slots are occupied. Do not admit a third or duplicate
queued PubMed checks. Afterall6successful and PubMed releases the slot, collect
all6new selectedVALID banks through the frozen structural procedure, reuseold33,
and analyze the complete39bank/45fit family. OriginalBUDDY,NCN,composedMolBACE
andnodeCPU checks already passed. ComposedBUDDY andNCNC CUDA still wait.
Choose/admit the next scientific study from complete outcomes and actual cost.
The late snapshot and this handoff are stored locally/on allocation; they are
post-commit operational receipts to include in the next selected publication.
