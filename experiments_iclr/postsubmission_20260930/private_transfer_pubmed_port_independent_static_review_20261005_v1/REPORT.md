# Independent static source review of the disabled Pubmed port

5 October 2026. This review concerns source correctness, wrapper behavior and custody. It is not a paper review, predictive result, acceptance verdict, execution release or adoption of the proposed scientific comparison.

The reviewed immutable packet is `private_transfer_pubmed_disabled_port_preparation_20261005_v1`. Its packet manifest SHA256 is `21fcc254e0bab5639895351a87feda54eda39f3e00cf688bca8fe540adb3a926`; its source manifest SHA256 is `a814a23e3ee7bba0b9e2391a6f59570a7c92a97a4ae090c460e3c766433a1954`.

I verified both supplied manifest pins, all 36 packet-manifest entries, all 21 source-manifest entries, all 33 saved input bindings and all 16 declared exact copies. No mismatch was found. I parsed the 21 source files as AST data and independently checked the declared model edits, selected helper equality, native training-body equality, disabled flags and parameter arithmetic. The input/source manifests saved beside this report identify the exact reviewed bytes. Supplemental plan inputs used for the requested original30/companion9 distinction are separately listed in INPUT_MANIFEST.json.

The work used local text reads, AST parsing, stdlib JSON and hashes only. No target or numerical module was imported or executed. No fixture, sampler, feature/data/model/checkpoint/history/score payload, server, network, MacLink or 18.77 access occurred. No extra agent was used. The source packet and its bound saved inputs were not edited. Only this new review directory was written.

All source references below are relative to the reviewed packet unless another directory is named. P2 denotes a material source safeguard or protocol-definition issue to resolve before a future release. These findings do not establish that a current execution occurred or that any saved numerical result is invalid.

## Findings

### F1 — P2: The candidate FP32 receipt is not bound to the current input/runtime identity

Locations: `run.py:36–50`, `qualify_training_step.py:133–144`, `custody.py:149–167` and `custody.py:175–195`.

The qualifier records its actual `runtime` and `inputs` in RESULT.json. The runner's `admissible()` checks the receipt hash, passed/scope flags, exact source-manifest hash, architecture names and first-episode sizes, but never compares the receipt's runtime or TRAIN/feature identities with the prospective job/current loaded inputs. The runtime and input loaders subsequently authenticate the *current* job; they do not make this missing comparison. Thus a receipt from the same sealed source and sizes but a different feature member or numerical runtime satisfies this particular receipt gate. The exact TRAIN hash is fixed in custody, which limits the TRAIN mismatch risk, but feature bytes and runtime versions are prospective job fields.

The separately required root runtime/feature authorities can prevent such reuse if root checks their content carefully. They are authority references, not an implemented receipt-identity comparison: `authorize()` only checks their approved flag, nonempty references and reference hashes. Preserve this distinction when describing what the program itself guarantees. Bind the qualification receipt's recorded TRAIN/feature identities and runtime to the actual job/input loader identity before accepting it for cost or fit. Seed0 qualification can remain the prescribed engineering fixture; this finding requests no seed grid or new tolerance.

### F2 — P2: Native fit admission omits the cohort's cost and selection-budget checks

Locations: `native_ncnc.py:46–62`, `run.py:64–84`, `COHORT_PROPOSAL_DISABLED.json:149–152,248–252`, `custody.py:113–117`.

The candidate runner requires nonempty byte-authenticated complete-cycle cost evidence and `selection_budget_fairness_approved=True` plus a paid-budget description in its frozen cohort. `validate_fit_plan()` for the native members checks the root-adoption flag, TEST closure, member identity and native schedule, but omits those cost and budget conditions. It also does not inspect the plan's `fits_authorized` flag. As a concrete static counterexample, a future plan with root_adopted_after_TRAIN_cost=true and the correct native members/schedule, while retaining the current empty cost evidence and false budget flag, passes `validate_fit_plan()` once the separate job's fit/VALID flags are enabled.

`custody.authorize()` does require an approved, byte-pinned `Pubmed_cost_and_evaluator_admission` reference for native fitting. That makes manual root admission a prerequisite, but the evidence contents are not parsed and the cohort omission remains. Enforce the prospective cost/budget conditions consistently for the complete candidate/native comparison, including native epoch/full-VALID/replay costs and horizon practicality. The current file remains disabled and this finding proposes no shortened native schedule.

### F3 — P2: Native replay opens an early member payload before checking complete-four artifact custody

Locations: `native_pool_replay.py:21–49`, `custody.py:24–30`, `NATIVE_WRAPPER_CONTRACT.json:owned_complete_fit_custody`; predecessor `pubmed_native_predictive_program_source_20261004_v1/common.py:106–112` and `replay_selected.py:44–50`.

Before numerical setup, the replay program checks only that four rows exist in member order and calls `require_evidence(owned_complete_native_fit_family)`. That helper verifies an approved flag and hashes of supplied reference files. It does not parse a complete-family inventory, successful owned supervisor terminal status, member coverage or the relationship between those references and the supplied four freezes/states/scores.

The subsequent loop authenticates and reads member0's freeze, then calls `torch.load()` for member0's state and scores before member1–3 artifact existence, hashes, COMPLETE status, source/input/runtime identity and freeze-to-selected-file bindings have been checked. A missing, incomplete or mismatched later member therefore fails only after earlier payload access. The program prevents final partial-family pooling, but does not provide the stated complete-family gate before the first selected payload opens. A root-approved reference alone supplies no executable semantic or all-four artifact check.

Use an initial custody pass over all four members and their successful owned terminal/inventory evidence, including exact freeze/state/score/source/input/runtime relationships, before deserializing any selected member payload. A second pass can perform reconstruction, exact state/RNG/metric replay and pooling. The bound predecessor explicitly checked successful supervisor/fit custody before replay, whereas this new wrapper leaves those semantics to root. The future supervisor's concrete evidence format is still unresolved; this review did not read any terminal, fit, state or score payload.

### F4 — P2: Proposal wording conflates original30 promotion with descriptive companion9 analysis

Locations: `COHORT_PROPOSAL_DISABLED.json:247`, `REPORT.md:74`, `custody.py:113–115`; supplemental `shared_private_transfer_fixed30_plus9_mechanism_analysis_plan_20261005_v1/REPORT.md:80,84,114,120`, original `shared_private_transfer_paired_pilot_execution_root_20261005_v2/ROOT_RELEASE.json:3–30`, and `shared_private_transfer_row0_single_companion_preparation_20261005_v1/COMPANION_PLAN_DISABLED.json:222,227`.

The Pubmed proposal sets `original30plus9_quality_gate_required=true`, and its report calls this an “original complete30+9 lead gate.” The saved mechanism plan instead requires the complete fixed39 family for analysis, applies the original30 quality/attribution policy unchanged, and adds descriptive F4–F1 and member-count-credit contrasts with no new threshold. The companion plan explicitly adds no companion acceptance gate and leaves the original promotion gate unchanged. The original release contains the frozen positive-mean/two-positive-block quality policy and separate attribution policy.

The Pubmed wording does not accurately preserve those two roles. It describes an unsupported combined gate and could invite companion diagnostics to become new promotion conditions. No numerical companion threshold is actually implemented in this packet: the proposal key is descriptive, while the future fit gate only requires a root evidence reference named `original_complete_family_lead`. Clarify the prospective requirement as the unchanged original30 quality/promotion decision plus complete39 descriptive mechanism analysis, preserving unavailable diagnostics and adding no companion threshold. This is a protocol-definition issue, not an assessment of outcomes.

## Source properties checked

The candidate remains nonrecursive NCN. `models.py:15–24` preserves the width256, one-layer puregcn, input projection, JK=True, .4 input/.3 encoder-and-head dropout and unchanged unframed four-row predictor. Comparing against bound `shared_backbone_private_transfer_training_source_20261005_v2/models.py` showed exactly input width3703→500 and the two outer-count assertions1409026→589058. All declared Adam, transfer, head, geometry, cycle, adjoint, runner, qualifier and vendor copies match their saved inputs.

`private_adam.py:39–62` detaches old moments, applies the fixed current-gradient Adam rule with the stable zero-history root, and rejects its stated nonfinite/underflow conditions. `transfer_step.py:124–171` computes the virtual private response, differentiates the outer half-aggregate/half-member BCE, commits the shared Adam step, then recomputes and commits private Adam from unchanged old private parameters/moments. Virtual state is discarded; only recomputation advances private dropout streams. The detached variant removes adaptation credit at the same virtual values. Serving averages raw logits without adaptation. `ordinary_episode_step()` retains three joint Adam commits. I found no additional concrete numerical-learning defect by static inspection; this is not numerical parity or derivative qualification.

The source count derivation is consistent with the bound constructors and head flags:

| Role | Source-derived count |
| --- | ---: |
| Encoder: 500×256 projection +256 bias +1 JK scalar | 128,257 |
| Dense head bases/biases: seven256→256 and one256→1 linear | 460,801 |
| F4 shared | 589,058 |
| One private row: 3,841 factors +2,560 LayerNorm affine +1 beta | 6,402 |
| F4 private total | 25,608 |
| Untied4 outer, qualification only | 2,356,232 |
| Capable-single outer/private, qualification only | 128,257 /463,362 |

`models.py:26–34,132–154` partitions factors, normalization affine and beta as private and the encoder/bases/biases as shared; the F4/untied assertions match that arithmetic. Runtime construction, actual role/shape coverage and Parameter identity preservation after device conversion remain part of qualification.

`native_bodies.py` is byte-identical to its bound source-v2 module, with candidate/reference NCNC train AST bodies equal after normalizing the function name. Its native factory retains IncompleteCN1Predictor depth1 completion, native Pubmed dropout/clamp/degree/splitsize settings and own encoder/head Adam groups. `native_ncnc.py:81–121` creates one complete member per invocation, applies its own normalized native loss with no1/4 scaling, accounts36 dropped-tail commits per epoch, validates every5 epochs with no initial serve, selects the first strict improvement in rounded4 complete MRR, and stops on11 misses or9999 epochs. It never loads a donor state during fit. The fixed seed policy produces12 native fits, with member0 reused for the single, plus9 candidate fits for21 unique fits and15 served comparisons.

After the custody issue in F3 is addressed, the replay body has the expected computational structure: fresh reconstruction, serialized full-state/RNG restore, inherited128×float32-epsilon logit tolerance, exact per-query/rounded metrics and selected-MRR identity, exact post-serve state/RNG, and mean raw-logit pooling with no optimizer step or ensemble reselection. Static equality and control-flow inspection cannot establish that this integration executes successfully on the allocation.

All six supplied job recipes have false source-review, fit and VALID flags, null available-manifest hash and soft bound, and false external-hard-bound confirmation. The source authorizer checks root approvals and exact packet bytes before numerical imports. Current candidate fit/cost entry points restrict the future scientific arms to sharedF4 endpoint live/detached/ordinary joint. The proposal itself has false root adoption and budget approval, empty costs, and no release. Nothing in this review enables those files.

## Unresolved requirements and present blockers

The compact bound ROOT_ADOPTION establishes exact TRAIN member authentication and feasible paired seed0/cycle0 geometry for all589 episodes including the44-row tail. It expressly does not admit model training. I read only that compact design adoption and its report; I did not read a raw census/query payload. Other seed/cycle feasibility remains unqualified, and the runner is written to fail without redraw or skipped episodes.

| Requirement | Static status and consequence |
| --- | --- |
| New allocation available-input manifest | Absent by contract; the jobs have null hash pins. This blocks every numerical path today. |
| Allocation features, provenance and applicability of any raw-feature-equivalence evidence | Unresolved. Width/shape and a historical receipt reference do not establish new allocation feature authority. |
| VALID positives and original fixed500 pool custody/semantics | Unresolved. Loader source preserves row/candidate association, duplicates and other-VALID-positive collisions; it supplies no current authenticated values or global nonlink claim. |
| Exact allocation runtime and candidate FP32 Adam/direct+mixed/committed-serving gate | Pending new engineering. Citeseer receipts cannot admit the changed source; F1 additionally records the missing input/runtime crosscheck. |
| Allocation native factory/train parity | Pending. The two-epoch disabled native qualification runs only the candidate body and commit accounting; it does not compare reference state/numerics or qualify VALID/replay. The source report correctly discloses this limit. |
| Full VALID evaluator/selected-state/pooling integration | Pending fresh allocation qualification. The candidate formula is consistent with the native tie-aware rank, but complete logits, rounded dictionaries and serialized replay were not run. |
| Model/train/full-serving/replay time, RSS/CUDA/output and practical horizons | Unmeasured. TRAIN geometry costs do not substitute for these. The live cost recipe alone cannot release the other comparisons. |
| Owned supervisor/terminal custody and enforceable caps | No new supervisor/launch adapter is supplied. Root must bind a reviewed owned implementation, measured bounds, fresh outputs and successful terminal relationships; F3 identifies the replay-side semantic gap. |
| Original scientific prerequisites and prospective release | Original30 promotion remains required; complete39 companion analysis is descriptive with no new gate. Current jobs have no such admission or scientific release. |

No runnable invocation, target-source repair, data acquisition or execution was attempted. No P0/P1 numerical blocker was established from static source inspection. The four P2 findings remain open, and the intentionally missing authorities, input manifest, qualification and bounds are concrete present execution blockers. A future review must assess the revised exact source and actual bound engineering evidence; this report supplies no requested outcome or fit approval.
