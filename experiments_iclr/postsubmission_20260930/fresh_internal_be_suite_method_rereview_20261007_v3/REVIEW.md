# Bounded technical successor method rereview — v3

Reviewed 7 October 2026. Exact v3 MANIFEST SHA256: `cbbb1396ddd1d5479c7b98ef4d709b6e36c19cb55daef8c26bcf76ee883ea4b9`. This is a technical successor review using prior v1/v2 source context, not a fresh paper review, literature search, novelty clearance or manuscript acceptance verdict. Prior packets and reviews are preserved.

**Recommendation: approve exact v3 method/experimental-design source for eligibility to run the fixed pilot, subject to the independent data/export, runtime, resource and root-release gates.** M1–M4 are closed within this method scope. No remaining method implementation or adopted-family closure blocker was identified. This approval does not authorize an exporter, scientific fit, predictive-score opening, later tranche or TEST access, and does not certify competent learning or feasibility.

## Bounded verification

Independently checked all 27 sealed files against MANIFEST sizes and SHA256 and confirmed the requested manifest digest. Parsed all 12 sealed Python files with the standard-library AST parser without imports. Compared v3 with v2 and read the changed closure/placement protocol and transition-test source, their author receipts and relevant config changes.

The following are byte-identical to v2: models.py, factors.py, objectives.py, run.py, selection.py, data.py, native dependency bindings, RUNTIME_PIN and ATTRIBUTION. Thus the actual fixes and method source traced in the prior rereview remain the same. All nonbudget config fields are identical, including eight arms, model settings, loss coefficients/temperature/object cap, initialization, pilot and confirmation seeds, label/view opportunity, pool, checkpoint policy, normalization, optimizers and full horizons. Config additions only bind active-compute/cleanup portions within the same absolute cap; the separate role/runtime reviewer owns their enforcement verdict. No hidden ninth arm or changed scientific recipe was introduced.

No model/framework imports, model execution, remote access, dataset/checkpoint/logit reads, exporter execution, GPU work or source edits occurred. Named installed provider bytes remain uninspected locally. The author's 30-row CPU receipt binds v3 and reports synthetic checks only; it was not rerun here and is not independent proof of real-data quality, custody or resource feasibility.

## M4: adopted-family closure is now consistent

PROTOCOL.md:41 explicitly requires all eight arms and all three seeds, 24 cells, to complete or preserve failure status within each separately adopted task family before opening that family's predictive comparisons. The same paragraph prohibits score-driven stopping, arm pruning and seed replacement within an adopted family and defines the 72-cell matrix as maximal/unlaunched.

PROTOCOL.md:51 locks the adopted family's data/source/runtime/config/checkpoint hashes, missing/failure cells, pooling and selectors after its closure and explicitly says unadopted later families need not finish before that family's opening. The staged section at :75 and clarification at :80 agree: WikiCS first, separately adopted Collab next, separately adopted molecules last; no automatic next tranche; skipped families/reasons disclosed; no unconditional 72-cell score barrier.

This resolves the v2 all-task versus per-family conflict without shrinking any released family. Every adopted family keeps the full representative population, fixed full horizons and all declared paired controls/seeds. The 72-cell parameter/path/cap arithmetic describes the maximal suite, not completed runs or measured need. Later adoption may depend on closed first-family development evidence and cost, but that does not turn first-family signs into confirmation or permit retrospective within-family search. A later distinct question after null/weak evidence remains exploratory and cannot rescue the original recipe or establish unconditional broad validity.

## M1–M3 remain closed by unchanged repaired source

M1: run.py's exact ordinary-independent flag invokes selection.py's own transition only for independent4. Packed independent4_contrastive restores one pooled whole-model checkpoint and the full optimizer bank. Live end-local streams/RNG are preserved. Ordinary independent4 still serves its own-selected bank with recorded per-body stage flags. Source bytes are identical to the verified v2 repair.

M2: the runner supplies the exact bounded TRAIN query subset to alignment_loss. Canonical unordered pair equality makes repeated/reversed records same-target positives and rejects contradictory labels. Own supervision retains every record and complete tails; all duplicate/reversed supervised positive targets remain removed from support. No heldout identity enters these losses. Source bytes are identical to the verified v2 repair.

M3: the protocol retains the correct GCNConv -> LayerNorm -> dropout(.1) -> ReLU description and excludes learned edge attention. The actual native/channel architecture is unchanged; fixed aggregation and common-neighbor support do not become member-specific learned topology. Non-factor count naming remains corrected.

## Discriminating model and Adam transition evidence

The v3 CPU fixture now creates actual Adam history with a supervised backward and optimizer step, saves the joint state, and constructs distinct own candidates (check_cpu.py:132–146). It perturbs live model parameters by +13 and **every live Adam tensor by +113**, including moments and step tensors (check_cpu.py:147–152). Own optimizer histories also receive distinct member-dependent perturbations before they are saved.

It then calls the same helper used by the scientific runner, checks the precise ordinary versus joint filenames, compares all restored model state and selects the expected own or pooled optimizer bank (check_cpu.py:157–167). Both branches now compare param groups and every current optimizer-state tensor to the declared candidate's tensor (check_cpu.py:168–172), in addition to live CPU RNG and global-mode checks.

Those changes make omission of optimizer restoration discriminable from a test that merely rereads an unchanged live optimizer. The author receipt's last two records report live Adam perturbation and both-branch moments/steps assertions. This is useful source-bound engineering evidence. It remains synthetic CPU evidence, not an end-to-end CUDA trajectory, kernel-repeatability guarantee or real-learning certificate; approval here rests on the actual runner/helper bindings already inspected plus this correctly targeted test source.

## Original method and placement claims

PROTOCOL.md:84 now explicitly requires a prospectively matched original GNNM/boundary-projector port before claiming that internal placement improves the original method. The port must use the same new backbone, data, supervision, stochastic-view and selection opportunity, with placement as the declared contrast.

This is the correct scope distinction. Both be_init and be_init_contrastive already place factors internally; their primary difference measures the **combined auxiliary loss**, not internal-versus-boundary placement. The required matched port is a post-pilot follow-up, not a ninth current arm or a recalculation of original paper numbers. Its eventual recipe/role/selector comparison must itself be prospectively bound; current outcomes cannot establish placement causality. Strong native competence and proper DICE/CDLG-style comparator limits remain explicitly required for the stronger claims.

## Comparative and novelty limits retained

The pilot supports a narrow fixed paired question about adding the combined alignment/residual recipe to sign-initialized internal BE, with initialization and unit-start contrasts and ordinary single/untied controls. It does not isolate alignment from residual repulsion, individual factor placements or the molecular pre-message vector. It does not prove a unique advantage from common-body sharing. Single/independent/BE capacity, gradient sharing and total stochastic computation differ; no strong single model matched on capacity and compute has been added.

Real learned internal steering and categorical semantics remain supported by source: trainable common weights/biases/attention/normalization/embeddings/virtual node; trainable private dense rank-one factors and molecular channel vectors; complete member paths; categorical atom/bond encoders before modulation; every member's full own supervised loss; two native stochastic views per update; and persistent member-specific dropout streams. GINE with virtual node and LayerNorm remains an explicit baseline adaptation, not reproduced author benchmark scores. First-factor Rademacher signs are a different adapter initialization, not four independent common-matrix initializations.

The residual class-centered proxy does not guarantee preserved class information or useful independent predictions. Same-label chemistry is coarse, class-center estimates can be imbalanced/noisy, and member separation can occupy irrelevant or gauge directions. It is not DICE conditional mutual information, verified author CDLG loss or an independence theorem. Established BatchEnsemble/TabM and closest prior remain attributed; no new novelty is created by correcting code or closure prose.

Three paired seeds per adopted family remain weak development evidence. Source/CPU success proves no real learning, calibrated costs, evidence specialization, broad graph dominance or acceptance. Stronger claims require the matched original-method placement port where relevant, credible native/prior/capable controls, member/common-error and separately bound evidence-response diagnostics, meaningful capacity/cost comparisons, and separately reviewed unused TEST confirmation. All failures, selected horizons and unadopted families must remain visible.

## Exact approval scope

This review records `approved:true` for **v3 method and experimental-design source only**. It closes the earlier method findings and allows this source to be considered for representative closed-score qualification and the fixed full first-family pilot after the independent gates pass. All job templates remain disabled; root adoption and exact-source/data/runtime/resource/supervisor requirements remain necessary. There is no permission here to start fits, run exporters, open scores, launch later tranches or read TEST.

The separate role/runtime reviewer must decide the updated measurement types, cleanup/supervisor contract and raw-export metadata boundary. No verdict on those distinct changes is implied by this bounded method rereview. If those gates fail, this method approval does not override them.

Exact source bindings and machine-readable scope/dispositions are saved beside this report in SOURCE_BINDINGS.json and REVIEW.json.
