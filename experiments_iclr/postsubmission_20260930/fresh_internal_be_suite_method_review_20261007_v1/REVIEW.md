# Independent static method and control review

Reviewed 7 October 2026. Packet: `learnable_internal_be_contrastive_multitask_suite_20261007_v1`. Exact MANIFEST SHA256: `3c0c88031dfc30178f0942794e64f4ae3677d3a2def1f2035adaa9e745474211`.

**Recommendation: do not approve v1 for scientific execution.** A concrete WikiCS transition defect changes the jointly regularized untied control's declared learning trajectory. Prepare and review an immutable successor before release. This is a source review, not a real-data competence, efficiency, novelty or manuscript acceptance verdict. Real-data custody and runtime/selector admission are covered by a separate reviewer.

## Scope and evidence

Read PROTOCOL, ATTRIBUTION, DEPENDENCIES, all three configs, factors/models/objectives/run/runtime/data source, check_cpu source and CPU_CHECK receipt. Read the exact locally retained Polynormer and NCN model/utils dependencies named by DEPENDENCIES; their SHA256 values match the bindings. Independently checked every one of the 19 sealed files against MANIFEST bytes and SHA256. Parsed sealed Python source using the standard-library AST parser without importing it.

No model/framework imports, execution of CPU_CHECK, dataset/checkpoint/logit reads, GPU work, remote access, or modifications to v1 occurred. RUNTIME_PIN's named remote installed PyG/OGB provider files are unavailable on this local host; the runtime receipt binds their purported identities, but this review did not inspect those installed bytes or independently attest that remote runtime. Standard PyG/OGB operator semantics and the locally inspectable wrapper/native source support the architecture reading below; full installed-operator qualification remains a separate gate.

## Findings that require action

### M1 — coupled untied WikiCS state is spliced from separately selected epochs (release blocker)

`Ensemble.independent` is true for both ordinary `independent4` and `independent4_contrastive` (models.py:169–184). The latter receives cross-member contrastive gradients (objectives.py:46–74,84; run.py:131–133), so its four model and Adam states form one jointly regularized training state.

At the WikiCS global transition, run.py:110–113 branches on `model.independent` and loads every body's `own_local_m.pt` and corresponding optimizer. Those files are chosen separately using each member's VALID metric (run.py:145–152). The packed untied contrastive bank can therefore be assembled from four mutually incompatible epochs. It then resumes a coupled objective from this hybrid state. PROTOCOL's pooled-selector description for `independent4_contrastive` does not describe this implementation. This is a control/trajectory defect, not a novelty limitation or a tiny numerical parity issue.

Ordinary `independent4` correctly needs its own local transitions and own final checkpoints; single and single_contrastive have only one body, so their own/pool choice is identical. The fix should make only the packed untied multi-member control restore the complete pooled local checkpoint, including all jointly trained body and optimizer states. Preserve the prescribed live end-local member/dropout and outer RNG state. The existing `selected_local.pt` saves the full bank and optimizer list (run.py:96–100,143–144). Do not alter the ordinary independent bank into a pooled-selected comparator.

### M2 — collab identity positives can repel another record of the same target pair (conditional scientific correctness gate)

Collab alignment defines positives as the diagonal identity matrix (objectives.py:38), treating every other sampled query row as a negative. Training preserves all positive records, concatenates them with sampled negative rows, and does not canonicalize or deduplicate target queries (data.py:155–167). The duplicate-removal rule applies to support, not to the supervised query population (data.py:117–129).

If two repeated or reversed records of the same undirected TRAIN pair occur in the contrastive subset, one view of that target becomes a negative for another row of the same target. In deterministic evaluation they even have the same endpoint-derived representation; stochastic row dropout may make the representations differ without creating a different edge instance. This does not match a claim that the same target edge is the positive and distinct other edges are the negatives.

No data was read, so this review does **not** assert that official hydrated batches actually contain such collisions or quantify their frequency. Before adopting the edge-instance interpretation, bind TRAIN-only canonical-pair duplicate evidence for the actual projection and sampled contrastive object population, or implement declared canonical-pair-aware positives/deduplication while retaining every record's own supervised loss. The synthetic support-removal fixture contains duplicates, but its contrastive collab query fixture has no repeated positive pair (check_cpu.py:30–38), so the present engineering receipt does not settle this issue. A pre-outcome successor may address it without tuning on predictive results.

### M3 — the protocol calls a nonlinear one-layer GCN linear (factual correction)

The pinned native `GCN` constructor installs `GCNConv -> LayerNorm -> dropout(.1) -> ReLU` for `num_layers=1` (native model.py:148–163,180–188); models.py:90–91 selects exactly this path with `ln=True`. The 128-to-64 dimensions also prevent its optional residual addition at this layer. Therefore the complete one-layer encoder is nonlinear. The protocol's “One-layer linear GCN” description should be corrected.

The substantive narrower statement remains correct: this GCN has fixed degree-normalized aggregation and no learned edge attention. Factors can alter projected channels before the native LayerNorm/ReLU and NCN nonlinear transformations. They do not turn the encoder into member-specific learned graph attention. The supplied `taildropout=.05` is unused by the selected non-pure one-layer constructor path; the active encoder dropout is its xdropout `.25`, adjacency dropout `.25`, and post-convolution dropout `.1`. Record actual active settings when comparing native recipes.

## What the method actually implements

### Shared body and private factors

`FactorLinear` retains each source matrix and bias Parameter object and adds trainable member-indexed r/s vectors (factors.py:18–26). It implements `diag(s_m) W diag(r_m) x + b`, with bias added after output scaling as declared. No weight or factor is frozen. Replacement covers materialized `nn.Linear` and PyG dense Linear children, including native GAT/GCN projections (factors.py:29–48). It does not factor normalization scales, attention vectors, embeddings, scalar GINE epsilon, virtual-node seed, or NCN beta; those remain common and trainable. Four complete nonlinear paths execute against the same body, using different private factor rows (models.py:200–214).

Each BE member's own supervised loss contributes to the common parameter gradient; private row gradients also flow through the entire member path. Shared BE minimizes the member mean own loss plus the auxiliary loss, with one Adam optimizer for the common body and all factor tensors. Untied members have separate complete parameter sets/optimizers and minimize their sum of own losses; multiplying the auxiliary mean by M keeps the intended auxiliary/own relative scale (objectives.py:79–84; run.py:45–50,123–138). This is real learned internal steering, not a frozen teacher, output-only adjustment or a pooled prediction loss substituted for member competence.

### WikiCS / Polynormer

Private maps occur in the input projection, every local feature-gate h map, every GAT projected channel map, every local residual dense map, global h/k/v maps, global output map, and both prediction heads. The native global default ties q to k (`qk_shared=True`); there is no independently parameterized q map in this configuration (native_polynormer.py:6,14,22–34,57–63). The factors can change local attention logits through projected node channels and change the polynomial h*x paths before the native nonlinear stages (native_polynormer.py:148–175). Common learned attention vectors, beta gates and LayerNorm parameters continue to train.

Local and global representation hooks capture the selected head's input for exactly the requested TRAIN node IDs (models.py:72–83). No private contrastive projection head is added. Both native modes and all modules are retained, with inactive local/global heads naturally receiving no gradient when unused.

### Collab / GCN + NCN cn1

The first r is on the GCNConv dense projection, followed by the native nonlinear encoder described in M3. All NCN dense maps in xlin, xcnlin, xijlin and final predictor are factorized. Endpoint products and sum-aggregated common-neighbor features then undergo the native nonlinear decoder (native model.py:546–592). NCN beta stays common/trainable; `cndeg=-1` retains complete common neighbors. The contrastive representation is the final prediction map's input (models.py:99–109).

Support passed to the decoder is the same target-masked adjacency passed to the encoder, while native encoder DropAdj makes each stochastic message-passing view different (models.py:103–104; native model.py:180–184). The decoder's own DropAdj has edrop zero, so the common-neighbor support is identical across those views. This is member-specific channel computation over shared structural support, not learned member-specific topology or an NCNC completion mechanism.

### MolHIV / residual GINE and virtual node

The adaptation uses all nine atom and three bond integer categorical fields and OGB encoders. Atoms are embedded before the input factorized map; bonds are embedded separately at each layer before GINE (models.py:121–126,145–156). Integer category indices are never multiplied by steering factors. The additional per-layer private message vector multiplies `LayerNorm(h + virtual)` before the GINE neighbor message `relu(x_j + bond)` and self term. MLP maps inside GINE and the virtual-node update are also factorized. The virtual-node seed, bond/atom embeddings, GINE epsilon, normalization and residual structure are common/trainable (models.py:127–162).

Thus members can induce different nonlinear neighbor-channel responses while topology and chemical fields are preserved. The architecture is a five-layer residual GINE + virtual-node + node LayerNorm baseline adaptation; it is not a numerical reproduction of an OGB author GIN/GINE recipe. Its common categorical embeddings do not establish chemically meaningful evidence specialization. Permutation equivariance is consistent with the source and tiny fixture; that fixture is not a real-data chemical competence certificate.

## Initialization, dropout and supervision

All factors begin at one except `be_init`/`be_init_contrastive` first-input r entries, which receive independent Rademacher signs from the dedicated CPU generator `seed+900001` (factors.py:51–62; models.py:185–192). The matrices/biases/common parameters are constructed from the same base seed across the BE arms and single member 0. This is a different first adapter initialization, not four different common-matrix initializations or deeper random-factor distributions. For molecules the signs operate on already embedded atom channels.

Untied constructors use `seed+1009*m`. The runner creates persistent CPU/CUDA dropout streams `seed+1009*m+300001`, restores each before its forward, and advances it afterwards (run.py:51–65). This mechanism applies to BE as well as untied members. Two views advance each member's stream independently; contrastive losses and bounded index selection introduce no additional random draw. Data order is intentionally aligned. The helper constructor seed context preserves Python/NumPy/CPU torch state, but does not independently preserve CUDA RNG; the runner's explicit member streams are the relevant stochastic forward guarantee. This review does not attest CUDA kernel repeatability.

Every arm makes exactly two own-supervision forwards of every TRAIN target in each update. WikiCS uses member CE; molecular targets use BCE; collab retains positive-mean plus negative-mean BCE. Ordinary independent members receive unscaled own gradients, while BE averages own loss across member paths. No target specialist mask or sole pooled CE is present. This provides matched per-member label opportunity. It does not make the single body and shared four-member body equal in total stochastic gradient samples or compute.

## Contrastive objective and interpretation

Alignment normalizes member/object representations and uses symmetric cross-view softmax. WikiCS and molecules make all same-label TRAIN objects in the opposite view positives, including the paired object; opposite labels are negatives. Collab uses only the identity paired row, subject to M2. This is cross-view supervised/instance contrast, not verified semantic equivalence of molecules or arbitrary edge deletion invariance.

Residual contrast differentiably subtracts the current batch's class mean separately for each member/view, skips class singletons, normalizes residuals and classifies the correct member among same-object opposite-view member representations. M=1 is zero. Labels and representations enter the same TRAIN-only subset; there is no stop-gradient center or private auxiliary projector (objectives.py:46–74). The runner applies both terms with weights `.05`, temperature `.2`, and at most 512 deterministic aligned objects (run.py:125–133); full own supervision still covers all targets. The code hardcodes those fixed coefficients/temperature while configs agree, so the sealed recipe has no present config/source coefficient mismatch.

Centering makes the residual objective invariant to a common within-class representation shift. It does not protect predictive class information under parameter updates: representation Jacobians can differ by object, and both objectives can change class means through the common network. Same-label chemistry is a coarse grouping, batch class centers can be noisy/imbalanced, and member separation can live in gauge or irrelevant feature directions without useful prediction diversity. These are known limitations, already substantially disclosed by PROTOCOL, not newly discovered implementation errors. Tiny deterministic BE-unit checks also cannot show that residual loss alone breaks exact member symmetry; actual dropout or first-factor signs supplies asymmetry.

## Eight arms and causal meaning

| Arm | Parameter ownership / initialization | Training objective | Intended selection |
|---|---|---|---|
| single | One complete ordinary body, base seed | Two-view own supervision | Its own metric; identical to pool at M=1 |
| single_contrastive | Same ordinary body/seed | Own + alignment; residual exactly zero | Same as single |
| independent4 | Four complete untied bodies, distinct member seeds | Four separate own losses, no joint regularizer | Each member's own local and overall checkpoints; serve assembled bank |
| independent4_contrastive | Four complete untied bodies | Own + joint alignment/residual terms | Pooled bank; M1 violates pooled local-transition custody |
| be_unit | One common trainable body, all factors one | Member mean own supervision | Pooled local/overall checkpoint |
| be_init | Same BE capacity; first r random signs | Member mean own supervision | Same as be_unit |
| be_unit_contrastive | Same unit BE initialization | Own + alignment/residual | Same pooled selector |
| be_init_contrastive | Same sign-initialized BE | Own + alignment/residual | Same pooled selector |

The primary `be_init_contrastive - be_init` comparison isolates the declared combined auxiliary recipe in the same architecture, initialization, member count, own supervision and selector. `be_init - be_unit` tests first-adapter initialization; `be_unit_contrastive - be_unit` tests the auxiliary recipe at unit start. All must be analyzed as fixed paired seeds/cells with failures retained.

Single/untied contrastive controls test whether the combined recipe helps ordinary bodies. The eight arms do not isolate alignment from residual repulsion, nor each factor placement or molecule message-factor contribution. They cannot establish either term's necessity, an optimal placement, or a unique benefit from shared BE. Ordinary `independent4` has independently learned weights with a deliberately common target order/negative population; that is a legitimate matched-order ordinary ensemble, not fully independent data randomness. Its final own-selected bank is implemented at run.py:159–169. The jointly regularized untied comparator must continue to be called packed untied, not independently learned predictors.

## Capacity, normalization and claim boundaries

Stateless LayerNorm is consistent within each task across single/untied/BE arms, avoiding cross-member BatchNorm running-statistic contamination. Its trainable scales and biases are shared in BE and private inside each complete untied model. GINE's adaptation is disclosed and should be judged as its own baseline recipe. Each task uses the same backbone width/depth within the pilot, but parameter capacity differs among controls.

Static dimension counts, without framework construction: Polynormer shared base 7,537,172 parameters and BE private vectors 122,112 (union of local/global modules); GCN+NCN shared base 33,922 and BE private vectors 4,100; molecular BE private vectors 63,492. These formulas match the corresponding tiny CPU fixture dimensions/counts when substituted, but are not executed production-model receipts. Molecular common capacity also depends on installed AtomEncoder/BondEncoder embedding cardinalities, which were not locally inspected here. An independent4 bank owns four complete common-base copies; BE owns one plus factors. `factor_counts` labels all non-factor parameters “shared,” including the four actually untied copies in independent4, so interpret that output as non-factor count for that arm, not sharing across members.

Shared BE still performs every complete member path twice per update. Factors add multiply operations and gradients; untied models add four common optimizer/checkpoint states; different selection policies add different bookkeeping. Similar parameter counts do not establish equal compute, accuracy/compute tradeoffs or sufficient capable-single baselines. Capacity/compute matched strong single controls and representative real-data resource qualification remain future requirements for superiority/efficiency claims, not hidden pilot alternatives. Primary BE-vs-BE comparisons avoid these capacity changes.

The attribution properly acknowledges established BatchEnsemble rank-one maps, TabM-inspired first-adapter initialization, label conditioning motivated by DICE and CDLG-inspired cross-member pairing. The implemented residual proxy is not DICE conditional mutual information, verified CDLG author loss, an independence theorem or established new graph principle. Learned channel differences are not evidence specialization; a separately bound task-safe evidence-response panel and prediction/member/common-error evaluation are needed. No implementation or synthetic CPU evidence establishes real learning, broad graph dominance, official custody, calibrated costs, publication novelty or acceptance.

## What the CPU receipt supports

CPU_CHECK is an existing synthetic engineering receipt, not rerun by this review. Its source uses tiny double-precision eval-mode graphs, zero native dropout probabilities for the WikiCS/molecule fixtures, synthetic labels and miniature dimensions. It checks forward shape/finite gradients, some shared/private optimizer updates, native unit-factor function/shared-gradient identity in selected modes, a chemical permutation, support duplicate removal and a direct in-memory role-key rejection.

It does not run the scientific training loop, persistent stochastic dropout runner, coupled local transition, complete-data support/batching, production parameter construction, real chemical categories, real learning or resource ceilings. The “full member label gradient” fixture verifies nonzero gradient for each random logit/object, rather than separately perturbing every label through each production backbone. These checks are useful and their numerical parity is not a reason for additional tiny FP32 gates; the actual M1 trajectory defect and scientifically meaningful qualification gaps determine release.

## Release conditions

1. Keep v1 immutable and make the pooled joint transition fix in a separately sealed successor. Reinspect the exact branch and checkpoint/optimizer/RNG custody for packed untied WikiCS; retain ordinary independent own selection.
2. Correct the nonlinear GCN description and disclose its active native dropout/residual behavior accurately.
3. Resolve or bind M2's TRAIN target identity condition before an edge-instance contrastive interpretation. Preserve full record-level own supervision and close predictive outcomes during this pre-outcome choice.
4. Require independent role/runtime/resource review and full representative TRAIN update + complete VALID evaluation qualification with scores closed. This method review does not approve actual data hydration, execution, TEST access or score opening.
5. After a successor passes source review, limit initial claims to the fixed pilot's combined-recipe effects with member competence and failures. Additional mechanistic, novelty, efficiency or broad-superiority claims need the specified independent evidence.

Machine-readable bindings/findings and the nonapproval receipt are in `REVIEW.json` and `SOURCE_BINDINGS.json` beside this report.
