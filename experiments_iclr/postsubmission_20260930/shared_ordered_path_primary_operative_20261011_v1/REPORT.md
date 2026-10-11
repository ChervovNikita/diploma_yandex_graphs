# One integrated shared pathway from ordered three edge walks

**Decision:** a two-edge ordered path does not establish additional reach beyond current depth-two SAGE or endpoint VALUE. Retain one inactive, teacher-free **three-edge walk residual inside the four native routes**. It can supply additional third-hop feature context when the selected walk reaches outside radius two, and can retain a joint sequence before pooling. Its utility and any benefit specific to sharing remain unmeasured. PathMLP is the strongest operator collision; PathNet and Graph Substructure Networks prevent a new path or motif principle claim. This is an operative hypothesis and source reuse plan, with no new compute family or launch.

## What the primary methods actually add

| Primary scope | Relevant operation | Boundary for this question |
| --- | --- | --- |
| PathMLP, arXiv:2306.13532v2, saved §4.1, §5 and Algorithm 1 | Feature-inner-product path selection; nonlinear node encoding in the printed equations; ordered concatenation and path MLP before path pooling; own TRAIN CE. Full model also has node/path-specific coefficients and an adjacency-row encoder. | Ordered sequences and node/path fusion are established. Its adjacency-row map has N input coordinates and learned node/path coefficients add transductive capacity; neither belongs to the proposed small branch. The saved PolyFormer token-collision argument does not transfer to nonlinear depth-two SAGE. |
| PathNet, IJCAI 2022, DOI 10.24963/ijcai.2022/310, new scoped §§3.1–3.2 and §4.2 | Maximal-entropy walks, reversed ordered recurrent encoding, shortest-distance-specific maps U_d, path attention and end-to-end prediction. | Fig. 3's triangle/open-path distinction uses root-distance information. Plain ordered feature concatenation does not inherit that structural argument. Its sampling theorem is not a guarantee that four walks capture useful evidence here. |
| Graph Substructure Networks, arXiv:2006.09252, saved §§3.1–3.2 reread | Explicit node/edge orbit counts enter structure-aware messages. | A triangle/closure/distance identifier would introduce a known structural signal. The proposed branch has no such identifier, and receives no motif-expressivity credit. |

For fixed graph, weights and node-wise preprocessing/LayerNorm, two native message layers depend on the two-edge message unfolding. Endpoint VALUE changes the learned message using live sender/receiver states, but does not add another propagation round. Changing raw features strictly beyond radius two cannot affect that current two-layer factual logit. This is a receptive-field statement, not a diagnosis of the saved errors.

A root plus two successive neighbors uses at most two edges and the same possible feature radius. Nonlinear tuple encoding may preserve correlations before compression, but there is no established universal separation from a sufficiently expressive two-layer endpoint-conditioned network. Three edges add a real information opportunity: a selected terminal at distance three can affect the path logit while the current two-layer logit is unchanged. Walks that return inside radius two do not supply a new terminal feature; finite feature-guided sampling may choose only such walks. The actual fraction of roots gaining outside-radius-two observations is unknown until a later authorized source/data qualification. A deeper native endpoint-VALUE model is therefore an indispensable comparator.

PathNet's inspected qualitative example is particularly useful here: two-edge attribute sequences can match in a triangle and an open branch, while root distances differ. Removing its distance maps and keeping only concatenation does not preserve that argument. The proposed three-edge attribute branch may change such sequences, but has no graph-isomorphism or beyond-1-WL theorem. Repeated-feature walks can still collide.

## The author code is a reference rather than an exact paper port

The PathMLP URL in the saved primary abstract resolves to **GISec-Team/PathMLP**, pinned commit `4525359bc683d89b4b805251ace6f20c74d0b8ee`. README, model, sampler and main were read as source text; none was imported or run.

At this commit, `path_project(x)` is linear and has **no activation before concatenation**. The path map is activated afterward. The paper's Eq. 11 nonlinear node encoding is therefore not an exact description of this code path. The code stores `beta=args.beta` as a fixed scalar, while the paper describes a trainable gate. Its learnable path weights are N×P parameters. The adjacency-row `mlpA` is constructed and evaluated even at beta zero. These differences matter for capacity and source equivalence.

The sampler's `depth` counts **node positions**: depth 4 is three edges/four positions and has 2×3×4=24 branch candidates. It reverses sampled neighbor order, appends the root, and selects one common random subset of candidate columns for all roots. Low-degree branches duplicate the last available selected neighbor; its zero-neighbor fallback inserts the root. It uses outgoing records, whereas the operative contract below follows incoming native message edges. These are declared adaptations, not certified author-equivalent defaults. The code's ordinary train-mask loss supplies no independent teacher ensemble.

## One concrete operative hypothesis

**H:** one shared three-edge ordered-walk residual, with native rank-one private factors and full own CE, acquires correct alternatives on fixed baseline common-rival errors and retains enough of them to improve served full-population predictions over competent native and same-information references.

Use the ordinary depth-two native SAGE bank as the base, without modifying or stacking onto the active E4 source. For route m, capture its already computed nonlinear native stem `e_m(v)` **before the first graph module**. Reuse that stem once; do not acquire another node encoder. Obtain the native two-layer final hidden state `h_m(v)` before output normalization. Fix H=128, P=4 and four node positions as the single illustrative operation, with no length/count/strength search.

For each root v, the label-free path input contains four fixed incoming walks `v0=v, v1, v2, v3`, with actual edges `v1→v0`, `v2→v1`, `v3→v2`. Then

`g_m(v,k) = GELU(F_path,m(e_m(v0) || e_m(v1) || e_m(v2) || e_m(v3)))`

`q_m(v) = (1/P) sum_k g_m(v,k)`

`h_new,m(v) = h_m(v) + F_out,m(q_m(v))`.

Apply the unchanged native output normalization and classifier to `h_new,m`. Both added maps use the existing `s_m ⊙ W(r_m ⊙ z) + b` convention, with W/b common and r/s private; bias remains outside s. `F_path` has shape 4H→H and `F_out` H→H. Every member's full TRAIN true-label CE trains the complete native body and the common path maps through its live private route. Serve the ordinary mean of four member probabilities. There is no teacher bank, pseudo-label target, pretrained side network, route-owned label subset, path ownership assignment or later student retraining.

Initialize the new output W/b to zero from a fresh native initialization; this preserves the initial deterministic native function. It guarantees neither learned member competence nor final pool quality. The path input-map gradient and some private-factor gradients are initially zero behind the zero output map, then become active as the output learns. Keep this initialization and all harms visible; it is not a competence theorem.

### Fixed path construction

Use the exact public X and native source-to-target edge records. For the branch, create incoming unique nonself neighbor lists while retaining the entire original native support, loops, order and multiplicity in the native forward. Rank these branch neighbors by the raw feature inner product, with a fixed label-free tie convention. Choose top 2, top 3 and top 4 neighbors at successive steps, enumerate at most 24 three-edge walks and select four fixed candidate columns with one public seed shared by every arm. Backtracking is allowed only over actual nonself edges.

Duplicate the last available neighbor when a nonempty list is shorter than a branch width. If a walk reaches a node with no incoming nonself neighbor, stop it and zero-mask the remaining gathered positions; do not invent a self edge. Store indices plus validity masks in the label-free batch, not as learned node parameters. All controls use those same selected node tuples and masks. This keeps the operation finite while exposing its selection bias and duplicate weighting. No labels, model probabilities or error cohorts select a walk.

The branch intentionally omits PathMLP's adjacency-row map, per-node path coefficients, PathMLP+ feature smoothing, PathNet's entropy eigenvector and root-distance recurrent maps, and GSN counts. The single changed operation is a nonlinear joint walk encoding before mean pooling, integrated through existing member factors.

## Controls that can explain the gain

| Required control | What it decides |
| --- | --- |
| Native own-CE shared C and the eventual completed depth-two E4 under exact source identity | Whether the path method improves the competent baseline/current operator. No active E4 outcome is opened here. |
| Same-parameter marginal control | For the identical tuples, average each of the four positions across paths first, concatenate those four means, and apply the same F_path/GELU/F_out. Every map coordinate remains active. This retains sampled observations and position marginals while removing within-walk pairing before the nonlinearity. It uses fewer path-map evaluations, which must be disclosed. |
| Same-parameter native-hidden control | Feed `[e_m(v), h_m^1(v), h_m^2(v), GELU(h_m^2(v))]` through those same maps. All input blocks and weights are functional. This supplies capable extra nonlinear capacity using the existing two-hop native state without walk context. |
| Retrained order control | Fix a label-free per-root/per-walk swap of the two intermediate positions, retaining the root and terminal. Train and serve with it. A single global swap only relabels positions. Recoverable order is a limitation of the control, not a reason to report an evaluation-only shuffle. |
| Complete union-view M1, genuine independently fitted I4 and matched joint untied4 | Every body gets the full native-plus-walk architecture, same tuple union and full labels. M1/I4 keep capable own optimizers/selectors; joint untied4 matches initial route functions, mean-own-CE and coherent bank selection while removing sharing. These are controls, not acquired teachers. |
| Three-layer native endpoint-VALUE control | It receives at least the same third-hop raw context through one additional native propagation round. Its actual parameter count, initialization and work must be bound; it cannot be replaced by the weaker two-layer control to claim a path-specific gain. |

A path-order gain, a pairing gain and a sharing benefit are separate claims. Compare each to its own competent control. If marginal pooling or native-hidden capacity explains the gain, close the joint-path explanation. If the deeper native operator explains it, credit wider receptive field. If M1/I4 or matched joint untied4 explains it, credit a known architecture acquisition rather than new useful sharing. Report the path shared-minus-untied contrast beside the original native shared-minus-untied contrast; these remain full-procedure comparisons.

On one fixed C bank, report gross new/removed coverage, true correct alternatives on its fixed strict-rival and all-member-wrong cohorts, acquisitions served/lost, retained/lost baseline successes, full-population repairs/harms, member mean/worst/per-seed accuracy and raw/calibrated NLL/Brier. Do not pool the fresh F repeat's cohorts with the different historical context BASE bank. A greater oracle count without served utility fails H. Prediction distance or TRAIN fit is insufficient. Root must freeze practical accuracy/probability criteria, exact source/roles and a finite complete-control budget before any future fit; none is admitted here.

## Source reuse and cost are concrete but unexecuted

Reuse the project `FactorLinear`, `member_context`, original native body, `NativeModelAdapter`, `wrap_shared` and persistent member RNG streams. A future narrow adapter copy computes the original stem once, preserves every native residual module, adds the path residual before output normalization and then calls the original head. Register both path maps before optimizer creation; dynamic member context must cover them. Preserve source equality at zero branch without consuming extra training-dropout draws. Update the parameter-role/map ledger to distinguish added shared path weights from original native weights. The detailed file/operation plan and exact pins are in `SOURCE_REUSE_PLAN.md`; this folder supplies no module, owner or launcher.

At H=128/M=4, the two maps add `5H²+2H=82,176` shared W/b and `7MH=3,584` private factors, total **85,760** parameters. Against the recorded ordinary native F inventory of 216,674, this gives **302,434** served parameters for the proposed SAGE adaptation; E4 and deeper controls have their own inventories. Four route predictions still run. The added dense forward work is `M*N*(P*4H²+H²)` MACs, about 13.04 billion at N=11,701, before gathers, nonlinearities, backward and the native model. A materialized FP32 `[N,P,4,H]` buffer is about 95.85 MB per route; indices are about 1.50 MB, before masks and tapes. Stem reuse and deterministic indices are actual reusable work, not a wall-time or memory-saving result.

No actual graph/provider compatibility, new-state timing or peak memory has been qualified. The older frozen F-plus-S diagnostic and its 18-body cost bound remain unchanged and do not certify this integrated design. One new primary method body (PathNet) and one scoped author-code set were retrieved; saved PathMLP/GSN bodies were reused. Existing paper scores, current E/Penn sources and TEST closure remain unchanged.
