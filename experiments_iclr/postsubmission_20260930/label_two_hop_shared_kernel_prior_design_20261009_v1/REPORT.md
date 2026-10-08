# Inactive shared-kernel two-hop label posterior assessment

9 October 2026. One source-only hypothesis survives: replace the one-hop label
operator with two applications of the same member-specific feature attention
kernel. This tests whether additional permitted anchor evidence is useful within
the frozen-native posterior recipe. It is an adaptation of known label
propagation and masked-label learning, not a new primitive or a novelty claim.
No method implementation, import, fit, launch, source mutation or new outcome
read was performed. The query-conditioned value-gate branch is outside this
assessment.

## Attributable operation and surviving question

The saved closest-prior report and conditional hypothesis already establish
multi-hop label propagation and query-label removal before propagation as
C&S/GAMLP/UniMP/Echoless-LP ancestry. Echoless-LP explicitly masks every current
partition label before its potentially multi-hop label precomputation. Its
learnable final encoder combines feature and label tensors; its per-node
retained-label normalization is different from the scalar below. The saved
UniMP source conclusion describes complementary TRAIN label exposure/CE targets,
all-TRAIN label serving, and several attention layers over combined feature and
label states. Neither a second hop nor masked-label learning is new here.

The saved GOODIE appendix/code follow-up resolves its inspected path as
persistent TRAIN anchors, iterative clamped LP, a trainable label decoder,
joint feature gradients, combined-classifier CE, and learned embedding fusion.
This provides direct decoder/LP ancestry. Its inspected path differs from the
fixed native capture, direct route CE, member probability mean and fixed native
probability mixture here. Those scoped distinctions do not establish absence
elsewhere. No primary-paper or author-code reread/search was needed.

The surviving utility question is narrow: with competent detached native H,
can two applications of one feature-only kernel deliver useful extra label
evidence without harming complete-population quality or making the extra graph
work too expensive? A capable single receiving the same two-hop messages can
explain any gain. Parameter sharing alone supplies no evidence of useful
ensemble diversity.

## Exact inactive candidate

Let A be the 580 permitted TRAIN IDs, Q the common 290 TRAIN targets, N=11701,
F=512 the captured native width, d=64, and C=10. Preserve the original directed
nonself edge records, including order and multiplicity. For each member m:

1. Compute q_m=Q_m(H) and k_m=K_m(H) for **all N nodes**, from fixed detached H
   and the existing BE maps. For incoming edge j->i, score q_m(i)^T k_m(j)/sqrt(d).
   Normalize over every incoming nonself record at i, including sources with
   zero label value. Compute these scores and this normalization once for the
   member at its current parameters.
2. Build Z0_m[N,d] from zero. Only rows a in A\Q receive
   (579/290) V_m(e(y_a)) during training. At serving, only A receives V_m(e(y_a))
   with scale1. No raw H value, native probability, pseudo-label, readout bias,
   new value map, residual or nonlinear intermediate transformation is added.
3. Compute Z1_m=P_m(H) Z0_m for **all N nodes**, then
   Z2_m=P_m(H) Z1_m for **all N nodes**, using the same attention tensor in both
   multiplications. Gather target rows only after the second multiplication.
   Apply the existing O_m class readout there. All four maps remain shared
   dense BE maps with their existing private factors.
4. Train mean own masked-label CE across C4 routes, directly on these label
   logits. Keep the existing staged zero-base adapter, so native logits do not
   enter this loss. Four backwards use the same old parameters before one
   candidate Adam update. Fully untied routes use their existing four Adams;
   the joint single uses one CE/Adam on its existing nonlinear four-head
   readout; one-path uses one CE/Adam.
5. For every arm, serve exact native probabilities only when structural
   **exact two-edge** permitted-anchor support is empty. Otherwise use
   .2 p_native + .8 p_label per route and mean the actual route probabilities.
   The joint single uses its one posterior at the same .8 family weight.
   Use the existing stable log mixture and complete-population selectors.

P has row sum1 on nonempty incoming neighborhoods and a zero row at nodes with
none. Calling the full matrix row-stochastic requires this isolate convention;
P^2 need not have row sum1 when a first-step source has a zero row. There is no
renormalization of label mass or second-step support. A structurally supported
row can have tiny mass, numerical underflow or a canceled message; none of those
conditions changes the declared fallback.

## Reach is exact length two, not radius two

The root supplied saved counts are one-hop reach 2762 of 5274 development nodes,
directed BFS radius<=2 reach 5025, no directed path at any length 161, and 88 reached
only beyond radius2. Thus 249 are outside radius2. These are motivating saved
counts, not a new measurement. The corrected source definition is
`native_CS_selected_recipe_error_analyzer_source_20261008_v1/analyze.py:78–89,123`:
it starts BFS at permitted TRAIN anchors and follows directed source->receiver
nonself records; distance -1 means no path. The root supplied the saved distance
supports {-1:161,1:2762,2:2263,3:81,4:5,5:1,6:1}. No RESULT file or arrays were
opened here and no counts were recomputed. The earlier pointer to the separate
one-hop diagnostic source was corrected in this scoped record.

For positive attention, pure P^2 support is the set of anchors connected by
**exactly two incoming edges**. Without self-loops, this need not contain
one-hop support. A query whose sole anchor is a direct leaf neighbor may lose
its label evidence at step2. Consequently 5025 is an upper bound on the
exact-two-edge development support, not the candidate's known reach or fallback
count. In particular, 161 is not its known fallback count. No overlap, exact
P^2 count, support benefit or accuracy benefit is inferred.

A standard clamped-LP alternative would replace Z1_m[a] by Z0_m[a] for visible
anchors a before computing P_m Z1_m. This preserves direct anchor values and can
retain one-hop evidence while allowing two-edge walks through other nodes.
Queries are never clamped and the source scale is still applied once. Clamping
is known LP ancestry (including saved GOODIE). It is an inactive, unselected
alternative: it does not change the pure-P^2 candidate in this assessment and
does not introduce a second recipe or a hop/clamp search.

## Mask, echo and capacity boundaries

Every Q label must be absent from Z0 everywhere, including all member contexts.
Z1 must be rebuilt from that masked Z0 for each draw and current parameter state.
Q nodes may relay other visible anchors in Z1; zeroing their entire Z1 rows would
change the declared operator. A stale unmasked Z1 or cached PZ0 admits literal
query-label echoes. Neither removing graph self-loops nor masking only the
final queries blocks two-edge return walks. Masking Q at the source blocks
their literal labels on all such walks. Visible anchor return walks remain
permitted.

For a fixed old-state, mask-independent P and linear label-value route, and
conditional on target i in Q, each other TRAIN anchor is retained with
probability 290/579. Applying 579/290 once to Z0 gives the conditional first
moment of the full permitted-label operator with i's own source excluded.
Its P^2 diagonal coefficient can be nonzero, so that self exclusion matters.
Rescaling Z1 again is wrong. This statement is about the target row's linear
message, not all intermediate rows, posterior probabilities, CE, gradients or
unbiased out-of-sample evaluation.

The frozen native H already encodes TRAIN supervision and development selection.
Detachment prevents new label-stage gradients into the native model; masking Q
does not remove its earlier supervised influence from H. The comparison is
transductive and staged. TEST truth remains outside construction and selection.

Pure P^2 preserves the restricted class-prototype geometry: for fixed maps,
Z2 is a weighted sum of the same C class value vectors. With only one reachable
visible class, each linear route's logits remain a nonnegative scalar times
one fixed class-logit vector. A second hop may expose more classes; it does not
solve that direction constraint or reproduce the separate query-value gate.

P must remain differentiable through both applications. The derivative includes
(dP)PZ0 + P(dP)Z0, as well as the value derivative. Detaching Z1, rebuilding a
different second kernel, or caching P across optimizer steps changes the
candidate. Reusing P inside a member is exact; fixed H does not make learned
Q/K parameters fixed.

## Required source changes if separately elected

These are prospective changes to a new source copy; the listed current files
were read only.

| Source boundary | Exact required change |
|---|---|
| `core.py:206–253`, `:260–272`, `:304–335` | Replace the query-row one-hop route operation with all-N Q/K and one full-edge P, rebuild masked Z0, apply P twice to full-N fields, then gather/read out queries. Remove the one-hop empty-row assertion; assert exact zero on structurally empty two-edge support. Training and serving must use this same operation. |
| `controls.py:149–194`, `:203–216`, `:283–324` | Make every ordinary route/head use its own full-edge kernel twice. Concatenate four **second-hop** messages for the existing nonlinear joint single. Keep controls' information, masking, source scale, optimization and serving aligned with C4. |
| `posterior.py:35–41` | Replace one-hop reach by two successive Boolean incoming-neighbor propagations from permitted TRAIN IDs. Use A at serving; use A\Q for training assertions. Do not use message magnitude or heldout labels for support. |
| `stage.py:236–268`, `:308–345`, `:438–445` | Construct/bind the new two-hop sources and same support across arms, preserve one fixed capture, zero-base direct label CE and existing bounded serving. Replace one-hop work checks with separate score, hop1, hop2 and full-field counters. |
| source descriptors, protocol, binding manifests and endpoint receipts | Declare exactly two shared-kernel applications, full nonself graph, mask rebuilding, no new maps, exact-two-edge fallback, updated work accounting and newly hashed source identities. Existing one-hop seals cannot certify this change. |

No native implementation, capture policy, trainable map dimensions or nonlinear
joint readout needs changing. The candidate/control parameter inventories remain
C4=76328, joint4head=349184, fully untied4=283648 and one-path=70912. These are
unchanged saved source inventories, not new parameter runs. No parameters or
Adam moments are added by the second propagation.

## Full graph work, not a nominal factor of two

Let E be the number of preserved nonself edge records, E_T the number ending in
the T queried rows, L the visible anchors, and d=64. The corrected BFS source
asserts 431206 nonself records after removing 11701 self records from 442907 input
records. These are source contracts, not newly observed data counts. Count
dense multiply-accumulates per route, before backward, separately from
edge dot products and sparse weighted message elements:

| Work | Existing one-hop query-row source | Declared pure P^2 source |
|---|---|---|
| Q/K dense maps | (T+N) F d | 2 N F d; Q adds (N-T) F d |
| Value dense map | L d^2 | same L d^2, once at Z0 |
| Class readout | T d C | same T d C after gathering Z2 |
| Edge scores and one softmax | E_T d dot work; E_T records | E d dot work; E records, computed once |
| Weighted edge messages | E_T d | 2 E d, with all-N intermediate/output fields |
| Structural support checks | one propagation | two propagations |

T=290 during training and T=5274 for complete development selection, L=290 and580
respectively. The candidate creates full-N Z1 and Z2 and retains differentiable
edge attention for both uses. Full-N q replaces the smaller query q; Q/K, edge
score, softmax, label-field, gather/scatter and backward buffers contribute to
peak memory. The float32 payloads for one route's Z1/Z2 alone are2*N*d*4 bytes;
this is neither a measured peak nor the total increment. Sequential route
backward can release each route's graph before the next; the joint single must
retain four head paths until its one backward.

BE shares weights and optimizer inventory. The inspected route loop still
performs member-specific dense forwards, edge scoring and both propagations;
private input factors precede dense maps. Four-route C4, joint4head and untied4
therefore each require four head kernels and eight message passes per call.
One-path requires one kernel and two passes. Thus a four-head two-hop call scores
4*431206 directed records once each and traverses 8*431206 weighted message
records, before backward; the one-hop comparison scores/traverses4*E_T records.
Native acquisition/capture, all 1100
complete selections, source/role hashing, I/O, optimizer and snapshot work remain
part of total cost. Fullgraph training/selection wall time and process peak
memory require fresh qualification before any efficiency claim. No numerical
cost or runtime ratio was measured.

## One fixed, decisive comparison

If separately elected, compare fresh **C4 one-hop** with **C4 two-hop,
joint4head two-hop, fully untied4 two-hop, one-path two-hop** on the same fixed
competent native captures at 6101/6203/6307. Use the same paired common-Q draws,
initialization conventions, original Adam settings, all 1100 updates per arm,
and the strict first maximum correctcount on all 5274 development rows. All
two-hop controls receive identical two-hop information and exact-two-edge
fallback. No coefficient, hop, seed, subgroup, horizon or fallback grid is
introduced. Existing outcome files are not comparison inputs or a reason to
choose this candidate.

Open comparisons only after all 15 fresh bank endpoints close and their exact
source/capture/parameter/Adam/mask/work/hash custody is complete. Preserve the
original three-block native acquisition cost in the total accounting.

C4-two minus C4-one tests this complete depth adaptation, including changed
support/fallback. C4-two minus the capable nonlinear joint4head-two is the
ensemble falsifier; minus untied4-two tests whether sharing remains useful.
One-path is secondary. Apply the saved full-family accuracy/NLL/member gates
without relaxation: positive gain at each of three seeds, mean gain at least
.2pp in both co-primary candidate/control contrasts, nonpositive mean pool NLL
delta, and the saved member safeguards (mean/worst member accuracy deltas at
least -.1/-.2pp and mean/worst member NLL deltas at most .01/.02, averaged over
the three seeds). Require a positive complete-population
depth contrast as well, then account for extra total wall/memory cost. A depth
gain common to a capable single is useful label-propagation adaptation, not
ensemble-specific evidence. No ensemble-specific utility or efficiency conclusion survives if
the capable single explains C4, extra propagation harms full-population quality,
or added work defeats the claimed resource advantage.

This remains an inactive design. A positive fixed screen would still leave
source-native label-aware references and ordinary independent GNN4 obligations;
these five arms establish neither a faithful LP/UniMP reproduction nor global
superiority. No scientific result, experiment authorization, novelty clearance
or absence certificate is issued by this report.
