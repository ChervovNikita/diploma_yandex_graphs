# Benchmark review: exact scope and decisions still needed

**For root review. This document changes no roster, scientific setting, access
permission or statistical decision.** It reads the frozen 6,460-fit design and
the current v3 compact descriptor index. It does not inspect training progress,
tensors or new outcomes. `FACTS.json` contains the exact IDs, counts and source
hashes; `PROPOSED_PROTOCOL.json` separates candidate choices from existing rules.

## 1. What the benchmark actually contains

There are seven datasets, eight author architecture families and 27 fixed
presets. PolyFormer's Mono/Cheb choices and OGB's dropout 0/.5 choices are separate
source settings, not extra datasets or a new tuning grid.

| Dataset | Fixed presets | Cohorts per preset | Source fits | Matched native bodies | Shared fits | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CiteSeer | PolyFormer Mono, Cheb | 10 | 20 | 80 | 380 | 480 |
| PubMed | PolyFormer Mono, Cheb | 10 | 20 | 80 | 380 | 480 |
| Coauthor-CS | PolyFormer Mono, Cheb | 10 | 20 | 80 | 380 | 480 |
| Coauthor-Physics | PolyFormer Mono, Cheb | 10 | 20 | 80 | 380 | 480 |
| ogbg-molbace | GCN, GCN-VN, GIN, GIN-VN × dropout 0/.5 | 10 | 80 | 320 | 1,520 | 1,920 |
| ogbg-molhiv | Same eight molecular presets | 10 | 80 | 320 | 1,520 | 1,920 |
| ogbl-collab | NCN, NCNC, BUDDY | 10 | 30 | 120 | 550 | 700 |
| **Total** | **27 presets** | **270 matched cohorts** | **270** | **1,080** | **5,110** | **6,460** |

Each node/molecular preset has 19 shared recipes and 240 fits. NCN and NCNC
have 18 shared recipes and 230 fits each; BUDDY has 19 and 240. The difference
is private BN: only the molecular models and BUDDY have the native target.

The queue separately contains **270 I4 aggregation/export tasks**, for
**6,730 tasks**, not 6,730 training fits. The 540 scheduling groups comprise
270 source-reference groups and 270 matched groups; they are not 540 paired
replicates. Same-bank readouts create no extra fit. Engineering checks, failed
checks, replay work and the separate 24-, 3- and 9-fit development studies have
their own costs and do not change 6,460. The proposed 120-fit confirmation study
is also outside this count and remains a proposal.

### Why each preset needs all four reference roles

| Role | What it answers | Training and selection |
| --- | --- | --- |
| Source-setting native reproduction | Did the port preserve the reference setting well enough to interpret new results? | Ten separate source-horizon fits; published numbers are context, never paired observations. |
| Matched M1 | Does sharing/ensembling improve on one native model given the same extended controller? | Exactly native I4 member0; zero extra fits. |
| Native I4 | Does the shared method compete with four ordinary independently trained models? | Four separately initialized bodies, optimizers and own selected checkpoints; average raw logits. |
| G0, original boundary GNNM | Does the proposed change improve the original shared baseline? | One native body, four factor routes, original own-risk training and mean-logit serving. |

These required roles account for 1,620 physical fits; the other 4,840 are
additional shared methods and ingredient controls.

Source-setting fits cannot replace matched M1: different horizons/selectors
would mix training policy with the ensemble effect. M1 reuse makes the
single-versus-I4 comparison exact and avoids duplicate training. It also makes
M1 and I4 contrasts dependent. I4 contributes one bank score per cohort, not
four independent experimental observations. Its pool does not select a new
joint epoch. G0 is required even when I4 is the strongest reference.

“Source-setting reproduction” is an attempted replication with disclosed
limits. PolyFormer's source uses advancing RNG from global seed 42; the paired
reseeded comparison is an adaptation. OGB's source has no seed/repeat option,
so ten seeded runs are a declared repetition adaptation. Published-score
agreement, a shared method's improvement and exact source-trajectory fidelity
are separate questions.

### Exact task × recipe applicability

Numbers below are physical shared fits, already included in 5,110. “—” means
absent, not failed or awaiting an invented substitute. The 25 recipe identities
include controls and known ingredients; they are not 25 novelty claims.

| Recipe | Node: 8 presets | Molecule: 16 | NCN/NCNC: 2 | BUDDY: 1 | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0: original boundary GNNM | 80 | 160 | 20 | 10 | 270 |
| Private half-pool credit | 80 | 160 | 20 | 10 | 270 |
| Fixed positive bootstrap responsibility | 80 | 160 | — | — | 240 |
| Earlier node within-route SupCon | 80 | — | — | — | 80 |
| Class alignment | 80 | 160 | — | — | 240 |
| Class alignment + conditional variance | 80 | 160 | — | — | 240 |
| Link same-query cosine | — | — | 20 | 10 | 30 |
| Link same-query global contrast | — | — | 20 | 10 | 30 |
| Link same-query endpoint contrast | — | — | 20 | 10 | 30 |
| Structural class-quartile responsibility | 80 | 160 | — | — | 240 |
| Permuted quartile control | 80 | 160 | — | — | 240 |
| Link endpoint responsibility v2 | — | — | 20 | 10 | 30 |
| Link matched query responsibility v2 | — | — | 20 | 10 | 30 |
| Common context-affinity target | 80 | 160 | 20 | 10 | 270 |
| Split context-affinity targets | 80 | 160 | 20 | 10 | 270 |
| BE + private input residual | 80 | 160 | 20 | 10 | 270 |
| Four-route input residual without BE | 80 | 160 | 20 | 10 | 270 |
| Mean aggregation under global J | 80 | 160 | 20 | 10 | 270 |
| Learned global aggregation under J | 80 | 160 | 20 | 10 | 270 |
| Conditional aggregation under J | 80 | 160 | 20 | 10 | 270 |
| Geometry × input correction | 80 | 160 | 20 | 10 | 270 |
| Responsibility × private pool credit | 80 | 160 | 20 | 10 | 270 |
| Input correction × mean J | 80 | 160 | 20 | 10 | 270 |
| Input correction × conditional J | 80 | 160 | 20 | 10 | 270 |
| Private native BN state | — | 160 | — | 10 | 170 |
| **Total shared fits** | **1,520** | **3,040** | **360** | **190** | **5,110** |

The full recipe ID and its declared controls are in `FACTS.json`. Uniform
responsibility reuses G0. `aggregation_mean_globalJ_v1` is the exact alias of
the predecessor global half-pool rule and is counted once.

## 2. The ten cohorts do not all mean the same thing

- **Nodes:** paired optimizer seeds 0…9 accompany split indices 0…9, each bound
  to its retained literal 60/20/20 split seed. There are ten rows, not a
  ten-by-ten grid. Variation combines split and training variation on one graph.
- **Molecules and Collab:** seeds 0…9 share the official split 0. Variation is
  training randomness conditional on that fixed data protocol.
- **I4:** body seeds are cohort seed plus `[0,100003,200003,300007]`. Each body
  owns its state. Matching cohort labels supplies pairing; it does not prove
  algorithm-label exchangeability or independence of all reported contrasts.

Source horizons are PolyFormer cap 2,000 with its native patience 250, and
100 epochs for OGB/NCN/NCNC/BUDDY. Matched controllers use PolyFormer
minimum 2,000/cap 10,000/patience 100 and the other families
minimum 100/cap 1,000/patience 50, preserving source phases and strict first-best
whole-VALID selection. A cap without the declared plateau remains unresolved.
The exact native settings and source/repetition caveats remain in
`expanded_study_design_v1/NATIVE_SETTINGS.json`.

### “Contrastive” is not one common loss

| Recipe/task | Positive target and denominator | Attribution requirement |
| --- | --- | --- |
| Retained node within-route SupCon | Whole-TRAIN same-class records within each route; predecessor's unbalanced denominator | Keep its original identity; do not rename it as the new cross-route method. |
| Node class geometry | Same-class existing route/node records, including the same node in another route; class-balanced denominator/anchors over a fixed, class-preserving budget of at most 512 TRAIN nodes | Native full TRAIN CE remains; variance uses distinct nodes within each route/class. |
| Molecular class geometry | Same observed-class route/graph records in the unchanged source minibatch; class-balanced auxiliary | Native observed-label BCE remains; variance needs distinct graphs. Report rare-class and zero-support rates, especially MolHIV. |
| Link query contrast | The same authenticated query in another route is positive; opposite source-label, same-relation existing alternatives form the denominator, optionally sharing an endpoint | Compare cosine, support/count-matched global, and endpoint contrast. Sampled negatives are not certified nonedges; no link variance term. |

Node/molecular geometry uses the declared all-block auxiliary gradient; the
link recipe retains shared native-risk credit and private auxiliary credit.
NCN/NCNC preserve separate positive/negative native reductions; BUDDY preserves
its mixed-minibatch mean. The interaction recipe uses each task's own rule.
This difference must appear in tables instead of one generic “SupCon” column.

Common/split **context affinity** is another question: all routes learn the
same mixture of two factual similarity targets, or routes 0/1 and 2/3 learn one
target each. Those targets are native versus propagated node features;
atom versus bond histograms; or query-excluded TRAIN degree versus witness
statistics for links. Equal donor opportunities and work make the comparison
interpretable. The targets do not add inference features or extra task queries.

## 3. Scope/removal check

The current method IDs and compact fit IDs retain no factorized-single/F1,
internal/all-map placement grid, jointly trained untied ensemble, equipped
native reference or separate capacity-control campaign. No original-five
dataset rerun is present; original paper scores remain unchanged. There is no
new CAL/SELECT split, HPO, TEST-selected preset or implicit Cartesian method
grid. The four interaction recipes are explicitly listed above.

The residual-only ablation is **four private residual routes around one shared
native body** with BE factors removed. It is not the removed factorized-single
arm or an equipped native reference. Private BN adds no BN to BN-free models.
C7, flow/probabilistic-factor proposals and private-BN × adapter combinations
are not silently added. The separate nine-fit follow-up does not modify this
roster. Deferred architecture cells in the predecessor are outside these 27.

## 4. Proposed report and uncertainty contract — review required

The existing design already requires all outcomes and costs. The following is
a concrete proposal for implementing that requirement across the expanded
matrix; its statistical choices have **not** been adopted.

**One paired row per preset/cohort.** Publish exact fit/control IDs, split and
seed, source/controller/readout identities, native metric in declared units,
proper task loss where meaningful, selected/trained epochs, stop reason,
failures and cost. For I4 retain all four own epochs and component IDs. For a
learned gate, report the actual selected served predictor and the fixed mean
of that same bank; the diagnostic mean cannot reselect the checkpoint or
silently become the primary predictor. Preserve auxiliary support counts and
member/pooling diagnostics with their added inference cost.

**Pair before summarizing.** Within a preset use candidate minus matched M1,
G0 and I4 as applicable, plus the recipe's ingredient controls and predeclared
factorial interactions. Show every signed difference, mean, sample SD, median
and win/tie/loss/missing counts. Do not concatenate presets into a larger n.
Presets sharing a dataset are dependent views of it; 27 presets are not 27
independent datasets. Source-reproduction tables remain separate.

**Candidate interval choice:** paired percentile bootstrap over whole cohort
rows, preserving the candidate/control vector, 10,000 resamples with the
retained analysis seed 130102026, 95% marginal intervals. Root must explicitly
accept or replace this approximation before results guide a choice. Ten rows
give weak tail information. Node intervals combine split/training variation;
fixed-split molecular/link intervals concern training variation. None is an
interval for unseen graphs, molecules or datasets. More bootstrap resamples
do not add observations. Do not resample individual routes or correlated link
queries to enlarge the training n.

SD is dispersion of individual runs, not uncertainty of the paired effect.
Its variance depends on both methods' variability **and their covariance**.
Separated mean±SD bars, tiny SD, or a nearly degenerate bootstrap interval do
not establish superiority. Use paired differences and state the tested claim.

Missing, nonfinite, capped and interrupted outcomes remain visible. A proposal
is to withhold a complete-preset inference when required pairs are missing,
while showing the available descriptive rows and denominator. Do not retry
failed science automatically, silently remove hard seeds, impute scores or
claim convergence from reaching a cap. Numerical repair and its historical
attempt need an explicit disposition and cost record.

### Multiplicity and independent confirmation

The retained **226-contrast** testing rule belongs to the **2,480-fit ancestor**;
it is not a complete comparison registry for 6,460. With ten nonzero pairs, its
two-sided sign-flip minimum is 2/1024≈.001953, above Holm's first threshold
.05/226≈.000221. Even perfect signs cannot pass that family. A mean-zero null
alone also does not justify independent sign flipping. These are known design
limitations, not permission to choose a different test after seeing outcomes.

**Candidate route for review:** keep the full matrix exploratory, publish all
declared effects without confirmatory win labels, then use the existing
`confirmation_protocol_proposal_v1` only after completing the decisions below.
It proposes at most one method in one setting, 20 fresh paired cohorts and
120 new fits, with M1 reused from I4 body0. Its three one-sided sign tests ask
whether the candidate wins in more than half of repeats against **each** of
M1/G0/I4. The joint intersection-union claim uses the maximum component
p-value, each at .05; 15/20 wins are needed per comparator. This claim concerns
win probability, not mean improvement. Additional settings or separate claims
require a prospective multiplicity plan. Twenty is a proposed budget, not a
power guarantee, and no new fits are authorized here.

Concrete gaps in that proposal:

1. Freeze the eligible candidate/control-only list, screening rule, tie rule
   and practically meaningful effect criterion. Its strongest-baseline maximum
   is a screening statistic, not a newly served baseline. AUC, accuracy and
   Hits gains all lie on 0–1 scales but do not have identical practical meaning;
   root must accept cross-setting screening or predeclare a task/setting scope.
   Reconcile this with the retained parent `no_cross_preset_winner_selection`
   flag: a later confirmation nomination cannot silently replace the complete
   preset reports with one favorable setting.
2. Freeze all 20 seed streams and check collisions against prior development,
   component offsets and checks. For a node winner, specify the confirmation
   split law and estimand: fixed-split repeats and fresh split/training draws
   answer different questions. Do not silently change the existing ten rows.
3. Freeze the primary endpoint, failure/tie treatment, comparison family,
   interval assumptions and any meaningful-effect threshold. The existing
   proposal treats ties/unresolved outcomes as non-wins in a fixed denominator;
   quality-only descriptive statistics must retain their missingness separately.
4. Reconcile historical outcome exposure before describing confirmation as
   independent. New seeds do not erase validation-guided method/dataset choice
   or prior TEST access. All earlier selected-VALID studies remain development.

## 5. Proposed access and publication protocol — review required

No consolidated exposure ledger or confirmation seed registry was located in
the reviewed E source/metadata filenames. That is a missing artifact, not proof
that no historical TEST exposure occurred.

| Stage | Candidate allowed information/action | Evidence needed before moving on |
| --- | --- | --- |
| Development | Native TRAIN fitting; whole VALID checkpoint selection and declared diagnostics | Complete fit/outcome ledger; all amendments timestamped; TEST metrics/predictions unavailable for decisions. |
| Review freeze | Root fixes candidate, comparator family, seeds/splits, controller, evaluator, actual served readout and claims | Hashed registry and real review decision; no field silently defaults from a favorable result. |
| Heldout readiness | Source-bound selected-state reconstruction and declared VALID parity checks | Each own checkpoint, native role/query/topology receipts, complete official population, qualified runtime/resources; no fallback current/recovery state. |
| TEST release | Score only the exact reviewed request after actual human authorization | Access ledger entry binding authorization, request/source/input/checkpoint hashes and the eventual export; no tuning from returned results. |
| Publication | Release the complete declared report and failures/costs | Immutable raw-scalar/export references and separate exploration/confirmation labels. Later changes require a new evaluation rationale. |

Each exposure entry should record who/what read which role, when, whether raw
labels/predictions/metrics or only opaque integrity metadata were seen, the
authorized purpose, artifacts affected and any subsequent design decision.
Unknown historical exposure stays “unknown.” TRAIN/VALID graph data and the
native Collab TEST topology rule must be distinguished: adding native positive
VALID edges at TEST is a declared source operation; adding TEST query edges or
using TEST outcomes in selection is not. Current closed-TEST operation must
also be reconciled with source programs' diagnostic/RNG behavior; the review
must identify a qualified no-TEST emulation or disclose the adaptation.

If the user later chooses a complete exploratory TEST matrix, report it
completely and keep its exploratory label. It cannot revise the frozen primary
candidate or make that same set untouched again. No protocol proposed here
creates TEST authorization or changes data-role boundaries.

## 6. Readiness and cost decisions that remain concrete

Overlay v3 supplies all 6,460 source descriptors (100 replaced source bindings,
6,360 reused), preserving all 2,480 ancestor IDs/contracts. **Descriptor coverage
does not equal executable campaign admission.** The current readiness delta
still names whole-epoch/selected-state/device branches, source-reference
fidelity, runtime/dispatcher binding, measured resources and statistics/access.
Link final serving explicitly lacks complete actual selected-state/provider/
VALID-parity/authorization evidence. Its interface's coverage must also be
checked for each new extension and actual learned readout before any TEST plan.
These are source-snapshot findings; no current training or resource probe was
performed for this review.

Report physical fits, optimizer attempts/completions, total worker time,
elapsed time, preprocessing/cache construction, selected inference/export time,
peak device/host memory and storage separately. Charge I4's four bodies and all
failed/resumed/replayed work. Reuse is zero **new** training cost, not zero
historical cost. Short smoke checks establish engineering behavior, not
accuracy or mature full-run cost. The current launch-cost catalogue transfers
pilot observations to only 60 exactly matching intentions; it supplies no
campaign-wide ETA. New source/settings need scoped measurements before timing
is transferred. Any two-child concurrency needs measured headroom, not a
nominal GPU count. Only the allocation is authorized; future 77 assignments
are not available capacity.

**Root review outcome needed:** select or revise the reporting/confirmation
choices in `PROPOSED_PROTOCOL.json`, commission the exposure/seed registries,
and attach their reviewed hashes to the eventual execution/reporting binding.
This package sets every decision to pending and grants no launch, monitoring,
TEST or new confirmation work.
