# Native Polynormer Q/K source interface

This is source/metadata preparation only. No historical outcome, model/data
tensor, checkpoint or prediction was opened. No remote access, CUDA, fit,
optimizer update or GO occurred. The nine developed operator/ownership scopes
are preserved; native_tied and ordinary/boundary controls are additional arms.

## Native architecture and operator

This source is **Polynormer**, distinct from the PolyFormer implementation in
v8. Pin native_model SHA `9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8`.
The source recipe has seven local GAT layers, width512, one head, two global
layers, beta−1, no pre-LayerNorm. Input/local/global dropouts are separate knobs.
The source WikiCS dimensions are300 features and10 classes; new graph providers
bind input/output dimensions, factual COO edges and original TRAIN labels.

Every global layer normally forms a tied positive sigmoid query/key. The exact
installer source is `pre_sigmoid_qk_operator_adapter_source_20261009_v1`:

- `active_reversible_exp`: one bias-free large projection a=W(x*r), then
  u=sigmoid(a*s+b)*exp(.5*gamma). Query and key both equal u. Zero-start gamma
  is trainable, not an identity-bypass branch. The symmetric positive kernel
  structure is retained. No separate query weight matrix is introduced.
- `pre_sigmoid_split`: use the same a, but q=sigmoid(a*(s+delta)+b) and
  k=sigmoid(a*(s−delta)+b). Zero-start delta is trainable. Bias stays outside
  scales, and query/key use one shared large projection. It changes the directed
  affinity without introducing a second dense weight matrix.
- `full_qk`: clone the initialized complete k FactorLinear as q, with separate
  W/b/r/s storage. The clone starts equal, with no constructor/reset draw. Set
  qk_shared=False and use the original native untied global forward.
- `native_tied`: no gamma/delta/query copy. Equipped ordinary references still
  receive global r/s coordinates, as in the exact source installer. A bare
  native ordinary reference is therefore a separate necessary control.

Gamma/delta has2*members*512 parameters. Full Q/K adds2*(512*512+512+
2*members*512) copied query parameters. W/b is retained when global-only r/s
is added to ordinary references. Shared all-map BE already has those factors.

## Ownership, initialization and training

The source supports single, genuine independent4 and be_init. Single/I4 bodies
are ordinary native models; only the global Q/K maps gain r/s equipment. BE has
one shared complete body, FactorLinear around every dense/PyGLinear map and four
routes. It uses first-stem Rademacher seed+900001 and later factors equal one.
Normalizations, local attention vectors and other native parameters remain live.

Native model m initializes at base_seed+1009*m. Private dropout stream m starts
at base_seed+1009*m+300001, preserving Python/NumPy/CPU constructor isolation and
forked persistent CPU/CUDA dropout streams. For a new separately fitted I4 body,
set actual_base_seed=base_seed+1009*body and use a one-member source session;
this preserves the corresponding source model/dropout seeds and optimizer law.
Each body is fitted and selected separately. It is not four repeated logits.

The exact streamed F callback takes two full own-CE views, view then member.
Every backward completes at the same old parameter state before any Adam action.
Shared loss gradients scale each view/member by .5/4. Independent member losses
scale each view by .5, with a genuinely private Adam. One Adam per physical body
uses source defaults, epsilon1e−8 and weight decay0. There is no contrastive
auxiliary, teacher or outcome-dependent router in this source port.

## Local/global transition and new controller

The source driver trains100 local updates, then restores the strict first SELECT
accuracy maximum's **model and Adam state** from those local updates. It retains
the live end-local global/private RNG, then turns the native global branch on.
It does not rewind RNG to the selected local step. The ordinary independent
control restores each body's own local state; shared routes restore one joint
local state. Final selection permits local/global snapshots, with each snapshot's
native `_global` flag. The source's total1100 updates/100local history is preserved.

The proposed standardized packet prospectively uses5000 complete updates for
every arm:100local then4900global. Earliest SELECT accuracy maximum selects the
final checkpoint; no checkpoint0/early stop. Native local/global dropout values
are separately recorded. Proposed nine-point search is three LRs
(.0005,.001,.002) times these explicit dropout triples:

| Input | Local | Global | Role |
| ---: | ---: | ---: | --- |
| .5 | .5 | .5 | literal source anchor |
| .5 | .3 | .5 | local dropout neighbor |
| .5 | .5 | .3 | global dropout neighbor |

Thus local and global dropout are not silently collapsed into one knob. The
extended horizon/new roles/seeds/search/CAL are declared adaptations, not an
author CLI or published score reproduction. Native source and historical
outcomes remain unchanged.

## Original GNNM reference

The all-map internal BE model is not original GNNM. The original reference must
wrap the native Polynormer input and both active local/global prediction heads
with the actual source BEBlock R/S/B affine maps. Preserve bias-free W with
source Xavier uniform initialization, Rademacher input R, output R=1, all S=1
and B=0. Route selection applies the original member row to those boundaries.
Local/global source blocks stay shared and contain no internal diagonal factors.
Use mean raw logits for its SELECT/serving; original scores are never edited.

This is an explicit original-GNNM adaptation to a new backbone. It removes the
replaced native boundary biases and adds native original-BE W/R/S/B, exactly
declared. It uses the same prospective two-view/full-budget native Adam protocol
for fair comparison; it is not the original paper's backbone/optimizer recipe.
The reference cannot be materialized as an all-map model with a different label.
Until root numerical/source checks pass, it remains an unqualified source
candidate, not an accuracy baseline result.

## Exact proposed count

Two studies:20 WikiCS masks and10 Amazon Photo masks, two paired HPO seeds and
three paired comparative seeds. Both are new to the original five-dataset paper;
both retain encountered-development status. Roman Empire, Amazon Ratings,
Minesweeper, Questions and Tolokers are excluded for every arm.

Ten atomic fit families: four Q/K operators including native_tied × single/BE,
bare native single, original boundary GNNM. Five genuine I4 banks reuse their
single body0 and pay three additional independently selected bodies per bank.
Fifteen served banks per mask/seed. Every fit uses the same declared domain,
except I4 bodies use their own corresponding single family's selected point.

| Planned unit | Count |
| --- | ---: |
| New standardized studies |2 |
| HPO fit groups |360 |
| Comparative fit groups |2250 |
| Total model-fit groups / Adam starts / physical bodies |2610 |
| Model optimizer calls at full5000 |13,050,000 |
| Selection jobs |20 |
| Comparative banks / exports / CAL jobs / evaluations, each |1350 |
| Genuine I4 banks |450 |
| All job records |6680 |
| Scalar CAL updates at500 |675,000 |
| Actual fitted models/updates/scores |0 |

The exact source constructor creates an initial unstepped Adam before the Q/K
installer replaces it with the active fresh owner. Across this roster those
2,088 discarded pre-install optimizer objects perform zero updates; active
trajectory owners remain2,610. The metadata ledger charges45,000,000 TRAIN
forwards and45,000,000 streamed backwards. Include the unused constructor costs
without pretending they are additional model fits or optimizer actions.

The source driver does not have a generic Phase2 compressed-head shortcut.
Model costs include both own views and native local/global attention; all I4
bodies are paid. Hardware memory, serialized local-snapshot/restore growth,
endpoint serving, exact runtime compatibility and whole-study cost remain root
prelaunch gates. Trained-state replay and convergence diagnostics are checked
during approved execution, not by an unapproved pilot.

## Candidate adapter representation

Use the pinned source Ensemble/native model constructor. Reuse exact `_attach`,
`_global_forward` and `_streamed_f_step` AST. The only numerical provider bindings
inside their AST are the fixed580 TRAIN guards and the declared Adam LR value;
width512/head1/global2 and every Q/K/operator/objective statement are unchanged.
Exclude the obsolete installer route/runtime/role gate from execution; the new
prospective source packet's normal GO/route/controller guards govern new fits.
The pinned historical `install`/`make_session` are never called or modified.

Capture external streams, factor/boundary member selectors, native global flag,
Session counters/failure state, NumPy RNG and the separate best-local model/Adam
snapshot. Wrap SELECT observation to retain only the strict first best-local
state during1–100. At step101 load model/Adam only, preserving end-local RNG;
selected final capture includes that branch and transition history. No recursive
checkpoint inside the external state is needed.
