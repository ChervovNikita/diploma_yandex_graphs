# Independent decoder launch source check

10 October 2026. **No material launch blocker found in the inspected source.**
This is a source-readiness conclusion with the limits below, not scientific
utility evidence or a manuscript verdict. No launch-packet file was changed.

## Checked acquisition and control identity

The exact AST-generated constructor, fit and run were inspected without compiling
or executing the generated Family. The pinned native constructor is truncated
before device/optimizer creation; the extension registers B before one AdamW
optimizer and checks that every trainable parameter belongs to it exactly once.
The optimizer includes native maps, factors and B with the fixed lr .001 and
zero weight decay.

Genuine factorized I4 runs four separate members=1 fits per seed. Each constructs
its own complete native body and B, follows the declared native/factor/dropout
seed stride, selects its own decoded VALID accuracy and is restored separately
before terminal probability pooling. The untied/common-B control instead
constructs four fresh complete factorized bodies, registers one common B on a
ModuleList, and trains them together under one mean route loss and pool selector.
It is explicitly marked non-independent. Shared arms wrap one native body with
four private factor rows; private-B uses four matrices rather than a common one.

The generated run covers all seven arms and three fixed seeds: 21 banks,
30 optimizer fits, 39 native bodies and 66 route trajectories per full roster
forward. Completion refuses any different bank/fit/body total. The finite owner
waits on one scientific child and requires the complete-family receipt plus
absence of its child/CUDA PID before declaring success.

## Checked state, gradients, restore and selection

All five recurrence steps use current live route distributions. The neighbor
index selection, index-add, degree scaling, B product and softmax remain in the
TRAIN autograd path. No detach, cross-route state averaging or human label value
is introduced into the decoder. Loss is .5 native CE plus .5 decoded CE averaged
across route/node pairs; there is no CE on the pooled distribution. Four routes
share parameters/gradients where declared, while their class states stay separate.

VALID evaluation uses model.eval() and no_grad(), then mean decoded probabilities
for the strict accuracy selector and patience counter. The chosen model state
includes B, native parameters and private factors. Strict model restoration,
optimizer state and saved member RNG streams precede the selected full-graph
forward. Native readouts are captured from that same decoded-selected model;
archives label decoded scores as raw_logits and store native_logits separately.
Native comparisons therefore include selection/stopping and do not isolate a
training causal effect.

The active spatial P is incoming D^-1(A_off+I): all factual off-diagonal record
multiplicity remains and exactly one identity record replaces factual self
records. The decision states this pre-outcome amendment to the scout's nonself
operator. Consequently the scout's isolate-native-for-all-B statement is not
an active guarantee: an isolate now has P=I and can undergo class refinement.
The explicit P=I arm provides the declared no-neighbor control. Native backbone
graph support is unchanged.

## Evidence and limits

The supplied qualification record reports seven actual full-graph SAGE TRAIN
cases, one update per case, finite gradients and zero-B identity, with about
1.38 GiB peak allocated memory. It scores no VALID quality and is not scientific
quality evidence. Its very small initial common-B off-TRAIN VJP is only the
reported plumbing check; it establishes neither material neighbor influence nor
accuracy benefit. This review did not rerun qualification or test FP32 parity.

CPGNN/GBPN/CRF-RNN/collective ensemble ancestry is stated honestly in the decision:
no new primitive, exact CRF/posterior, stability or novelty claim. AMCE's source
resolution remains in its separate sealed literature successor. The encountered
WikiCS screen, three-seed uncertainty, repeated method selection, competent
same-decoder controls and later unused confirmation limits remain explicit.

All 23 locally available non-data research-tree freeze entries match their stated
hashes. The remote repository's three top-level native-source hashes and role
archives were not independently opened or hashed by this source-only reviewer;
the frozen launcher checks them on the authorized host. Source syntax and the
exact unique AST replacements were checked using the standard library only.

Zero scientific/model/data execution, remote commands, role payload reads,
new/partial outcomes, TEST access or canonical state edits. Root retains launch
and complete-family interpretation custody.
