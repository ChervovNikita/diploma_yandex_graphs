Private steering: complete public TRAIN/VALID interface
=====================================================

Source readiness only. This separate wrapper edits neither the sealed steering
adapter nor public V2. No numerical imports, model/data/checkpoint/score/TEST,
server or GPU operations were performed. Execution, runtime permission catalog
and costs remain unverified. No current family, strength, extra arm, campaign,
scientific novelty claim or deployment is adopted by this packet.

Identity flags source_preparation_adopts_execution=False and
prior_interface_runtime_qualification=False describe this packet's
prequalification status. They are not a denial of observations in a future
actual COMPLETE receipt or its charged counters. No completed fit currently
exists. Future run observations must be kept distinct from this static source
readiness/prequalification statement.

Callable
--------
run_complete(task, mode, base_arm, train, valid, output, seed=6101,
             device='cpu', lambda_value=None, polynormer=None, ncn_model=None,
             ncn_utils=None, public_root=None, adapter_root=None)

Tasks: wikics, collab, molhiv. Modes: alignment_private, hidden_private,
gncl_private. Base constructor: be_unit or be_init only. GNCL requires an
explicit finite lambda in [0,1], with no default/adopted strength. Reference
modes reject lambda. Public/adapter roots default to sibling sealed packets.
The callable handles one explicitly requested fresh output, never a campaign.
Calls must be serial per process because original train.main reads sys.argv.

CLI templates -- documented only, not executed
--------------------------------------------
python train_steering.py --task molhiv --mode alignment_private \
  --base-arm be_init --train TRAIN.npz --valid VALID.npz --output FRESH_OUTPUT

WikiCS additionally supplies --polynormer /path/to/pinned/model.py.
Collab additionally supplies --ncn-model /path/to/pinned/model.py and
--ncn-utils /path/to/pinned/utils.py.
For GNCL use --mode gncl_private --lambda-value EXPLICIT_DECLARED_SCALAR.
Optional --seed, --device, --public-root and --adapter-root remain explicit.
There are no epoch/batch/population/auxiliary-weight/selector overrides, TEST
arguments, resume, retry, family queue or automatic cross-task progression.
Parsing/help and import are stdlib only; this preparation did not execute them.

Unchanged full driver and data
-----------------------------
The wrapper hash-verifies the existing public V2 and steering manifests, then
loads private in-memory source modules. It constructs the original raw public
Session with the requested BE base arm before attaching PrivateSteeringAdapter,
so the adapter's original-forward/parameter permission audit sees the exact
source Session. A delegating SteeringSession retains all original model,
optimizer, streams, fields, serving and forward functions. Only train_step
uses the sealed adapter. No training loop, data loader, evaluator, local
transition or selection implementation is copied.

Original train.main runs complete source epochs and batches, with original
validation and joint snapshots: WikiCS1100 epochs (100 local then1000 global),
one update/epoch, TRAIN580 and VALID5274 on the complete fixed graph; Collab100
epochs,18 updates/epoch, all1179052 TRAIN events through2017,60084 VALID2018
positives plus all100000 authentic negatives including the one self-pair;
Molhiv100 epochs,258 updates/epoch, all32901 TRAIN and4113 VALID graphs and
original atom/bond fields/scaffold split. Public numeric input checks do not
invent official provenance or author-runtime equivalence. GNCL pool is the
sealed adapter's native served-pool objective, never a validation target.

Shared BE has four members, one physical body and one Adam. Eight full own
supervised member/view forwards are retained. Two autograd.grad collections
come from the same old state before one original Adam transition: shared and
boundary blocks get mean-own gradients, audited internal factors get the chosen
private objective. Both reverse passes are charged. There is no theta-first
update, second optimizer, new router/head, own-selection conversion or new
inference rule. WikiCS restores the original joint local model/Adam while
retaining live end-local RNG. Strict first maximum of the complete unrounded
VALID metric, all exposures and no early stop remain the original policy.

Identity and diagnostics
------------------------
The full driver receives a visible composite arm, e.g. be_init__gncl_private.
The base constructor remains separately identified; outputs cannot masquerade
as an unchanged be_init fit. Every RUN, selected/local selected snapshot,
COMPLETE and FAILURE includes private_steering_control, mode/lambda, source
manifest/program/interface digests, audited permissions where available and
the two-pass contract/actual counters. Source setup failures also preserve a
fresh labelled FAILURE; existing outputs are never overwritten. COMPLETE checks
all original steps plus2 reverse collections and8 member forwards per step.

The original driver requires result['auxiliary']; the bridge supplies
private_loss-own_mean. This is a diagnostic difference for the private block,
not an auxiliary gradient sent to the shared block or a common scalar objective
for all parameters. TRAIN trace/snapshots report own_mean, private_loss, actual
private supervised mixture mean, served-pool mean, alignment and residual
separately. GNCL mixture is (1-lambda)*own + lambda*pool before its alignment;
reference supervised mixture is own. No magnitude/sign of this diagnostic is
interpreted as a strength or scientific decision.

All original finite checks and failure/partial-output preservation remain.
Selected snapshots are source joint snapshots plus explicit method metadata;
exact resume remains unsupported and unannotated continuation methods refuse.
Static syntax/schema/source checks only establish integration readiness, not
gradient/runtime parity, real-data suitability, full costs or predictive merit.
