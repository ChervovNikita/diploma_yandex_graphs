# Restore-boundary repair

## Failure and diagnosis

The preserved actual v1 structured CPU attempt failed at `fresh.load_state`
after two constructions, 29 native forwards, four reverse collections, one Adam
transition and one complete VALID call. Its initial native parity, first
F/shared and W/private gradients, zero-up gradient pattern and BN update checks
passed. Fresh VALID restoration and the resumed update did not qualify.
`PARENT_FAILURE.json` binds the exact outcome, traceback, work and v1 manifest.
The original v1 folder, failed checkpoint and result are unchanged.

The unchanged parent `OGBSmokeSession.load_state` calls
`self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))` at
`reference_wrapper_redesign_20261013_v1/smoke_20261014/ogb_port/session.py:364`.
V1 subsequently requires `group is old` for every construction-time group dict.
The identity of these Python dictionaries is not a stable PyTorch restore
contract. A load may recreate them from saved groups while mapping saved
parameter slots onto the existing live Parameters. V1 would reject that valid
case even if all parameter ownership and option values were unchanged.

The failure receipt did not record separate before/after option/default maps,
so it cannot establish that dictionary identity was the *only* difference.
V2 checks those semantics exactly and adds separate native-qualifier reporting
instead of assuming or ignoring them.

## Minimal change

At construction v2 freezes:

- The exact live Parameter IDs, in order **within each group**. This also binds
  group count and group boundaries, strengthening v1's flattened owner check.
- Every group option key/value, excluding only the `params` list whose members
  are checked separately.
- Every optimizer default key/value.

Ordinary updates, state export and entry to restore still require the current
group dictionary objects and unchanged semantic contract. After the unchanged
parent restore, v2 compares against the same construction-time contract. Only
after that succeeds does it renew the local references to the current group
dictionaries. It never adopts new options, defaults or owners from a checkpoint
as its baseline. The final full identity check still covers exact native/private
Parameter objects, exhaustive disjoint partition, optimizer object, model object,
all native Parameters and private BN identity.

No PyTorch-added option fields are exempted. Own checkpoints produced/restored
by the same Adam implementation should already carry its constructor defaults.
Fields such as `foreach`, `fused`, `amsgrad`, `maximize`, `capturable` and
`differentiable` are checked like learning rate, betas, epsilon and weight decay.
A newly observed benign field difference would require its own explicit review;
this repair does not silently normalize one away.

The two producer IDs and outer STATE schema advance to v2. V1 checkpoints are
not reinterpreted as v2. Native initialization, rational weights/permutation,
parameter partition, four forwards, F/shared versus W/private, same-old-state
reverse order, original Adam and mean-logit serving are unchanged.

## Verification and remaining work

Ten stdlib checks pass: the original six source/table/partition/update checks
plus four focused restore checks. They demonstrate valid dictionary replacement
at the restore boundary, rejection outside that boundary, strict option/default
checks, exact group boundaries/order/live objects and restore-check ordering.
One test preserves flattened parameter order while moving a boundary and verifies
rejection. These are source/fake-scalar checks, not PyTorch or native execution.

The copied bounded qualifier changes only its version identity and reporting.
On an actual successful fresh restore it will separately disclose dictionary
replacement, live parameter/group-boundary preservation, full group-option
preservation and default preservation. Existing per-variant limits remain two
constructions, two updates, three complete VALID calls, 73 native forwards and
eight reverse collections including four diagnostic derivatives. Root will run
structured and then permuted in fresh outputs under its own CPU admission.

Both native qualifiers remain **unexecuted for v2**. This source repair is not a
scientific launch decision, quality result or permission to reuse an engineering
checkpoint for training. No remote command, Git operation, scientific job file,
new native run or change to v1 occurred during this repair.
