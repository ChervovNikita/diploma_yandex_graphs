# Reporting-only proposal for a future production generation

The frozen v3 qualification passed, with one declared reporting caveat. A
controlled exception after durable epoch1 PROGRESS occurs before the current
in-memory RESULT summary update. The failed invocation reports
`training_epochs_this_invocation=0` while selector/completed_epochs, recovery,
CURVE and optimizer deltas correctly record one full epoch and38 steps.

`REPORTING_FIX.diff.txt` moves the existing four-field `result.update` block
immediately after `selector.observe`/`observe_validation` returns. This writes
the summary as soon as the selector epoch commits, before checkpoint/curve/
progress I/O can raise. The final failure receipt then retains the correct
invocation epoch count. No update arguments or computation change.

This is a proposal for incorporation into the eventual production generation.
The patch is not applied to frozen runtime v3 or v2. No runtime binding,
pilot/launcher, source model, optimizer, loss, policy, selector, checkpoint,
RNG or resume behavior is changed here. No additional training was launched.

Static checks confirm the exact original update statement was moved and the
entire remaining AST is identical; the proposed file compiles. Patch context
must be reviewed against the eventual production source before applying.
The v3 checkpoint/runtime/evidence hashes remain frozen.

The change addresses this completed-selector-epoch summary timing. Work before
selector commit or after the last durable checkpoint must still be charged
from optimizer deltas and retained epoch/work receipts, including failed runs.
