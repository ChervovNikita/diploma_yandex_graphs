# WikiCS qualifier v2 narrow repair

V1 failed before copy checks: native Polynormer registers a top-level learned `betas` parameter, and `name.split('.', 1)` could not unpack that undotted name. The earlier static review missed this fixture bug. The paid v1 failure and all v1 bytes remain preserved.

V2 changes one qualifier line to `prefix, _, suffix = name.partition('.')`. Undotted `betas` therefore follows the existing core branch and maps to `core.betas`. Boundary weight/bias mapping, exact parameter copy checks, all-parameter storage checks, native stages, dropout, gradients, Adam, fields, tolerances and budgets are unchanged. The constructor/core/RNG helper ASTs still equal the sealed trainer.

Seven pure string fixtures execute the actual repaired AST assignment/mapping expression, covering undotted beta, dotted nested beta and boundary weights/biases. This is the repair author's self review. No torch import, model, tensor, field generation, numerical gate, dataset/checkpoint/outcome read or launch was performed. Root must independently inspect the one-line delta before activating a new v2 job/output. No thresholds or caps were relaxed.

The disabled command/job now target fresh v2 paths; command metadata binds the same existing matched-reference run_fit/helper and root adapter. Trainer, weights, protocol and TRAIN projection remain pinned unchanged. Full fits remain unauthorized.
