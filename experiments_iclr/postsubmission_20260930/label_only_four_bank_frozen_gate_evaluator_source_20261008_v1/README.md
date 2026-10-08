# Disabled frozen four-bank gate evaluator

Source only; no scores opened. This helper binds the exact frozen protocol and
the sealed v1 post-family collector source. Root handles family opening and
collection. It requires an exact enabled source-reviewed release, successful
complete collector and cost-receipt hashes, the whole-family closure hash and
the exact `SAME_STATE_CORRECTION.csv` hash. It accepts all36 rows:3seeds,
4arms and3cohorts. Twelve selected-state identities, full5274 whole supports and
single-predictor cardinality for both singles are checked. Incomplete custody
produces failure, never a partial pass.

For **each** C4−S_joint4head and C4−U4_sharedB comparison, all seven conditions
are required:

| Component | Frozen condition |
|---|---|
| Served accuracy signs | C4 gain strictly positive in all3seeds |
| Mean served accuracy gain | ≥0.2percentage points |
| Mean served NLL difference | ≤0nats |
| Mean member accuracy difference | ≥−0.1percentage points |
| Worst member accuracy difference | ≥−0.2percentage points |
| Mean member NLL difference | ≤0.01nats |
| Worst member NLL difference | ≤0.02nats |

Every mean weights the three optimizer seeds equally. Member means/minimum
accuracies and means/maximum NLLs are computed within each seed first. The joint
single's one predictor defines its mean and worst statistics. All14 components
must pass. S_one_path, Brier, cohorts and intervals cannot rescue a failure.
All differences are C4−reference; positive accuracy and negative risk favor C4.

Accuracy gates derive from integer correctcounts. NLL gates compare exact
rational representations of the recorded binary floating-point values against
the original decimal thresholds. No rounding or comparison tolerance changes
the decision. Reports retain original per-seed values, supports, selected-state
identities, all member values, exact means and every failed component.

For seven paired whole-population effects, each co-primary report also includes
the descriptive two-sided95% t interval `mean ± t(0.975,df2)×SD/√3`, with
`t=4.302652729911275`. The units are three paired optimizer-seed differences,
not independent nodes or graphs. Small-sample distribution assumptions are not
established; intervals supply no significance, confirmation or promotion claim
and never enter the frozen gate.

`run(release_path=..., release_sha256=..., later_execution_authorized=False)` and
the CLI refuse before input access by default. Root fills the nine variable
fields in `ROOT_RELEASE_TEMPLATE_DISABLED.json`; fixed gate scope and interval
settings cannot be changed. Inputs may be root's compact local copies or the
allocation collector directory, with exact hashes. Raw logits, probabilities,
training data and checkpoints are never opened by this helper.

Fresh output: `GATE_RESULT.json`, then `TERMINAL.json`; errors retain
`FAILURE.json` and terminal CPU/RSS/wall costs. A passed exploratory gate does
not authorize promotion. There is no retry, training, gate relaxation or server
operation. Preparation used AST/compile/JSON and source metadata bindings only;
no evaluator execution, numerical fixtures, outcomes or data were accessed.
