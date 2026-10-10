# Complete SAGE family: interpretation that changes the next decision

Prepared prospectively on 2026-10-10. No current output, logits, errors or checkpoint was opened. The training source remains frozen. Run this analysis only after the complete 21-group/39-acquisition-unit roster closes, using its selected VALID payloads and root's frozen criteria. This plan defines diagnostics, not replacement acceptance thresholds.

| Symbol | Arm | Main question |
|---|---|---|
| O1 / O4 | ordinary M1 / genuinely independent I4 | Can the ordinary model and independently selected members predict competently? |
| F1 / F4 | all-map factorized M1 / genuinely independent I4 | Does factorization explain the benefit before sharing or exchange? |
| S | unchanged shared4 | What does shared storage lose against capable independent members? |
| D | equal-size separable | Does a connector help through route-wise computation? |
| X | exchange | Does the exchange condition improve useful predictions over D and S? |

Use each member's own selected state in O4/F4 and the selected bank state in S/D/X. Preserve recorded selection accuracy, selected steps and restored readout separately. Use saved original float32 logits for float32 probability-mean decisions and float64 `log_softmax`/`logsumexp` NLL. Compare identical ordered VALID IDs and labels across all arms. Do not select a different checkpoint, averaging rule or best member after seeing these outputs.

## 1. Start with capable references and paired uncertainty

For every seed report pooled accuracy/NLL, all four member accuracy/NLL values, mean/worst member performance, per-class accuracy/NLL, acquisition cost, inference parameter bytes and observed readout cost. Report O1/F1 as their actual singles; a validation-selected best member is only a diagnostic and cannot replace a frozen reference.

Primary paired deltas are X−D, X−S, X−F4, S−F4, F1−O1 and F4−O4. For accuracy use percentage points; for NLL preserve the sign so a negative change means improvement. Show all three seed deltas, their mean, sample SD and min/max. A descriptive 95% paired t interval is `mean ± 4.303*SD/sqrt(3)` under its distributional assumptions. With only three seed pairs, even three same-sign effects have minimum exact sign-flip p-values 0.125 one-sided and 0.25 two-sided. Do not turn repeated node resampling or a large bootstrap count into additional training replicates.

These seeds estimate optimization variation on one encountered development graph/split, after selection on its VALID labels. They do not estimate uncertainty across datasets, splits or unseen confirmation. Keep root's frozen accuracy, NLL, member and class protection rules unchanged. A wide interval is uncertainty, not permission to weaken a failed rule.

## 2. Distinguish member weakness, coverage and pooling loss exactly

For arm a and VALID node i define member-correct flags `c[a,m,i]`, any-member coverage `V[a,i]=any_m c[a,m,i]`, and probability-pool correctness `P[a,i]`. Compute:

- unavailable alternatives: `~V`;
- available alternatives lost during pooling: `L=V & ~P`;
- pooled correct without a top-1-correct member: `U=~V & P`;
- served member alternatives: `V & P`.

The exact identity is `#P = #V − #L + #U`. Therefore `Δ#P = Δ#V − Δ#L + Δ#U`. **Include U:** averaging can choose the true class even when every member has a different wrong top-1 class. Omitting U can misattribute an improvement to acquired or unlocked member predictions.

For X/D versus S, and S versus F4, tabulate the complete 4×4 node-state transition table (`V&P`, `V&~P`, `~V&P`, `~V&~P`). Also count newly acquired coverage, lost coverage, new alternatives served and new alternatives still lost. This catches the earlier failure mode where pooling loss decreased only because competent alternatives disappeared.

Interpretation changes the next action:

- Low member accuracy and low V: acquisition remains the problem; a reduction in L is not a remedy.
- Competent members/high V with large L: correct alternatives exist but the frozen serving rule loses them.
- Increased P mainly through U: aggregation supplies useful combined evidence; describe this contribution explicitly rather than calling it improved member competence.

Coverage is an oracle diagnostic using known VALID labels. It is not an achievable predictor, a training target or a reported accuracy reference.

## 3. Explain repairs and harms rather than only net accuracy

Relative to reference b, define repairs `R[a|b]=~P[b]&P[a]` and harms `H[a|b]=P[b]&~P[a]`. Report counts and IDs by seed/class against S, D and F4. For X repairs over S, use three disjoint causes:

1. `V[X]&~V[S]`: a new correct member alternative was acquired and served;
2. `V[X]&V[S]`: an already available alternative became served;
3. `~V[X]`: the pool became correct without any correct member top-1.

For harms, distinguish X retaining/acquiring an alternative but losing it in pooling (`V[X]`), removing S's available alternatives (`~V[X]&V[S]`), and losing an aggregation-only answer (`~V[X]&~V[S]`). Record true-class pooled probability and strongest false-rival identity on these exact cohorts, with margins summarized by class. This asks whether exchange fixes common false rivals or creates confident new errors; it does not infer hidden routing causality from outputs.

Compare D and X repair-set overlap, D-only repairs and X-only repairs against S, plus whether each set survives against F4. Repeat for harm sets. Track the same-ID intersections across all three seeds and report per-seed sets as well; persistent development-node repairs are evidence about this graph, not unused confirmation. Useful exclusive repairs can exist alongside negative total quality. They cannot waive member/class/NLL protection or rescue a worse final predictor.

## 4. Estimate the interaction that the actual roster supports

The genuine O1/O4/F1/F4 quartet supports a factorization × independent-ensemble interaction:

`I(Q) = (Q[F4] − Q[O4]) − (Q[F1] − Q[O1])`,

where Q is accuracy or −NLL. Report this quantity per seed and with the same paired uncertainty summary. If F1 explains essentially all of F4's gain, do not label it an ensemble-specific accuracy benefit. A positive interaction still needs better final quality and competent members; it cannot rescue F4 losing to a capable single.

X−D is the prospective exchange-condition comparison at equal connector parameter count; X−S measures its total addition. The attention and route-wise MLP have different function classes, so their difference alone does not identify a unique causal communication pathway. D and X are alternative complete conditions, not two ingredients of an observed A+B factorial. Their exclusive repairs can motivate a future interaction hypothesis, but there is no measured D×X interaction in this roster. Likewise, there is no measured exchange-without-factorization arm.

## 5. One bounded decision after closure

The final report should end with one disposition and at most three concrete next actions:

- If X meets the frozen quality/protection requirements against S/D and the capable references, identify which exact repair category drives the gain and take matched references plus unused confirmation next. Extra exploratory ablations have lower priority.
- If X fails but reveals competent, persistent alternatives lost during pooling, retain the measured serving diagnosis and request one distinct mechanism/prior-method review targeted to that cohort before specifying another study. Do not repackage a closed diversity loss or frozen-logit processing family.
- If member/coverage loss or stronger F1/F4 explains the result, close this exchange recipe within its measured SAGE/development scope. Do not claim that all shared GNNs or all cross-route mechanisms are ineffective.

A future A/B/A+B proposal requires two separately measured, competence-preserving ingredients, a mechanistic reason their exclusive repairs should survive jointly, matched capable references, explicit expected harms and frozen costs. Complementary error sets are a hypothesis lead, not admission of an unmeasured combination. No partial family or TEST result chooses a continuation.
