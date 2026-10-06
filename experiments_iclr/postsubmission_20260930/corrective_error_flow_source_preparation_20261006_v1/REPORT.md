# Account for repaired and newly introduced prediction errors

This disabled source implements the already frozen common400→endpoint error accounting. It changes no training arm, serving rule, population, threshold or selection procedure. It cannot read labels, load states, run a model or write output. A caller and independent source review are still required before use on scientific evidence.

The baseline and endpoint inputs are the evaluator's immutable native logits and served FP32 probabilities, aligned by the same node IDs. The program reports the four correct/wrong transitions, wrong-class churn, and their exact net accuracy identity. The baseline all-members-wrong pooled-error cohort is fixed before examining endpoint predictions. Class and degree slices use the existing classes0–4 and degree bands0,1–2,3–5,6–10,11–20,>20.

Paired NLL, Brier, true-class probability and confidence differences accompany the transitions. Confidence on baseline errors uses the same nodes at both endpoints, so a changed error population cannot create an apparent confidence benefit. Stable NLL uses float64 arithmetic on the saved native logits; classification and Brier use the saved served probabilities. Raw member argmax determines the all-members-wrong cohort, with the existing first-class tie convention. The caller must compare aggregate accuracy/NLL/Brier against the original evaluator scores before accepting output.

The output is descriptive. Graph nodes are dependent; the code adds no independent-node significance test, acceptance gate or causal claim. No A, VALID or TEST data has been supplied, and scientific scoring remains closed. The source remains disabled. Synthetic accounting checks are arithmetic verification, not a dataset experiment.

Decision reference: `amazon_error_analysis_corrective_method_decision_note_20261006_v1/REPORT.md` (SHA256 `cbb404eb2a752e68435358934deae0a10af3b7e8e67189e3bac875bb1b70c43f`).

The six-node arithmetic check passed, including exact repair/new-error/churn counts, the net accuracy identity, fixed cohort confidence, ID-order rejection, probability underflow and empty slices. `SYNTHETIC_ACCOUNTING_RESULT.json` SHA256 `edec9ba903b6e7f117a47d4fe367a8c3ee6e6a4dfaf4bbf55add0317f256c70a`. This is a synthetic source check; no scientific outcome or source release.
