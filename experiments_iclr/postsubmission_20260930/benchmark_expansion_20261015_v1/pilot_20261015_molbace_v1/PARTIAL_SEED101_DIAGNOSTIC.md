# First completed molecular seed: information, not promotion

These are four completed fits from the frozen seed101 MolBACE/GIN development
cohort. Each used149 complete TRAIN epochs, the agreed minimum100/cap1000/
patience50 controller, native data and own whole-VALID AUC selection. The
official VALID set has151 graphs,130positive and21negative. TEST remains closed.

| Method | Selected epoch | VALID AUC | VALID BCE |
| --- | ---: | ---: | ---: |
| Native single/body0 | 98 | 0.73846 | 0.92069 |
| Base GNNM | 2 | 0.62784 | 0.49848 |
| Graph alignment + variance | 4 | 0.71062 | 0.55421 |
| Private nonlinear input correction | 38 | 0.67729 | 0.54666 |

The graph objective improves selected AUC over base GNNM by0.08278 in this
seed, and the private correction by0.04945. Both remain below the native
single and both have higher BCE than base GNNM. AUC and BCE measure different
properties; neither tradeoff is hidden. The graph objective's mean member AUC
also increases, but a weak, oppositely biased route persists. The earlier
near-complete member-correct union was not evidence of a usable oracle gate.

The combined ingredient fit is still running at the saved observation. The
other two planned seeds and three remaining independent native bodies for this
seed are not yet complete. There is no completed I4 comparison or estimable
three-seed interaction, uncertainty, method promotion or novelty conclusion.
Different selected epochs also prevent a causal same-weight geometry claim.

The curve diagnostic preserves every completed epoch. A dot identifies each
method's own AUC-selected checkpoint; it is not a selected BCE minimum. The
logged `convergence_established` field means the frozen validation-patience rule
was met. It does not prove that training gradients vanish, that all future
epochs would fail to improve, or that the study has solved undertraining.
TRAIN mean update losses include different auxiliary criteria, so they are not
plotted as a common native-risk comparison.

OBSERVATION_12.json pins the four results and raw server observation time.
The source `first_seed_curves.py` creates only a compact CSV, SVG and source
receipts on the server. It constructs no model, changes no selection and runs
zero optimizer steps. This diagnostic does not alter the frozen pilot.
