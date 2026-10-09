# Inactive source-supply helper

This is concrete reusable source for the saved persistent source-supply policy. It constructs no model, graph, dataset, optimizer, training loop, supervisor or checkpoint selector. Runtime operations default disabled and import Torch only after explicit enablement. No active/frozen source was edited.

Files:

- `source_supply.py`: complete parameter ownership checks; categorical and Bernoulli-marginal supervised objective; output cotangents and private-only replay; small competence-cone projection; one finite guarded private trial with rollback; explicit native full-input serving.
- `INTEGRATION_CONTRACT.md`: native typed-architecture/view/state/role requirements and integration sequence.
- `BERNOULLI_MARGINAL_AMENDMENT.md`: separately specified prospective mean-BCE amendment, with categorical modes retained.
- `PRIOR_AND_LIMITS.md`: strongest collisions and correctness/scientific limits.
- `verify_analytical.py` and `ANALYTICAL_VERIFICATION.json`: synthetic stdlib checks, with no Torch/model/data execution.
- Disabled template, input/scoping metadata and manifest/seal: source integrity and inactive status.

The meaningful analytical audit checks categorical and mixed positive/negative Bernoulli gradients by central differences, distinguishes marginal probability pooling from a joint-label likelihood mixture, checks absence of artificial cross-label normalization, verifies the anti-gaming guard and cone scale invariance, and checks complete shared/private/untied ownership with non-tensor fixtures. It does **not** qualify Torch autograd, native source views, replay, GPU memory, an architecture, baseline competence or predictive utility.

Run the narrow audit locally with `PYTHONDONTWRITEBYTECODE=1 python3 -S verify_analytical.py`. This runs only hardcoded synthetic algebra and ownership fixtures. There is no scientific execution entry point.

Typed HGB IMDB remains contingent on the separate source/schema/backbone work. Keyword availability, per-type features, native loss, eligible factor sites and competent contemporary architecture are not established by this helper. Root must complete coherent native integration/qualification and prospectively fix the representative pilot before any source-steering scientific run.
