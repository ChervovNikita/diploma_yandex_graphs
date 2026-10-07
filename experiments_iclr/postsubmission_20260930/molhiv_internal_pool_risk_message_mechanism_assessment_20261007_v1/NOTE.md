# MolHIV internal pool risk: one conditional message mechanism

7 October 2026. Source/closest-prior assessment only. No current quality, TEST, models, data, server or scientific runtime access. The implementation agent's wrapper for the exact 18 fits and pending Wiki24 family are outside this assessment and remain unchanged.

## Finding

As a learning principle, **I uses established GNCL risk with prescribed block gradients**. Shared/private ensembles, channel modulation and block-dependent supervision also have direct prior. No new loss, graph operator or optimization theorem is established.

A defensible conditional method contribution remains possible: **pool-risk credit assigned to existing post-normalization, pre-bond-activation member factors may produce useful atom–bond message differences while shared features and prediction boundaries retain own supervision.** This is a graph-computation hypothesis grounded in the actual source. It requires served ROC AUC, competent members, message-specific attribution and ranking repair; a positive implementation check or a generic I win alone does not establish it. There is no acceptance or exhaustive-novelty verdict.

## 1. What the model and objective actually do

The public source is a bond-aware residual GINE/virtual-node adaptation with categorical AtomEncoder/BondEncoder, five layers, shared LayerNorm and a scalar graph head. The code is not a qualified reproduction of an OGB leaderboard recipe. MolHIV's official metric is ROC AUC on its scaffold split. This method's declared pool is **mean raw logits**; its TRAIN loss is BCE, not an AUC objective.

For member m and layer l, the relevant computation is

```
u_m,v = LayerNorm_l(h_m,v + virtual_m,graph(v))
x_m,v = c_m,l ⊙ u_m,v
GINE input aggregate:
  a_m,v = (1+eps_l)x_m,v
          + sum_{j→v} ReLU(x_m,j + e_l,jv)
update_m,v = MLP_m,l(a_m,v)
h'_m,v = h_m,v + dropout(ReLU(update_m,v)).
```

This uses standard GINE semantics invoked by the source; dependency/runtime equivalence is not audited here. **The same c affects GINE's self term and neighbor terms. It is not a pure edge gate or learned edge probability.** Bond embeddings e and eps are shared. Virtual-node states differ because member computations differ, while their base parameters are shared. Graph representation is the mean of output-normalized node states.

The private internal block includes c **plus** fast factors in GINE and virtual-node MLPs. Input and scalar-head fast factors are private boundaries. Shared atom/bond embeddings, dense matrices/biases, norm parameters, virtual base and eps receive mean-own supervised gradients. In I, internal factors receive the existing fixed `.5 L_own + .5 L_pool` supervised mixture plus the unchanged phi-only alignment. No independent teachers, new router or extra branch is added.

For one binary object, the internal logit cotangent is

`[(sigmoid(z_m)−y)/2 + (sigmoid(mean_j z_j)−y)/2]/M`.

All members receive the same pool residual through different Jacobians. Wiki probability responsibilities do not apply. The saved exact gap is `D=mean softplus(z_m)−softplus(mean z_m)`; the internal mixture retains half its derivative, whereas own-supervised shared/boundary coordinates retain the full own-risk gap derivative. That is established algebra and allocation ancestry. Neither I nor the matched own policy with phi-only alignment should generally be called the gradient of one global potential; saved separability/optimizer qualifications remain.

Own risk and the retained individual component provide training pressure toward competent members. They do not preserve member functions, ensure monotone loss steps or guarantee AUC. At exact identical deterministic routes, the gap gradient is zero; existing initialization/streams supply possible asymmetry rather than a specialization guarantee.

## 2. Why this particular factor can affect graph evidence

c acts **after** this LayerNorm and **before** addition of the shared bond embedding and ReLU. Its relative scaling of node channels against bond channels can change which incoming messages survive. Later LayerNorm can suppress some amplitude effects, but does not universally remove altered activation patterns. With no relevant bond term and positive c, scaling can commute with ReLU/sum and be absorbed into a downstream input factor; that degenerate case cannot establish a graph-specific advantage.

A source-level local witness shows the nonlinear distinction. In one channel, consider equal-self-term contexts:

| Context | Normalized neighbor u | Shared bond e | At c=1 | At c=2 |
|---|---:|---:|---:|---:|
| A | 1 | −.5 | ReLU(.5)=.5 | ReLU(1.5)=1.5 |
| B | −1 | 1.5 | ReLU(.5)=.5 | ReLU(−.5)=0 |

The original aggregated channel is equal, but member scaling before bond addition separates the contexts. Multiplying the already aggregated `.5` cannot produce these two outcomes. Other channels can have equal/suppressed contributions and the receiver's first normalized channel can be zero. LayerNorm can produce these signed normalized states from nonnegative pre-norm hidden states. This is a local non-commutation witness, not a whole-network expressivity separation, a trained model or evidence of real HIV-relevant substructures. Other layers may recover or erase the difference.

Thus a **possible** mechanism is different member admission of shared atom–bond contexts before irreversible aggregation. Internal pool gradients can train that mechanism; there is no explicit objective requiring it, no member-to-chemotype assignment and no biological causal interpretation.

## 3. Closest priors already bound

- **GNCL `2011.02952v2` Eq.5 and pinned author implementation:** the member/pool risk is established; the author code supplies one global mixture backward. I's own-core/boundary restriction is the operational allocation to test against all-private P and global-supervision G.
- **BatchEnsemble `2002.06715v2`, TabM `2410.24210v3`, Kim graph-BE thesis:** shared matrices, private factors and graph member trajectories are prior. Internal factorization itself is not the contribution.
- **GNN-FiLM `1906.12192v5`, saved method blocks 10–38 and Eq7/8 fragments:** receiver/relation-conditioned featurewise affine incoming-message modulation, with nonlinearity-placement variants, is direct graph-message ancestry. Current c is a static member/layer/channel scale rather than that dynamic receiver generator; bond-conditioned effects arise through u+e. No new modulation primitive is claimed.
- **PCL `2006.04147v2` complete saved method, Ensemble++ `2407.13195v6`, and GAR `2609.36724v1`:** differential gradient exposure with shared/live features is prior. Their fusion/KD/uncertainty/router targets and recipients differ from supervised internal GINE pool risk.
- **CDLG, DICE, SuGAr and FoRDE saved scopes:** channel/conditional/evidence/sensitivity diversity has prior. I adds neither hidden redundancy estimation, independently learned edge masks nor sensitivity repulsion. A bond-response diagnostic does not create a new diversity principle.

The old project reverse own/pool policy and nonconservative-field assessment are reused rather than rediscovered. GNN-FiLM was already method-read, so no genuinely missing primary section was needed. New primary retrieval/method/whole-paper/code/performance credits are zero.

## 4. One falsifiable hypothesis and minimal attribution

**Hypothesis:** compared with the same model whose existing message factors retain own-risk supervision, I's pool-risk exposure on those factors produces net repair of common positive–negative ordering errors, supported by changed bond/neighbor responses beyond global score recalibration, while member quality remains competitive.

First, the frozen allocation study can test whether I has useful served quality. Known GNCL P/G and matched own O are necessary before claiming the overall restriction matters. **O/I/P/G do not isolate message factors:** other private MLPs and the virtual-node path change too. Do not assume an unlisted cell exists in the wrapper for the exact 18 fits, or append any arm to it.

Only if the completed fixed result warrants the mechanism question, freeze a **separate two-condition attribution comparison** on the same declared task/constructor/paired blocks:

1. The existing I policy.
2. The identical model/update, with **only the supervised objective for c changed back to own risk**; other internal factors retain J, shared/boundary factors remain own, and phi-only alignment retains identical permissions including c.

Every parameter, factor initialization, graph/labels, views, optimizer/state, fixed coefficient, horizon, pool and selector is otherwise matched. This changes loss exposure of one existing graph-computation block; it does not freeze/remove capacity. It is one conditional comparison, not a factor/location/strength grid and not an adopted run. Establishing necessity of pre-aggregation placement would require a further placement control; this note makes no such necessity claim.

Retain capable same-source single, genuinely independently acquired four and packed untied references for quality/sharing context, with normalization, views and selection differences disclosed. Independent models are references, never compression teachers. Charge complete member graph work and reverse collections; shared parameter storage is not a training-speed claim.

## 5. Ranking and evidence readout; null interpretations

For a positive molecule a and negative b, let `d_m=z_m(a)−z_m(b)`. If every d_m is strictly negative, the mean-logit pool also reverses that pair. Fix the common-reversal cohort from matched O; I must change predictions to repair it. Report full-population AUC contributions, not only this favorable cohort: each pair contributes 0, .5 or 1 according to its pooled margin, and net gain is the mean contribution change including ties and newly introduced reversals. AUC uses all positive–negative ordering, not a fixed binary threshold.

A common global score shift changes BCE but leaves AUC unchanged. Positive per-member affine rescaling preserves each member's ranking and can still change the raw-mean pool by changing relative weights. Therefore score variance or pooled AUC alone does not prove new graph evidence. Record whether globally affine score changes explain the difference, and use one fixed label-blind bond/neighbor-response panel scored by relative positive–negative margins. Such probes measure model sensitivity; chemically altered bonds are not automatically valid molecules or label-preserving causal interventions. Probe selection cannot use favorable errors. No new perturbation or training objective is adopted here.

- If I lacks served utility beyond the required fixed references, this graph-method explanation is unsupported for the representative task.
- If P/G match I, the internal/core/boundary allocation advantage is unestablished.
- If message-own matches I, pool supervision of c is not the demonstrated mechanism; generic private MLP/virtual-node optimization remains plausible.
- If the gain is explained by score rescaling, unchanged ranking/evidence responses or confidence alone, narrow it to that result.
- If common-pair repairs are canceled elsewhere, or member competence collapses, useful complementary repair is unestablished.

Gating differences with no pooled gain are not success. Conversely, an I gain can support an attributed training recipe without establishing a new graph primitive. Same split/paired optimizer seeds remain development evidence; chemically related molecules/scaffolds and VALID selection do not become independent dataset replications. TEST remains closed.

## Disposition

A conditional methodological contribution could be **an empirically justified supervision allocation in a compact bond-aware shared ensemble**, with message-specific and strongest-prior attribution. The loss and block optimization are known ingredients; the concrete graph effect above is possible but currently unmeasured. No novelty certificate, acceptance verdict, source edit, wrapper change or new experiment admission is supplied.
