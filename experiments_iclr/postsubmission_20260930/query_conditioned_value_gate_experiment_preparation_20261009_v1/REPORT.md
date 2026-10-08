# Prospective query-conditioned value gate — disabled preparation

9 October 2026. This is a distinct inactive hypothesis prompted by the fixed
one-visible-class direction of a bias-free linear label-value route. The running
P0 experiment, its sources, seals, targets and result scope are preserved.
No launch is admitted and no current outcome was read.

The candidate modulates each **aggregated 64D dense label message before its
10-class readout** by `2*sigmoid(A*stopgrad(H_i)+b)`. C4 shares one gate across
its four BE routes. A=b=0 gives the identity gate without a random draw. There is
no affine output shift, class-logit gate or native update. Empty label context
still gives zero route logits and the existing explicit native fallback.
Receiver-conditioned featurewise modulation is credited to GNN-FiLM
[1906.12192v5](https://arxiv.org/html/1906.12192v5) from the saved conclusion;
FiLM and BE components carry no novelty claim.

## Capability, parameter and optimizer inventory

Each gate adds `64*(512+1)=32,832` float32 learned scalars. All new gates start at
zero. Every arm uses the same frozen native state, mask draws, feature information,
permitted label context, fixed selection and 0.2/0.8 serving weights.

| New arm | Message/readout capability | Gate ownership | Total learned scalars | Fresh Adam owners |
|---|---|---|---:|---:|
| C4_gate | Four BE 64D messages and four bias-free 10-class readouts | One shared A,b | 109,160 | 1, all learned maps + shared gate |
| S_joint4head_gate | Four untied 64D messages; gated before concat; nonlinear bias-free256→256→10 joint readout | One shared A,b | 382,016 | 1, all heads/readout/gate |
| U4_sameB_full_untied_gate | Four fully untied 64D routes/readouts on the same B | Four independent A_m,b_m | 414,976 | 4, each complete route including its gate |
| S_one_path_gate | One 64D route and10-class readout; secondary only | One A,b | 103,744 | 1, all one-path maps + gate |
| C4_gate_identity_erased | Same C4 architecture; permanent class-independent anchor value input | One shared A,b | 109,160 nominal | 1, same as C4_gate |

The fully untied reference also unties the gate. Its extra capacity is explicit;
a shared-gate untied variant is not part of this first family. No optimizer owns
the same learned coordinate twice and no gate parameters are omitted. C4 and
its erased control accumulate four CE/4 gradients at old parameters before one
Adam step. U4 uses the original four CE/4 backwards before its four private Adam
steps. The joint and one-path singles each use one CE/backward/Adam. Adam settings
remain P0's lr 0.001, betas (0.9,0.999), eps 1e−8 and weight_decay 0.

Gradients reach the gate, label embedding/value/readout and feature-only Q/K
projection parameters; they do not reach detached H, native logits, B or any
native optimizer. The zero-initialized output/final readout makes gate gradients
zero on the initial forward/backward; later learning is conditional on the
readout receiving nonzero updates. This is a source/algebra statement, not a
measured training claim. The gate generator is shared within the candidate,
while U4's generators are private. There is no learned sharing across arms.

## Fixed prospective task and failure rule

All five banks run the original **1100 label updates** for each seed 6101/6203/6307,
requiring **15 complete bank endpoints**. Split 0 retains 11,701 nodes, 580 TRAIN labels,
10 classes and all 5,274 development nodes (val|stopping), without TEST truths.
The common Q is 290 of 580 with the original isolated mask stream and identical
positions/IDs/draw IDs across every arm. All Q labels are removed from every
label-derived context before propagation. The one-hop all-nonself-record
attention denominator and multiplicity/order, training scale 579/290 once,
serving scale 1 and zero-base direct label CE stay as P0. Each arm selects its
strict first maximum complete development correctcount over all 1100 updates.

The two co-primary comparisons are gated C4 versus the gated nonlinear joint
single and versus fully untied U4. Inherit P0's fixed complete-family gate:
mean accuracy gain at least 0.2 pp, positive in all three seeds, mean pool NLL delta
at most 0, and its mean/worst-member safeguards. The one-path arm is secondary.
For a **useful label-identity interpretation**, also require the same frozen gate
on C4 versus its permanent identity-erased control. Exact thresholds and all
fixed settings are in `EXPERIMENT_DESIGN.json`, declared before results.

Failure of either co-primary contrast fails this task's family utility hypothesis.
Failure against identity erasure leaves the label-identity interpretation
unsupported, even if the extra conditional classifier is useful. An incomplete
family or a mask/native/optimizer/zero-context violation is unqualified. There
is no seed, arm, epoch, subgroup, coefficient, initialization or horizon rescue.
The five arms alone do not isolate a gate causal effect versus ungated P0.

## Permanent value-identity control and its limit

For **every** training and serving call of C4_gate_identity_erased, each visible
anchor supplies the fixed uniform class code `u=1_10/10` to the original10 × 64
embedding matrix. This is its learned row mean, followed by the unchanged
bias-free BE value map and then the same query gate. The control preserves
matrix shapes and nominal parameter count, but only the 64D row mean is
identifiable through these value inputs: 576 embedding row-contrast coordinates
form an extra nullspace. Other parameter symmetries are not audited. This
contrast changes effective class-conditioned value capacity; a positive contrast
does not prove that every gain uses semantically correct neighbor labels.

Targets remain the original supervised TRAIN labels. Q membership and visibility,
anchor positions/mass, denominator, native TRAIN supervision/H and selector
truths are unchanged. The control erases message class identity only. Native H
still carries native TRAIN supervision; Q masking in message fields does not
make native predictions on TRAIN out of sample. It can expose a nonlinear H classifier
activated by anchor presence; exact zero output without context does not establish
useful label identity. A fixed global class permutation or a post-training value
swap would not implement this permanent control.

## Source readiness

`query_value_gate_hook.py` contains a small disabled factory and a value-only
lookup helper. `UNAPPLIED_CORE_PATCH.diff` and `PATCH_PLAN.md` specify the exact
insertion/wiring. C4's sealed core has no message callback, so an installed hook
would require a new numerical source release. Preparation stops at this concrete
patch design; it does not duplicate a trainer or patch the original core.
Only static parsing/source-hash checks are admitted here. Numerical identity,
gradient/optimizer, masking and serving qualification remain unperformed.
