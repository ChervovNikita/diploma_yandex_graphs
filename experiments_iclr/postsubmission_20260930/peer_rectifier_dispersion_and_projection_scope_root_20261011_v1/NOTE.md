# What the nonlinear peer operation can retain

This note explains the frozen SX4/SM4 operations from the actual core. No pilot quality was opened when this note was written. It changes no model, coefficient, endpoint, metric or scientific criterion. The identities use known rectification, averaging and Euclidean projection properties; they are not a claim of a new primitive or improved prediction quality.

## The information discarded by averaging first

An attribute hub receives a hidden vector U_n from each donor member. The shared feedback map first multiplies it by B, applies ReLU, multiplies by V, and projects the result onto the unit Euclidean ball. The recipient averages the other three mapped donors. The control averages those donor vectors before applying the same map.

Before the feedback map's projection, put a_n=U_n B and let E denote the mean over the three donors. The two orders differ by

    E[ReLU(a_n)] V − ReLU(E[a_n]) V
      = (E[|a_n|] − |E[a_n]|) V / 2.

This follows componentwise from ReLU(a)=(a+|a|)/2. The quantity in parentheses is nonnegative in each coordinate. It is zero when all donor activations in that coordinate have the same sign. It can be positive when donors lie on different sides of the rectifier threshold. Thus averaging first can remove information about which donor directions activated, even when the donors' mean is unchanged.

The learned V can turn that nonnegative vector into either sign or any allowed output direction. The difference is not an intrinsically beneficial diversity signal. Nonlinear set aggregation and this rectifier identity are established operations; the identity does not certify a novel ensemble method or an expressivity hierarchy over complete GNNs.

## Both projections matter

The actual map is phi(U)=Pi(ReLU(UB)V), where Pi is projection onto the unit ball. Define c(U)=Pi(ReLU(UB)V)−ReLU(UB)V. Its exact mapped-donor versus mapped-mean difference is

    E[phi(U_n)] − phi(E[U_n])
      = (E[|a_n|] − |E[a_n]|) V / 2
        + E[c(U_n)] − c(E[U_n]).

The simpler rectifier expression is therefore exact only if the relevant map projections are inactive. The projection correction has no fixed sign. The outer projection after adding feedback creates another nonlinear interaction. Omitting either projection would misdescribe the executed core.

There is nevertheless a conditional bound for the actual operation. Every mapped donor has norm at most one. Donor averaging preserves that bound, and each row of D is a nonnegative average over the node's attributes (or zero for an empty row). Consequently the received node vector has norm at most one. The native first-layer hidden vector is already in the unit ball: its layer projects first and ReLU cannot increase its norm. At the fixed feedback scale alpha=.25,

    ||Pi(H_m + alpha D R_m) − H_m||_2 <= alpha = .25

for each node. This uses the nonexpansiveness of Euclidean projection and Pi(H_m)=H_m. At fixed hidden states, maps and graph inputs, the SX4 and SM4 outer hidden states differ by at most 2 alpha=.5 per node. These are perturbation bounds, not contraction of member disagreement or accuracy guarantees. Separately trained arms have different states and maps, so the bounds do not compare their final acquired predictors.

## Use with the complete experiment

The fixed roster already compares self feedback, the two operation orders, ordinary sharing, capable singles, genuine independent equipped models and coupled unshared models. A useful complete result must improve predictions under those controls. A larger rectifier dispersion alone would not establish that members learned better decisions. The same feedback can propagate a useful alternative or reinforce a common wrong class.

Keep this explanation outside the main paper unless the complete experiments justify discussing this mechanism. It can then support a short explanation of what members exchange, with the projection caveat preserved. It supplies no new threshold or outcome-dependent continuation.
