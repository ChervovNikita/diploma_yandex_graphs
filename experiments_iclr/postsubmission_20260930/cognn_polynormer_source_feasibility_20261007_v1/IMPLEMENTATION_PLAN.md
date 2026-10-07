# A gradient-preserving Polynormer graft

This is a proposed isolated source implementation contract, **not implemented or released**. It refines the earlier unadopted sketch: do not renormalize surviving attention messages. Source evidence is the pinned native Polynormer forward and CoGNN's zero-weight-preserving message aggregation.

## Exact location and semantics

Retain the native full WikiCS input/support,512-wide seven-local/two-global architecture, all original dense/attention parameters, private factors, local skip accumulator, nonlinearities, dropout, beta mixtures and local/global transition. Prepared WikiCS support already includes one self-loop/node; native `GATConv(add_self_loops=False,bias=False)` consumes those loops and adds a separate `lins[i](x)` own-state route.

For each local layer and member, compute native attention coefficients α on the **unchanged original support**. Private receive/send policies observe the current member state on that support and produce ST binary keep values r_v and b_u. Multiply external message values after the native attention softmax:

```text
gate(u→v) = 1                       if u=v
            receive(v)*send(u)      otherwise
message(u→v) = alpha(u→v) * gate(u→v) * projected_state(u)
```

Sum these messages and add the original native `lins[i](x)` route. Continue the existing forward exactly. The action policies are reused across local depth; the global attention module consumes the resulting native local accumulator. No cross-member hidden mixing, new edge support or teacher is introduced.

**Do not** filter edge indices with booleans, convert gates to detached probabilities, use `log(gate)` at hard zero, or softmax again after multiplication. Such changes can lose policy gradients, create infinities, or change the declared operator. The static self-loop predicate may choose constant gate1; external gate products remain live tensors. This is weighted native attention with attenuated message mass, **not conditional normalization over surviving neighbors and not an exact native CoGNN/GAT reproduction**.

For an external sum `m_v=r_v Σ_u α_vu b_u V_u`, its direct receive derivative is `Σ_u α_vu b_u V_u`. A recipient factor inside a normalized row could cancel from numerator and denominator; this construction has no such denominator. It still has legitimate zero-gradient cases: no external edges, zero messages, every sender off, recipient off for sender credit, downstream nullspaces or normalization compensation. A product of two hard-off bits gives no direct edge gradient. Nonzero ST gradients and useful learning require qualification, not a blanket guarantee.

## Parameter, RNG and serving ownership

- Keep common native parameters θ, existing factor tensors φ, and private policy/temperature parameters Π exhaustive and disjoint. Policies live outside the factor-installation traversal, so they are not accidentally converted into tied BE maps. Preserve native parameter objects, constructor draws and reset order; initialize policies using isolated RNG state.
- A small source-derived starting module is one MeanGNN layer per binary policy plus a linear learned temperature. At one layer `act_dim` does not add capacity. The exact policy initialization/temperature choice must be frozen separately; near-standard biases or policy warmup are not silently assumed to preserve competence.
- Training uses independent policy noise streams per member/view/layer, separate from native dropout. Preserve the native live RNG at local/global restoration, and save policy generators with checkpoint and optimizer states. Existing parameter moments cannot be selectively reused after an architecture change.
- For a reproducible research readout, propose one predeclared fixed Gumbel noise panel per member/layer at serving, reused at every selection/replay call. This retains hard stochastic-policy mechanics conditional on a fixed label-blind draw, but **differs from the author's fresh eval draws**. Generate it independently before fitting; no draw selection, hidden Monte Carlo averaging, or argmax substitution. All credit conditions use the same policy-serving rule. Final source admission must bind this choice explicitly.
- All θ/φ/Π remain live. The inherited conditional controls keep the same model and own-label opportunity, changing only whether the established member/pool risk supplies gradients to Π or φ. No revised strength, epoch, seed or task grid is adopted here.

## Required source/numerical qualification before a fit

1. **All-keep identity:** force every external gate to1 and compare complete native versus adapted outputs, loss gradients for all original θ/φ, actual Adam displacement/state and both local/global modes under identical tensors/RNG. Retain original loop ordering and parameter identities.
2. **Action and support:** send-only, receive-only, isolate and standard fixtures must select the stated directed paths; all external-off leaves original self/root contributions. Empty rows remain finite. Tests must cover prepared loops and duplicate/coalesced-edge conventions.
3. **Policy backward:** on a small nondegenerate directed fixture, verify receive and send logits have the expected finite nonzero ST surrogate gradients in both hard-on and hard-off cases where a path is available. Compare against the declared soft surrogate with the same Gumbel draw; do not finite-difference the discontinuous hard forward and call it an ST failure.
4. **Multi-member graph retention:** two views and all four members must keep distinct live gate tensors across forward/backward; overwriting a module cache must not route a previous graph to another member. Loss routing must affect only declared parameter blocks while retaining gradient paths through sampled actions and current states.
5. **Custody and resources:** TEST-free target/role admission, exact selected-state replay, policy RNG/restoration, finiteness after updates and measured full-graph local/global two-view cost are required. No such check has been run in this packet.

The fetched PyG2.7 GAT source matches the adopted runtime-file SHA `a7b2353003394ab433f909c0dbd7e5a06b7ae1289939857245ae3062f1eb5901`. Its forward invokes `edge_updater` to form α, then `propagate`; `edge_update` applies softmax/dropout, and `message` multiplies α by projected source values. The integration site is that message-value operation. Retain native score construction/softmax and parameter objects; stock GAT has no declared `edge_weight` input that implements this plan. An isolated subclass/explicit weighted message path must still qualify PyG's signature inspection, gate tensor lifetime and backward; the author's PyG2.3 mean aggregator is not a runtime substitute.

## Evidence-selection versus attenuation

The operator changes both which sources contribute and total external message mass. A graph-evidence interpretation needs more than action disagreement. In a frozen diagnostic, compare the learned external mask with uniform external attenuation that retains each row/head's same total native-attention mass but leaves relative neighbor weighting unchanged. Freeze policies, states and the diagnostic roster before labels score it. If uniform attenuation explains the margin/repair benefit, attribute it to generic message damping rather than complementary source selection. This is a diagnostic obligation, not another fitting grid or a claim of semantic/causal intervention.

The candidate must still repair shared wrong competitors by changing member predictions; gating cannot rescue a convex pool whose member rankings remain jointly wrong. Native attention/private nonlinear states may already contain the useful evidence directions. CoGNN, attention, graph message gating, BE and GNCL remain credited ancestry; a successful implementation supplies no novelty or performance claim.
