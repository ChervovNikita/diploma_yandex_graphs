# Native recurrent class decoder source

This inactive packet supplies a stateless five-step decoder and a thin extension of the existing native `Family`. Root has authorized source preparation, with scientific execution awaiting the complete current retrieval-family decision. No outcomes were used to choose this recipe. The CLI refuses launch; root can later obtain the class through `family_class()` and construct it under the existing qualified lifecycle.

`decoder.py` takes live full-node native logits, caller-owned B and label-free graph support. `native_extension.py` reuses the pinned native model/factor constructors, route seed policy, safe TRAIN/VALID loading, progress, fit loop, strict decoded-accuracy selection, patience, complete checkpoint restoration, metrics, archives and full-roster completion. Its bounded AST edits change the training loss, selected native-logit capture, acquisition roster and accounting. There is no new model wrapper, process owner, launcher, retry loop or separate fit lifecycle. The untied bank uses a `ModuleList` parameter container and one optimizer.

## Fixed decoder and graph amendment

```text
q0 = softmax(Z)
repeat exactly five times:
    S = Z + (P q) B
    q = softmax(S)
loss = mean_routes(.5 CE(Z, TRAIN) + .5 CE(S5, TRAIN))
serve = mean_routes(q5)
```

B starts at zero, is registered before device movement, optimizer construction and parameter accounting, and is shared over all five updates. Gradients remain live through native logits, predicted class states, incoming messages and B. No truth label is a graph message value. TRAIN truth enters only the two losses; VALID truth enters only selection/metrics. Native graph blocks, heads and factor locations retain their original operation.

**Root's 10 October source instruction amends the sealed proposal's nonself mean to `P=D^-1(A_off+I)`.** A_off contains every factual off-diagonal directed record, in its original order and with duplicate multiplicity retained. For the decoder only, remove all factual self records and append exactly one identity record per node. D counts incoming records after that operation, so `Pq[v]` averages source distributions over records ending at v, including its one identity record. The native graph is unchanged. This explicitly replaces the older zero-empty-row/isolate claim: an isolate follows the P=I recurrence and may change after B learns. At B=0 all arms still reproduce native scores algebraically. No convergence, BP, CRF or quality guarantee is asserted.

The P=I arm performs the same five recurrent softmax/dense updates with no spatial gather. It controls the added neighbor computation while keeping recurrent class capacity. All class states are current predicted probabilities, including unlabeled graph nodes; this is ordinary full-graph transductive computation and supplies no extra raw information. The native C10 runner is unchanged. The helper also maps a binary native logit z to the equivalent `[0,z]` coordinates when used separately with C2 B.

## Seven prospective arms

| Arm | Native bodies and decoder ownership | Optimizer fits per seed |
|---|---|---:|
| Ordinary M1 graphP/commonB | One ordinary full native body; own B | 1 |
| Factorized M1 graphP/commonB | One full native body with all-map r/s coordinates; own B | 1 |
| Factorized genuine I4 graphP/ownB | Four separately initialized, trained and selected full factorized native+own-B models | 4 |
| Joint untied4 graphP/commonB | Four factorized full native bodies with untied native weights; one shared B and joint mean-route loss | 1 |
| Shared4 graphP/commonB | One shared full native body, four private factor rows; one B | 1 |
| Shared4 graphP/privateB4 | Same shared native body and private factors; four B_m | 1 |
| Shared4 P=I/commonB | Same shared native body and private factors; one B | 1 |

The joint untied control retains the same factorization and private-factor initialization opportunity as Shared4, while untying the native W/b and other native body parameters. It is a coupled joint bank, never genuine independent I4. Genuine I4 retains the capable factorized-M1 opportunity and each member's own decoder/optimizer/selection. Preserve the closed ordinary raw I4 archive as contextual reference; acquisition neither reads nor re-fits that archive. The new seven-arm comparison remains complete by itself.

One backbone with three paired seeds has **21 banks, 30 optimizer fits, 39 native bodies and 66 route trajectories per family-wide forward**. Joint untied4 contributes one fit and four bodies. Shared4 contributes one fit, one stored body and four route trajectories. A three-backbone rollout is not admitted. SAGE is the proposal's first representative; the inherited interface can also accept GCN/GAT through an explicit future root decision.

## Seeds loss optimizer and selection

The configuration supplies three distinct base seed IDs that root must freeze before execution. For route m:

`native_seed=base_seed+1,000,003*m`, `factor_seed=native_seed+2,000,003`, `dropout_seed=native_seed+3,000,007`.

Shared4 constructs one native body at route0's native seed. Genuine I4 and joint untied4 construct each route at its native seed. Every factorized body has a row-wise Rademacher input-stem r drawn under that route's factor seed; all other r/s start at one. The same persistent private dropout streams are used throughout training, evaluation and restoration. B's zero construction consumes no random draw.

All arms retain depth2, width128, dropout .2, learning rate .001, 1000 maximum updates and 300 stale-update patience. Native parameters, factors and B use one AdamW group per fit with zero weight decay and no decoder multiplier. Shared and joint banks average the two own-route losses over four routes; a genuine I4 member has its own unscaled M1 loss and optimizer. These optimization differences define the bank controls and remain explicit.

Joint banks select whole-role mean decoded-probability VALID accuracy. Genuine I4 members select their own decoded VALID accuracy. Strict improvement selects a later checkpoint; ties retain the earlier state. NLL is reported, never a new tie-breaker. Selected model, B, optimizer and route streams are restored strictly using the existing lifecycle. Every selected native metric is measured at that decoded-selected state, so contrasts include selection/stopping.

## Archives and counted work

The unchanged `selected_VALID.npz` fields `raw_logits`, `probability_mean`, `member_errors` and `pooled_errors` now refer to the final decoded scores and native float32 probability pool. Additional `native_logits`, `native_probability_mean`, `native_member_errors` and `native_pooled_errors` preserve the original native prediction at the **same** selected checkpoint. Both come from one native route forward. `RESULT.json` reports native and decoded member/pool metrics and arm semantics. Stable pooled NLL follows the existing FP64 log-softmax/logsumexp calculation; archived float32 decisions remain their own authority.

Native trajectories, graph-block calls, Adam/backward updates, parameter/checkpoint bytes, acquisition and serving/readout time, CUDA/RSS peaks retain the existing accounting. Each fit additionally records five decoder updates per route for every TRAIN, VALID-selection and selected VALID forward; spatial passes are zero for P=I. Decoder edge-class elements and dense class MACs are counted. Family completion aggregates those counts and asserts all21 banks/all30 fits/all39 bodies. Label-free decoder support preparation and storage are recorded once per family in `DECODER_GRAPH.json`; repeated fit metadata does not incur repeated preparation cost.

For C10, a common B has100 parameters and privateB4 has400. Zero initialization gives the native-gradient coefficient one at the initial step algebraically, but the learned recurrence may reinforce wrong rivals or overconfidence. No isolated speed or memory advantage is claimed. Actual graph record count, gradients, restoration, peak memory and runtime need root's bounded qualification before scientific admission.

The utility decision requires full paired final accuracy/NLL, native/decoded competence, common-rival repairs, harms and alternatives lost during pooling. Failure against capable same-decoder M1/I4, or explanation by the joint untied/commonB or P=I control, closes the corresponding sharing or neighbor hypothesis. CPGNN, GBPN, CRF-RNN and collective classification supply the established ancestry; this packet makes no novelty claim. A favorable screen would require competent source-native references and unused confirmation. Original scores, sealed proposal bytes and TEST custody remain unchanged.
