# Label information and attention capacity after complete-family opening

The previous goal turn made scientific progress: the full three-seed correction screen was completed, its unchanged gate failed, and a stronger known C&S reference completed all 72 records. This turn adds error attribution and a structural capacity check. Original scores and running recipes remain unchanged.

## What the graph supplies

Of 5,274 development nodes, 2,512 have no permitted TRAIN-labelled neighbour and 1,733 have neighbours carrying exactly one distinct TRAIN class. Those two cases cover 4,245 nodes, 80.49% of the population. There are 1,029 nodes with at least two visible classes. Structural counts are independent of predictions. The present one-hop recipe cannot use literal TRAIN-label evidence beyond that immediate neighbourhood.

Across C4's three own-selected native states, the one-class cohort contains 276, 270 and 266 native errors. Its visible class equals the truth for only 49, 62 and 49 of those errors. Thus the observed available labels are often misleading specifically where the base model is wrong. This is retrospective consumed-development evidence, not a confirmation dataset or a new gate cohort. Missing/unlabelled nodes were not assigned a negative class.

## Exact restricted algebra

At fixed parameters, the current bias-free linear label-value route has label logits ell_m(i)=T_m g_m(i). If all visible source labels have class c, g_m(i)=a_m(i)e_c with nonnegative attention mass a_m(i). For strictly positive mass, ell_m(i)=a_m(i)T_m[:, c]. Changing attention mass therefore changes confidence but cannot change the ordering of that route's label logits. Zero mass gives zero logits. This statement is before adding native logits or taking the new bounded native/posterior mixture. Different routes can have different T_m columns, and their pooled probabilities plus the native prediction can still have query-dependent decisions. It is not an impossibility result for the whole ensemble.

The algebra is an explanation of this particular operator. No new mathematical principle or guaranteed loss of accuracy is claimed. Query-dependent featurewise modulation before class readout could change the class-logit direction, but this is established FiLM/GNN-FiLM ancestry. It might also create an extra feature classifier that ignores label identity. A capable joint single and a label-identity control are necessary before calling any gain an ensemble or label-evidence contribution. That extension remains inactive.

## What the stronger reference actually repairs

The selected common C&S recipe repairs 39, 49 and 34 native errors, while introducing 22, 25 and 29 errors. Its repaired native predictions have median top-one confidence about 0.605, 0.603 and 0.562. These are much less confident than the typical unrepaired native mistake. The useful repair signal is concentrated in marginal decisions.

At one-hop-covered nodes, repair/harm counts are 30/18, 37/22 and 22/24. At nodes without one-hop TRAIN anchors they are 9/4, 12/3 and 12/5. The third seed's net gain therefore comes entirely from outside the current one-hop support. The helper uses native prediction fields and multiple graph propagation steps, so these improvements cannot be assigned solely to longer-range literal labels. C4 and C&S native-selected epochs differ in seed 6101(178 versus 143), and match in the other two seeds. Cross-pipeline correct-overlap counts are descriptive and cannot be used as an oracle selector.

A two-hop literal-label path would reach 5,025 of 5,274 development nodes, compared with 2,762 at one hop. Some 161 nodes have no directed path to a TRAIN anchor. Wider reach is available structure, not proof of useful evidence or a newly adopted multi-hop method.

## Next action

Keep the staged posterior P0 source inactive pending complete source and runtime review. It tests direct masked-label supervision plus bounded probability serving on a frozen native predictor. It does not claim to fix the single-class direction constraint or unreachable-label cases. Assess the conditional value-modulation alternative as a separate attributed hypothesis before any source or gate change. Continue complete graph-relation 12 and molecular 18 families under their existing fixed rules. No new quality advantage, novelty clearance or paper acceptance has been established.
