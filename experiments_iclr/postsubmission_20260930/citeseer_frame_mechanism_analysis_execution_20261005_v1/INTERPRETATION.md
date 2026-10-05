# Citeseer private-frame mechanism description

Status: successful execution of the fixed post-completion analysis, **not a successful method confirmation**. Root retains scientific adoption. This packet does not change canonical status, decision ledgers or original paper scores.

## What was inspected

All seven frozen families and three paired blocks were retained. The analysis describes the same 227 VALID positives, each ranked against all 500 released negatives. Two separate TRAIN-graph partitions were used: common-neighbor support and minimum endpoint degree. No split combination or subgroup model selector was fitted. All 21 selected framed checkpoints and 39 frame vectors were included. TEST, optimizer updates, inference replay and member-logit analyses were absent. Child execution took 4.582 seconds. Authenticated local retrieval contains 114,988 bytes of compact metadata; checkpoints and tensors remain on the authorized one-GPU server.

## Aggregate quality remains the primary evidence

| Frozen family | VALID MRR mean | Private-frame minus family | Private higher blocks |
|---|---:|---:|---:|
| native_single | 0.267811 | +0.013984 | 3/3 |
| ordinary_independent4 | 0.279204 | +0.002591 | 2/3 |
| unframed_f4 | 0.284115 | -0.002320 | 1/3 |
| shared_frame_f4 | 0.280053 | +0.001742 | 1/3 |
| private_frame_f4 | 0.281795 | — | — |
| same_four_frames_single | 0.269768 | +0.012027 | 2/3 |
| independent_frame4 | 0.280116 | +0.001680 | 3/3 |

Private-frame F4 improves the complete VALID mean over the four primary references, but its privacy contrast is weak: it exceeds shared-frame F4 in only one of three blocks. It also trails the plain unframed F4 mean, losing that comparison in two blocks. These are validation-selected development results on one graph and three seeds. They establish neither heldout superiority nor statistical significance nor generality across graphs.

## Fixed structural partitions

The CN-zero partition has 157 queries. Private-frame minus ordinary ensemble is +0.012411 MRR, positive in all three blocks, with descriptive seed t95 interval [-0.011189, +0.036011]. The CN-positive partition has 70 queries. That contrast is -0.019435, negative in all three blocks, with interval [-0.040731, +0.001861]. The candidate's improvement over ordinary ensembles in the first partition does not identify a reflection benefit: unframed F4 is essentially tied there (0.286431 versus private 0.286388) and is higher in the CN-positive partition (0.278922 versus 0.271493).

Endpoint-degree bins preserve 119, 83 and 25 queries. Private-frame minus ordinary ensemble is respectively +0.011945, -0.008581 and -0.004847. All intervals span zero; no bin selects or rescues a method. Exact all-family values and all six frozen contrasts per bin remain in STRATIFIED_MRR.json. T intervals describe variation under an approximate normal seed model, conditional on the one fixed graph/split; they are not independent-query intervals or corrected significance tests.

## What the frame descriptions establish

Selected private-frame checkpoints have off-initial-axis energy between 0.057446 and 0.135292. The parameters therefore do not remain exactly at their axis initialization. However, within-checkpoint reflection-map distances are 2.816613–2.828427, very close to the distinct-axis initialization distance sqrt(8). The corresponding same-four-frames single distances are 2.816645–2.828427. This describes map parameters; it does not establish complementary predictions, a causal mechanism or successful avoidance of predictive collapse. Norms depend on scale gauge. Independent encoder coordinates were never pooled into map distances.

## Scientific implication for root triage

The aggregate unframed control is stronger than this candidate, and the shared/private comparison does not provide consistent evidence that private learned frames help. A subgroup cannot override those observations. The evidence supports retaining this as a completed mechanism investigation, not a confirmed new methodological contribution. Any new training rule or stronger interaction-capable single must be a new prospective hypothesis with its own prior-work assessment and paired quality controls; this packet admits no such execution.

## Provenance

Execution RESULTS SHA256: `942992e040cb1cc820965707bf7c20b071efce5c29857dee03ab0a2eca06d421`.

Authenticated file identities and exact host/GPU route are in compact_retrieval_v1/RETRIEVAL_RECEIPT.json. All fixed partition outcomes are preserved. No existing source, protocol or predecessor was changed.
