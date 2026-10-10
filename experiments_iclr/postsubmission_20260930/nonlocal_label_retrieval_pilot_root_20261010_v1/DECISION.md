# Prospective pilot of live nonlocal label retrieval

10 October 2026. Freeze this decision before qualification, training or inspection of new validation outcomes. The root admits one complete WikiCS development comparison across SAGE, GCN and GAT, with paired seeds 7301, 7403 and 7507. Use the original graph, features, 580 TRAIN nodes, 5274 VALID nodes, width 128, at most 1000 updates and patience of 300 updates.

## Mechanism and ancestry

The same current native forward supplies each route's logits and hidden states. A query reads class values from permitted TRAIN anchors, using softmax of cosine similarity at temperature 0.1. Each update uses one common, approximately half-TRAIN query set, stratified by class. Remove every query from every route's support before retrieval. Labels never enter the hidden states or a subsequent message pass.

Train with native cross-entropy on all TRAIN nodes plus retrieval cross-entropy on the query set, with coefficient 1 for each. Retrieval gradients reach both query and support hidden states. At serving, use all TRAIN anchors and mix native and retrieval probabilities with weights 0.5 and 0.5. Average the resulting route probabilities.

Matching Networks supplies the learned kernel decoder. Neighborhood Components Analysis is the earlier family of stochastic neighbor learning. Its original probability objective differs from this log-loss. TPN and UniMP establish related similarity and label-aware graph learning. This pilot assumes no new primitive or methodological novelty.

## Complete roster and references

The new arms are live_shared4, detached_shared4, ordinary_M1_live and genuine_ordinary_I4_live. The detached arm keeps retrieval and serving identical, but removes all gradients through retrieval similarities. Every arm and independent member receives the same query draws from a separate CPU generator initialized with seed + 5000011. Preserve the existing native, factor and dropout seeds.

Three backbones, three seeds and four arms require 36 new banks and 63 new fits. Reuse closed original shared, single and independent reference archives. Preserve their reported scores and do not rerun those baselines. No VALID or TEST truth may enter retrieval values, masks or training losses.

Complete all 21 fits within each family and all three families before scientific comparison. Qualification uses actual TRAIN losses, derivatives and output shapes. It inspects no validation accuracy and preserves its costs separately. No coefficient, temperature, rank, support fraction or optimizer grid follows a partial result.

## Fixed screening criteria

Served accuracy is primary. On every backbone, live_shared4 must gain at least 0.2 percentage points over the same-information ordinary I4, and at least 0.1 over the same-information single and detached shared bank. Every contrast must be nonnegative at all three seeds and positive at least twice. Preserve every contrast against original shared and ordinary/factorized raw references as context.

Report NLL protection separately. Against the same-information controls, deterioration must be at most 0.02 on average and 0.05 in each seed. These are prospective screening rules, not statistical significance or population guarantees. Member and native quality remain separate diagnostics. A useful pooled ensemble can contain weaker individual members, whose weaknesses must remain visible.

Store native logits separately from mixed log probabilities. New arms select checkpoints by mixed VALID accuracy. Native-component scores at that selected checkpoint therefore include selection and stopping differences. Do not interpret them as a pure causal measure of acquisition. Report member quality, correct-member coverage, pooling losses, common-rival repairs and newly introduced errors across all nodes and classes.

A live-over-detached gain without superiority to capable same-information controls does not satisfy the shared-wrapper objective. A passing screen prioritizes capable published label-aware references, relevant factorized single/I4 controls, and a frozen pipeline on unused graph evidence. The screen alone supplies no manuscript claim. Failure closes this exact recipe and preserves any narrower kernel-decoder effect.

## Provenance and limitations

Original paper scores and failed studies remain unchanged. Stratification uses TRAIN truths, so no uniform-mask unbiasedness claim follows. Native loss also supervises the masked queries, so this is episodic training on representations learned from their labels. Half-support training and all-anchor serving are different estimates. Three optimization seeds on one connected graph are not independent graph replications.

WikiCS TEST remains unclassified after older access and stays closed. Record every additional similarity, retrieval and native operation. Keep models, labels, logits and checkpoints inside the authorized repository. No sudo, GENLINK, PDF compilation or unrelated-file operations.
