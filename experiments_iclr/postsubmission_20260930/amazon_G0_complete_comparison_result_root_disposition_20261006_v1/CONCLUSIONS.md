# Frozen Amazon comparison: negative quality result

The candidate classified 1,028 of 2,449 reserved assessment nodes correctly (41.98%). The independent ensemble classified 1,134 correctly (46.30%), the single model 1,111 (45.37%), and the ordinary shared pooled model 1,110 (45.32%). The unchanged seven-condition gate failed. This frozen candidate configuration does not advance to confirmation.

The graph-free control has the same accuracy, and its NLL is lower by approximately 0.0000024. The present experiment provides no useful predictive advantage attributable to the candidate's graph conditioning. Lower log loss and Brier score than the warm model or long-trained references do not establish accuracy superiority; the candidate loses 79 correct nodes relative to its unchanged warm state.

This is one seed and split, with 16 SGD updates versus the references' 2,300 Adam updates and different training hardware/optional extensions. Serving used one common allocation runtime. The references have low accuracy and high NLL, so their training curves and configuration need scrutiny before treating them as strong benchmarks. No cause of that poor generalization has been identified here.

All twelve models, error flows, frozen predictions and failed setup attempts remain in the evidence. All 44 forwards and twelve artifacts were frozen before the first assessment-label read. The original paper scores and original validation/test labels are unchanged. The independent result interpretation is being prepared.
