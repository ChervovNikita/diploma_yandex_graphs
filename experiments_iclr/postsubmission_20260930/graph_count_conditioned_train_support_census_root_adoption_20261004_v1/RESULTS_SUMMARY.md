# Complete TRAIN support census

All three seed mask streams completed 17 native batches, with 65,536 positive and 65,536 sampled negative queries per batch. The exact original source, data and runtime custody passed. Full histogram JSON remains on 18.77; a pinned compact projection rechecks every histogram scalar and joint/marginal count using standard-library integer arithmetic.

Across 3,342,336 positive query occurrences, 194,664 (5.8242%) had nonconstant patterns on both sides, 45,453 (1.3599%) had genuine subset choices on both sides and 306,577 (9.1725%) had a genuine subset on either side. Of 3,342,336 negative occurrences, one had both sides nonconstant, one had either side genuine and none had both sides genuine.

This supports testing the hypothesis on real positive patterns. It also reveals that the current auxiliary provides essentially no cross-side association signal on native sampled negatives. These are repeated TRAIN observations, not independent graphs or accuracy evidence.

One positive side can have 958 genuine exact groups, 86,629 Python group-slot iterations, 749,572 dynamic-program transition cells and 1,661,851 total residual slots. Mathematical work counts do not establish practical GPU time or backward memory. The sequential single-model control also needs exact vectorization before native feasibility is assumed.

No loss, model, feature, optimizer, VALID or TEST was evaluated. A fresh audit of compact projection, source and custody is pending; it will explicitly distinguish its checks from an independent raw histogram reread.
