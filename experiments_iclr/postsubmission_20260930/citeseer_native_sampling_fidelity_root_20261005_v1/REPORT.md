# Native TRAIN negative-sampling fidelity

The suspected difference is resolved: the pinned HeaRT loader explicitly replaces `data.edge_index` with the undirected TRAIN positives. Its negative sampler therefore uses TRAIN topology, as does the running Citeseer runner. The source does not require exclusion of withheld links from this native training sampler.

No running source, cohort, scores or TEST inputs were changed. This static check does not establish bitwise equivalence across PyG versions; the actual pinned modern runtime and RNG protocol remain stated in every run configuration.
