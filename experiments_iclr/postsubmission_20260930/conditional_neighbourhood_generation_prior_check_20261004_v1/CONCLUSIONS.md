# Neighbourhood reconstruction: inspected prior and current hypothesis

## GraphLP

Source: *Generative Graph Neural Networks for Link Prediction*, arXiv:2301.00169v1, primary PDF retrieved and retained in `primary/generative_gnn.pdf`. Read scope: introduction, related work and method through Sections 4.4–4.5; dataset/protocol discussion at the start of PDF page 7. This is a scoped method read, not a full-paper certification or reproduction.

GraphLP combines a global low-rank self-representation of adjacency with high-order propagation, multiscale representations and adjacency reconstruction from perturbed graphs. Its reconstruction target uses all-pairs binary cross-entropy. The inspected method establishes prior credit for training graph representations through structural reconstruction and multiple generative operations.

The inspected method does not implement our count-conditioned candidate-subset likelihood with one member responsibility shared between both endpoints. This bounded observation is not a certificate of novelty. Its seven small-graph, random observed-edge protocols do not establish competitive performance under the official OGB split and evaluation recipe.

## GAD-NR

Source: *GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction*, WSDM 2024, arXiv:2306.01951v8, primary PDF retained in `primary/gad_nr.pdf`. Read scope: PDF pages 1–5 and the continuation of Section 4.4 on page 6, ending before Section 5 experiments. The primary method diagram on PDF page 4 was inspected. This is a scoped method read, not a full-paper certification or reproduction.

GAD-NR reconstructs self-features, node degree and the distribution of neighbour representations. Degree is predicted with an MLP and squared error. The neighbour-distribution target is approximated by a Gaussian; the decoder generates samples through a learned distribution and neural transformation, and the reconstruction loss compares Gaussian moments using KL divergence. Target moments stop gradients, and covariance regularization addresses singular matrices. The loss is also used to score node anomalies.

GAD-NR explicitly builds on NWR-GAE's neighbourhood reconstruction with optimal transport and Hungarian matching. Its motivation is to reduce computation and avoid fitting anomalies too closely, not to maximize reconstruction expressiveness. This rules out a claim that generic neighbourhood reconstruction, neighbour-distribution modelling or topology supervision is our new contribution. Gaussian-mixture density models and multiple GNN views are also cited in its related work.

Our current auxiliary conditions on observed subset cardinality and models which candidate identities belong to the residual neighbour subset. It does not predict node degree or reconstruct neighbour-feature moments. Sharing the same latent member across both endpoints is a narrower adaptation hypothesis. It still needs closest-prior analysis and predictive evidence. GAD-NR's reasoning also cautions against treating a lower auxiliary reconstruction loss as sufficient evidence of better downstream decisions.

## Consequence for research

Retain the paired target-only, shared-responsibility and independently mixed endpoint comparison. A useful claim requires improvement of the unchanged, count-free served link ranker, rather than only a better training proxy. Preserve failures and unsuccessful transfer. Do not describe either the general mixture likelihood or reconstruction idea as new.

Primary-file hashes, versions and retrieval failures are preserved in the accompanying retrieval records. No author code, dataset, model fit or paper experimental score was reproduced in this reading packet.
