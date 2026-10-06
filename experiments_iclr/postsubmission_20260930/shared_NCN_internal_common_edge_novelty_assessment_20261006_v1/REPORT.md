# Shared NCN internal edge interaction assessment

**A learned nonseparable internal-edge residual can serve as a quality hypothesis in shared NCN. No novel method gap is established.** It changes the restricted current NCN interface while retaining learned shared encoder/message weights. Internal common-neighbor topology and conditional messages are established prior; the candidate must be attributed to local-community link prediction and query-subgraph GNNs. The frozen native plus H2GCN composition remains a control only.

## Exact closest overlaps

For query q=(u,v), let C(q)=N(u) intersect N(v), F(q)=E(G[C(q)]), and d_C(w)=|N(w) intersect C(q)|.

Cannistraci, Alanis-Lobato, and Ravasi, Scientific Reports2013 DOI10.1038/srep01613, explicitly use these internal edges:

`LCL(q)=|F(q)|=0.5 sum_(w in C(q)) d_C(w)`;

`CAR(q)=|C(q)| LCL(q)`;

`CRA(q)=sum_(w in C(q)) d_C(w)/deg(w)`;

`CAA(q)=sum_(w in C(q)) d_C(w)/log2(deg(w))`.

Thus the graph observation, internal edge count, and internal-degree-weighted unary pooling are direct prior. The appended 2015 erratum corrects JC/CJC denominators to neighborhood unions; it does not alter these four formulas. Read original PDF pp1–3 and appended erratum p14; visually checked p2. No source implementation or reported performance is adopted.

OCN arXiv2505.19719v2 was already scoped in the saved ledger. Its walk-based coefficient vector is `CN_k(u,v)=sum_(2(k−1)<a+b<=2k,a<=k,b<=k) (A^a)_u ⊙ (A^b)_v`; orthogonalized/normalized coefficients pool shared MPNN embeddings. Eq7 combines endpoint product with `sum_k alpha_k OCN_k H`. This is a strong higher-order structural comparator, rather than an exact statement that its finite configured model equals arbitrary induced-CN edge messages. Current PDF pp1–5 and7 were a targeted rereview, not a new identity/read credit or proof audit.

The vendored author NCN source already has CNhalf2/CNRes and CN2. CN2 (model.py918–979) computes A/A² overlaps in four combinations, then combines `alpha0*xcn1 + alpha1*xcn2*xcn3 + alpha2*xcn4 + beta*xij`. Generic higher-order overlap or bilinear structural residual claims are therefore closed. SEAL's target-labelled enclosing-subgraph learner and the broader edge-message subgraph-GNN family can use F(q); the candidate is a restricted shared-weight port into that family. An exact finite SEAL/OCN/NCN2 implementation identity is not certified.

Saved Neo-GNN uses learned structural weights in endpoint overlap and higher-order neighborhoods. LPFormer supplies pair-conditioned structural attention. ECC, GNN-FiLM, GATv2, HyperBatchEnsemble, and BatchEnsemble supply conditioned messages and shared-core/private-factor ancestry. Their saved method scopes are reused; no generic conditional-filter, attention, hypernetwork, or ensemble principle is new.

## What actually distinguishes the candidate

The current route sums nodewise functions of CN embeddings, plus an endpoint product. Equal endpoint products and equal CN embedding multisets force equal route scores, even with unrestricted nodewise transforms. A conditioner observing only those same statistics also collides. Query-constant linear diagonal factors commute outside the sum; placing input modulation before a nonlinearity can change the unary function, but cannot recover unseen induced adjacency from identical inputs.

The theory candidate instead computes one learned shared symmetric edge message:

`t_theta(q,w,z)=Psi_theta([h_w+h_z;h_w⊙h_z;h_u⊙h_v])`;

`g_m(q,e)=sigmoid(r_m^T xi(q,e))`;

`f_new,m(q)=f_base,m(q)+a_m^T sum_(e in F(q)) g_m(q,e)t_theta(q,e)`.

This is an edge-feature network with private weighting/readout over a query subgraph. If t decomposes as phi(h_w)+phi(h_z), its edge sum equals `sum_w d_C(w)phi(h_w)`, the unary local-community family. A nonseparable interaction can retain more information: fixed h=(1,1,-1,-1), with internal edges (0,1),(2,3) versus (0,2),(1,3), has identical CN count4, LCL2, internal degrees1, CAR8, and unary multiset, but `sum_edges h_w*h_z` is +2 versus -2. This is an interface witness conditional on fixed H, not a full-GNN theorem or a diagnosed Citeseer error.

The theory packet's constant-feature witness is already solved by CAR/counts. It proves a current-interface limitation, not novelty or necessity of learned nonseparable messages. Setting a_m=0 preserves initial base scores; later competence is not guaranteed. Repairs require a positive mean residual margin large enough to overcome the negative base mean margin; residual diversity with zero mean cannot repair it.

## Minimal strong comparison

Keep original ordinary loss, draws, target-masked support, three-pass rule, optimizer, selector, and raw-mean serving. Compare learned private-edge residuals with: original E_joint; count-only CAR/local-community augmentation; the same learned edge residual with tied gates/readouts; equal-capacity structure-blind residual; and a capable single given the same structural observations. Retain original J4 and add its structural counterpart before claiming a sharing-specific benefit. NCN2/OCN and an admitted strong subgraph learner are quality comparators; published table numbers are not executable baselines.

Primary evidence is pooled MRR against the strong E_joint0.296971/J4_joint0.291541 references, member MRR, and aligned strict common-negative repairs versus new errors. E's74.7% versus J4's45.2% is the fraction of pooled strict negative-slot errors that every member shares, not a query error probability. These selected VALID associations do not identify structural causality. Neither lower overlap alone nor gains reproduced by count-only/tied-gate/single controls establish a private structural ensemble contribution.

The code prototype is delegated to the existing theory agent: learned shared encoder/Psi, small private r/a, private/tied/count-only/blind/single controls, symmetry, current masked support, zero-a inclusion. No scientific job is launched here. New private encoders, frozen-only shared operators, routing and diversity penalties are outside this candidate. Internal-edge enumeration and serving cost must be measured, especially when F(q) is empty or large.

## Accounting

One new bounded primary read (CAR); one targeted OCN rereview; zero full-paper reads and zero new paper identities for reused methods. Source fragments and saved conclusions were inspected without imports or model execution. The canonical ledger, prior results and jobs are unchanged. Exact new-method duplication remains unestablished, but the consulted priors are sufficient to reject new internal-CN-count, generic conditional-message, or subgraph-ensemble principle claims. **Proceeding with an attributed architecture utility test is scientifically coherent while the broader novelty search continues.**
