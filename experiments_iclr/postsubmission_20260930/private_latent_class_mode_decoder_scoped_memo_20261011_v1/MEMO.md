# One inactive mechanism: private latent class modes

**Nomination:** replace each route's final affine class decoder by a small, factual two-mode decoder, jointly learned with the shared native backbone from scratch. Preserve the complete graph, endpoint VALUE operation, earlier private factors, own-label CE, optimizer/selector policy and probability-mean serving. This is one attributed prediction-capacity hypothesis, not a new ensemble principle or an admitted fit. It does not change the running Physics/Penn protocols.

## Why this question remains after complete E/PE

The complete WikiCS E study closed15 banks/24 fits and75 calibration attempts. E4 improves pooled accuracy by.5688pp over BASE and.3034pp over equipped genuineI4, with all three paired signs positive. E4's mean members improve1.0729pp over BASE but remain.7632pp weaker than equippedI4. I4 retains better raw/calibrated NLL and Brier; the descriptive E4−I4 accuracy interval includes zero. Against BASE, E4 removes106 covered alternatives but removes196 pooling losses, net90 additional correct repeated readouts. These facts do not diagnose a classifier bottleneck.

The complete PE join closes18 banks/27 optimizer bundles/36 body records. E4−PE is+1.4094pp and BASE−PE+.8406pp, all three signs positive. PE matches BASE's parameter budget; E4 is about10% larger and has worse rawNLL thanPE. Thus a different allocation of a small weight budget did not automatically produce stronger served predictions. All these results are encountered development, not unused confirmation or proof of a general sharing advantage.

The remaining question is whether a restricted affine class boundary contributes to weak own members. A nonlinear decoder could separate two within-class hidden regions without changing graph information or adding an auxiliary objective. It cannot reconstruct distinctions absent from every factual hidden state. Neither the E result nor common-false-rival accounting establishes that actual errors have this cause.

## Exact proposed operation

For each native route's final hidden state h_m(v), learn two ordinary affine scores per class:

```
a_m,c,k(v) = w_m,c,k^T h_m(v) + b_m,c,k,  k=1,2
z_m,c(v) = logsumexp_k a_m,c,k(v) - log(2)
p_m(v) = softmax_c z_m,c(v)
L = mean_m mean_TRAIN[-log p_m(y)]
serve = mean_m p_m
```

All head weights are route-owned; the expensive native graph backbone and endpoint builders retain their declared sharing. No mode is assigned a subset of labels, and no mode has a separate CE. The correct class's two modes jointly receive its marginal likelihood. Head and backbone are live from fresh constructors. Mode heads receive separate declared initialization streams, so identical duplicated starts cannot lock a symmetry. There is no fitted donor, teacher, clustering, pseudo-label, TRAIN-label input, diversity penalty, center repulsion, graph propagation of scores or inference-time adaptation. Fix two modes and unit logsumexp scale; a failed outcome does not select another count/temperature.

This changes factual class scores at inference. It can reverse a common false rival by changing the member vectors; it is not a nonnegative router over unchanged predictions. With one mode it is an ordinary private affine classifier. With two modes, each class region can follow more than one affine direction. This is a boundary of the decoder at fixed h, not a theorem that the full native network cannot represent the same function. Modes can collapse, overfit or learn the same rankings across routes.

## Concrete novelty boundary

The probability is exactly a2C-state affine softmax marginalized over its two latent states per observed class: p(c)=sum_k exp(a_c,k)/sum_d,j exp(a_d,j). Marginalization and latent mixture classifiers are established. Do not market this algebra as a new loss, probabilistic guarantee or discovery of class modes.

For the new question about factual multi-center scores, I revisited only retained SoftTriple arXiv1909.05235v1 blocks37–61 and Eqs5–8. It explicitly supplies multiple class centers, a hard maximum and a smoothed within-class assignment. Its relaxed similarity is an assignment-weighted center similarity followed by a margin/scale class loss. The proposed marginal logsumexp with affine biases and ordinary CE is not an exact SoftTriple reproduction; the multi-center prediction principle is already occupied. Saved Mixture-of-Softmaxes1711.03953v3§4 conclusions also establish multiple conditional outputs under a shared representation; no new read or vocabulary-rank theorem is adopted here.

The saved class/affinity/SupCon recipes trained auxiliary geometry behind the existing native class head; their failed scopes do not test this served head. The recurrent class decoder exchanges class messages over the graph, and frozen GPN correction fits density/transport heads after base acquisition. This proposal uses neither operation. Earlier private-last/node-capacity ideas already cover untying small prediction capacity, so ownership alone is not novel either. The unverified narrow empirical question is whether this particular class-marginal decoder improves the current shared native predictor beyond a capable private affine head and equally equipped single/I4.

History researcher coordination: their one lane is learned sparse graph-layer ownership/topology with prune/regrowth. This memo changes a fixed nodewise class function and learns no connection masks or graph-layer allocation. They identified no exact served two-affine-mode/logsumexp pilot in the completed-history scopes they revisited; this is scoped nonidentification, not global novelty clearance.

## Decisive reference and falsifier

A later bounded comparison would retain six fixed conditions: unchanged E4; E4 with private two-mode heads; E4 with private one-mode affine heads; E4 with one common two-mode head shared across its still-private backbone routes; same-operation equipped M1; and same-operation genuineI4 with independent optimization and own selection. Private-affine/common-head controls differ in functional capacity; do not label them parameter-matched. The common head controls whether route-owned decoder estimation is needed, and the private affine head controls whether generic head ownership explains the change. The capable M1/I4 receive the entire two-mode operation and full TRAIN supervision.

The primary falsifier is no complete served-accuracy increment over unchanged E4, private affine and common-mode controls, or no practical advantage against equally equipped genuineI4. For the specific weaker-member use case, require retained/improved mean and worst own-member quality, report NLL/Brier and preserve every class/seed harm. A pooled gain obtained by weakening members does not support this nomination. Record common-rival true-label acquisitions actually served, new common mistakes, coverage turnover and pooling loss. Better proxy separation, mode usage or TRAIN loss is insufficient. If the same-operation single explains the gain, retain a general decoder improvement but reject an ensemble-specific explanation. Close a failed exact mechanism without a rescue grid.

## Estimated work, not execution authority

At h128,C10,M4, the private two-mode heads store M·2C(h+1)=10,320 parameters (41,280FP32 bytes); one shared two-mode head stores2,580. These are standalone head counts, not verified whole-model totals. Full routed head arithmetic is2MNCh MACs versus MNCh for an ordinary affine head: about119.8million versus59.9million per full WikiCS forward. The four-route mode-score tensor is about3.74MB; sequential route execution needs about.94MB for this tensor. Backward/optimizer/workspace and backbone activations are extra. There are no new edge builders or sparse propagations.

Six conditions over three paired development seeds mean18 banks/27 full optimizer fits, including12 genuineI4 bodies. With sequential own-CE and a1000-update ceiling: at most27,000 Adam steps,63,000 TRAIN routes/backwards and63,000 SELECT routes, plus63 selected serving routes. All reference fits and fixed calibration are paid; no favorable subset substitutes for closure. The full original five-arm endpoint acquisition took4315.8s under its observed concurrent scope. Because this proposed roster contains four endpoint-shared conditions, a coarse complete bill is **2–5 GPU hours**, plus the small fixed readout, rather than scaling only by fit count. This is an unqualified forecast; early stopping and full-model preparation dominate. No GPU/time grant, qualifier, source consumer, reviewer or fit is requested.

**Disposition:** one compact inactive decoder hypothesis. It is operationally different from endpoint messages and the history topology lane, has strong established decoder ancestry, and needs an empirical utility result before any scientific contribution claim. Current confirmation remains the priority.
