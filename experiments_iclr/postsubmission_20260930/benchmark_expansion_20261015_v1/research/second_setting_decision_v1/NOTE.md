# Second setting: Coauthor-CS / native PolyFormer-Cheb

15 October 2026. Source/metadata decision only; no launch or new scientific code.

**Recommend Coauthor-CS / PolyFormer-Cheb as the next setting to prepare**, conditional on the current MolBACE decision and exact residual/structural lifecycle and resource admission. It tests a different task and backbone without native BN, uses a documented author preset, and has substantially more VALID records. It is not presently a prepaid six-fit extension or a measured cheap experiment.

## Native support and validation population

The pinned author `node_classification.sh:39` explicitly declares CS/Cheb: hidden 64, K=4, one layer, eight heads, FFN 32, dropout .8, attention dropout 0, q=1, multi .5; ordinary/attention Adam learning rates .0005/.01 and both weight decays .001. `training.py` implements full-graph masked CE and first strict maximum VALID accuracy. `mymodels.py` and `PolyFormerBlock.py` retain polynomial tokens, attention and LayerNorm; no BN is present. The native-settings catalogue records the paper's 96.49 ± .17% result as contextual 95% CI over ten runs, not a reproduced result or expected pilot score. Its high reported accuracy also cautions against assuming a large repair opportunity.

The saved provider audit has **18,333 nodes, 9,105 TRAIN and 3,667 VALID on each of all ten literal source splits**. VALID has 24.28 times MolBACE's 151 records. For a bounded follow-up, use a prospectively fixed literal split, naturally split 0 / source seed 1941488137, and keep complete TRAIN/VALID. Paired optimizer seeds are separate from the split seed. One split with three optimizer seeds is development evidence, not the source's ten-split reproduction.

**The nominal “60/20/20” description needs qualification.** The actual native splitter takes at most 733 TRAIN nodes per class and all nodes from smaller classes. Classes 0/1/3/6/9/12 have 708/462/429/371/118/420 nodes, respectively, all in TRAIN. These six of 15 classes have no VALID support; VALID can assess only the other nine. Do not change this source split silently or claim all-class validation. Correlated nodes in one transductive graph are not 3,667 independent graphs, and prior exposure is not certified absent. Even with those limits, this is a more informative second task than another adaptive loop on 151 molecules.

## Horizon and exactly what is already paid

| Item | Verified scope |
| --- | --- |
| Native source reference | At most 2,000 full epochs, native patience 250; ten literal source split seeds and advancing model RNG from global 42. |
| Matched pilot controller | Minimum 2,000 full epochs; then patience 100 on strict whole-VALID accuracy improvement; cap 10,000. One full TRAIN graph update per epoch; VALID every epoch; first ties; no checkpoint 0. This is an explicit extended controller, not the native source horizon. |
| Exact completed matched Cheb controls | **None verified in the inspected authorities** for M1, genuine I4, original G0, portable BE-plus-residual or residual-only. Historical SAGE, Mono and different-controller studies cannot substitute without exact protocol/state receipts. |
| Paid Cheb engineering | Seven passing legacy smoke cases: native single, G0, global half-pool, private half-pool, fixed private responsibility, within-route SupCon (four optimizer callbacks each), and independent4 (16 total callbacks): 40 charged updates. These are not selected full-fit controls. |

The existing residual-transfer protocol requires native M1/I4, G0, BE-plus-residual and residual-only: **21 physical trajectories across three paired seeds**, with M1 reused as I4/body0. Adding structural and permuted residual arms makes 27 trajectories, not six from a paid base. A structural-only question would still require 24 trajectories from scratch (12 native bodies plus G0/uniform/structured/permuted shared banks); omitting the BE-plus-residual arm would narrow the residual-removal question and needs an explicit freeze. No new matrix is adopted here.

For the 27-trajectory combined question, the declared matched minimum/cap implies 54,000/270,000 optimizer transitions, excluding repeated work. Route forwards, the second reverse collection, VALID and checkpoints add costs. An epoch cap is not a convergence certificate.

## Measured cost and the one alternative

| Evidence | Actual measurement and limit |
| --- | --- |
| Cheb native single smoke | Four callbacks; 12.563 s invocation; 5.425 GiB peak CUDA allocated; 5,756,874-byte checkpoint. |
| Cheb G0 smoke | Four callbacks; 22.835 s; 12.430 GiB allocated; 6,101,642-byte checkpoint. |
| Cheb fixed private-responsibility smoke | Four callbacks; 13.390 s; **21.731 GiB allocated**; 6,393,298-byte checkpoint. This is old BE private credit, not residual structural training. It warns against budgeting from G0's 12.43 GiB alone. |
| Alternative: MolHIV / native GIN dropout .5 | 32,901 TRAIN / 4,113 VALID molecules; 1,029 native batch 32 updates per epoch. Source 100 complete epochs; matched minimum 100 / patience 50 / cap 1,000. No exact completed controls for this proposed native setting were verified. |
| Nearby MolHIV cost evidence | The later GIN-VN shared engineering receipt completed one full TRAIN epoch in 582.446 s, plus two whole VALID calls and finalization in 613.996 s total. Its 100-epoch TRAIN-only extrapolation is 16.179 h for that measured method. **This is not plain-GIN residual timing or a completed scientific fit.** It supersedes the earlier budget note's “MolHIV whole-epoch time unknown” only in this scope. |

Prefer Cheb for its different prediction unit/operator and absence of BN. MolHIV modestly enlarges VALID beyond CS but has much more minibatch work and keeps the molecular setting; its measured sibling does not establish a cheaper transfer. There is no defensible full-fit ETA for Cheb residual/structural arms. Reuse existing shape/provider proofs, then qualify only the exact new composition, state restore, complete epoch/VALID and measured memory/checkpoint/runtime costs before budgeting.

Freeze one structural rule and exact within-class vector-permutation control, keep uniform residual and native/G0 references visible, and report mean/worst member accuracy, NLL and pooled repairs/harms. A larger VALID population does not justify choosing quartiles, strengths, seeds or settings after outcomes. `SOURCE_REFS.json` pins the local sources and read scopes; no remote, Git, tensor, checkpoint or new TEST-score access occurred.
