# Bounded reading conclusions

Two new primaries were read: GraphLand arXiv:2409.14500v5 and GraphPFN arXiv:2509.21489v4. The existing graph-ensemble uncertainty conclusions and literature index v7 were consulted first; that uncertainty paper was not reread. Cached PolyFormer, Polynormer and heterophily-reassessment conclusions were reused. Published results below were not reproduced.

## GraphLand v5

GraphLand supplies the exact industrial leads `tolokers-2` (worker-ban binary classification) and `artnet-views` (social-user views regression), with typed features and fixed random/temporal roles. RL uses 10/10/80; TH and THI use 50/25/25, with later nodes absent during THI training. These are different questions, not interchangeable replications.

Its full-graph residual GNN recipes use three aggregation blocks, width 512, interleaved two-layer GELU MLPs and multi-layer heads. RL Tolokers2 favors GAT/local GT over GCN; RL Artnetviews and TH on both tasks make GCN a defensible common native backbone. LightGBM+NFA remains a necessary competitive reference. Recent GFMs evaluated in this version have classification-only implemented support. GraphPFN is cited, not evaluated by GraphLand v5.

Paper and released optimizer descriptions differ (Adam versus source AdamW). Native scripts expose test scores during selection; prospective closed-label studies need a guarded wrapper. Provider preprocessing, categorical encoding, missing-target masks and self-loop policy must be bound. Metadata reports raw feature counts, not final encoded widths or labeled role counts.

## GraphPFN v4

This newer primary evaluates the same GraphLand RL releases, including both targets. GraphPFN FT reports AP 62.80±0.39 and R² 65.35±0.06 (×100), substantially above its competent native GNN/LightGBM references. G2T-LimiX is also strong on both tasks. This supports including a modern pretrained baseline, with separate pretraining, tuning, inference and ensemble accounting.

GraphPFN combines pretrained LimiX with graph-attention adapters and synthetic graph pretraining. Its ten-member ensemble shares model weights and varies preprocessing/feature samples; it is not an independent deep ensemble or the R17 four-route architecture. Published empirical superiority on fixed RL splits cannot be extended to TH/THI, new graphs, epistemic calibration or equal training cost.

Static paper configs verify that its GNN recipe differs from the GraphLand release, including two-logit CE, stopping, loop choices and transform search. GraphPFN FT configs bind ten members, eight random features, AdamW, query length 1024, evaluation epoch size 10 and patience 4. Selected FT learning rates are 2.320794010302052e-5 for Tolokers2 and 8.340500244230498e-6 for Artnetviews. No checkpoint or dataset was downloaded.

## Decision

Prepare one Tolokers2 RL/residual-GAT confirmation after the existing pilot demonstrates utility. Use a qualified native independent ensemble, LightGBM+NFA and GraphPFN FT references. Keep AP, NLL, calibration and route-separation interpretations distinct. Add Artnetviews RL/residual-GCN only after a new squared-loss regression initializer and numerical qualification. Freeze any temporal test separately.

The saved NFA script has static fraction-input and inductive graph-scope issues; audit or replace the exact implementation before new use. This observation does not invalidate published tables. The source environments and A100 hardware reports supply preparation leads, not a local feasibility result.

No numerical scientific execution, model/library import, dataset/label acquisition, training, SSH, remote execution or sealed-packet modification occurred. This packet is a source recommendation, and grants no execution permission.
