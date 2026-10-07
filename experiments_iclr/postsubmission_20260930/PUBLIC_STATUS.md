# GNNM research status

Updated 2026-10-07T15:32:43.138260+00:00. Goal active: establish a useful methodological extension with verified gains and a fresh neutral manuscript review. **No verified superiority over an ordinary independent ensemble or new manuscript acceptance yet.** Original submitted scores are unchanged.

## New completed evidence

The full WikiCS comparison finished 24/24 fits: eight configurations, three seeds, 1100 epochs per fit. Prediction analysis also finished. On the 5274-node split 0 development population used for checkpoint selection:

- Unit initialization plus the contrastive package improves over plain unit factors by 0.51percentage points; all three seeds improve. Its mean 81.63% is above the matched single 81.48%, but below the ordinary independent ensemble 82.04%.
- Randomized first-factor initialization weakens members. It produces more correct-member coverage than unit factors, but poorer served accuracy.
- The randomized-init contrastive primary yields 349 repairs and 321 new errors across three seed blocks: a small net change, with one negative seed. Selected development analysis cannot establish generalization.
- Common wrong competitors dominate the remaining shared errors. Reweighting existing probabilities cannot repair a node when every member ranks the same wrong class above truth. Better member learning is required for those cases.

These are meaningful diagnostic results, not a new method victory. Full eight-arm outcomes and negatives remain recorded. The gain in the unit arm mainly reflects member competence; it does not establish a contrastive diversification mechanism. Independent scientific interpretation and its proposed bounded tests are saved in `wikics24_scientific_interpretation_independent_20261007_v1`.

## Actual work now

The molecular graph study began on the authorized allocation. At 15:23 UTC its first single-model run reached 25/100 epochs. The frozen 18 fits compare single, ordinary independent4, and four shared-ensemble learning rules O/I/P/G on full official scaffold TRAIN 32901/development 4113. Candidate I was fixed before outcomes. No molecular comparative scores have been opened. This family is not duplicated on 18.77.

18.77 reconnected successfully at 15:18 UTC. Both prior four-member native prefix banks for seeds 29/43 completed 1100 epochs/member while detached. Their original checkpoints and clean terminal receipts remain on that server. The four remaining continuation/private-path controls are admitted from these completed banks. Both continuation fits actually started at 15:41 UTC, one per GPU; each first member reached epoch 1200 by 15:42 UTC. The private-path fit follows its continuation in each lane. Owner 3614607/start 1753763677 is detached. No original prefix was restarted. Both expected A100 80 GB GPU identities are verified. The Git checkout was synchronized to 8dd796bf91f65e1de2ed99a3ee6a3efe6a3d2bf3 without replacing working files or touching jobs.

## Research decisions

Earlier Citeseer initialization and private-growth studies failed their declared references and remain stopped. A local Collab gain over single still lost to independent4. Keep those results and their costs. Hidden separation, stronger average members, parameter savings, or an engineering source pass alone do not establish the required quality contribution.

We continue studying training rules and graph-specific evidence differences. The molecular factors already overlap member-conditioned normalization/GNN-FiLM; no new primitive or global-descent theorem is claimed. Literature notes retain exact source scope and failures. An attribution control is prepared but inactive until its scientific question is warranted. New paper text and acceptance review require meaningful supported evidence, separate frozen confirmation and honest uncertainty.

## Boundaries

Scientific execution stays in the project repositories on the authorized one-GPU allocation and18.77. The seven-GPU account is MacLink forwarding only. No sudo, host mount/namespace changes, PDF compilation, GENLINK, Desktop operations or unrelated-data access. Normal incidental runtime caches are allowed. Raw arrays and checkpoints stay on servers; compact active analysis is mirrored locally and committed with source/decisions. Reviewers must start fresh with the supplied skill and receive no desired verdict.
