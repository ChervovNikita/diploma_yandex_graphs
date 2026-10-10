# PubMed masked-context development screen

Frozen before acquisition, qualification or new numerical outcomes on 10 October 2026.

Each route learns to classify the original graph and its own graph with one persistent quarter of node features missing. A common decoder teaches the route to recover those missing features from the remaining information. The hypothesis is that these different training tasks yield useful complementary predictions on the original graph. More distant hidden vectors alone do not satisfy this hypothesis.

Use the complete 19,717-node, 500-feature PubMed graph loaded by PyG Planetoid with NormalizeFeatures. Retain the published PolyFormer monomial configuration and source exactly as bound by masked_context_be_source_prototype_20261010_v2. The native author split replaces the public Planetoid masks. Use its exact class-balanced TRAIN sampling and global remainder VALID sampling with split seed 190111. This is a new exploratory protocol on previously encountered public data, not unused confirmation. Three optimizer seeds share this fixed split. Report class counts because equal TRAIN quotas can leave an unequal held-out population. Stop preparation if any class has fewer than 50 VALID or 50 TEST nodes. These aggregate counts are data-construction checks, not model outcomes.

TRAIN has 3,943 labels per class, 11,829 total. VALID has 3,943 labels. TEST is the remainder and stays closed for scoring. The model and engineering worker receive only normalized features, edges, TRAIN IDs and TRAIN labels. A separate custodian creates the split and retains its public-data provenance. No model uses TEST labels. This protocol does not recalculate an original paper result.

The prospective scientific selector is the first epoch with the highest factual pooled VALID accuracy. Ties retain the earlier epoch. Stop after 250 epochs without a strict improvement, with at most 2,000 updates. No TEST monitoring. For independent4, use one common pooled selector for the bank in the primary comparison and retain member values at that same epoch. This bank selector differs from individually selected independent members, and any eventual superiority claim requires the latter reference as well. Qualification is three full TRAIN updates only and does not establish learning competence.

Stage 1 includes shared4_own, shared4_core and genuine independent4_native for optimizer seeds 9101, 9203 and 9307. Complete all nine records before interpreting the comparison. No seed replacement or outcome-based retry. All fixed model, loss, mask, temperature, decoder, negative sampling and initialization settings remain those in V2.

For a continuation to the other 18 records, require all of the following on VALID, using percentage points for accuracy:

- Candidate minus shared4_own has a mean gain of at least 0.20 and is positive for all three paired seeds.
- If the mean independent4 minus shared4_own gap is positive, the candidate closes at least one quarter of that gap. Otherwise the candidate mean is at least the independent4 mean.
- Mean pooled negative log likelihood is no worse than shared4_own.
- Mean member accuracy falls by no more than 0.10 points and worst-member accuracy falls by no more than 0.20 points, averaged across seeds at each bank's selected epoch.

Passing admits further mechanism comparisons, not a paper claim that the candidate beats independent ensembles. Keep every outcome, cost and failed condition. The 18 additional records separate masking alone, persistent ownership, graph alignment, a native single, a single receiving every auxiliary view, and untied auxiliary bodies coupled by the common decoder. A positive final method requires paired comparisons with capable singles and ordinary independent ensembles, uncertainty estimates, and unused confirmation on other splits or tasks.

## Immediate engineering envelope

Acquire or load public data in this project repository. Bound acquisition and split export to 600 seconds. Bind the existing repository runtime, exact TRAIN arrays and source manifest before running a fresh worker. Run three complete TRAIN updates for shared4_core, seed 9101, with an external 1,800-second timeout, 32 GiB GPU and 16 GiB RSS ceilings. Preserve failures and do not extend or retry automatically. Store arrays and outputs on the allocation. Fetch only compact metadata to the Mac.
