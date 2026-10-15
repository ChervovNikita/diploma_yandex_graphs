# Complete molecular route diagnosis

This posthoc analysis extends the first-seed diagnostic to every completed
MolBACE pilot cohort: seeds 101, 202 and 303, all four native independent
models, base GNNM, alignment/variance, private input correction and their
combination. It uses the existing selected-VALID exports and existing route
reconstructions. INPUTS.json binds all 24 fits and route tensors. No model,
forward, optimizer step, fitted combiner, TEST access or selection change is
permitted. Original fit scores remain unchanged.

For every four-member bank report member AUC, the actual reconstructed mean's
AUC, their difference, common reversed positive-negative pairs, disagreements,
and reversed pairs despite a correct route or a strict correct majority. Use
the exact stored mean in its original dtype. For I4 use the four saved native
exports to form the original arithmetic mean and retain reconstruction
differences separately. Also report both fixed mean-logit and mean-probability
readouts for every bank; do not select a readout. Preserve BCE tradeoffs.

Compare every candidate with base on identical positive-negative graph pairs:
strict repairs, strict harms, ties and change in AUC credit. Summarize the
existing class geometry only as descriptive geometry. Individual embeddings,
graph pairs and routes are not additional experimental repetitions. The same
151 graphs (130 positives, 21 negatives) occur in every seed. Selected VALID
does not provide independent confirmation or a causal mechanism test.

Run on the authorized allocation, CPU only, two threads, one invocation with
a five-minute cap. Read about 11 MB of saved route tensors plus small exports;
reserve 1 GiB RAM. No retries or tolerance search. Write fresh results, source
and input hashes, actual wall/RSS costs, all cohort rows and their unweighted
three-seed summaries. Preserve any reconstruction warning without changing a
checkpoint or re-running inference.
