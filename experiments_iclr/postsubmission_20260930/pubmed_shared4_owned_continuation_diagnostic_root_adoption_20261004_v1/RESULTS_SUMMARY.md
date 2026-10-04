# Pubmed saved-state continuation diagnosis

The original restore helper shares all 40 CPU Adam step tensors with the saved state. One 36-update epoch advances the saved counters from 180 to 216. The second reconstruction therefore restores step 216, although the immutable epoch5 observation has step 180. Its next epoch advances the shared counters to 252, including those of the idle first optimizer. Only these 40 step paths change in the saved and pre-epoch comparisons; sampled edges, batch permutations and RNG streams agree exactly.

The second loss is 0.3458168929 versus 0.3455345068 for the first reconstruction. The difference 0.0002823861 exceeds the unchanged allowance 0.0000205312 by 13.754 times. This is a real state-isolation defect; the diagnostic does not establish that it alone caused the earlier full qualification failure.

Both complete native epochs finished: 72 Adam updates, zero VALID serves and no scientific fit or TEST access. Physical supervision exited 0 after 25.59 seconds, with no signal or resource breach. Logical result, physical terminal, source/release and final custody hashes match. COMPLETE means diagnostic evidence was collected; it grants no qualification PASS or scientific donor state.

Next: clone the complete saved tree before each restoration, independently review that source successor and repeat the full fresh private/pooled bridge qualification with the original model, numerical bodies, streams, inputs, profile and tolerances. Preserve all failed attempts and avoid using any diagnostic state to initialize research fits.
