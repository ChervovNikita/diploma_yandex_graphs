# Additional scoped TabM passages

The current read of correct primary2410.24210v3 additionally covered AppendixA.2,
A.3 and D.2–D.3. This is additional scope of a reused paper, not another unique
paper or a full-paper reading.

- A.2 applies feature embeddings before the MLP and initializes the first
  multiplicative adapter from a standard normal distribution for those variants.
  That is not direct evidence for changing native categorical molecular
  AtomEncoder factors. An adapted initializer needs its own explicit definition.
- A.3 observes larger optimal learning rates for TabM in its study. The current
  graph benchmark retains paper-native LR/WD and performs no such tuning.
- D.2 uses standard tabular preprocessing, AdamW, global norm clipping1,
  patience16 and generally50–100 TPE tuning iterations. These are TabM protocol
  choices, not settings silently imported into our native GNN comparisons.
- D.3 generally uses15 evaluation seeds and a score/SD-based descriptive tier
  rule. That ranking heuristic is not a formal confidence interval, paired
  hypothesis test or proof of reliability. The current statistical amendment
  handles repeated paired trials and multiplicity explicitly.

The new molecular pilot uses three seeds for exploration; it is not promoted as
TabM-scale evaluation. No native model, scientific run list or frozen pilot was
changed by this additional reading.
