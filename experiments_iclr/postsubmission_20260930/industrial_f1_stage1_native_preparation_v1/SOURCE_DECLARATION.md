# Source and scientific status

The model and training recipe come from official RelBench v1.1.0 at
`9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`, not the RelGT release. The
unchanged official `examples/model.py` is loaded from its saved snapshot;
critical installed RelBench/PyG/Frame modules must byte-match their snapshots.
Receipts in `evidence/source_receipts.json` identify each copied or newly retrieved
source. Newly retrieved RelBench helpers match the official Git blobs. Frame
GBDT source is verified against the SHA256-pinned PyPI 0.2.3 wheel; that wheel was
read in memory only to extract source text. No package was installed or imported.
Release-source URL failures during discovery did not retrieve data or weights.

The engineered-control basis is the user-study commit
`445bb7a3b1230f49f8e5890ae81754d3e365680f`; its original SQL, stypes and trainer
are preserved in `sources/`. The new `history_features.sql` and
`ENGINEERED_STYPES.json` are declared adaptations, not author-provided artifacts.
The README lists all changes, including key retention, future-feature removal,
chronological conversion, deterministic result duplicates, fixed validation
sampling, validation-only evaluation and seeded Optuna accounting.

No results, runtime timing, numerical readiness, GPU compatibility, data admission,
filesystem isolation, heldout evaluation, initializer adoption, ensemble adoption,
extension-grid execution or paper claims are established by this packet. The user
default is four members. Stage 1 contains three independent native fits only.
