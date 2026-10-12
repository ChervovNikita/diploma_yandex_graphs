# Paid source and continuation training

A staged method first fits a source model and then initializes a separate training segment from its validation-selected parameters. Each segment has its own declared full training budget and a fresh optimizer. The source acquisition is a paid fit in the same prospective study. Its optimizer state is not reused.

Factories that need the source declare an explicit `execution_context` argument. The dispatcher provides the registry, output root and verified paid-source checkpoint entries. Each source must finish its full budget, select a positive training step, and match the recorded checkpoint bytes and SHA256 before tensor loading. Direct declared ancestors are the only available sources.

The lifecycle schema is `paid-parent-continuation-v1`: one `parent_fit_ids` entry, `new_optimizer=true` and `parameter_initialization=selected_parent_no_optimizer_reuse`. The PT port identifies `pt_stage` and `pt_arm`; it is not counted as a generic BatchEnsemble fit. The master counts the initial and continuation segments separately. A four-member independent control needs four distinct source acquisitions with equivalent learning/data/selection contracts and distinct initialization streams. Member0 is reused as its matching singleton.

The continuation and its source must retain the same split, seed, member index and search/comparative phase. Only completed source-selected parameters may enter. Context-aware source loading does not waive whole-search, whole-comparative, calibration or outcome gates.

Cost-matched ordinary and factorized continuations are new controls with the same acquisition plus continuation budget. They must be labeled adaptations; they are not unchanged author-protocol reproductions. Source and continuation costs are reported together for comparisons and separately for diagnostics.

No human GO, new benchmark fit or predictive score is produced by this implementation. The new runtime remains unfrozen until engineering and final-master integration close.
