# Fixed seven-arm H16 frozen training-diagnostic CPU summary

Source preparation only. No prepared-source import/execution, Torch/numerical import, diagnostic-tensor load, data/label/checkpoint/served-prediction access, model/forward/autograd, staging or allocation contact occurred. The existing ordinary77 metadata schedule remains separate.

## Exact input and scope

The seven SHA/byte-bound inputs are the six B_diagnostics descriptors in the actual completed Amazon COMPLETE metadata and the utility B_diagnostics descriptor in the actual terminal/resource-closed H16 utility receipt. Root review/execution is required. The reader is hard disabled and the release template approval is false. All seven diagnostic files must pass exact custody before the first CPU weights_only=True load. There is no file discovery, dataset/evaluator/model import, arm/seed/pair selection or fallback input.

The source runs only at the allocation repository, hides CUDA in its own process, uses the already installed Torch2.1.2+cu118 with CPU deserialization, and verifies that CUDA was not initialized. It reads only these frozen training B/S/R artifacts. Copied terminal metadata contain descriptors, not checkpoint/prediction tensor contents. The reader produces one compact SUMMARY.json plus compact scalar stdout; raw Q/cost matrices remain on allocation.

## Fixed complete summaries

All seven arms, all16 episodes, all ten unordered class pairs and all four members are retained. Every pair retains its item/left/right/Q-entry counts for every episode. Series are keyed by fixed class pair and episode index; means/minima/maxima give equal weight to all160 pair-episode cells, with overlapping S-item denominators explicitly disclosed.

- Author scalars: centered response RMS, centered assignment-cost RMS, smooth cost scale, epsilon/scale, normalized cost RMS, Q entropy/deviation/relative RMS/minimum and row/column marginal residuals.
- Derived observed-response summaries: recorded centered response RMS/epsilon and RMS/sqrt(RMS²+epsilon²), with the fixed epsilon0.001. These are explicitly distinct from recorded margin/first-order cost normalization.
- Raw Q descriptions in CPUfloat64: entropy, relative RMS, maximum deviation and marginal residuals; exact nonuniform-row/entry counts versus0.25 retain raw denominators and use no threshold.
- Every episode's virtual query loss and four probe-own-CE changes, with fixed-member summaries.
- Author-recorded endpoint-versus-common400 shared L2 and all four private-member L2 distances. Per-episode displacement is marked unavailable because it is not stored. No checkpoint or state read can be used to reconstruct it.
- Only inside the utility episode/state, raw first-order versus paid finite-response matrices yield raw/centered difference RMS, centered maximum difference, RMS/epsilon and the two centered RMS values. No finite assignment solver is rerun. Cross-arm trajectory differences are not interpreted as Taylor approximation error.

The actual caller pins utility_control v1, and that is the operative schema bound here. A later v3 locator was encountered during source navigation but supplies no assumed actual diagnostic behavior. The author finite-response scalar rows and pair matrices are distinguished from current-margin costs and first-order utility costs throughout.

## Interpretation

This can describe whether responsibilities numerically differ from uniform, whether the recorded response scale is small relative to its smoothing epsilon, and how the first-order and paid finite costs differ at the same utility state. It cannot establish accuracy/generalization, useful specialization, net error repair, sharing benefit or novelty. A capable single's absorption of a forward bank limits a capacity-exclusivity claim; it does not reject possible optimization/inductive-bias utility of a precisely attributed training map. Such utility still requires reproducible complete quality beyond competent controls. No training/horizon/gate/selector or decision is changed by this reader.

## Admission and verification

ROOT_RELEASE_TEMPLATE keeps approval false and source-manifest/review authority unresolved. The exact disabled and anticipated flag-only source hashes are in SOURCE_MANIFEST. Root must adopt the reviewed flag-only source and a separate immutable CPU training-diagnostic release. No supervisor or new analysis framework is provided. Static verification uses only stdlib AST/hash/JSON and in-memory syntax compilation; it is not a numerical test or actual summary result.
