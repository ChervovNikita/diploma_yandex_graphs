# C2 native node bridge

This separate port applies the frozen C2 assignment to the actual native PolyFormer node pipeline. It imports the pinned pure table helper from `structural_class_responsibility_v1`; molecular sources remain unchanged.

For each observed TRAIN class, incoming degree comes from the actual post-source-preprocessing `data.edge_index`. Each target edge record contributes one, retaining duplicates and present self-loop records. This port adds no graph edges, loops, coalescing or propagation normalization. The full public transductive graph may contribute topology; only source TRAIN IDs and labels define assignments.

The helper assigns equal fractional quartile overlap to tied counts. It gives each route exactly one quarter of each class's assignment mass, with positive weights `w=.5+2q`, member/class weight mass equal to class size and per-node mean member weight one. The matched control permutes each whole four-route weight vector among TRAIN nodes within its class, using fit seed+7030031 without redraw. All ties may make the assignment uninformative; no artificial diversity is introduced.

Shared native parameters receive the source unweighted CE gradient F; boundary factors receive the fixed-table weighted CE gradient W. Both use the complete native TRAIN reduction, with all reverse collections at the same old state and one source Adam transition. The original operators, dimensions, parameters, optimizer groups/options, native activation checkpointing and mean-logit VALID readout are unchanged. F/W are logged through a session-local delegate to the unchanged training rules. They do not define a claimed global scalar objective. Later shared trajectories may differ because earlier private updates differ.

The launcher must bind the absolute actual repository root explicitly and pass this package's frozen manifest SHA. State records bind method, helper, source, topology, TRAIN IDs/classes/counts, exact rational table, fit seed and split. Restore refuses mismatched state; all native Parameter identities are retained.

Qualification scope is finite CPU engineering only: the actual source CiteSeer-Cheb graph, two complete TRAIN updates per variant, full VALID before/after fresh-session checkpoint restore, two CPU threads and hidden CUDA. No Physics/link/GPU fits, TEST scoring, convergence or predictive claim is implied. Other node cells and full workers/paired resources/final serving still require their own qualification. Scientific comparisons must be prospectively frozen; this package supplies no new campaign GO.

Source generation v2 repairs the v1 guard that incorrectly expected molecular wrapper module names. The failed v1 check and its two native constructions with zero optimizer attempts remain preserved. The actual node wrapper is `native` plus the original two-bank `banks` ModuleList, with exact lin1/lin3 bindings and four-member factors. No numerical rule changed.
