# Native link-geometry checks: NCN only

All three frozen link geometry recipes passed real NCN/Collab CPU lifecycle
checks. Native operators, source minibatches and positive/negative reductions
were retained. Each method had two actual updates, three complete VALID calls
(60,084 positives plus 100,000 negatives) and fresh-state reconstruction.

| Frozen recipe | Updates | Complete VALID calls | Wall seconds | Peak RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| link-same-query-cosine-private | 2 | 3 | 62.19 | 9.71 |
| link-global-opposite-label-private | 2 | 3 | 66.42 | 9.62 |
| link-endpoint-contrast-private | 2 | 3 | 66.18 | 9.41 |

Every first update had an active auxiliary over 632 authenticated anchors.
The global and endpoint variants each used 1,104 matched alternatives; the
cosine control had no denominator alternatives. Existing preheads were used
with no extra native forward, and final boundary factors had no direct
auxiliary path. Native task gradients still trained those factors.

The counts total six optimizer transitions, six native constructions and nine
full VALID calls. Whole TRAIN epochs, convergence, predictive comparisons,
NCNC/BUDDY, CUDA and final TEST serving are not qualified by these checks.
No scientific fit, TEST score, method promotion or automatic retry occurred.

CPU contention with the NCNC and bounded runtime qualifiers was disclosed in
LAUNCH.json. Runtime caches were reclaimed normally; no host setting or cache
was changed. Full raw results and checkpoints remain inside the server repo.
READINESS.json contains compact costs, active exposure and exact result hashes.
