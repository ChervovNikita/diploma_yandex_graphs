# Complete native PENCIL resource result

The single authorized probe passed on `anogena-2-0` and singleton GPU `GPU-44039938-fd82-41d2-fefd-de71514e2fac`. It traversed all 7,740 TRAIN queries in 31 minibatches, then all 102,255 unique VALID queries in 400 minibatches and restored all 113,727 original occurrences. Native data, source and workload settings were preserved. Parameters stayed byte-identical; outputs and gradients were finite. No optimizer updates, scientific fits, ranking metrics, saved scores, checkpoints or TEST access occurred.

| Resource | Observed peak | Authorized ceiling |
| --- | ---: | ---: |
| CUDA allocated | 9.395 GiB | 24 GiB |
| CUDA reserved | 20.178 GiB | 28 GiB |
| Probe and loader-descendant RSS | 41.615 GiB | 128 GiB |

TRAIN forward/backward took 17.85 seconds. Complete VALID and inverse restoration took 93.38 seconds. Probe timing was 124.73 seconds; the owned child, including pre-probe admission/startup, took 141.41 seconds. This run was co-resident with the existing large GPU job, so these are observed costs under contention, not uncontended throughput claims. No existing job was signalled.

There were 22,693,889 trainable parameter elements. Two FP32 Adam moments would add 181,551,112 bytes, but no moments or optimizer-step scratch were allocated. The prior provisional dispatch rule (peak reservation + 16 bytes per trainable element + 2 GiB safety) gives 22.516 GiB of required freshly observed physical free memory. That arithmetic does not establish full-fit readiness: a future separately authorized scientific fit must validate actual moment allocation and finite state at its first ordinary update, preserve its bounds thereafter, and record any later failure.

Repeating exactly this epoch-0 TRAIN/VALID workload 300 times would arithmetically take 9.27 hours, before Adam updates, scientific saves or later-epoch variability. It is not a completion-time forecast. Validation dominates the observed cost.

The initial missing-directory staging failure is preserved. Read-only evidence showed that no probe child had started; the staging continuation then launched the one authorized probe. No resource retry or recipe downgrade occurred. Complete logs and the 7,620-file dependency binding stay on the server; only small receipts and source are copied locally.
