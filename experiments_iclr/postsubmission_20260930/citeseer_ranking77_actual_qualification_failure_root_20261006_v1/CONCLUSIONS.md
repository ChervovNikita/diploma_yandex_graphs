# First 18.77 ranking qualification failed

Both assigned numerical qualifiers completed their first three-pass ordinary update and failed the original committed-parameter comparison. They exited1 with observed terminal closure, no signals and no cap violations. The frozen reference tolerance remains unchanged. No full TRAIN cost cycle, validation read or full fit started. The detached owner stopped before fit admission.

The discrepancy has not been causally diagnosed. The inspected CUDA sources use different FP32 expressions for SoftplusBackward and sigmoid, and Adam can amplify gradient differences near zero. These observations motivate a matched-output/first-pass-gradient diagnostic; they do not prove rounding caused this failure. All raw failures and costs are preserved.

The connection and GPU availability checks succeeded. This is a numerical qualification failure, not a connection or GPU availability failure. A new corrected execution requires a separately reviewed source and fresh output paths; no silent rerun or tolerance relaxation is authorized.
