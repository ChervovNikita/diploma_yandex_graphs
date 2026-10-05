# Complete local synthetic episode qualification

The fixed V2 successor completed all nine engineering checks at the original synthetic state and full native architecture. It changes only float64 derivative displacements to 1e-6 and 1e-7, after the separately preserved same-state diagnosis; the original tolerances and all remaining checks are unchanged. This is a retrospectively informed local derivative check. The original coarse-step run remains FAIL_ENGINEERING.

The shared directional errors are 1.004e-10 and 4.763e-9. Stopped-Q and independent fixed-Q derivatives match every shared coordinate exactly. The unused local-head finite tangent is zero. The complete native public episode and independent recomputation from the original private state agree exactly; all 9,112,064 shared derivative coordinates are included. Native parameters, modes, buffers, inputs, RNG and the temporary port gate are restored. Constructed next states are discarded.

The worker took 32.981 seconds and peaked at 5,012,553,728 bytes RSS. No dataset labels, fits, persistent parameter updates, scoring or prediction-quality evidence are supplied. Synthetic local consistency does not qualify full-graph memory/time, float32 feasibility, or all-direction/global differentiability. A complete full-graph engineering check remains necessary before any new candidate training.

While preparing full mode, root also found a path integration defect: the qualifier passes the Git repository to an accessor whose artifact paths are relative to the research phase. A separate disabled V3 repair is being prepared before actual full execution. No data copy, alias or path workaround is used.
