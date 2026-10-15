# One NCN CUDA lifecycle check

Run the already prepared common factual-context recipe on the dedicated GPU
after the three-fit molecular BN queue has ended and its GPU process has
exited. The older GRU worker may remain active. Maximum work is unchanged:
two native constructions, three optimizer transitions (including the one
replayed transition), and four complete VALID passes. No TEST or scientific
training fit is involved. A 20-minute watchdog stops only this child if needed.

This successor fixes an unexecuted check's demand for bit-identical repeated
CUDA computations. Loading the stored checkpoint must still restore native
parameters, optimizer, sampler and RNG state exactly. IDs, role order, integer
state and original parameter identities also remain exact checks. Repeated
floating computations receive the existing project diagnostic thresholds
rtol=1e-4 and atol=1e-5, with actual maximum/RMS differences and counts recorded.
An outside-threshold result remains a warning, not a passed check; it does not
trigger a rerun or threshold adjustment. Method source is unchanged.

Before launch require at least 32 GiB GPU memory free, at least 12 GiB
cache-aware host/cgroup headroom after the BUDDY 8 GiB growth reserve, and no
own active GPU process other than the existing GRU PID. This is admission for
one check, not a measured two-worker campaign capacity or an ETA.
