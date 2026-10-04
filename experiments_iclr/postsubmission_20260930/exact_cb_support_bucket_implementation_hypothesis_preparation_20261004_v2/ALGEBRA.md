# Preserved exact law and changed dispatch

The mathematical proof and bucket decision are inherited from the pinned v1 `ALGEBRA.md`; no tuning decision is added here. This packet binds that implementation to unchanged core-v3 CB source.

For each active query, preserve its ordered real support, detached teacher pattern, k and r=min(k,n-k), differentiable first-slot centering and complement. Its NLL remains log e_r(exp a)-sum(a*b). A bucket contains only one exact r. Real slots occupy the original prefix; suffix logits are masked to -inf for ESP weights and suffix teacher bits to0 for the finite observed-score product. Therefore padded subsets have weight0, real subsets appear once, and real derivatives retain the same pi-z law. Because n>=2r, every retained state is reachable before padding, avoiding logaddexp(-inf,-inf). Masking and zero bits block repeated-anchor padded derivatives. r0 uses the original empty-slice connected zero.

All shared/separate/uniform mixture arithmetic and denominator/max-count reduction remain the byte-identical v3 body. The candidate only packs queries for the same vector recurrence. No real support, neural input, subset or query is discarded, sampled, recentered differently or cast to another precision.

For each exact r>=2, merge a power-of-two width W only if W is strictly smaller than the sum of distinct original n descriptors. Otherwise retain direct groups. Thus total recurrence slot loops never increase; this actual metadata estimate is132447→13609. Padding increases arithmetic, group peak and retained graph work. Real-arithmetic preservation is not bitwise or universal finite-precision equivalence. The unchanged float32/64 independent-oracle tolerances and future native resource limits remain required.
