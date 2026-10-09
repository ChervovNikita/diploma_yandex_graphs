# Cached Gram projection: exact formulas and finite policy

Let t be the original target and let A contain the exact same nonzero risk rows normalized with the predecessor's `hypot` rule. Zero rows are omitted, never replaced. The closed cone is K={v:Av≤0}. Normalization by a positive row norm preserves that cone in real arithmetic.

For an active index set S, define G=A Aᵀ and q=A t. The predecessor solves G_SS λ=q_S with `_solve`, then reconstructs v=t-A_Sᵀλ. This successor computes G and q once and uses that identical small linear system for each subset.

The three compressed candidate tests follow by substitution:

- Multiplier condition: λ≥0 (the same inherited numerical admission is λ≥−tol).
- Full cone feasibility: A v = q-G_:S λ, compared to the same original tol.
- Squared distance: ||v-t||² = ||A_Sᵀλ||² = λᵀG_SSλ.

For exact nonsingular solves, active equality, multiplier nonnegativity and full feasibility are the KKT conditions for the unique Euclidean projection onto this convex cone. A dependent active support can be reduced to an independent support while preserving its conic normal combination, so enumerating all subsets retains an independent representation of the projection. This is the same active-set mechanism, not a new optimization method. The predecessor's positive pivot tolerance can skip ill-conditioned nonsingular systems; that existing finite policy is retained, not claimed to be an exact symbolic solver.

`_dot` and `_solve` bodies, Config/defaults, row validation/normalization, subset order, strict minimum-distance comparison and every score/pullback/direction/finite-step guard are unchanged. Gram symmetry mirrors the upper-triangle `_dot` result; multiplication and coordinate summation order for a reversed row pair give the same dot. The original tol1e-9 remains unchanged and caller-supplied original tolerances retain their API.

Only the chosen candidate is reconstructed with the predecessor's exact `x-fsum(lam*a[j])` expression. It is checked for finite coordinates, full-space `_dot(row,v)≤tol` for every normalized row, and finite original squared distance. Nonfinite compressed arithmetic or negative compressed squared distance raises; a compressed/full-space feasibility disagreement also raises. There is no clamp, retolerance, second candidate reconstruction, retry or invented zero direction. A genuine projected zero is still valid.

Finite arithmetic can alter feasibility or ranking near the inherited boundary because q-Gλ and λᵀGλ rearrange products/sums. Source mathematical equivalence is not bitwise trajectory equivalence; the explicit full check catches a chosen infeasible direction, and a bounded engineering check cannot certify every conditioning case. No numeric tolerance is changed to rescue a discrepancy.

With r≤11 rows and d private coordinates, full-space work is O(r²d) for the single Gram/target cache plus O(rd) for one reconstruction/check. Active enumeration and unchanged elimination use only≤11 dimensions. The predecessor recomputes full dots/reconstructs candidates across up to2^r subsets. The new source still stores the normalized full rows; its memory is not O(r²) alone.
