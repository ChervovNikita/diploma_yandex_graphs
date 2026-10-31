# Prospective correction for the two constant-incidence NSD methods

The exact failed callbacks were localized without an optimizer update. All degree matrices were positive diagonal blocks (minimum above 1.579), all source singular-value masks admitted all four values, and all matrix-power output adjoints were finite. Nonfinite input adjoints matched repeated singular-value nodes exactly, with no bad nonrepeated node. Only per-layer constant-incidence `raw` parameters received NaNs. The original absolute training jitter remains present but rounds to eigenvalue collisions at large float32 scalar degrees.

For positive definite symmetric A=U diag(lambda) U^T, the derivative of f(A)=A^(-1/2) is U (K * (U^T E U)) U^T. The divided difference has the continuous expression

`K_ij = -1 / (sqrt(lambda_i) * sqrt(lambda_j) * (sqrt(lambda_i) + sqrt(lambda_j)))`.

It equals `f'(lambda_i) = -.5 lambda_i^(-3/2)` at repeated values. The adjoint uses the symmetric incoming matrix gradient. It therefore differentiates the matrix function without differentiating an arbitrary singular-vector basis.

The prospective implementation retains the exact original SVD forward expression and the original normalization function code, including all source jitter draws, degree matrices, sparse indices, clamps, initializers, factors, Adam and objective. Only the inverse-square-root first derivative is replaced for `layer_shared_incidence` and `node_capacity_right`. It is installed on each owned builder instance; passed factories and the original native source files remain unchanged. The guard rejects non-SPD, non-symmetric, other-power, or source-masked spectra. No jitter magnitude, seed, recipe or finite check is changed. Higher derivatives are unsupported by this prospective first-order correction.

CPU proof is required before root source review. A corrected actual-input repeat is not authorized by staging this proposal. The original failed v2 attempts remain failures; successful future qualification would belong to this new implementation revision.
