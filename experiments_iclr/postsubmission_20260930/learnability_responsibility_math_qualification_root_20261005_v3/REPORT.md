# Engineering check of the corrective-learning operator

The repaired operator passed a CPU check on a small deterministic nonlinear tensor fixture under the existing project Torch2.1.2 runtime. This verifies implementation arithmetic and derivatives. It uses no scientific dataset and supplies no evidence that a predictive method works.

The independent plain-autograd calculation agrees exactly with the intended private gradient when responsibilities are held fixed. The shared-core directional gradient agrees with central finite differences: absolute error is 1.04e-8, 9.35e-10 and 1.05e-10 for step sizes 1e-3, 3e-4 and 1e-4. Stopping responsibility credit agrees exactly with differentiating a fixed-responsibility function; the live responsibility path has a nonzero gradient contribution in this fixture.

Four assignment fixtures, including one-item and one-member degeneracies, retain positive cells and row/column residuals at most 2.22e-16. The declared objective decreases in the two nontrivial fixtures. These observations do not prove universal floating-point stability, native sparse derivatives, optimizer suitability or graph generalization.

Two preceding attempts are preserved. V1 could not import Torch from the isolated native-NCN virtual environment without the existing repository package path. V2 used the normal project package path, then exposed an unsupported constant creation from a functorch-wrapped tensor. The repaired source subtracts the scalar `math.log(M)`, preserving the mathematical probability-average loss. V3 reruns the unchanged fixture; it changes no scientific recipe or old result. No installation, host setting or mount changed.

Measured fixture calculation took 0.625 seconds, excluding import/SSH overhead. Source remains disabled. Whole-native member-forward reconstruction, sparse affinity equivalence, full-graph higher-order/resource checks and the prospectively fixed TRAIN-only response comparison are still required before any empirical claim.
