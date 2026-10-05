# Source proof and derivative ownership

The only allocation substitution is c_FO[p,i,m]=-eta_probe * D_phi b[p,i,m][g_m], where g_m=partial_phi mean own CE_S and b=softplus(z_competitor-z_target). Both class-pair directions and all original physical private R/S/B coordinates are retained. No shared coordinate, cosine, clipping, rectification, gradient normalization or preconditioner appears in the dot product. Eta=.01 remains inside utility; the fixed epsilon=.001 prevents treating scale cancellation as exact.

## Matrix-free value extraction

For one original-state native forward, define independent dummy vectors a_p shaped like each complete pair-margin vector. Ordinary reverse AD computes p=partial_phi sum_p <a_p,b_p> with create_graph=True. The second dummy reverse computes partial_a sum_coordinate p*g_detached = D_phi b[g], and the source multiplies by -eta_probe. This avoids an item-by-parameter Jacobian. The own-CE backward retains that same native forward graph until the dummy reverse; the g values are detached only in this value calculation. A mathematically zero dummy anchor keeps fully disconnected zero directions defined. It does not establish native second-AD support: independent coordinate parity and nontrivial native support checks remain mandatory.

The before logit values and actual own-CE gradient are reused for the paid private probe, followed by the original native after forward. Actual finite softplus response and CE diagnostics are retained. Utility raw banks and observed finite banks have separate fields; centered_response_rms remains the observed finite response, while centered_cost_rms/scale describe the substituted utility.

## Exact live utility credit

The inherited complete original centering/RMS/eight-step graph map returns the raw-cost cotangent t. Hold this cotangent fixed for a VJP, and write h=partial_phi sum_{p,i} t[p,i,m]*b[p,i,m]. One original-state native forward supplies both original own CE and weighted margins. The source constructs BOTH g and h with create_graph=True, then differentiates -eta_probe * sum_coordinate g*h to shared theta and inspection-only private phi. Product rule gives

(D_theta c_FO)^T t = -eta_probe*((D_theta h)^T g+(D_theta g)^T h).

Neither gradient factor is detached in this credit helper; only the VJP's already computed t is held fixed. Private inspection gives the analogous two private Hessian products. A zero parameter anchor adds no mathematical derivative. The result remains a full mixed-Hessian update: first order describes the probe-step approximation, not Hessian-free outer learning.

## Unchanged composite map

A subclass of the exact SHA-bound reviewed sequential engine reuses its input/partition validation, independent-Q main private partial, detached main/query value passes, original joint all-member own/probability-pool query cotangents, direct member theta/Q/private VJPs, and complete raw-to-Q map VJPs. It replaces only the raw response value construction and finite cost-credit helper with the utility versions. It never edits or monkeypatches the original source/classes/control set.

The theta-time direct and utility shared credits are accumulated before the original .001 shared SGD step. The inherited response then recomputes own gradients/utility/Q/main partial at returned theta+ from self.phis, which are the ORIGINAL input rows. The virtual probe and theta-time adapted state never become committed private inputs. Only the original .01 independent-Q main private SGD result commits; outer-phi inspection is never a commit gradient. Episode output detaches remain. The existing served arithmetic probability mean remains the future serving policy; no serving/evaluator code is introduced.

All original G0 coefficients, own-CE anchor, extra-margin normalization, graph gamma/entropy/hard-balance/eight finite steps, public native callback and supplied class-pair conventions remain. The utility is a Taylor approximation to finite response only along a smooth bounded-Hessian segment; branch crossings/kinks retain existing AD/FD limitations, and normalization/Q can amplify or suppress the remainder. This source establishes neither a numeric pass nor a predictive effect.
