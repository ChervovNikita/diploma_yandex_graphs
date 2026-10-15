# Prospective molecular BN follow-up

This frozen exploratory follow-up adds **three** scientific fits, one for each
existing MolBACE/GIN seed101,202,303. It reuses the complete parent study's own
base-GNNM and native-single/independent4 controls. It does not train duplicates.
The only change is route-private BN running state with shared affine parameters.
The same native model, LR/WD, mean-own loss and complete-epoch selection apply.

The question follows the source mechanism in the catalogue. Early parent VALID
observations were already available when this follow-up was frozen; it is
development, not independent confirmation. Every paired trial and failure is
required. TEST remains closed, and no latent/state difference counts as a
quality improvement. There is no novelty or conference-acceptance claim.

`SPEC.json` specifies all three jobs, the immutable source pins and parent control
cohorts. `FREEZE.json` binds it before new training. `supervisor.py` waits at most
36hours for the parent24-fit queue to complete successfully, then starts only
one new GPU child at a time. A parent failure prevents launch. A child failure
stops admission, with actual costs retained and no automatic retry. Starting
this bounded supervisor is separate from the giant campaign's pending GO.

Prepared only until a distinct launch receipt exists. The current24-fit queue,
its code, controls, selection and source hashes remain unchanged. Final paired
analysis is performed from the immutable selected VALID exports after all three
candidate fits and all parent controls are complete.
