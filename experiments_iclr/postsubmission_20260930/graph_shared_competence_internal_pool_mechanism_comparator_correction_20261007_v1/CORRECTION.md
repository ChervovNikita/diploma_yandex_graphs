# Historical comparator identity clarification

7 October 2026. Read with the sealed synthesis v1; that packet is unchanged.

The v1 phrase **“jointly trained native four” refers specifically to completed39 cell `J4_end_joint`**, not to the separately trained independent-four external anchor or the later one-seed BPR joint-untied family.

Use this exact replacement for the historical comparison:

> In completed39, the ordinary shared control `E_end_joint` had saved mean VALID MRR 0.2969714303811391 versus 0.2915405531724294 for `J4_end_joint` (arm `ordinary_native4`, rule `ordinary_joint`, blocks b0/b1/b2), while having lower mean Hits10 and more common-negative burden.

The completed39 summary and its bound cohort specification agree on this identity. The supplied root shorthand “native independent four” for MRR0.291541 does not match those fields. `EXTERNAL_ANCHORS.json` separately identifies independently trained full encoders/heads and says their original training and independently selected constituents differ from the new episode-cycle joint controls. No assertion about the later BPR family's scores or recipe is made here; it was not reopened.

The v1 wording follows the completed39 conclusion, but the exact cell/rule wording above removes its ambiguity. The qualitative competent-yet-common-error motivation and prospective hypotheses are unchanged. No raw arrays, fitted states, current scores, new analysis, literature search, code, compute or variant were opened or created.

`IDENTITY_EVIDENCE.json` records exact saved fields and byte bindings. The MRR values are copied from existing summary fields, not recalculated.
