# Qualification input repair

The first qualification exited with KeyError: backbone before model or data construction. The original SAGE runner implied its backbone and its configuration omitted the key. The new generic runner requires it explicitly.

Preserve CONFIG_V1_FAILED_QUALIFICATION.json, the initial freeze, source, terminal failure receipt and empty server output. The only recipe input correction is backbone = SAGE. No training, scientific score, model construction or TEST read occurred. The corrected qualification uses a fresh v2 source, freeze and output. Original study criteria and sealed learning code remain unchanged.
