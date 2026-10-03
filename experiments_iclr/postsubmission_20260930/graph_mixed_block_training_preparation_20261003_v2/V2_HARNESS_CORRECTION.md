# V2: one recursive mapping-comparison correction

V1 remains immutable at manifest SHA256 `dcbd2d40cfc9fd4fb97ce8c56f21adc78004d6455304f8478e82975c3b3600af`. Its first root qualification failed in the harness before real graph access: `model.state_dict()` returned an `OrderedDict`, which skipped the exact-dict branch and fell into tensor-containing scalar equality.

V2 changes only the qualification comparator: dictionary-subclass dispatch uses `isinstance(left, dict)`, and the right mapping must have exactly the same concrete type as the left mapping. Existing key checks, recursive traversal, tensor comparisons and every other Python source remain unchanged. The exact diff and byte-bound preserved failure are in `V2_HARNESS_CORRECTION.json` and `V1_TO_V2_QUALIFICATION.diff`; the original remote-result metadata is copied byte-for-byte into the packet.

The global_BE model, losses, block gradient field, native own/own fit, optimizer/OneCycle, selector, checkpoints/RNG, seeds/splits, practical thresholds and scientific design are unchanged. The original README and prior source-custody metadata are preserved byte-for-byte as historical preparation records. The parent-provided completed CP outcome summary did not change this independently frozen design; `fresh_all40` remains the default when all ten baseline controls cannot qualify.

V2 is source preparation only. The existing stdlib static checker may compile and verify source custody; no numerical witness, resource probe, training or new fixture battery is run here. Releases remain false. Root's final reviewer independently inspects this harness correction before any new qualification release.
