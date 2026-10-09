# Inactive SeHGNN grouped member factors

Concrete source adapter for exact SeHGNN HGB IMDB commit `e92bd37d0b803457339555684f139b4c8f3e160d`, model SHA256 `0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a`.

`grouped_member_factors.py` installs private unit r/s factors at six native affine sites: both grouped feature projection layers; semantic Q/K/V; semantic `fc_after_concat`. All original slow Parameters are shared by actual identity. Native heads/task residual blocks, normalization, activations, dropout, embedding dictionaries and zero-start semantic gamma remain intact. Modules and every registered buffer are separate member contexts. The original model file, native reference driver and already sealed source-supply helper are unchanged.

The adapter calls each exact native affine forward with factor-scaled input and adds a centered output correction keeping native bias outside the output factor. It casts effective factors to native activation/output dtype to preserve AMP dtype flow. There is no unit-only branch that would suppress initial factor derivatives.

Graph propagation remains the native precomputed feature/label aggregation. Private factors steer the learned semantic processor over those graph-derived channels; they do not learn per-edge transport or a live sheaf operator. Native source-family support and derived channels must be rebuilt/validated outside this adapter. Native biases and pointwise/remaining-source explanations remain visible.

The caller supplies the native initialized/placed prototype, installs the adapter, then constructs the optimizer. The installer performs no original model reset, slow RNG draw, dataset read, graph propagation, training, scoring or orchestration. Runtime defaults disabled. M1 is reserved for the full-native/unit qualification witness; the representative bank is M4.

See `DRIVER_INTERFACE.md`, `PRIVATE_SITE_JACOBIANS.md` and `NATIVE_QUALIFICATION_CONTRACT.md`. Actual source/probe Jacobians are site dependent: removed first-layer rows can lose probe dependence, later LayerNorm/semantic sites can retain it, and Q/K/V predictor derivatives vanish while gamma is zero. This is a specific known factor adaptation, not a universal adapter, new primitive, private graph geometry theorem or established performance result.

The source-only preparation used no dataset/model/server execution. Static source verification does not qualify native runtime or science. Root must first establish the credible full-task native reference, then qualify this concrete integration and fix a representative pilot before a scientific release.
