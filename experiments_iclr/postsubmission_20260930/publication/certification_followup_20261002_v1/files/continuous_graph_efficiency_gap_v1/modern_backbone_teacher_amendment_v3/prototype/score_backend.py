"""Explicit deterministic CPU APS postprocessor, preserving native score formula.

No neural forward/backward, precision, tolerance or selection change. Do not
disable deterministic algorithms or use warn_only. Copies/CPU work are charged.
SOURCE ONLY: never imported/executed in child authoring.
"""
import sys

POLICY = 'randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2'


def install():
    import aligned_score_correction as aligned
    if not hasattr(aligned, '_modern_native_randomized_aps'):
        aligned._modern_native_randomized_aps = aligned.randomized_aps
    native = aligned._modern_native_randomized_aps

    def cpu_aps(probabilities, uniforms):
        device = probabilities.device
        # Preserve dtype and the native stable tie/cumsum/randomization formula.
        # Device copies remain differentiable if a future caller needs gradients.
        return native(probabilities.to('cpu'), uniforms.to('cpu')).to(device)

    aligned.randomized_aps = cpu_aps
    # A previously imported CF module may hold its original from-import alias.
    cf = sys.modules.get('cf_gnn_source_adapter')
    if cf is not None:
        cf.randomized_aps = cpu_aps
    return POLICY
