"""Unexecuted source candidate for the frozen K1 final-head selector closure.

No imports of numerical packages occur until the explicit binder is called.
Use only after separate equivalence qualification and verified source loading.
This candidate contains no warm, optimizer, trial, continuation, or launch API.
"""
from __future__ import annotations

import copy


SELECTOR_SHA256 = "03544a2b480875cc3894b3186ab98d2d8c90d8d1d2959b19b1cba59fcb445ccb"
BOUNDARY_SHA256 = "699b606ead00cb7bdd9be6cd58730a0687157c40cf594af620d1edc4c93b6bac"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def bind_cached_head_only(model, backbone, model_args, selector_api, boundary_api):
    """Capture one frozen eval prehead; preserve only head.R differentiation.

    Inputs are a disposable, unchanged K1 wrapper and the exact verified modules
    from the v3 qualification loader/driver. Binding and calls require the native
    FP32 context without autocast or inference mode. Graph inputs, topology,
    source, stage and all upstream state are fixed for this binding's lifetime.
    Rebind from the required native endpoint whenever that snapshot changes.
    """
    import torch

    # Explicit no-argument context APIs supported by the bound Torch 2.1.2 runtime.
    is_inference_mode_enabled = getattr(torch, "is_inference_mode_enabled", None)
    is_cuda_autocast_enabled = getattr(torch, "is_autocast_enabled", None)
    is_cpu_autocast_enabled = getattr(torch, "is_autocast_cpu_enabled", None)
    _require(callable(is_inference_mode_enabled),
             "Torch 2.1.2 query is_inference_mode_enabled() unavailable")
    _require(callable(is_cuda_autocast_enabled),
             "Torch 2.1.2 query is_autocast_enabled() unavailable")
    _require(callable(is_cpu_autocast_enabled),
             "Torch 2.1.2 query is_autocast_cpu_enabled() unavailable")

    def check_context():
        _require(not is_inference_mode_enabled(), "AD needs ordinary tensors")
        _require(not is_cpu_autocast_enabled()
                 and not is_cuda_autocast_enabled(), "Native no-autocast context required")

    check_context()
    _require(getattr(selector_api, "__executed_sha256__", None) == SELECTOR_SHA256,
             "Require the exact verified executed v3 selector")
    _require(getattr(boundary_api, "__graph_curvature_executed_sha256__", None) == BOUNDARY_SHA256,
             "Require the exact verified executed boundary source")
    wrappers = {"polyformer_mono": boundary_api.PolyFormerBoundaryFamily,
                "polynormer_r": boundary_api.PolynormerBoundaryFamily}
    _require(backbone in wrappers and type(model) is wrappers[backbone],
             "Only the exact bound native boundary wrapper is supported")
    _require(all(not module._forward_pre_hooks and not module._forward_hooks
                 and not module._backward_hooks for module in model.modules()),
             "Do not bind a wrapper with external module hooks")
    _require(all(torch.is_tensor(value) and not value.requires_grad for value in model_args),
             "Cache only fixed native graph inputs")

    # Reuse every original identity, K1, final-stage, factor and eval-mode guard.
    theta0, uncached_logits, binding = selector_api.bind_head_only(model, backbone, model_args)
    _require(theta0.dtype == torch.float32, "Exact native FP32 head snapshot required")
    name, width = selector_api.HEADS[backbone]
    head = getattr(model, name.rsplit(".", 1)[0])
    _require(type(head) is boundary_api.BoundaryProjector and head.members == 1,
             "Exact final K1 BoundaryProjector required")
    captured = []

    def capture_prehead(module, args):
        _require(module is head and len(args) == 2 and args[1] == 0,
                 "Need the final predictive member-zero head call")
        x = args[0]
        _require(tuple(x.shape) == (model_args[0].shape[0], width),
                 "Final prehead must be [N, final-head input width]")
        # This hook runs while the original strict functional_call has installed
        # its frozen parameters and fresh buffers; snapshot those exact values.
        captured.append((x.detach().clone(memory_format=torch.preserve_format),
                         {key: value.detach().clone() for key, value in module.named_parameters()},
                         {key: value.detach().clone() for key, value in module.named_buffers()}))

    hook = head.register_forward_pre_hook(capture_prehead)
    try:
        # no_grad, not inference_mode: the cache can be an ordinary AD constant.
        with torch.no_grad():
            reference = uncached_logits(theta0)
    finally:
        hook.remove()
    _require(len(captured) == 1, "Final predictive head must be reached exactly once")
    prehead, frozen_head, buffers = captured[0]
    _require(set(frozen_head) == {"weight", "R", "S", "B"} and not buffers,
             "Bound final-head state inventory changed")
    _require(prehead.dtype == theta0.dtype == reference.dtype
             and prehead.device == theta0.device == reference.device,
             "Native prehead, head slice and logits dtype/device must agree")
    expected_shape = tuple(reference.shape)
    # Isolate the callable and its Python attributes from subsequent caller edits.
    frozen_module = copy.deepcopy(head)
    del uncached_logits, reference, captured

    def logits_fn(theta):
        check_context()
        _require(tuple(theta.shape) == (width,), "Head slice shape changed")
        _require(theta.dtype == theta0.dtype and theta.device == theta0.device,
                 "Native head slice dtype/device changed")
        replacements = dict(frozen_head)
        replacements["R"] = theta.view_as(frozen_head["R"])
        fresh_buffers = {key: value.clone() for key, value in buffers.items()}
        # Execute the unchanged projector: x*R -> F.linear -> *S -> +B.
        member = torch.func.functional_call(frozen_module, (replacements, fresh_buffers),
                                            (prehead, 0), strict=True)
        result = torch.stack([member], dim=0)
        _require(tuple(result.shape) == (1, *expected_shape) and result.shape[2] >= 2,
                 "Need the original final predictive K1 class logits shape")
        return result[0]

    binding = dict(binding, candidate="cached_final_prehead_v1",
                   cache_shape=list(prehead.shape), cache_full_k1_forwards=1,
                   cache_scope="one_frozen_eval_K1_snapshot",
                   derivative_scope="final_head_R_only",
                   numerical_equivalence_pending=True, runtime_qualification_inherited=False)
    return theta0, logits_fn, binding
