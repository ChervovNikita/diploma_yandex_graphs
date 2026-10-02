"""Meaningful qualification hooks for a future authorized runtime; unexecuted."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ports import (MEMBERS, BOUNDARY_CAP, FIXED_WIDTH, MIMO_WIDTH,
    native_parameter_count, mimo_parameter_count, member_loss, optimizer_for)


def check_parameter_budget(model):
    """Run immediately after construction, before any labels or validation."""
    actual = sum(p.numel() for p in model.parameters())
    if model.specification["family"] == "cached_token_mimo_graph_port":
        width, step = MIMO_WIDTH, 4
        expected = mimo_parameter_count(width)
        next_count = mimo_parameter_count(width + step)
    else:
        width = FIXED_WIDTH[model.backbone]
        step = 4 if model.backbone == "polyformer_mono" else 8
        expected = MEMBERS * native_parameter_count(model.backbone, width)
        next_count = MEMBERS * native_parameter_count(model.backbone, width + step)
    cap = BOUNDARY_CAP[model.backbone]
    if actual != expected or not (expected <= cap < next_count):
        raise AssertionError(f"Fixed complete-source budget mismatch: actual={actual}, "
            f"expected={expected}, cap={cap}, next legal={next_count}")
    return dict(actual_parameters=actual, cap=cap, fixed_width=width,
        next_legal_width_parameters=next_count)


def _reference_parameters(reference):
    return [dict(member.named_parameters()) for member in reference.models]


def _assert_member_gradients(packed, references, *, rtol, atol):
    for name, key in packed._parameter_keys.items():
        stacked = packed.stacked[key]
        for member, native in enumerate(references):
            grad = native[name].grad
            if (stacked.grad is None) != (grad is None):
                raise AssertionError(f"Gradient availability mismatch: {name}, member{member}")
            if grad is not None:
                torch.testing.assert_close(stacked.grad[member], grad, rtol=rtol, atol=atol)


def _assert_adam_state(packed, references, packed_optimizer, reference_optimizer, *, rtol, atol):
    for name, key in packed._parameter_keys.items():
        stacked = packed.stacked[key]
        state = packed_optimizer.state.get(stacked, {})
        for member, native in enumerate(references):
            param = native[name]
            torch.testing.assert_close(stacked[member], param, rtol=rtol, atol=atol)
            other = reference_optimizer.state.get(param, {})
            if set(state) != set(other):
                raise AssertionError(f"Adam state availability mismatch: {name}")
            for field in state:
                # Adam step metadata is shared ONLY for synchronized members.
                value = state[field] if field == "step" else state[field][member]
                torch.testing.assert_close(value, other[field], rtol=rtol, atol=atol)


def check_independent_equivalence(reference, packed, graph, train, *, rtol=1e-5, atol=1e-7):
    """Mutates two supplied copies; deterministic outputs/gradients/two Adam steps.

    Dropout is disabled through eval mode while autograd/Adam remain enabled.
    Both Photo stages are checked with one continuing synchronized optimizer.
    RNG-identical stochastic training and runtime efficiency are not certified.
    """
    if reference.backend != "sequential" or packed.backend != "vmap":
        raise ValueError("Supply explicit sequential and vmap copies of identical members")
    if reference.backbone != packed.backbone:
        raise ValueError("Backbones differ")
    check_parameter_budget(reference)
    check_parameter_budget(packed)
    reference.eval()
    packed.eval()
    refs = _reference_parameters(reference)
    opt_ref, opt_packed = optimizer_for(reference), optimizer_for(packed)
    stages = (False, True) if packed.backbone == "polynormer_r" else (False,)
    rows = []
    for global_stage in stages:
        reference.set_global_stage(global_stage)
        packed.set_global_stage(global_stage)
        for step in range(2):
            opt_ref.zero_grad(set_to_none=True)
            opt_packed.zero_grad(set_to_none=True)
            a, b = reference(graph), packed(graph)
            torch.testing.assert_close(b, a, rtol=rtol, atol=atol)
            member_loss(a, train).backward()
            member_loss(b, train).backward()
            _assert_member_gradients(packed, refs, rtol=rtol, atol=atol)
            opt_ref.step()
            opt_packed.step()
            _assert_adam_state(packed, refs, opt_packed, opt_ref, rtol=rtol, atol=atol)
            rows.append(dict(global_stage=global_stage, adam_step=step + 1,
                max_absolute_logit_difference=float((a.detach() - b.detach()).abs().max())))
    return dict(backend="vmap", checks=rows, deterministic_equivalence=True,
        dropout_replay_equivalence=False, runtime_measured=False)


def check_member_independence(packed, graph, train, *, global_stage=False):
    """A member-only parameter perturbation and a member-only gradient probe."""
    packed.eval()
    packed.set_global_stage(global_stage)
    head = ("core.lin3.bias" if packed.backbone == "polyformer_mono" else
        "core.pred_global.bias" if global_stage else "core.pred_local.bias")
    key = packed._parameter_keys[head]
    with torch.no_grad():
        before = packed(graph).clone()
        original_bias = packed.stacked[key][0, 0].clone()
        packed.stacked[key][0, 0].add_(0.125)
        after = packed(graph)
        torch.testing.assert_close(before[1:], after[1:], rtol=0, atol=0)
        if torch.equal(before[0], after[0]):
            raise AssertionError("Target member did not respond to its private head bias")
        packed.stacked[key][0, 0].copy_(original_bias)
    packed.zero_grad(set_to_none=True)
    logits = packed(graph)
    F.cross_entropy(logits[2, train.nodes], train.labels).backward()
    seen = False
    for parameter in packed.parameters():
        if parameter.grad is not None:
            seen = seen or bool((parameter.grad[2] != 0).any())
            if bool((parameter.grad[[0, 1, 3]] != 0).any()):
                raise AssertionError("One member's supervised loss reached another member's parameters")
    if not seen:
        raise AssertionError("The selected member has no nonzero supervised derivative")
    return dict(private_perturbation=True, private_selected_member_gradients=True)


def check_mimo_correspondence(graph, train, positions, tuples):
    """Detect slot reorder, token truncation and compact-label/global-row confusion."""
    torch.testing.assert_close(tuples.positions, positions, rtol=0, atol=0)
    torch.testing.assert_close(tuples.nodes, train.nodes[positions], rtol=0, atol=0)
    torch.testing.assert_close(tuples.labels, train.labels[positions], rtol=0, atol=0)
    feature_count = graph.teacher_input.shape[-1]
    for slot in range(MEMBERS):
        expected = graph.teacher_input[train.nodes[positions[:, slot]]]
        actual = tuples.tokens[..., slot * feature_count:(slot + 1) * feature_count]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return dict(complete_rows=True, compact_tuple_labels_match=True, slots=MEMBERS)
