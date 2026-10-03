"""Prospective synthetic native references; source prepared, never run here."""
from dataclasses import asdict
from types import SimpleNamespace
from native_source.adapter import build_all_layer_polyformer, parameter_algebra


def unequal_factors(model, *, signed=False):
    """Fixed engineering values at every site; no warm/study initialization rule."""
    import torch
    with torch.no_grad():
        for site_id, name in enumerate(model.site_names):
            site = model.core.get_submodule(name)
            for factor_id, factor in enumerate((site.R, site.S)):
                coordinates = torch.arange(factor.numel(), dtype=factor.dtype, device=factor.device).reshape_as(factor)
                value = .7 + .5 * torch.sin(coordinates * .37 + site_id * .19 + factor_id * .41).square()
                if signed:
                    value = torch.where((coordinates.to(torch.long) + site_id + factor_id) % 5 == 0, -value, value)
                factor.copy_(value)
            if site.shared.bias is not None:
                coordinates = torch.arange(site.shared.bias.numel(), dtype=site.shared.bias.dtype, device=site.shared.bias.device)
                site.shared.bias.copy_(.13 + .03 * torch.cos(coordinates + site_id))


def native_heterogeneous_reference(spec, tokens):
    """Complete unwrapped author paths with effective W_m and scaled b_m.

    functional_call keeps factor/shared/input gradients alive; it substitutes
    effective affine tensors without changing any native forward. Each member
    gets its own independently constructed unwrapped author module.
    """
    import torch
    from torch.func import functional_call
    from native_source.native_polyformer_outer import PolyFormer
    model = build_all_layer_polyformer(spec, seed=17,
                                      composition_receipt="engineering_unequal_reference_only").eval()
    unequal_factors(model, signed=True)
    x = tokens.detach().clone().requires_grad_()
    native_members = []
    for _ in range(4):
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            native_members.append(PolyFormer(None, SimpleNamespace(**asdict(spec))).eval())
    common = {name: parameter for name, parameter in model.core.named_parameters()
              if not any(name == site + suffix for site in model.site_names
                         for suffix in (".R", ".S", ".shared.weight", ".shared.bias"))}
    references = []
    for member, native in enumerate(native_members):
        effective = dict(common)
        for name in model.site_names:
            site = model.core.get_submodule(name)
            effective[name + ".weight"] = site.S[member, :, None] * site.shared.weight * site.R[member, None, :]
            if site.shared.bias is not None:
                effective[name + ".bias"] = site.S[member] * site.shared.bias
                assert (site.shared.bias != 0).all() and (site.S[member] != 1).any()
        assert set(effective) == set(dict(native.named_parameters()))
        references.append(functional_call(native, effective, (SimpleNamespace(list_mat=x.unbind(1)),), strict=True))
    oracle = torch.stack(references)
    actual = model(x)
    assert actual.shape == oracle.shape == (4, tokens.shape[0], spec.num_classes)
    assert torch.allclose(actual, oracle, atol=1e-5, rtol=1e-4)
    coefficients = torch.cos(torch.arange(actual.numel(), dtype=actual.dtype, device=actual.device)).reshape_as(actual)
    parameters = tuple(model.parameters())
    left = torch.autograd.grad(actual.square().mean() + (actual * coefficients).mean(), (x,) + parameters)
    right = torch.autograd.grad(oracle.square().mean() + (oracle * coefficients).mean(), (x,) + parameters)
    for a, b in zip(left, right):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.allclose(a, b, atol=2e-5, rtol=2e-4)
    permutation = torch.tensor([2, 0, 3, 1], device="cpu")
    with torch.no_grad():
        for name in model.site_names:
            site = model.core.get_submodule(name)
            site.R.copy_(site.R.index_select(0, permutation))
            site.S.copy_(site.S.index_select(0, permutation))
        permuted = model(tokens)
    assert torch.equal(permuted, actual.detach().index_select(0, permutation))
    # Every affine family and both factor types: another member must be bitwise
    # unchanged even if a perturbation lies on a dead ReLU coordinate.
    for name in model.site_names:
        for factor_name in ("R", "S"):
            factor = getattr(model.core.get_submodule(name), factor_name)
            with torch.no_grad():
                saved = factor[1].clone()
                factor[1].mul_(1.07)
                changed = model(tokens)
                factor[1].copy_(saved)
            assert torch.equal(changed[[0, 2, 3]], permuted[[0, 2, 3]])
    algebra = parameter_algebra(spec, 4)
    assert sum(p.numel() for p in model.parameters()) == algebra["complete_parameters"]
    model.begin_private_continuation(warm_bank_receipt="synthetic_not_a_fitted_warm_bank")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == algebra["stage_b_private_parameters"]


def native_split_reset_reconstruction(spec, tokens):
    import torch
    from core.transaction import StateSnapshot
    from train_roles import select_native_train_column
    mask = torch.zeros((7, 3), dtype=torch.bool, device="cpu")
    mask[[0, 4, 6], 1] = True
    selected = select_native_train_column(mask, node_count=7, expected_split_count=3,
                                         official_split_id=1, release_and_mask_receipt="synthetic_unequal_axes_only")
    assert selected.ids == (0, 4, 6)
    assert torch.equal(mask.permute(1, 0)[1], mask[:, 1])
    assert mask[1].shape != mask[:, 1].shape
    model = build_all_layer_polyformer(spec, seed=17, composition_receipt="synthetic_checkpoint_only").eval()
    unequal_factors(model)
    model.begin_private_continuation(warm_bank_receipt="synthetic_nonone_checkpoint")
    custody = model.stage_b_custody(audit_receipt="synthetic_reset_audit",
                                  optimizer_resolution_receipt="parent_modified_step_successor_rule")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    before = StateSnapshot.capture(model, optimizer, custody)
    reset_calls = [model.core.reset_parameters]
    for block in model.core.attn:
        reset_calls += [block.attnmodule.reset_parameters, block.ffnmodule.ffn_net.reset_parameters]
    for reset in reset_calls:
        try:
            reset()
        except RuntimeError as error:
            assert "reconstruct" in str(error)
        else:
            raise AssertionError("Post-wrap reset must fail before changing any stored warm state")
        assert before.exact_equal(StateSnapshot.capture(model, optimizer, custody))
    replay = build_all_layer_polyformer(spec, seed=17, composition_receipt="synthetic_checkpoint_only")
    replay.load_state_dict(before.model_state, strict=True)
    # State_dict does not encode mode, permissions or warm/custody receipts.
    replay.begin_private_continuation(warm_bank_receipt="synthetic_nonone_checkpoint")
    replay_custody = replay.stage_b_custody(audit_receipt="synthetic_reset_audit",
                                          optimizer_resolution_receipt="parent_modified_step_successor_rule")
    assert replay_custody.private_site_source_receipt == custody.private_site_source_receipt
    assert replay_custody.mutable_state_audit_receipt == custody.mutable_state_audit_receipt
    assert replay_custody.optimizer_resolution_receipt == custody.optimizer_resolution_receipt
    assert replay_custody.moment_policy == custody.moment_policy and replay_custody.zero_policy == custody.zero_policy
    assert tuple((h.name, h.source_receipt) for h in replay_custody.external_hooks) == tuple(
        (h.name, h.source_receipt) for h in custody.external_hooks)
    assert replay.stage == model.stage and replay.warm_bank_receipt == model.warm_bank_receipt
    assert replay_custody.private_parameter_names == custody.private_parameter_names
    assert {n for n, p in replay.named_parameters() if p.requires_grad} == set(custody.private_parameter_names)
    with torch.no_grad():
        assert torch.equal(replay(tokens), model(tokens))
