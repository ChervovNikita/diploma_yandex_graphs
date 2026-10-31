"""Unexecuted synthetic numerical checks; no provider/data import or access."""


def run(device, *, tolerance, boundary):
    import torch
    from torch.nn import functional as F
    from types import SimpleNamespace
    from propagation_cost_engineering_v1.complete_models import CompleteCSR, NoAdjointCSR, CompleteFamily, OperationLedger
    from propagation_cost_impl_v1.dropout import DropoutKey
    from propagation_cost_impl_v1._pinned.frozen_centered_paths import PlainSAGE
    from propagation_cost_impl_v1.models import SourcePlainSAGE
    from propagation_cost_engineering_v1.complete_models import construct_source_arm
    from .counts import ARMS, ordinary_count, ensemble_count, matching_width
    from .models import construct, shared_aggregation, forward_logits, train_step, make_optimizer, pooled_inference, split_logits
    checks = []
    boundary("synthetic_imports_complete")
    def close(name, actual, expected, atol=None, rtol=None):
        torch.testing.assert_close(actual, expected,
            atol=tolerance["atol"] if atol is None else atol,
            rtol=tolerance["rtol"] if rtol is None else rtol)
        checks.append(dict(check=name, passed=True,
            maximum_absolute_error=float((actual.detach() - expected.detach()).abs().max().cpu())))
        boundary(name)
    def check(name, value):
        if not value:
            raise ValueError("Synthetic check failed: " + name)
        checks.append(dict(check=name, passed=True))
        boundary(name)
    generator = torch.Generator(device=device).manual_seed(20260930)
    edges = torch.tensor([[0, 1, 2, 0, 3, 1, 4, 5, 5], [1, 1, 1, 2, 2, 3, 3, 4, 5]], device=device)
    dense = torch.zeros(7, 7, device=device)
    for dst in range(7):
        ids = edges[0, edges[1] == dst]
        if ids.numel():
            dense[dst].index_add_(0, ids, torch.full((ids.numel(),), 1 / ids.numel(), device=device))
    metadata = dict(nodes=7, canonical_edges=int((dense != 0).sum()), P_edge_only_fingerprint="synthetic-only")
    def family(detached=False):
        ledger = OperationLedger()
        operator = (NoAdjointCSR("full", dense.to_sparse_csr(), ledger, 7) if detached else
                    CompleteCSR("full", dense.to_sparse_csr(), dense.T.to_sparse_csr(), ledger, 7))
        return CompleteFamily({"full": operator}, metadata, ledger, device)
    spec = dict(features=7, classes=3)
    features = torch.randn(7, 7, generator=generator, device=device)
    train_ids, targets = torch.tensor([0, 2, 4, 6], device=device), torch.tensor([0, 1, 2, 0], device=device)
    key = DropoutKey(0, 17, 1)
    for name, call in (("old_fixture_width_guard_preserved", lambda: SourcePlainSAGE(7, 3, dim=65)),
                       ("old_seed_guard_preserved", lambda: construct_source_arm(spec, "single_base", optimizer_seed=17))):
        try:
            call()
        except ValueError:
            check(name, True)
        else:
            raise ValueError("Original guard unexpectedly weakened: " + name)
    models = {arm: construct(spec, arm).to(device) for arm in ARMS}
    for arm, model in models.items():
        expected = (ordinary_count(7, 3, 64) if arm == "single_D64" else matching_width(7, 3)["single_parameters"]
                    if arm == "single_matched" else ensemble_count(7, 3))
        check("runtime_parameter_count_" + arm, sum(p.numel() for p in model.parameters()) == expected)
    for arm in ("shared_mean", "detached_shared_mean"):
        check("paired_initialization_" + arm, all(torch.equal(value, models["exact_member"].state_dict()[name])
            for name, value in models[arm].state_dict().items()))
    for training in (False, True):
        models["shared_mean"].train(training)
        models["detached_shared_mean"].train(training)
        first = forward_logits(models["shared_mean"], family(), features, dropout_key=key)
        second = forward_logits(models["detached_shared_mean"], family(True), features, dropout_key=key)
        close("shared_detached_same_forward_training%d" % training, first, second)
        close("keyed_dropout_reproducible_training%d" % training, first,
              forward_logits(models["shared_mean"], family(), features, dropout_key=key))
    # Double precision local Jacobian: entire common-message channel and stop.
    matrix = dense.double()
    local = torch.randn(7, 4, 3, dtype=torch.float64, generator=generator, device=device, requires_grad=True)
    upstream = torch.randn(7, 4, 3, dtype=torch.float64, generator=generator, device=device)
    class DenseOperator:
        def __call__(self, values):
            return matrix @ values
    local_family = SimpleNamespace(full=DenseOperator(), ledger=OperationLedger())
    shared = shared_aggregation(local_family, local)
    derivative = torch.autograd.grad((shared * upstream).sum(), local)[0]
    expected = (matrix.T @ upstream.sum(dim=1) / 4)[:, None, :].expand_as(local)
    close("differentiable_common_message_full_adjoint", derivative, expected, 1e-12, 1e-12)
    stopped = shared_aggregation(local_family, local, detached=True)
    check("detached_common_message_has_no_neighbor_state_autograd", stopped.requires_grad is False)
    zero = torch.autograd.grad(((stopped + local * 0) * upstream).sum(), local)[0]
    close("whole_neighbor_gradient_zero_not_cross_only", zero, torch.zeros_like(zero), 0, 0)
    individual = torch.einsum("ij,jmd->imd", matrix, local)
    check("shared_forward_differs_from_exact_member_messages", bool((individual - shared).abs().max() > 1e-6))
    # Downstream learned maps and the root path survive whole-neighbor stop.
    stopped_model = models["detached_shared_mean"].train()
    train_step(stopped_model, family(True), features, train_ids, targets, make_optimizer(stopped_model), dropout_key=key)
    boundary("detached_training_step_complete")
    check("detached_neighbor_learned_maps_train", all(block.neighbor.weight.grad is not None
          and bool(block.neighbor.weight.grad.abs().sum() > 0) for block in stopped_model.blocks))
    check("detached_root_learned_maps_train", all(block.root.weight.grad is not None
          and bool(block.root.weight.grad.abs().sum() > 0) for block in stopped_model.blocks))
    for arm in ARMS:
        ordinary = construct(spec, arm).to(device).train()
        checkpointed = construct(spec, arm, checkpoint_blocks=True).to(device).train()
        optimizer, checkpoint_optimizer = make_optimizer(ordinary), make_optimizer(checkpointed)
        train_step(ordinary, family(arm == "detached_shared_mean"), features, train_ids, targets, optimizer, dropout_key=key)
        boundary("ordinary_training_step_" + arm)
        train_step(checkpointed, family(arm == "detached_shared_mean"), features, train_ids, targets, checkpoint_optimizer, dropout_key=key)
        boundary("checkpoint_training_step_" + arm)
        for (name, value), (other_name, other) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
            check("checkpoint_parameter_name_" + arm + "_" + name, name == other_name)
            close("checkpoint_gradient_" + arm + "_" + name, value.grad, other.grad)
            close("checkpoint_AdamW_update_" + arm + "_" + name, value, other)
        check("one_AdamW_step_" + arm, all(float(state["step"].cpu()) == 1 for state in optimizer.state.values()))
    for checkpoint in (False, True):
        packed = construct(spec, "exact_member", checkpoint_blocks=checkpoint).to(device).train()
        serial = construct(spec, "exact_member", checkpoint_blocks=checkpoint).to(device).train()
        train_step(packed, family(), features, train_ids, targets, make_optimizer(packed), dropout_key=key)
        boundary("packed_exact_training_step_checkpoint%d" % checkpoint)
        train_step(serial, family(), features, train_ids, targets, make_optimizer(serial), execution="serial", dropout_key=key)
        boundary("serial_exact_training_step_checkpoint%d" % checkpoint)
        for (name, value), (_, other) in zip(packed.named_parameters(), serial.named_parameters()):
            close("exact_serial_gradient_checkpoint%d_%s" % (checkpoint, name), value.grad, other.grad)
            close("exact_serial_AdamW_update_checkpoint%d_%s" % (checkpoint, name), value, other)
        packed.eval()
        logits = forward_logits(packed, family(), features)
        close("serial_raw_pool_checkpoint%d" % checkpoint, pooled_inference(packed, family(), features, execution="serial"), logits.mean(0))
        member, pooled = split_logits(packed, family(), features, train_ids, execution="serial")
        close("serial_copy_before_pool_member_checkpoint%d" % checkpoint, member.to(device), logits[:, train_ids])
        close("serial_copy_before_pool_pooled_checkpoint%d" % checkpoint, pooled.to(device), logits.mean(0)[train_ids])
    for arm in ("single_D64", "single_matched"):
        single = construct(spec, arm).to(device).eval()
        plain = PlainSAGE(7, 3, dim=single.dim, layers=2, dropout=.2).to(device).eval()
        plain.load_state_dict(single.state_dict(), strict=True)
        close("ordinary_plain_source_block_forward_" + arm, forward_logits(single, family(), features), plain(lambda x: dense @ x, features))
    return dict(schema="gnnm-shared-message-synthetic-checks-v2", status="passed", device=device,
                checks=checks, check_count=len(checks), synthetic_only=True,
                real_data_or_heldout_labels_loaded=False, predictive_scores_computed=False,
                whole_detach_does_not_identify_offdiagonal_coupling=True)
