"""One callable fabricated CPU component check. No loader, fit or update.

Root supplies authenticated existing modules/helpers. This function has not
been executed by this source preparation. Run only under the separate bounded
root correctness-check authorization, with CUDA hidden before Torch imports.
"""
import torch
from single_control import make_control


def check_components(mods, graph_ops, make_native, adjacency_factory, teacher_type, loss_core):
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise ValueError("This bounded correctness check requires CUDA hidden")
    model, optimizer = make_control(mods, 0, torch.device("cpu"), graph_ops, adjacency_factory, make_native)
    # Six-node observed fixture. Three final records are hidden before
    # coalescing: both counterpart labels and the query edge. Neither side's
    # observed count nor labels enter the predictive model.
    pairs = torch.tensor([[0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [3, 5],
                          [1, 2], [0, 4], [0, 1]], dtype=torch.long)
    hidden_records = torch.tensor([6, 7, 8], dtype=torch.long)
    graph = graph_ops.Graph.mask_train_batch(pairs, hidden_records, 6)
    adjacency = adjacency_factory(graph)
    features = torch.sin(torch.arange(6 * 128, dtype=torch.float32)).reshape(6, 128)
    queries = torch.tensor([[0, 1], [2, 5], [2, 4]], dtype=torch.long)
    expected_neighbors = mods["native_utils"].adjoverlap(
        adjacency, adjacency, queries.T, calresadj=True, cnsampledeg=-1, ressampledeg=-1)
    actual_neighbors = graph_ops.enumerate_neighbors(graph, queries)
    for sparse, coordinate in zip(expected_neighbors,
                                  (actual_neighbors.common, actual_neighbors.left, actual_neighbors.right)):
        row, node, _ = sparse.coo()
        assert torch.equal(row, coordinate[0]) and torch.equal(node, coordinate[1])
    rows = model.auxiliary_emission.weight.detach()
    pair_differences = (rows[:, None] - rows[None, :]).abs().sum(dim=-1)
    assert bool((pair_differences[~torch.eye(4, dtype=torch.bool)] > 0).all())
    parity = {}
    for training in (False, True):
        model.train(training)
        h = model.encode(features, graph)
        before = torch.get_rng_state().clone()
        reference = model.decoder(h, adjacency, queries.T).flatten()
        reference_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        actual, _ = model.query_forward(h, graph, queries, emit_auxiliary=training)
        actual_rng = torch.get_rng_state().clone()
        # PyTorch's standard dtype-specific correctness tolerances, with no
        # predictive threshold or new optimizer/search parameter.
        torch.testing.assert_close(actual, reference)
        assert torch.equal(actual_rng, reference_rng)
        parity["train" if training else "eval"] = {
            "scalar_close": True, "CPU_RNG_equal": True,
            "maximum_absolute_difference": float((actual.detach() - reference.detach()).abs().max())}
    model.eval()
    h = model.encode(features, graph)
    scalar, detail = model.query_forward(h, graph, queries[:1], emit_auxiliary=True)
    teacher = teacher_type.from_train(pairs, 6)
    slot_rows, bits = teacher.labels(queries[:1], detail["neighbors"])
    left_rows, _ = detail["neighbors"].left
    right_rows, _ = detail["neighbors"].right
    assert torch.equal(slot_rows, torch.cat((left_rows, right_rows)))
    labels = loss_core.TrainPatternLabels(bits.to(detail["eta_left"].dtype),
                                         "complete_TRAIN_observation_membership")
    values = loss_core.training_pattern_losses(detail["eta_left"], detail["eta_right"],
                                              left_rows, right_rows, labels, 1)
    assert values["teacher_counts"] == [[1], [1]]
    named = {
        "encoder.convs[0].lin.weight": model.encoder.convs[0].lin.weight,
        "decoder.xlin[0].weight": model.decoder.xlin[0].weight,
        "decoder.xcnlin[0].weight": model.decoder.xcnlin[0].weight,
        "decoder.xijlin[0].weight": model.decoder.xijlin[0].weight,
        "decoder.lin[0].weight": model.decoder.lin[0].weight,
        "auxiliary_emission.weight": model.auxiliary_emission.weight}
    gradients = torch.autograd.grad(values["J_K"].mean(), tuple(named.values()), retain_graph=True)
    assert all(bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any()) for gradient in gradients)
    main = -torch.nn.functional.logsigmoid(scalar).mean()
    blocked = torch.autograd.grad(main, (model.auxiliary_emission.weight,
                                       detail["native_left_score"], detail["native_right_score"]), allow_unused=True)
    assert all(gradient is None for gradient in blocked)
    assert not optimizer.state and all(parameter.grad is None for parameter in model.parameters())
    return {"status": "PASS_FABRICATED_CPU_COMPONENT_CHECK", "fixture_nodes": 6,
            "support_coordinates_equal": True, "native_scalar_parity": parity,
            "distinct_auxiliary_weight_rows": True, "finite_nonzero_auxiliary_gradient_tensors": list(named),
            "main_completion_score_and_auxiliary_head_gradients_absent": True,
            "optimizer_steps": 0, "real_dataset_checkpoint_outcome_reads": 0}
