"""Exact source parameter formulas and deterministic widths; stdlib only."""
TASKS = {
    "Reddit2": dict(nodes=232965, features=602, classes=41, index=0),
    "ogbn-products": dict(nodes=2449029, features=100, classes=47, index=1),
}
ARMS = ("shared_mean", "detached_shared_mean", "exact_member", "single_D64", "single_matched")
SEEDS = (17, 29, 43)
DIM, MEMBERS, LAYERS, MAX_SINGLE_WIDTH = 64, 4, 2, 256
RECIPE = dict(hidden_width=64, members=4, layers=2, dropout=0.2,
              ffn_multiplier=1, optimizer="AdamW", lr=0.001, weight_decay=0.0001,
              betas=[0.9, 0.999], eps=1e-8, dtype="float32", AMP=False, TF32=False,
              train_loss="mean_own_member_cross_entropy", prediction="mean_raw_logits")
DROPOUT_POLICY = dict(key_type="propagation_cost_impl_v1.dropout.DropoutKey",
    domain="gnnm-propagation-dropout-v1", task_indices={"Reddit2": 0, "ogbn-products": 1},
    seeds=list(SEEDS), epoch="one-based complete AdamW update counter",
    member="0..3 for ensembles, 0 for singles", layer="-1 input; 0..1 residual blocks",
    sites=["input", "ffn_first", "ffn_second"],
    private_generator_per_member_site=True, training_colors_used=False)


def ordinary_count(features, classes, width):
    if any(type(v) is not int or v <= 0 for v in (features, classes, width)):
        raise ValueError("Positive integer source dimensions required")
    return 5 * LAYERS * width**2 + width * (features + classes + 3 + 5 * LAYERS) + classes


def ensemble_count(features, classes):
    ordinary_count(features, classes, DIM)
    return (features * DIM + DIM * classes + MEMBERS * features
            + 3 * MEMBERS * DIM + 2 * MEMBERS * classes
            + LAYERS * (5 * DIM**2 + 5 * DIM) + 2 * DIM)


def matching_width(features, classes):
    target = ensemble_count(features, classes)
    width = min(range(1, MAX_SINGLE_WIDTH + 1),
                key=lambda d: (abs(ordinary_count(features, classes, d) - target), d))
    actual = ordinary_count(features, classes, width)
    # If the best boundary were at the maximum, a larger search could matter.
    if width == MAX_SINGLE_WIDTH and ordinary_count(features, classes, width) < target:
        raise ValueError("Parameter target exceeds explicitly bounded single constructor")
    return dict(width=width, target_parameters=target, single_parameters=actual,
                signed_mismatch=actual - target, absolute_mismatch=abs(actual - target),
                relative_mismatch=(actual - target) / target,
                rule="min(abs(single_count-target), width), integer widths 1..256")


def plan(task):
    spec = TASKS[task]
    target = ensemble_count(spec["features"], spec["classes"])
    match = matching_width(spec["features"], spec["classes"])
    return dict(task=task, dimensions=spec, matched_single=match,
                dropout_policy=DROPOUT_POLICY,
                parameter_counts={a: (ordinary_count(spec["features"], spec["classes"], DIM)
                    if a == "single_D64" else match["single_parameters"] if a == "single_matched" else target)
                    for a in ARMS},
                symbolic_graph_channels_per_layer={"shared_mean_forward": DIM,
                    "shared_mean_adjoint": DIM, "detached_shared_mean_forward": DIM,
                    "detached_shared_mean_neighbour_adjoint": 0, "exact_member_forward": MEMBERS * DIM,
                    "exact_member_adjoint": MEMBERS * DIM, "single_D64": DIM,
                    "single_matched": match["width"]},
                member_state_float32_bytes=spec["nodes"] * MEMBERS * DIM * 4,
                common_message_float32_bytes=spec["nodes"] * DIM * 4,
                symbolic_only=True, measured_feasibility=False,
                parameters_do_not_match_activation_or_wall_cost=True)


def variants():
    result = []
    for arm in ARMS:
        for execution in (("packed", "serial") if arm == "exact_member" else ("packed",)):
            for checkpoint in (False, True):
                result.append(dict(arm=arm, execution=execution, checkpoint_blocks=checkpoint))
    return result


def variant_id(value):
    if value not in variants():
        raise ValueError("Outside the fixed outcome-independent qualification variants")
    return "%s__%s__checkpoint%d" % (value["arm"], value["execution"], value["checkpoint_blocks"])
