"""One shared warm native encoder and four independent full native NCN heads.

No numerical imports, construction, inputs or fitting occur at import time.
"""
from copy import deepcopy

CONDITIONS = ("shared_full_heads",)
RMS = None
EXPECTED_ENCODER = 948225
EXPECTED_HEAD = 463362


def make(torch, native, heads, models, warm, condition, seed, factor_seed, device, variations):
    if condition != "shared_full_heads" or variations:
        raise ValueError("Only the unchanged common-warm full-head control is admitted")

    class SharedFullHeads(native.nn.Module):
        member_count = 4

        def __init__(self, encoder, predictors):
            super().__init__()
            self.encoder = encoder
            self.heads = native.nn.ModuleList(predictors)

        def forward(self, features, support, positive, negative, route=None):
            if route is not None and (type(route) is not int or route not in range(4)):
                raise ValueError("Full native head route must be None or integer0..3")
            h = self.encoder(features, support)
            members = range(4) if route is None else (route,)
            return (torch.cat([self.heads[i](h, support, positive) for i in members], 1),
                    torch.cat([self.heads[i](h, support, negative) for i in members], 1))

    model = SharedFullHeads(deepcopy(warm.encoder),
                            [deepcopy(warm.predictor) for _ in range(4)]).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    ownership = ownership_check(torch, model, warm)
    names = dict(model.named_parameters())
    shared = [name for name in names if name.startswith("encoder.")]
    private = [name for name in names if name.startswith("heads.")]
    if set(shared) | set(private) != set(names) or set(shared) & set(private):
        raise ValueError("Full-head ownership is not exhaustive and disjoint")
    partition = {
        "arm": condition, "shared": shared, "private": private,
        "parameters": [{"name": name, "shape": list(value.shape), "numel": value.numel(),
                        "role": "shared" if name in shared else "private"}
                       for name, value in names.items()],
        "buffers": [{"name": name, "shape": list(value.shape), "role": "native_constant_buffer"}
                    for name, value in model.named_buffers()],
        "ownership": ownership,
        "initialization": "same_common_TRAIN_only_warm_encoder_once; four_full_native_head_copies",
        "all_dense_and_private_parameters_trainable": True,
        "inner_query_exposure": "each_full_head_uses_its_own_inner_route_list_as_F4",
        "optimizer": "ONE_joint_Adam_for_shared_encoder_and_all_four_full_heads",
        "selection": "first_maximum_complete_pooled_VALID_MRR_rounded4",
        "serving": "arithmetic_mean_raw_logits_as_initializer_pilot",
        "rank_one_factors": False, "factor_perturbation_RMS": None,
    }
    return model, partition


def ownership_check(torch, model, warm):
    """Runtime native identity/count/storage assertions; root executes this gate."""
    blocks = [model.encoder, *model.heads]
    if len(model.heads) != 4 or type(model.encoder) is not type(warm.encoder):
        raise ValueError("One native encoder and four native heads are required")
    if any(type(head) is not type(warm.predictor) for head in model.heads):
        raise ValueError("Full head must remain the exact native predictor class")
    counts = [sum(value.numel() for value in block.parameters()) for block in blocks]
    if counts != [EXPECTED_ENCODER] + [EXPECTED_HEAD] * 4:
        raise ValueError("Native encoder/full-head counts differ: " + str(counts))
    parameters = [value for block in blocks for value in block.parameters()]
    if len({id(value) for value in parameters}) != len(parameters):
        raise ValueError("Encoder/head parameter objects are aliased")
    pointers = [value.untyped_storage().data_ptr() for value in parameters]
    if len(set(pointers)) != len(pointers):
        raise ValueError("Encoder/head parameter storage is aliased")
    donor_pointers = {value.untyped_storage().data_ptr() for value in warm.parameters()}
    if set(pointers) & donor_pointers:
        raise ValueError("Control aliases the donor warm state")
    if any(not value.requires_grad for value in parameters):
        raise ValueError("Every shared encoder and full-head parameter must learn")
    for block, donor in zip(blocks, [warm.encoder] + [warm.predictor] * 4):
        actual, expected = block.state_dict(), donor.state_dict()
        if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in actual):
            raise ValueError("Native warm parameter/buffer copy differs")
    return {"encoder_copies": 1, "native_head_copies": 4,
            "encoder_parameters": counts[0], "parameters_per_full_head": counts[1],
            "all_private_head_parameters": sum(counts[1:]), "total_parameters": sum(counts),
            "parameter_objects_and_storage_disjoint": True,
            "no_donor_storage_alias": True, "native_warm_state_equal": True,
            "all_parameters_trainable": True, "factor_parameters": 0}


def gradient_check(torch, steps, model, x, support, inner, outer, streams):
    """One root-authorized TRAIN episode, replay only; no optimizer/RNG commit.

    Require every own route to connect its full head and the single encoder,
    leaving other heads unused. Then check joint mean-own and pooled+own paths.
    """
    if len(inner) != 4:
        raise ValueError("Qualification requires the existing four-route episode")
    parameters = dict(model.named_parameters())
    names, values = list(parameters), tuple(parameters.values())

    def own_loss(p, n):
        return -torch.nn.functional.logsigmoid(p).mean() - torch.nn.functional.logsigmoid(-n).mean()

    losses, records = [], []
    for member, queries in enumerate(inner):
        p, n = steps.functional_forward(torch, model, parameters, x, support, queries,
            training=True, stream=(streams, member), route=member, advance=False)
        if p.shape[1] != 1 or n.shape[1] != 1:
            raise ValueError("Own-route qualifier must emit one head")
        loss = own_loss(p, n)
        gradients = torch.autograd.grad(loss, values, retain_graph=True, allow_unused=True)
        active = 0
        for name, gradient in zip(names, gradients):
            expected = name.startswith("encoder.") or name.startswith("heads." + str(member) + ".")
            if expected:
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise ValueError("Disconnected/nonfinite own-route gradient: " + name)
                active += 1
            elif gradient is not None:
                raise ValueError("Own-route loss reaches a different full head: " + name)
        losses.append(loss)
        records.append({"member": member, "active_parameter_tensors": active,
                        "encoder_and_own_head_connected": True, "other_heads_unused": True})
    joint = torch.stack(losses).mean()
    joint_gradients = torch.autograd.grad(joint, values)
    if not bool(torch.isfinite(joint)) or any(not bool(torch.isfinite(g).all()) for g in joint_gradients):
        raise ValueError("Mean own loss has a nonfinite joint gradient")
    del losses, joint, joint_gradients, gradients, loss, p, n
    p, n = steps.functional_forward(torch, model, parameters, x, support, outer, training=False)
    if p.shape[1] != 4 or n.shape[1] != 4:
        raise ValueError("Joint outer qualifier must emit all four heads")
    pooled_own = .5 * own_loss(p.mean(1), n.mean(1)) + .5 * own_loss(p, n)
    outer_gradients = torch.autograd.grad(pooled_own, values)
    if not bool(torch.isfinite(pooled_own)) or any(not bool(torch.isfinite(g).all()) for g in outer_gradients):
        raise ValueError("Pooled plus own loss has a nonfinite joint gradient")
    return {"scope": "one_existing_masked_TRAIN_episode_before_fitting",
            "own_routes": records, "mean_own_all_parameters_connected": True,
            "pooled_plus_own_all_parameters_connected": True,
            "optimizer_steps": 0, "dropout_stream_advance": False,
            "shared_encoder_copies": 1, "all_head_parameters_independent": True}
