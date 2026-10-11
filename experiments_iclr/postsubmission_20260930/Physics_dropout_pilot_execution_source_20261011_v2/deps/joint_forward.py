"""Unexecuted native TABM wrapper: one controlled dropout slot, four native IID slots.

No constructor, data loader, numerical import, pilot or execution admission.
Caller supplies native model/torch/functional and a persistent mask sampler.
"""

PLACEMENTS = ("stem_pre_gelu", "first_normalized_sage_input")


def route_ids(model, *, multisample_single=False):
    """True four-row TABM, or four stochastic copies of one TABM1 body."""
    members = model.input_be_block.tabm_inits
    if multisample_single and members == 1:
        return (0, 0, 0, 0)
    if not multisample_single and members == 4:
        return (0, 1, 2, 3)
    raise ValueError("Use native TABM4 or one unchanged TABM1 multisample body")


def prepare_joint_masks(model, x, members, *, placement, torch, mask_sampler):
    """Detach four deterministic prefixes at current weights; return multipliers.

    Recompute these prefixes with autograd in each later native route. This adds
    four prefix forwards while allowing ordinary sequential suffix/backward.
    The native model is not modified. No label enters mask sampling.
    """
    if placement not in PLACEMENTS:
        raise ValueError("Unknown fixed placement")
    if model.model_name != "SAGE" or len(model.residual_modules) != 2:
        raise ValueError("This source supports the bound two-block native SAGE")
    if not model.dropout.training or model.dropout.p == 0:
        return None
    with torch.no_grad():
        values = []
        for member in members:
            h = model.input_be_block(x, tabm_seed=member)
            if placement == "first_normalized_sage_input":
                h = model.act(h)  # original stem dropout is moved, not duplicated
                h = model.residual_modules[0].normalization(h)
            values.append(h)
        values = torch.stack(values, dim=0)
        return mask_sampler(values, p=model.dropout.p, training=True,
                            site=placement)


def native_route_forward(model, graph, x, member, *, placement, torch,
                         multiplier=None):
    """One native route with original stem or relocated first graph-input dropout.

    multiplier is a cached detached node/channel mask divided by keep probability.
    None calls the actual native Dropout module, useful for TABM1 and genuine I4.
    The other four Dropout calls are inside unchanged native SAGEModule objects.
    """
    if placement not in PLACEMENTS:
        raise ValueError("Unknown fixed placement")
    if placement == "stem_pre_gelu" and multiplier is None:
        return model(graph, x, tabm_seed=member)
    h = model.input_be_block(x, tabm_seed=member)
    if placement == "stem_pre_gelu":
        h = model.act(h * multiplier)
        remaining = model.residual_modules
    else:
        h = model.act(h)
        first = model.residual_modules[0]
        normalized = first.normalization(h)
        normalized = (model.dropout(normalized) if multiplier is None
                      else normalized * multiplier)
        # SAGEModule uses this same tensor in its conv and explicit self concat.
        # Its native FFN executes both original IID Dropout calls. Identity h
        # is undropped: only the normalized residual-branch input is changed.
        h = h + first.module(graph, normalized)
        remaining = model.residual_modules[1:]
    for residual in remaining:
        h = residual(graph, h)
    h = model.output_normalization(h)
    return model.output_be_block(h, tabm_seed=member).squeeze(1)


def joint_forward(model, graph, x, *, placement, torch, mask_sampler,
                  multisample_single=False):
    """Four raw paths; intended for admitted no-grad readout, not low-memory training.

    In eval mode masks are off. Four-copy TABM1 outputs are identical; do not
    count them as four independent members or four parameter sets. Single-body
    selectors/serving call native_route_forward once, rather than this function.
    """
    members = route_ids(model, multisample_single=multisample_single)
    masks = prepare_joint_masks(model, x, members, placement=placement,
                                torch=torch, mask_sampler=mask_sampler)
    return torch.stack([
        native_route_forward(model, graph, x, member, placement=placement,
                             torch=torch, multiplier=None if masks is None else masks[r])
        for r, member in enumerate(members)
    ], dim=0)


def train_step_joint_routes(model, graph, x, train_nodes, train_targets, optimizer,
                            *, placement, torch, functional, mask_sampler,
                            multisample_single=False):
    """Four own CEs /4; sequential backwards, one native AdamW update.

    Current parameters stay fixed until all four backward calls finish. Detached
    masks have no graph; recomputation permits backward without retain_graph.
    TABM1/I4 singles use their full own loss externally, not this /4 helper.
    """
    model.train()
    optimizer.zero_grad()
    members = route_ids(model, multisample_single=multisample_single)
    masks = prepare_joint_masks(model, x, members, placement=placement,
                                torch=torch, mask_sampler=mask_sampler)
    total_loss = 0.0
    for r, member in enumerate(members):
        out = native_route_forward(model, graph, x, member, placement=placement,
                                   torch=torch,
                                   multiplier=None if masks is None else masks[r])
        loss = functional.cross_entropy(out[train_nodes], train_targets.long()) / 4
        loss.backward()
        total_loss += loss.item()
        del out, loss
    optimizer.step()
    return total_loss
