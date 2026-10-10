"""Fixed positive class-normalized weights and explicit shared/private gradients."""

WEIGHT_SEED_OFFSET = 6000119
MEMBERS, TRAIN_NODES, CLASSES = 4, 580, 10


def draw_weights(np, seed, ids, labels):
    """One local RNG table per family seed; no global RNG or outcome access."""
    if ids.shape != (TRAIN_NODES,) or labels.shape != ids.shape or ids.dtype != np.int64 or labels.dtype != np.int64:
        raise ValueError("Ordered int64 TRAIN580 IDs/labels required")
    if len(np.unique(ids)) != TRAIN_NODES or set(labels.tolist()) != set(range(CLASSES)):
        raise ValueError("Unique TRAIN IDs and every TRAIN class required")
    generator_seed = int(seed) + WEIGHT_SEED_OFFSET
    generator = np.random.default_rng(generator_seed)
    raw = generator.exponential(1., size=(MEMBERS, TRAIN_NODES))
    if raw.dtype != np.float64 or not np.isfinite(raw).all() or not (raw > 0).all():
        raise FloatingPointError("Every fixed Exp(1) draw must be finite and strictly positive; no redraw")
    weights, classes = raw.copy(), []
    for label in range(CLASSES):
        mask = labels == label
        count = int(mask.sum())
        denominator = raw[:, mask].sum(axis=1)
        if count == 0 or not np.isfinite(denominator).all() or not (denominator > 0).all():
            raise FloatingPointError("Every TRAIN class needs finite positive route mass")
        weights[:, mask] = raw[:, mask] * (count / denominator[:, None])
        mass = weights[:, mask].sum(axis=1)
        if not np.isfinite(mass).all() or not (np.abs(mass - count) <= 64 * np.finfo(np.float64).eps * count).all():
            raise FloatingPointError("Class mean-one normalization failed within float64 reduction roundoff")
        classes.append({"class": label, "nodes": count, "route_weight_sums": mass.tolist(),
                        "route_minimum_weights": weights[:, mask].min(axis=1).tolist(),
                        "route_maximum_weights": weights[:, mask].max(axis=1).tolist(),
                        "route_effective_sample_sizes": (np.square(mass) / np.square(weights[:, mask]).sum(axis=1)).tolist()})
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise FloatingPointError("Normalized weights must remain finite and strictly positive")
    return raw, weights, {"generator_seed": generator_seed, "seed_offset": WEIGHT_SEED_OFFSET,
                          "generator": type(generator.bit_generator).__name__, "numpy_version": np.__version__,
                          "shape": [MEMBERS, TRAIN_NODES], "dtype": "float64", "class_normalization": "mean_one_preserve_native_class_mass",
                          "classes": classes, "no_redraw_clipping_floor_or_outcome_choice": True}


def parameter_roles(torch, factors, model):
    """Identify private banks by actual FactorLinear Parameters, including extras."""
    diagonal_ids, extra_ids = set(), set()
    for module in model.modules():
        if isinstance(module, factors.FactorLinear):
            diagonal_ids.update(id(p) for p in (module.r, module.s))
            for attribute in ("u1", "u2", "lora_a", "lora_b"):
                parameter = getattr(module, attribute, None)
                if parameter is not None:
                    if not isinstance(parameter, torch.nn.Parameter):
                        raise TypeError("Explicit extra private banks must be registered Parameters")
                    extra_ids.add(id(parameter))
    named = list(model.named_parameters())
    named_ids = {id(p) for _, p in named}
    private_ids = diagonal_ids | extra_ids
    if not diagonal_ids or diagonal_ids & extra_ids or not private_ids <= named_ids:
        raise ValueError("Disjoint registered diagonal/extra private banks required")
    if any(not p.requires_grad for _, p in named):
        raise ValueError("Preserve the existing fully trainable native/private parameter roles")
    if any(p.ndim != 2 or p.shape[0] != MEMBERS for _, p in named if id(p) in private_ids):
        raise ValueError("Every private bank must retain four independently addressed route rows")
    shared = [(name, p) for name, p in named if id(p) not in private_ids]
    private = [(name, p) for name, p in named if id(p) in private_ids]
    if not shared or not private or {id(p) for _, p in shared} & {id(p) for _, p in private}:
        raise ValueError("Nonempty disjoint shared/private optimizer blocks required")
    extra_count = sum(p.numel() for _, p in named if id(p) in extra_ids)
    if extra_count != sum(site["extra_private_parameters"] for site in model._correction_sites):
        raise ValueError("Every installed correction bank must receive private weighted credit")
    record = {"shared_native_names": [name for name, _ in shared],
              "private_diagonal_names": [name for name, p in private if id(p) in diagonal_ids],
              "private_extra_names": [name for name, p in private if id(p) in extra_ids],
              "shared_parameters": sum(p.numel() for _, p in shared), "private_parameters": sum(p.numel() for _, p in private),
              "extra_private_parameters": extra_count,
              "private_bank_route_count": MEMBERS, "private_bank_route_axis": 0,
              "parameter_shapes": {name: list(p.shape) for name, p in named},
              "gradient_law": "shared=partial mean ownCE; private=partial mean(weight*ownCE); same forward; generally nonconservative",
              "reverse_mode_calls_per_update": 2, "adam_steps_per_update": 1,
              "unchanged_shared_loss_law_is_not_unchanged_shared_trajectory": True}
    return record, shared, private


def bank_gradients(torch, F, logits, targets, weights, shared, private):
    """Collect both block derivatives from the caller's single live route forward."""
    if logits.shape != (MEMBERS, TRAIN_NODES, CLASSES) or targets.shape != (TRAIN_NODES,):
        raise ValueError("Same full four-route TRAIN580 forward required")
    if weights.shape != (MEMBERS, TRAIN_NODES) or weights.dtype != torch.float64 or weights.device != logits.device or weights.requires_grad:
        raise ValueError("Fixed untrained float64 weight table on the logits device required")
    if not (torch.isfinite(weights) & (weights > 0)).all().item():
        raise FloatingPointError("Every applied weight must be finite and strictly positive")
    repeated = targets.repeat(MEMBERS)
    own = F.cross_entropy(logits.flatten(0, 1), repeated)  # Exact native mean loss call.
    per_node = F.cross_entropy(logits.flatten(0, 1), repeated, reduction="none").reshape(MEMBERS, TRAIN_NODES)
    weighted_per_node = per_node.to(torch.float64) * weights
    weighted = weighted_per_node.mean()
    if not torch.isfinite(own).item() or not torch.isfinite(weighted).item():
        raise FloatingPointError("Both full-label TRAIN losses must be finite")
    shared_gradients = torch.autograd.grad(own, tuple(p for _, p in shared), retain_graph=True, allow_unused=True)
    private_gradients = torch.autograd.grad(weighted, tuple(p for _, p in private), allow_unused=True)
    connected = [gradient for gradient in shared_gradients + private_gradients if gradient is not None]
    if not connected or not torch.stack([torch.isfinite(gradient).all() for gradient in connected]).all().item():
        raise FloatingPointError("Every connected shared/private gradient must be finite")
    unconnected = {"shared_native": [], "private": []}
    for role, named, gradients in (("shared_native", shared, shared_gradients), ("private", private, private_gradients)):
        for (name, parameter), gradient in zip(named, gradients):
            if gradient is None:
                unconnected[role].append(name)  # Preserve native unused-parameter semantics.
            parameter.grad = gradient
    member_losses = {"own_ce": per_node.detach().mean(1).cpu().tolist(),
                     "private_weighted_ce": weighted_per_node.detach().mean(1).cpu().tolist()}
    return own.item(), weighted.item(), unconnected, member_losses
