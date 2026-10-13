"""Four fixed training hypotheses; caller injects torch and native task losses.

No model, data provider, optimizer construction, numerical import or fit loop.
The caller supplies native predictions, passively captured raw logits and the
unchanged native reduction/regularizer. Existing source update/scheduler/RNG
and data-role custody remain frontend responsibilities.
"""


MEMBERS = 4
POOL_LAMBDA = 0.5
SUPCON_ALPHA = 0.05
SUPCON_TEMPERATURE = 0.2
SUPCON_EPSILON = 1e-12
RESPONSIBILITY_SEED_OFFSET = 6000119


def _finite_scalar(torch, value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value).item():
        raise ValueError(name + " must be a finite live scalar tensor")
    return value


def native_loss_terms(torch, predictions, raw_logits, *, native_loss, pooled_native_loss, native_regularizer):
    """Native regularizer once in F and L; J then includes it once.

    native_loss(prediction) uses the existing native output/loss interface.
    pooled_native_loss(mean_raw_logits) applies the original native activation
    where required and the native task loss, not a new probability-pool law.
    Masks, targets and source reductions are captured by caller-owned functions.
    native_regularizer is the unchanged source scalar over original parameters.
    """
    predictions, raw_logits = tuple(predictions), tuple(raw_logits)
    if len(predictions) != MEMBERS or len(raw_logits) != MEMBERS:
        raise ValueError("Four aligned native predictions/raw-logit tensors required")
    if any(not isinstance(z, torch.Tensor) for z in raw_logits) or any(
            z.shape != raw_logits[0].shape or z.dtype != raw_logits[0].dtype or z.device != raw_logits[0].device
            for z in raw_logits):
        raise ValueError("Aligned native raw-logit shape/dtype/device required")
    regularizer = _finite_scalar(torch, native_regularizer, "Native regularizer")
    own_task = torch.stack([_finite_scalar(torch, native_loss(p), "Native own loss") for p in predictions]).mean()
    pooled_logits = torch.stack(raw_logits).mean(0)
    pool_task = _finite_scalar(torch, pooled_native_loss(pooled_logits), "Native served-pool loss")
    own, pool = own_task + regularizer, pool_task + regularizer
    return {"F": own, "L": pool, "J": (1 - POOL_LAMBDA) * own + POOL_LAMBDA * pool,
            "pooled_raw_logits": pooled_logits}


def mean_raw_logits(torch, raw_logits):
    """One predetermined domain for both GNNM and ordinary I4."""
    values = tuple(raw_logits)
    if len(values) != MEMBERS or any(v.shape != values[0].shape or v.dtype != values[0].dtype
                                    or v.device != values[0].device for v in values):
        raise ValueError("Four aligned native raw-logit tensors required")
    return torch.stack(values).mean(0)


def draw_fixed_responsibilities(torch, labels, *, seed, reduction_groups=None):
    """Positive Exp(1), drawn once; mean-one per native reduction group/class.

    Only observed categorical/binary TRAIN labels belong here. The caller owns
    original loss groups and target missingness. This portable draw uses a
    declared isolated Torch CPU generator, rather than the old NumPy PCG64
    source stream; it preserves the fixed weight law, not the old exact draws.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Explicit nonnegative responsibility seed required")
    if labels.ndim != 1 or labels.dtype != torch.long or not len(labels):
        raise ValueError("Complete observed categorical/binary TRAIN labels required")
    if (labels < 0).any().item():
        raise ValueError("Missing targets cannot be turned into weight classes")
    classes = labels.detach().to(device="cpu")
    groups = torch.zeros_like(classes) if reduction_groups is None else reduction_groups.detach().to(device="cpu")
    if groups.shape != classes.shape or groups.dtype != torch.long:
        raise ValueError("Exact native loss-reduction groups required")
    generator = torch.Generator(device="cpu").manual_seed(seed + RESPONSIBILITY_SEED_OFFSET)
    weights = torch.empty(MEMBERS, len(classes), dtype=torch.float64, device="cpu").exponential_(
        1.0, generator=generator)
    if not (torch.isfinite(weights) & (weights > 0)).all().item():
        raise ValueError("Positive finite source draws required; no redraw/floor/clipping")
    for group in torch.unique(groups):
        group_mask = groups == group
        for label in torch.unique(classes[group_mask]):
            mask = group_mask & (classes == label)
            weights[:, mask] = weights[:, mask] * (mask.sum() / weights[:, mask].sum(1))[:, None]
    if not (torch.isfinite(weights) & (weights > 0)).all().item():
        raise ValueError("Every normalized responsibility must remain positive and finite")
    return weights.to(device=labels.device), {
        "distribution": "Exp(rate=1)", "normalization": "native reduction group and TRAIN class mean one",
        "generator": "isolated torch.Generator CPU", "seed": seed + RESPONSIBILITY_SEED_OFFSET,
        "redraws": 0, "clipping": False, "floor": False, "source_exact_random_stream": False}


def weighted_native_own_loss(torch, per_record_losses, weights, *, native_reduce, native_regularizer):
    """Apply fixed weights but keep the caller's original within-route reduction."""
    if per_record_losses.ndim != 2 or per_record_losses.shape[0] != MEMBERS or weights.shape != per_record_losses.shape:
        raise ValueError("Aligned [4,observed TRAIN records] losses and weights required")
    if weights.requires_grad or weights.device != per_record_losses.device or weights.dtype != torch.float64:
        raise ValueError("Fixed float64 TRAIN weights must align on the native loss device")
    if not (torch.isfinite(weights) & (weights > 0)).all().item():
        raise ValueError("Positive finite responsibilities required")
    weighted = [native_reduce(per_record_losses[m].to(torch.float64) * weights[m]) for m in range(MEMBERS)]
    return torch.stack([_finite_scalar(torch, value, "Weighted native loss") for value in weighted]).mean() + \
        _finite_scalar(torch, native_regularizer, "Native regularizer")


def within_route_node_supcon(torch, hidden, labels, *, train_ids, full_train_ids, task_kind):
    """Exact full-TRAIN within-route categorical source law, with no extra view."""
    if task_kind != "node_categorical":
        raise ValueError("SupCon is admitted only for node categorical classification")
    if hidden.ndim != 3 or hidden.shape[0] != MEMBERS or not hidden.is_floating_point() or not hidden.requires_grad:
        raise ValueError("Live [4,all TRAIN nodes,native prehead width] hidden tensor required")
    count = hidden.shape[1]
    if count < 2 or labels.shape != (count,) or labels.dtype != torch.long or labels.device != hidden.device:
        raise ValueError("Full categorical TRAIN labels must align with native hidden rows")
    if train_ids.dtype != torch.long or full_train_ids.dtype != torch.long or train_ids.shape != (count,) \
            or full_train_ids.shape != train_ids.shape or train_ids.device != full_train_ids.device \
            or not torch.equal(train_ids, full_train_ids) or len(torch.unique(train_ids)) != count:
        raise ValueError("Exact full original TRAIN anchor order required; no subsampling")
    if not torch.isfinite(hidden).all().item() or (labels < 0).any().item():
        raise ValueError("Finite live hidden and observed TRAIN labels required")
    normalized = torch.nn.functional.normalize(hidden, p=2, dim=-1, eps=SUPCON_EPSILON)
    scores = (normalized @ normalized.transpose(-2, -1)) / SUPCON_TEMPERATURE
    allowed = ~torch.eye(count, dtype=torch.bool, device=hidden.device)
    positives = labels[:, None].eq(labels[None, :]) & allowed
    positive_counts = positives.sum(-1)
    if (positive_counts == 0).any().item():
        raise ValueError("Every full-TRAIN anchor needs a nonself same-class positive")
    log_denominator = torch.logsumexp(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
    mean_positive_score = scores.masked_fill(~positives[None], 0).sum(-1) / positive_counts[None]
    return (log_denominator - mean_positive_score).mean()


def _partition(core, optimizer):
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [id(p) for p in optimized]
    if len(ids) != len(set(ids)):
        raise ValueError("Source optimizer parameter ownership must be disjoint")
    optimized_ids = set(ids)
    private = core.private_parameters()
    native = tuple(p for _, p in core._source_parameters)
    known = {id(p) for p in native + private}
    if not optimized_ids <= known or not {id(p) for p in private} <= optimized_ids:
        raise ValueError("Optimizer must own the unchanged native groups and all boundary banks")
    shared = tuple(p for p in native if p.requires_grad and id(p) in optimized_ids)
    private = tuple(p for p in private if p.requires_grad)
    if not shared or not private:
        raise ValueError("Nonempty source-shared and boundary-private gradient blocks required")
    return shared, private


def _old_state_step(torch, core, optimizer, collections, *, before_step=None):
    """All reverse collections precede assignment/callback/one optimizer step.

    The caller performs its exact native zero_grad before the forwards. A source
    clipping callback can run after complete gradient assignment. Scheduler and
    any AMP/DDP/accumulation adaptation remain source-specific frontend work.
    """
    supplied = []
    for index, (loss, parameters) in enumerate(collections):
        _finite_scalar(torch, loss, "Block objective")
        gradients = torch.autograd.grad(loss, parameters, retain_graph=index + 1 < len(collections),
                                        create_graph=False, allow_unused=True)
        for parameter, gradient in zip(parameters, gradients):
            values = gradient.coalesce().values() if gradient is not None and gradient.is_sparse else gradient
            if values is not None and not torch.isfinite(values).all().item():
                raise ValueError("Nonfinite native/private gradient")
            supplied.append((parameter, gradient))
    if len({id(p) for p, _ in supplied}) != len(supplied) or not any(g is not None for _, g in supplied):
        raise ValueError("Exhaustive disjoint gradient assignment with a live gradient required")
    for parameter, gradient in supplied:
        parameter.grad = gradient  # None preserves native unused-parameter semantics.
    if before_step is not None:
        before_step()
    optimizer.step()
    return {"reverse_collections": len(collections), "optimizer_transitions": 1,
            "all_gradients_from_old_state": True, "extra_forward_views": 0}


def step_global_half_pool(torch, core, optimizer, *, F, L, before_step=None):
    shared, private = _partition(core, optimizer)
    return _old_state_step(torch, core, optimizer,
                          [((1 - POOL_LAMBDA) * F + POOL_LAMBDA * L, shared + private)], before_step=before_step)


def step_private_half_pool(torch, core, optimizer, *, F, L, before_step=None):
    shared, private = _partition(core, optimizer)
    return _old_state_step(torch, core, optimizer,
                          [(F, shared), ((1 - POOL_LAMBDA) * F + POOL_LAMBDA * L, private)], before_step=before_step)


def step_private_responsibility(torch, core, optimizer, *, F, F_weighted, before_step=None):
    shared, private = _partition(core, optimizer)
    return _old_state_step(torch, core, optimizer, [(F, shared), (F_weighted, private)], before_step=before_step)


def step_own_supcon(torch, core, optimizer, *, F, A, task_kind, before_step=None):
    if task_kind != "node_categorical":
        raise ValueError("SupCon step is restricted to node categorical classification")
    shared, private = _partition(core, optimizer)
    return _old_state_step(torch, core, optimizer, [(F + SUPCON_ALPHA * A, shared + private)], before_step=before_step)
