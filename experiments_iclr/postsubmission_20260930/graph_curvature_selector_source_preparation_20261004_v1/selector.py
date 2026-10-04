"""Source preparation only; numerical and actual-warm qualification is pending.

Finite private-head covariance selection, not a graph-view training baseline.
Numerical dependencies are imported only inside explicitly called operations.
Only compact TRAIN labels enter construction, matching, safeguards or trials.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass


PAIRS = ((0, 1), (0, 2), (1, 2))
ARMS = ("common_only", "fixed_first_graph_pair", "selected_graph_pair",
        "selected_permuted_span", "selected_random_span")
HEADS = {"polyformer_mono": ("head.R", 256),
         "polynormer_r": ("global_head.R", 512)}


@dataclass(frozen=True)
class FrozenConstants:
    """No usable defaults: root must prospectively freeze these, once.

    radius is the sole graph reference radius; cap bounds every scalar match.
    Rank/sign rules use input order, never response scores or random redraws.
    These constants are implementation inputs, not a qualification certificate.
    """
    radius: float
    radius_cap: float
    rank_atol: float
    rank_rtol: float
    sign_atol: float
    mean_logit_atol: float
    mean_logit_rtol: float
    d_positive_min: float
    d_match_atol: float
    d_match_rtol: float
    bisection_steps: int
    tie_tolerance: float
    abstention_tolerance: float

    def validate(self):
        values = asdict(self)
        for name, value in values.items():
            if name != "bisection_steps":
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0,
                        "Every scalar constant must be finite and nonnegative")
        require(0 < self.radius <= self.radius_cap, "One positive radius within one cap")
        require(type(self.bisection_steps) is int and self.bisection_steps > 0,
                "Freeze a positive bounded scalar-search limit")
        require(self.rank_atol + self.rank_rtol > 0 and self.sign_atol > 0,
                "Rank and sign tolerances must be positive")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choose_pair(records, common_value, constants):
    """Fixed enumeration ties; strict improvement beyond abstention tolerance."""
    eligible = [record for record in records if record.get("status") == "eligible"
                and record.get("post_trial_pooled_train_ce") is not None
                and math.isfinite(record["post_trial_pooled_train_ce"])]
    if not eligible:
        return None, "no_eligible_pair"
    minimum = min(record["post_trial_pooled_train_ce"] for record in eligible)
    winner = next(record for record in eligible if record["post_trial_pooled_train_ce"]
                  <= minimum + constants.tie_tolerance)
    if not winner["post_trial_pooled_train_ce"] < common_value - constants.abstention_tolerance:
        return None, "common_trial_abstention"
    return winner["pair_index"], "selected"


def bind_head_only(model, backbone, model_args):
    """K1 identity warm snapshot; only final predictive head.R is replaceable.

    The model must be the unchanged raw boundary wrapper in the final stage.
    All upstream parameters, other factors and buffers are frozen snapshots.
    Common descent and every covariance candidate share this exact closure.
    """
    import torch
    require(backbone in HEADS and model.members == 1, "Admitted K1 wrapper only")
    require(all(not module.training for module in model.modules()), "Dropout-off K1 required")
    if backbone == "polynormer_r":
        require(model.core._global is True, "Photo construction uses final global head")
    name, width = HEADS[backbone]
    parameters = dict(model.named_parameters())
    require(name in parameters and tuple(parameters[name].shape) == (1, width),
            "Exact final head factor shape required")
    for key, value in parameters.items():
        if key.endswith((".R", ".S")):
            require(bool(torch.equal(value.detach(), torch.ones_like(value))),
                    "Every multiplicative factor must start at identity")
    frozen = {key: value.detach().clone() for key, value in parameters.items()}
    buffers = {key: value.detach().clone() for key, value in model.named_buffers()}
    theta0 = frozen[name].reshape(-1).clone()

    def logits_fn(theta):
        require(tuple(theta.shape) == (width,), "Head slice shape changed")
        replacements = dict(frozen)
        replacements[name] = theta.view_as(frozen[name])
        fresh_buffers = {key: value.clone() for key, value in buffers.items()}
        result = torch.func.functional_call(model, (replacements, fresh_buffers),
                                             model_args, strict=True)
        require(result.ndim == 3 and result.shape[0] == 1 and result.shape[2] >= 2,
                "Need final predictive K1 class logits")
        return result[0]

    return theta0, logits_fn, dict(name=name, dimensions=width, identity=True,
                                  affine_by_source_only=True, runtime_affinity_pending=True)


def install_head(model, slices, backbone):
    import torch
    name, width = HEADS[backbone]
    parameters = dict(model.named_parameters())
    require(model.members == 4 and tuple(slices.shape) == (4, width)
            and tuple(parameters[name].shape) == (4, width), "Exact K4 head slice required")
    with torch.no_grad():
        parameters[name].copy_(slices)


def projected_bank(logits_fn, center, S, target_nodes, train_rows, train_labels, source):
    """Original cubic TRAIN-remasked graph-error VJPs, evaluated at shared center."""
    import torch
    import torch.nn.functional as F
    z, pullback = torch.func.vjp(logits_fn, center)
    require(bool(torch.isfinite(z).all()), "Nonfinite construction logits")
    residual = torch.zeros_like(z)
    residual[train_rows] = (z[train_rows].softmax(-1).detach()
                           - F.one_hot(train_labels, z.shape[1]).to(z.dtype))/train_rows.numel()
    gradient = pullback(residual.detach())[0].detach()
    full = torch.zeros((S.shape[0], z.shape[1]), dtype=z.dtype, device=z.device)
    full[target_nodes] = residual
    bands = source.bernstein_cubic_bands(S, full).detach()
    partition_error = float((bands.sum(0)-full).norm()/full.norm().clamp_min(
        torch.finfo(full.dtype).tiny))
    require(partition_error <= source.ALGEBRA_TOLERANCE, "Cubic partition identity failed")
    columns = []
    for band in bands:
        cotangent = torch.zeros_like(z)
        cotangent[train_rows] = band[target_nodes[train_rows]]
        columns.append(pullback(cotangent)[0].detach())
    bank = torch.stack(columns).double()
    del pullback
    g = gradient.double()
    require(bool(torch.isfinite(bank).all() and torch.isfinite(g).all()), "Nonfinite VJP bank")
    sum_error = float((bank.sum(0)-g).norm()/g.norm().clamp_min(1e-20))
    require(sum_error <= source.ALGEBRA_TOLERANCE or float(g.square().sum()) <= 1e-20,
            "TRAIN-remasked bank does not sum to common gradient")
    return bank, g, dict(partition_relative_error=partition_error,
                        bank_sum_relative_error=sum_error, vjp_forwards=1, vjp_calls=5,
                        graph_sparse_products=3, output_support="TRAIN_remasked")


def ordered_basis(bank, gradient, constants, algebra_tolerance):
    """Two-pass projection/centering and reorthogonalized input-order Gram Schmidt."""
    import torch
    g2 = gradient.square().sum()
    if float(g2) <= 1e-20:
        return None, dict(status="rank_failure", rank=0, reason="zero_common_gradient",
                          pivots=[], redraws=0)
    projected = bank.double().clone()
    for _ in range(2):
        projected -= projected.mean(0, keepdim=True)
        if float(g2) > 1e-20:
            projected -= (projected @ gradient/g2)[:, None]*gradient
    cutoff = constants.rank_atol + constants.rank_rtol*float(projected.norm(dim=1).max())
    basis, pivots = [], []
    for index, column in enumerate(projected):
        value = column.clone()
        for _ in range(2):
            for vector in basis:
                value -= torch.dot(value, vector)*vector
        norm = float(value.norm())
        if norm <= cutoff:
            continue
        vector = value/norm
        coordinates = torch.nonzero(vector.abs() > constants.sign_atol).flatten()
        require(coordinates.numel() > 0, "No coordinate above fixed sign tolerance")
        if float(vector[int(coordinates[0])]) < 0:
            vector = -vector
        basis.append(vector)
        pivots.append(index)
    receipt = dict(rank=len(basis), pivots=pivots, rank_cutoff=cutoff,
                   sign="first coordinate above fixed tolerance positive", redraws=0)
    if len(basis) != 3:
        receipt["status"] = "rank_failure"
        return None, receipt
    vectors = torch.stack(basis)
    orth = float((vectors @ vectors.T-torch.eye(3, dtype=vectors.dtype,
                                              device=vectors.device)).abs().max())
    projection = float((vectors @ gradient).abs().max()/gradient.norm().clamp_min(1e-20))
    require(max(orth, projection) <= algebra_tolerance, "Basis orthogonality/projection failed")
    receipt.update(status="rank_three", orthogonality_error=orth, projection_error=projection)
    return vectors, receipt


def initial_metrics(logits_fn, center, vectors, pair, scale, train_rows, labels):
    import torch
    import torch.nn.functional as F
    a, b = (vectors[index].to(center.dtype)*scale for index in pair)
    slices = center[None, :]+torch.stack((a, -a, b, -b))
    with torch.no_grad():
        logits = torch.stack([logits_fn(row) for row in slices])
        reference = logits_fn(center)
        losses = torch.stack([F.cross_entropy(z[train_rows], labels) for z in logits])
        pooled = F.cross_entropy(logits.mean(0)[train_rows], labels)
        # Measurement only: FP64 CE/reduction for the finite Jensen match.
        # Actual trial training, guards and native pool retain source precision.
        z64 = logits[:, train_rows].double()
        d = F.cross_entropy(z64.reshape(-1, z64.shape[-1]), labels.repeat(4)) \
            - F.cross_entropy(z64.mean(0), labels)
        centered = logits[:, train_rows]-logits[:, train_rows].mean(-1, keepdim=True)
        rms = [float((centered[i]-centered[j]).square().mean().sqrt())
               for i in range(4) for j in range(i)]
        finite = bool(torch.isfinite(logits).all() and torch.isfinite(losses).all()
                      and torch.isfinite(pooled) and torch.isfinite(d))
        delta = float((logits.mean(0)-reference).abs().max()) if finite else None
    return slices.detach(), dict(radius=scale, finite=finite,
        initial_route_train_ce=[float(value) for value in losses] if finite else None,
        initial_pooled_train_ce=float(pooled) if finite else None,
        D=float(d) if finite else None, D_measurement="FP64_CE_and_mean",
        max_member_radius=float((slices-center).double().norm(dim=1).max()),
        offset_mean_max_absolute=float((slices-center).double().mean(0).abs().max()),
        mean_logit_max_absolute=delta,
        centered_pair_rms=rms if finite else None), logits.detach(), reference.detach()


def match_pair(logits_fn, center, vectors, pair, target_d, rows, labels, constants):
    """One cap, bounded monotone scalar bisection; no radius grid or cap increase."""
    def measure(scale):
        return initial_metrics(logits_fn, center, vectors, pair, scale, rows, labels)
    tolerance = constants.d_match_atol + constants.d_match_rtol*abs(target_d)
    lower, upper = 0.0, constants.radius_cap
    item = measure(upper)
    trace = [dict(radius=upper, D=item[1]["D"], finite=item[1]["finite"])]
    if not item[1]["finite"] or item[1]["D"] < target_d-tolerance:
        return None, dict(status="cap_or_nonfinite_failure", matching_trace=trace)
    for step in range(constants.bisection_steps+1):
        if item[1]["finite"] and abs(item[1]["D"]-target_d) <= tolerance:
            item[1].update(matching_trace=trace, scalar_search_evaluations=len(trace))
            return item, None
        if step == constants.bisection_steps:
            break
        if not item[1]["finite"]:
            return None, dict(status="nonfinite_match_failure", matching_trace=trace)
        if item[1]["D"] > target_d:
            upper = item[1]["radius"]
        else:
            lower = item[1]["radius"]
        item = measure((lower+upper)/2)
        trace.append(dict(radius=item[1]["radius"], D=item[1]["D"], finite=item[1]["finite"]))
    return None, dict(status="D_match_failure", matching_trace=trace)


def finite_guard(item, target_d, bound, rows, constants, source, center, gradient):
    """Original member+pooled TRAIN Armijo bound and finite response safeguard."""
    import torch
    slices, metrics, logits, reference = item
    if not metrics["finite"]:
        return "nonfinite_initialization"
    if metrics["max_member_radius"] > constants.radius_cap + source.ALGEBRA_TOLERANCE \
            * max(1.0, constants.radius_cap):
        return "finite_radius_cap_failure"
    offsets = (slices-center).double()
    scale = offsets.norm(dim=1).max().clamp_min(1e-20)
    zero_mean_error = float(offsets.mean(0).norm()/scale)
    projection_error = float((offsets @ gradient).abs().max()
                             /(scale*gradient.norm()).clamp_min(1e-20))
    metrics.update(offset_mean_relative_error=zero_mean_error,
                   offset_projection_relative_error=projection_error)
    if max(zero_mean_error, projection_error) > source.ALGEBRA_TOLERANCE:
        return "finite_offset_projection_or_centering_failure"
    if not torch.allclose(logits.mean(0), reference, atol=constants.mean_logit_atol,
                          rtol=constants.mean_logit_rtol):
        return "initial_mean_logits_differ"
    if abs(metrics["D"]-target_d) > constants.d_match_atol+constants.d_match_rtol*abs(target_d):
        return "D_match_failure"
    if not all(value <= bound for value in metrics["initial_route_train_ce"]) \
            or metrics["initial_pooled_train_ce"] > bound:
        return "original_TRAIN_safeguard_failure"
    centered = reference[rows]-reference[rows].mean(-1, keepdim=True)
    threshold = source.FUNCTION_RMS_TOLERANCE*max(1.0, float(centered.square().mean().sqrt()))
    if metrics["radius"] <= 0 or min(metrics["centered_pair_rms"])/metrics["radius"] <= threshold:
        return "class_centered_response_null_or_duplicate"
    return None


def one_adam_trial(prototype, frozen_optimizer, slices, backbone, model_args,
                   train_rows, train_labels, native_rng, adapter):
    """One complete coupled-Adam map; returned receipt holds no trial state."""
    import torch
    import torch.nn.functional as F
    trial = copy.deepcopy(prototype)
    optimizer = adapter.restore_named_optimizer(trial, frozen_optimizer)
    install_head(trial, slices, backbone)
    receipt = dict(updates=0, optimizer_step_attempted=False, optimizer="coupled_Adam",
                   dropout_off=True, buffers="eval_mode", trial_state_discarded=True,
                   trial_map_updates_all_live_shared_and_private_parameters=True)
    phase = "before_forward"
    try:
        adapter.rng_restore(native_rng)
        trial.eval()  # All dropout off; eval buffers; autograd and every live factor retained.
        optimizer.zero_grad(set_to_none=True)
        phase = "own_member_forward"
        logits = trial(*model_args)
        require(logits.ndim == 3 and logits.shape[0] == 4, "Full four-member logits required")
        loss = F.cross_entropy(logits[:, train_rows].reshape(-1, logits.shape[-1]),
                               train_labels.repeat(4))
        require(bool(torch.isfinite(loss)), "Nonfinite trial own-member TRAIN CE")
        loss.backward()
        phase = "gradient_checks"
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                    for p in trial.parameters()), "Nonfinite trial gradients")
        require(any(p.grad is not None and bool((p.grad != 0).any()) for p in trial.parameters()),
                "No nonzero trial gradient")
        phase = "optimizer_step"
        receipt["optimizer_step_attempted"] = True
        optimizer.step()
        receipt["updates"] = 1
        phase = "post_step_checks"
        require(all(bool(torch.isfinite(p).all()) for p in trial.parameters()),
                "Nonfinite trial parameters")
        for state in optimizer.state.values():
            require(all(not torch.is_tensor(v) or bool(torch.isfinite(v).all())
                        for v in state.values()), "Nonfinite trial Adam state")
        phase = "post_step_pooled_forward"
        with torch.no_grad():
            after = trial(*model_args)
            value = F.cross_entropy(after.mean(0)[train_rows], train_labels)
        require(bool(torch.isfinite(after).all() and torch.isfinite(value)), "Nonfinite pooled trial CE")
        receipt.update(status="eligible", post_trial_pooled_train_ce=float(value),
                       pre_trial_mean_member_train_ce=float(loss.detach()))
        return receipt
    except (ValueError, RuntimeError) as error:
        receipt.update(status="trial_failure", error=str(error), failed_phase=phase,
                       partial_optimizer_step_possible=phase == "optimizer_step")
        return receipt
    finally:
        adapter.rng_restore(native_rng)
        del optimizer, trial


def select_initializations(prototype, frozen_optimizer, logits_fn, theta0, S,
                           target_nodes, train_rows, labels, backbone, model_args,
                           native_rng, seed, constants, source, adapter):
    """Build all five arms jointly; one center and one common trial per block.

    Returns only independently cloned pre-trial initializations/optimizers/RNG.
    Construction failures are recorded as three null candidates per span.
    Exceptions in actual trials are failures, never a new donor or extra budget.
    """
    import torch
    constants.validate()
    require(type(seed) is int and seed in (17, 29, 43), "Retained development seeds only")
    modes = {name: module.training for name, module in prototype.named_modules()}
    record = dict(operation="graph_private_covariance_finite_selector_v1", constants=asdict(constants),
                  arms=list(ARMS), source_labels="TRAIN_only", pairs=[list(p) for p in PAIRS],
                  construction_seeds=dict(random=seed+70000, permutation=seed+80000),
                  warm_qualification_inherited=False, numerical_qualification_inherited=False)
    try:
        adapter.rng_restore(native_rng)
        common_slices, common_receipt = source.initialize_four_routes(
            logits_fn, theta0, S, target_nodes, train_rows, labels, tangent_mode="common_only")
        require(bool(torch.equal(common_slices, common_slices[0].expand_as(common_slices))),
                "Common safeguard produced unequal members")
        center = common_slices[0].detach()
        alpha = common_receipt["accepted_alpha"]
        bound = common_receipt["baseline_train_ce"] - source.ARMIJO_C*alpha*common_receipt[
            "common_gradient_squared_norm"]
        record["common_center"] = common_receipt
        common_trial = one_adam_trial(prototype, frozen_optimizer, common_slices, backbone,
                                     model_args, train_rows, labels, native_rng, adapter)
        record["common_trial"] = common_trial
        require(common_trial["status"] == "eligible", "Common trial failed: "+str(common_trial))
        spans, bases = {}, {}
        gradient = None
        try:
            graph_bank, gradient, bank_receipt = projected_bank(
                logits_fn, center, S, target_nodes, train_rows, labels, source)
            record["graph_bank"] = bank_receipt
            bases["graph"], spans["graph"] = ordered_basis(
                graph_bank, gradient, constants, source.ALGEBRA_TOLERANCE)
        except (ValueError, RuntimeError) as error:
            bases["graph"] = None
            spans["graph"] = dict(status="construction_failure", error=str(error))
        for name in ("permuted", "random"):
            try:
                require(gradient is not None, "Original common gradient unavailable")
                if name == "permuted":
                    permuted, _ = source.permute_topology_nodes(S, seed+80000)
                    bank, perm_gradient, bank_receipt = projected_bank(
                        logits_fn, center, permuted, target_nodes, train_rows, labels, source)
                    require(torch.allclose(gradient, perm_gradient, atol=constants.rank_atol,
                                            rtol=source.ALGEBRA_TOLERANCE), "Control common gradient changed")
                else:
                    generator = torch.Generator(device="cpu").manual_seed(seed+70000)
                    bank = torch.randn((4, center.numel()), dtype=torch.float64,
                                       generator=generator).to(center.device)
                    bank_receipt = dict(distribution="four_IID_standard_normal_rows")
                bases[name], spans[name] = ordered_basis(
                    bank, gradient, constants, source.ALGEBRA_TOLERANCE)
                spans[name]["bank"] = bank_receipt
            except (ValueError, RuntimeError) as error:
                bases[name] = None
                spans[name] = dict(status="construction_failure", error=str(error))
        record["spans"] = spans
        target_d, reference_item = None, None
        if bases["graph"] is not None:
            reference_item = initial_metrics(logits_fn, center, bases["graph"], PAIRS[0],
                                             constants.radius, train_rows, labels)
            record["D0_reference"] = reference_item[1]
            if reference_item[1]["finite"] and reference_item[1]["D"] > constants.d_positive_min:
                target_d = reference_item[1]["D"]
        selected_slices, candidates = {}, {}
        for span in ("graph", "permuted", "random"):
            candidates[span] = []
            for index, pair in enumerate(PAIRS):
                row = dict(pair_index=index, pair=list(pair))
                candidates[span].append(row)
                if bases[span] is None or target_d is None:
                    row.update(status="rank_or_D0_failure")
                    continue
                if span == "graph" and index == 0:
                    item, failure = reference_item, None
                else:
                    item, failure = match_pair(logits_fn, center, bases[span], pair, target_d,
                                               train_rows, labels, constants)
                if failure is not None:
                    row.update(failure)
                    continue
                row.update(item[1])
                guard = finite_guard(item, target_d, bound, train_rows, constants, source,
                                     center, gradient)
                row.update(item[1])
                if guard:
                    row.update(status=guard)
                    continue
                row["trial_started"] = True
                try:
                    row.update(one_adam_trial(prototype, frozen_optimizer, item[0], backbone,
                                              model_args, train_rows, labels, native_rng, adapter))
                except (ValueError, RuntimeError) as error:
                    row.update(status="trial_failure", error=str(error))
                    continue
                if row["status"] != "eligible":
                    continue
                selected_slices[(span, index)] = item[0]
        record["candidates"] = candidates
        selections = {"common_only": (None, "common_control"),
                      "fixed_first_graph_pair": (("graph", 0) if ("graph", 0) in selected_slices
                                                 else None, "fixed_pair" if ("graph", 0) in selected_slices
                                                 else "fixed_pair_ineligible_common_null")}
        for span, arm in (("graph", "selected_graph_pair"),
                          ("permuted", "selected_permuted_span"),
                          ("random", "selected_random_span")):
            index, reason = choose_pair(candidates[span], common_trial["post_trial_pooled_train_ce"], constants)
            selections[arm] = (None if index is None else (span, index), reason)
        output = {}
        for arm in ARMS:
            selection, reason = selections[arm]
            slices = common_slices if selection is None else selected_slices[selection]
            model = copy.deepcopy(prototype)
            install_head(model, slices, backbone)
            optimizer = adapter.restore_named_optimizer(model, frozen_optimizer)
            for name, module in model.named_modules():
                module.training = modes[name]
            output[arm] = dict(model=model, optimizer=optimizer, rng=copy.deepcopy(native_rng))
            record.setdefault("selection", {})[arm] = dict(pair=None if selection is None else list(selection),
                                                           reason=reason, pre_trial_state_returned=True)
        record["trial_maps_completed"] = 1 + sum(row.get("updates", 0)
            for rows in candidates.values() for row in rows)
        record["trial_maps_started"] = 1 + sum(bool(row.get("trial_started"))
            for rows in candidates.values() for row in rows)
        record["maximum_trial_maps"] = 10
        record["candidate_slots_per_selected_arm"] = 3
        record["failed_or_abstaining_arms_retained"] = True
        return output, record
    finally:
        adapter.rng_restore(native_rng)
