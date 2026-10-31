"""Common native warm lift and fixed TRAIN-only member initializations."""
from types import MethodType

CONDITIONS = ("random_signs_all", "tabm_first_normal", "warm_identity",
              "graph_covariance", "feature_covariance", "single", "independent_warm4")
RMS = 1.0


def warm_bias_arithmetic(torch, module):
    """Same rank-one map in real arithmetic; exact native bias path at r=s=1."""
    def forward_member(self, x, member):
        value = torch.nn.functional.linear(x*self.r[member], self.weight, self.bias)
        return value*self.s[member] + self.bias*(1-self.s[member]) if self.bias is not None else value*self.s[member]
    module.forward_member = MethodType(forward_member, module)


def copy_single(torch, target, warm):
    target.load_state_dict(warm.state_dict(), strict=True)


def lift(torch, bank, warm):
    bank.encoder.load_state_dict(warm.encoder.state_dict(), strict=True)
    signs = {}
    with torch.no_grad():
        for name in ("xlin", "xcnlin", "xijlin", "lin"):
            target = getattr(bank.predictor, name).ops
            source = getattr(warm.predictor, name)
            if len(target) != len(source): raise ValueError("Native head sequence differs")
            for index, (a, b) in enumerate(zip(target, source)):
                prefix = "predictor."+name+".ops."+str(index)
                if hasattr(a, "r"):
                    a.weight.copy_(b.weight)
                    if a.bias is not None: a.bias.copy_(b.bias)
                    signs[prefix+".r"] = a.r.detach().clone()
                    signs[prefix+".s"] = a.s.detach().clone()
                    a.r.fill_(1); a.s.fill_(1)
                    warm_bias_arithmetic(torch, a)
                elif hasattr(a, "normalized_shape") and hasattr(a, "forward_member"):
                    a.weight.copy_(b.weight.expand_as(a.weight)); a.bias.copy_(b.bias.expand_as(a.bias))
                    if a.eps != b.eps or a.normalized_shape != b.normalized_shape:
                        raise ValueError("Private LayerNorm native constants differ")
        bank.predictor.beta.copy_(warm.predictor.beta.expand_as(bank.predictor.beta))
    return signs


def variation(torch, warm, x, support, train, factor_seed):
    """PSD covariance roots: edge hidden variation versus centered hidden features.

    The feature covariance control uses the same graph-derived representation;
    only the additional TRAIN edge-pair covariance is absent.
    """
    modes = [(m, m.training) for m in warm.modules()]; warm.eval()
    try:
        with torch.no_grad(): hidden = warm.encoder(x, support).detach().cpu().double()
    finally:
        for m, mode in modes: m.training = mode
    edges = train.cpu(); delta = hidden[edges[:, 0]]-hidden[edges[:, 1]]
    centered = hidden-hidden.mean(0, keepdim=True)
    generator = torch.Generator(device="cpu").manual_seed(factor_seed+700001)
    white = torch.randn((2, hidden.shape[1]), dtype=torch.float64, generator=generator)
    result, records = {}, {}
    for name, rows in (("graph_covariance", delta), ("feature_covariance", centered)):
        covariance = rows.t().matmul(rows)/len(rows)
        values, vectors = torch.linalg.eigh(covariance)
        root = (vectors*values.clamp_min(0).sqrt()).matmul(vectors.t())
        directions = white.matmul(root)
        balanced = torch.stack((directions[0], -directions[0], directions[1], -directions[1]))
        magnitude = balanced.square().mean().sqrt()
        if not bool(torch.isfinite(magnitude)) or not bool(magnitude > 0):
            raise ValueError("Fixed covariance has no finite perturbation direction; no fallback/redraw")
        balanced = balanced*(RMS/magnitude)
        result[name] = balanced
        records[name] = {"covariance_trace": float(covariance.trace()), "smallest_eigenvalue": float(values.min()),
            "largest_eigenvalue": float(values.max()), "perturbation_RMS_before_FP32": float(balanced.square().mean().sqrt()),
            "maximum_member_sum_before_FP32": float(balanced.sum(0).abs().max()),
            "rows": len(rows), "width": hidden.shape[1]}
    return result, records


def make(torch, native, heads, models, warm, condition, seed, factor_seed, device, variations):
    if condition not in CONDITIONS: raise ValueError("Fixed initialization condition required")
    arm = "capable_single" if condition == "single" else "ordinary_native4" if condition == "independent_warm4" else "shared_f4"
    model, _, _, partition = models.build(native, heads, arm=arm, seed=seed, factor_seed=factor_seed, device=device)
    if condition == "single":
        copy_single(torch, model, warm)
    elif condition == "independent_warm4":
        for route in model.routes:
            route.encoder.load_state_dict(warm.encoder.state_dict(), strict=True)
            route.predictor.load_state_dict(warm.predictor.state_dict(), strict=True)
            original = route.forward
            def forward(self, x, support, positive, negative, route=None, original=original):
                if route not in (None, 0): raise ValueError("Independent single route differs")
                return original(x, support, positive, negative)
            route.forward = MethodType(forward, route); route.member_count = 1
    else:
        signs = lift(torch, model, warm)
        with torch.no_grad():
            if condition == "random_signs_all":
                for name, parameter in model.named_parameters():
                    if name in signs: parameter.copy_(signs[name])
            elif condition == "tabm_first_normal":
                generator = torch.Generator(device="cpu").manual_seed(factor_seed+700001)
                model.predictor.xlin.ops[0].r.copy_(torch.randn((4, 256), generator=generator).to(device))
            elif condition in ("graph_covariance", "feature_covariance"):
                model.predictor.xlin.ops[0].r.copy_((1+variations[condition]).float().to(device))
    for parameter in model.parameters(): parameter.requires_grad_(True)
    partition["initialization"] = ("same_common_warm_checkpoint; independent_postwarm_dropout_and_query_streams"
        if condition == "independent_warm4" else "native_single_from_same_common_warm_checkpoint"
        if condition == "single" else "shared_F4_lift_from_same_common_warm_checkpoint; "+condition)
    partition["all_dense_and_private_parameters_trainable"] = True
    partition["inner_query_exposure"] = ("each_independent_native_member_concatenates_all_four_inner_lists"
        if condition == "independent_warm4" else "single_concatenates_all_four_inner_lists"
        if condition == "single" else "each_F4_member_uses_its_own_inner_route_list")
    partition["selection"] = "first_maximum_complete_pooled_VALID_MRR_rounded4; independent_ensemble_uses_pooled_selector"
    return model, partition
