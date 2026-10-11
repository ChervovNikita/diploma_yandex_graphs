"""Unexecuted mask laws for four route activations at native dropout hooks.

No torch import, label input or learned sampler parameters. An admitted caller
supplies torch and a persistent generator on the activation device. Uniform
variates use float64; ideal probability-law statements still require a future
runtime check of finite-precision sampling. Not a bitwise nn.Dropout replay.
"""

MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def make_mask_sampler(policy, *, torch, generator):
    """policy = iid, balanced_08, or activation_sign_paired.

    Return detached mask/q multipliers, never modified activations. Call once
    after the native constructor; do not reset at sites or updates. site is an
    explicit controlled-slot name; every call consumes fresh generator draws.
    Remaining native FFN dropout uses its separate original global RNG stream.
    """
    if policy not in ("iid", "balanced_08", "activation_sign_paired"):
        raise ValueError("Unknown fixed policy")

    def sample(values, *, p, training, site):
        if values.ndim != 3 or values.shape[0] != 4:
            raise ValueError("Expected four [node, channel] activation tables")
        if not 0 <= p <= 1:
            raise ValueError("Invalid native dropout probability")
        if not training or p == 0:
            return torch.ones_like(values)
        if p == 1:
            return torch.zeros_like(values)
        q = 1 - p
        shape = values.shape[1:]

        def uniform(size):
            return torch.rand(size, device=values.device, dtype=torch.float64,
                              generator=generator)

        # No gradient through matching, signs, mask generation or keep count.
        with torch.no_grad():
            if policy == "iid":
                mask = uniform(values.shape) < q
            elif policy == "balanced_08":
                if abs(q - 0.8) > 1e-12:
                    raise ValueError("The proposed balanced control fixes p=0.2")
                # K=3 with probability .8, K=4 with probability .2.
                # The omitted route is uniform at every node/channel coordinate.
                all_keep = uniform(shape) < 0.2
                omitted = torch.randint(4, shape, device=values.device,
                                        generator=generator)
                route = torch.arange(4, device=values.device).view(4, 1, 1)
                mask = all_keep.unsqueeze(0) | (route != omitted.unsqueeze(0))
            else:
                fixed = values.detach().to(dtype=torch.float64)
                anti_cov = max(0.0, 2 * q - 1) - q * q
                same_cov = q * (1 - q)
                best = torch.full(shape, float("inf"), device=values.device,
                                  dtype=torch.float64)
                choice = torch.zeros(shape, device=values.device,
                                     dtype=torch.int64)
                for index, pairs in enumerate(MATCHINGS):
                    cost = torch.zeros_like(best)
                    for i, j in pairs:
                        product = fixed[i] * fixed[j]
                        covariance = torch.full_like(product, same_cov)
                        covariance.masked_fill_(product >= 0, anti_cov)
                        cost = cost + product * covariance
                    improve = cost < best  # deterministic first matching on ties
                    choice = torch.where(improve, index, choice)
                    best = torch.minimum(best, cost)

                # Two independent uniforms per coordinate suffice for whichever
                # matching was chosen from current activations before these draws.
                uniforms = uniform((2, *shape))
                masks = [torch.zeros(shape, device=values.device,
                                     dtype=torch.bool) for _ in range(4)]
                for index, pairs in enumerate(MATCHINGS):
                    chosen = choice == index
                    for pair, (i, j) in enumerate(pairs):
                        first = uniforms[pair] < q
                        anti = (1 - uniforms[pair]) < q
                        second = torch.where(fixed[i] * fixed[j] >= 0,
                                             anti, first)
                        masks[i] = torch.where(chosen, first, masks[i])
                        masks[j] = torch.where(chosen, second, masks[j])
                mask = torch.stack(masks, dim=0)
            multiplier = mask.to(dtype=values.dtype) / q
        return multiplier

    return sample
