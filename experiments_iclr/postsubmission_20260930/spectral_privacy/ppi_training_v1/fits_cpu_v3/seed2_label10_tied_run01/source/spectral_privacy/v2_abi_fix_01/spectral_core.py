"""Signed output-feature covariance and the frozen v2 discovery/screen rule."""

import torch

if __package__:
    from .contracts import pack_float64, tensor_fingerprint
else:
    from contracts import pack_float64, tensor_fingerprint


STAT_KEYS = ("mean_centered_member_gradients", "Sigma", "S", "N", "T")
RULE = {
    "version": "spectral-neighbor-v2-discovery-screen",
    "side": "output rows of original out-by-in weights",
    "center": "members separately within each graph unit",
    "covariance": "off-diagonal graph-pair U-statistic",
    "n0": 1,
    "rank_max_per_target": 16,
    "discovery_gate": "signed eigenvalue > 0",
    "screen_gate": "u.T @ T_screen @ u > 0",
    "ordering": "descending min(discovery eigenvalue, screening score), then discovery index",
    "zero": "strict positive tests; exact zero fails",
    "eigenvector_sign": "largest-absolute component is positive; first index breaks a tie",
    "interpretation": "fixed-projector denoising surrogate; no selected-projector unbiasedness",
}


def symmetric(value):
    return (value + value.T) / 2


def validate_gradients(gradients):
    if gradients.ndim != 4:
        raise ValueError("Expected [units,members,out,in] matrix gradients")
    n, members, outputs, inputs = gradients.shape
    if n < 2 or members < 2 or outputs < 1 or inputs < 1:
        raise ValueError("Invalid graph/member/matrix dimensions")
    value = gradients.detach().cpu().to(torch.float64).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite matrix gradients")
    return value


def output_statistics(gradients, n0=1):
    if type(n0) is not int or n0 < 1:
        raise ValueError("n0 must be a positive fixed graph batch count")
    value = validate_gradients(gradients)
    n, members, _, _ = value.shape
    centered = value - value.mean(dim=1, keepdim=True)
    means = centered.mean(dim=0)
    residual = centered - means.unsqueeze(0)
    sigma = symmetric(torch.einsum("nmoi,nmpi->op", residual, residual) / (members * (n - 1)))
    mean_covariance = symmetric(torch.einsum("moi,mpi->op", means, means) / members)
    signal = symmetric(mean_covariance - sigma / n)
    noise = sigma / n0
    result = {"mean_centered_member_gradients": means, "Sigma": sigma,
              "S": signal, "N": noise, "T": symmetric(signal - noise)}
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError("Output covariance overflowed or became nonfinite")
    return result


def canonical_signs(basis):
    basis = basis.clone()
    for column in range(basis.shape[1]):
        pivot = int(torch.argmax(basis[:, column].abs()))
        if float(basis[pivot, column]) < 0:
            basis[:, column].neg_()
    return basis


def descending_eigensystem(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric(matrix))
    # Reverse the ascending source index; record this index for exact ties.
    order = torch.arange(len(eigenvalues) - 1, -1, -1)
    return eigenvalues[order], canonical_signs(eigenvectors[:, order])


def choose_bases(discovery, screen, *, n0=1, rank_max=16):
    if type(rank_max) is not int or rank_max < 0 or n0 != 1:
        raise ValueError("V2 pilot requires n0=1 and a nonnegative rank ceiling")
    if not discovery or set(discovery) != set(screen):
        raise ValueError("Discovery/screen fixed-target keys disagree")
    bases, details, stats = {}, {}, {"discovery": {}, "screen": {}}
    for key in sorted(discovery):
        left, right = validate_gradients(discovery[key]), validate_gradients(screen[key])
        if left.shape[1:] != right.shape[1:]:
            raise ValueError("Discovery/screen member and matrix shapes disagree")
        ds, es = output_statistics(left, n0), output_statistics(right, n0)
        stats["discovery"][key], stats["screen"][key] = ds, es
        values, vectors = descending_eigensystem(ds["T"])
        candidates = []
        for index, eigenvalue in enumerate(values):
            vector = vectors[:, index]
            score = float(vector @ es["T"] @ vector)
            eigenvalue = float(eigenvalue)
            candidates.append({"discovery_index": index, "eigenvalue_D": eigenvalue,
                               "score_E": score, "stable_score": min(eigenvalue, score),
                               "eligible": eigenvalue > 0 and score > 0,
                               "direction_sha256": tensor_fingerprint(vector)["sha256"]})
        admitted = sorted((row for row in candidates if row["eligible"]),
                          key=lambda row: (-row["stable_score"], row["discovery_index"]))[:rank_max]
        indices = [row["discovery_index"] for row in admitted]
        bases[key] = vectors[:, indices].contiguous()
        details[key] = {"rank": len(indices), "rank_ceiling": rank_max,
                        "selected_discovery_indices": indices,
                        "eigenvalues_D": values.tolist(), "candidates": candidates}
    return bases, details, stats


def random_matched_bases(shapes, ranks, seed):
    if set(shapes) != set(ranks) or type(seed) is not int:
        raise ValueError("Invalid matched-rank random control specification")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for key in sorted(shapes):
        h, r = shapes[key][0], ranks[key]
        if type(r) is not int or not 0 <= r <= h:
            raise ValueError("Invalid matched rank")
        if r == 0:
            result[key] = torch.empty(h, 0, dtype=torch.float64)
        else:
            draw = torch.randn(h, r, dtype=torch.float64, generator=generator)
            result[key] = canonical_signs(torch.linalg.qr(draw, mode="reduced").Q)
    return result


def raw_gradient_matched_bases(discovery, ranks):
    if set(discovery) != set(ranks):
        raise ValueError("Invalid matched-rank raw-gradient specification")
    result = {}
    for key in sorted(discovery):
        value = validate_gradients(discovery[key])
        centered = value - value.mean(dim=1, keepdim=True)
        means = centered.mean(dim=0)
        covariance = torch.einsum("moi,mpi->op", means, means) / value.shape[1]
        _, vectors = descending_eigensystem(covariance)
        r = ranks[key]
        if type(r) is not int or not 0 <= r <= vectors.shape[0]:
            raise ValueError("Invalid matched rank")
        result[key] = vectors[:, :r].contiguous()
    return result


def serialize_selection(bases, details, stats, discovery, random_seed=800):
    ranks = {key: value.shape[1] for key, value in bases.items()}
    shapes = {key: tuple(value.shape[-2:]) for key, value in discovery.items()}
    random = random_matched_bases(shapes, ranks, random_seed)
    raw = raw_gradient_matched_bases(discovery, ranks)
    ceilings = {value["rank_ceiling"] for value in details.values()}
    if len(ceilings) != 1:
        raise ValueError("Inconsistent target rank ceilings")
    return {
        "rule": RULE | {"rank_max_per_target": next(iter(ceilings))},
        "ranks": ranks, "details": details,
        "basis": {key: pack_float64(value) for key, value in bases.items()},
        "statistics": {group: {key: {name: pack_float64(value[name]) for name in STAT_KEYS}
                                 for key, value in group_stats.items()}
                       for group, group_stats in stats.items()},
        "controls": {
            "random_seed": random_seed,
            "random_matched_rank_basis": {key: pack_float64(value) for key, value in random.items()},
            "raw_centered_gradient_matched_rank_basis": {key: pack_float64(value) for key, value in raw.items()},
        },
    }
