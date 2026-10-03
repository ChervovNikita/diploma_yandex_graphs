"""Explicit source-style frozen-common/all-private FoRDE graph adaptation.

One ordinary process evaluates one whole logical batch. No candidate guard,
projection, clamp, alternate epsilon, automatic backend fallback, or fit occurs.
Float64 is a named synthetic diagnostic, not a native float32 admission.
"""
from dataclasses import dataclass
import hashlib
import math
import torch


def tensor_digest(value):
    value = value.detach().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype), str(value.layout))).encode())
    if value.layout == torch.sparse_coo:
        digest.update(str(value.is_coalesced()).encode())
        digest.update(value._indices().contiguous().numpy().tobytes())
        digest.update(value._values().contiguous().numpy().tobytes())
    else:
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def setup_all_private(model, *, bank_receipt):
    """Enable all existing affine R/S, including the three boundary maps."""
    if not bank_receipt or model._in_forward:
        raise ValueError("An idle native composition and explicit bank receipt are required")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    names = tuple(f"core.{site}.{factor}" for site in model.site_names for factor in ("R", "S"))
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)
    model.eval()
    model.stage = "source_style_all_private"
    model.warm_bank_receipt = bank_receipt
    actual = tuple(n for n, p in model.named_parameters() if p.requires_grad)
    assert set(actual) == set(names)
    return actual


def source_sgd(model, *, learning_rate, weight_decay=5e-4):
    if model.stage != "source_style_all_private" or any(m.training for m in model.modules()):
        raise ValueError("Explicit source-style all-private eval setup required")
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9,
                           nesterov=True, dampening=0, weight_decay=weight_decay)


def tokens_from_x(x, operator, maximum_power):
    values = [x]
    for _ in range(maximum_power):
        values.append(torch.sparse.mm(operator, values[-1]))
    return torch.stack(values, dim=1)


def row_powers(operator, targets, maximum_power):
    with torch.no_grad():
        current = torch.zeros(operator.shape[0], targets.numel(), dtype=operator.dtype,
                              device=operator.device)
        current[targets, torch.arange(targets.numel(), device=operator.device)] = 1
        rows = [current.T]
        for _ in range(maximum_power):
            current = torch.sparse.mm(operator.transpose(0, 1), current)
            rows.append(current.T)
        return torch.stack(rows, dim=1)


@dataclass(frozen=True)
class BoundGraphBatch:
    operator: object
    complete_tokens: object
    targets: object
    grams: object
    powers: object
    maximum_power: int
    base: str
    feature_identity: str
    precision_receipt: str
    operator_sha256: str
    tokens_sha256: str
    targets_sha256: str
    grams_sha256: str
    powers_sha256: str
    diagnostic_float64: bool

    @classmethod
    def create(cls, operator, complete_tokens, targets, *, maximum_power,
               feature_identity, precision_receipt, diagnostic_float64=False):
        allowed_dtype = torch.float64 if diagnostic_float64 else torch.float32
        if (operator.layout != torch.sparse_coo or operator.dtype != allowed_dtype
                or operator.requires_grad or operator.ndim != 2
                or operator.shape[0] != operator.shape[1]):
            raise ValueError("Fixed square COO operator in declared dtype required")
        if (complete_tokens.shape[:2] != (operator.shape[0], maximum_power + 1)
                or complete_tokens.ndim != 3 or complete_tokens.dtype != operator.dtype
                or complete_tokens.device != operator.device or complete_tokens.requires_grad):
            raise ValueError("Complete fixed native mono token bank required")
        if (targets.ndim != 1 or targets.dtype != torch.long or not targets.numel()
                or targets.device != operator.device
                or targets.unique().numel() != targets.numel()
                or bool((targets < 0).any()) or bool((targets >= operator.shape[0]).any())):
            raise ValueError("Distinct ordered complete-graph targets required")
        if type(maximum_power) is not int or maximum_power < 0 or not feature_identity or not precision_receipt:
            raise ValueError("Power, feature and precision identities required")
        # The bank must actually contain this operator's fixed polynomial tokens.
        expected = tokens_from_x(complete_tokens[:, 0], operator, maximum_power)
        if not torch.equal(expected, complete_tokens):
            raise ValueError("Token bank does not equal the bound fixed mono construction")
        a = row_powers(operator, targets, maximum_power)
        g = torch.bmm(a, a.transpose(1, 2))
        if not bool(torch.isfinite(g).all()):
            raise FloatingPointError("Nonfinite graph Gram")
        return cls(operator, complete_tokens, targets, g, a, maximum_power, "mono",
                   feature_identity, precision_receipt, tensor_digest(operator),
                   tensor_digest(complete_tokens), tensor_digest(targets), tensor_digest(g),
                   tensor_digest(a), diagnostic_float64)

    def validate(self, targets=None):
        if targets is not None and tensor_digest(targets) != self.targets_sha256:
            raise ValueError("Ordered target identity mismatch")
        for value, expected, name in ((self.operator, self.operator_sha256, "operator"),
                                     (self.complete_tokens, self.tokens_sha256, "tokens"),
                                     (self.targets, self.targets_sha256, "targets"),
                                     (self.grams, self.grams_sha256, "Gram"),
                                     (self.powers, self.powers_sha256, "row powers")):
            if value.requires_grad or tensor_digest(value) != expected:
                raise ValueError(name + " custody mismatch")
        return self.complete_tokens[self.targets].detach().clone()


def quadratic(q, grams):
    return torch.einsum("bkl,bkf,blf->b", grams, q, q)


def finite_nonnegative(value, name):
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise FloatingPointError(name + " is nonfinite/negative; author recipe is not repaired")


def normalize_coefficients(q, grams):
    norm2 = torch.stack([quadratic(member, grams) for member in q])
    finite_nonnegative(norm2, "Gram full-X squared norm")
    return q / torch.sqrt(norm2[..., None, None] + 1e-24), norm2


def gram_distances(u, grams, reference=None):
    reference = u.detach() if reference is None else reference.detach()
    d = torch.stack([torch.stack([quadratic(live - ref, grams) for ref in reference])
                     for live in u])
    finite_nonnegative(d, "Gram full-X distance")
    return d


def column_bandwidth(d):
    if d.ndim != 3 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        raise ValueError("D[live,reference,target] with at least two members required")
    values = d.detach().sort(dim=0).values
    count = d.shape[0]
    median = values[count // 2] if count % 2 else (values[count // 2 - 1] + values[count // 2]) / 2
    return median / math.log(count) + 1e-12


def repulsion(d, frozen_bandwidth=None):
    h = column_bandwidth(d) if frozen_bandwidth is None else frozen_bandwidth.detach()
    if not d.shape[2] or h.shape != d.shape[1:]:
        raise ValueError("One whole logical target batch and column bandwidth required")
    if not bool(torch.isfinite(h).all()) or bool((h <= 0).any()):
        raise FloatingPointError("Nonpositive/nonfinite bandwidth; no replacement")
    b = d.shape[2]
    return (torch.logsumexp((-d / h.unsqueeze(0)).flatten(1), dim=1)
            - math.log(b)).sum() / b, h


def selected_coefficients(model, batch, channels):
    rows = batch.validate()
    if (model.spec.K != batch.maximum_power or model.spec.base != batch.base
            or model.spec.num_features != rows.shape[2]
            or model.stage != "source_style_all_private"
            or any(m.training for m in model.modules())
            or channels.dtype != torch.long or channels.shape != (rows.shape[0],)):
        raise ValueError("Qualified native all-private eval composition and score channels required")
    q, logits = [], []
    for member in range(model.members):
        leaf = rows.clone().requires_grad_(True)
        z = model.forward_member(leaf, member)
        score = z.gather(1, channels[:, None]).sum()
        q.append(torch.autograd.grad(score, leaf, create_graph=True, retain_graph=True)[0])
        logits.append(z)
    return torch.stack(q), torch.stack(logits)


def evaluate_coefficients(q, batch, *, backend):
    """Explicit direct backend is the full-X recipe, not an automatic repair."""
    if backend == "gram":
        u, norm2 = normalize_coefficients(q, batch.grams)
        d = gram_distances(u, batch.grams)
        similarity = torch.stack([torch.stack([
            torch.einsum("bkl,bkf,blf->b", batch.grams, live, ref.detach())
            for ref in u]) for live in u])
    elif backend == "direct":
        # Keeps target/member axes and evaluates the unchanged full-X metric.
        full = torch.einsum("bkn,mbkf->mbnf", batch.powers, q)
        norm2 = full.square().sum(dim=(-1, -2))
        u = full / torch.sqrt(norm2[..., None, None] + 1e-24)
        d = (u[:, None] - u.detach()[None, :]).square().sum(dim=(-1, -2))
        similarity = (u[:, None] * u.detach()[None, :]).sum(dim=(-1, -2))
    else:
        raise ValueError("Explicit 'gram' or 'direct' backend required")
    r, h = repulsion(d)
    return r, {"norm2": norm2, "normalized": u, "distance": d,
               "similarity": similarity, "bandwidth": h,
               "kernel": torch.exp(-d / h.unsqueeze(0)).mean(dim=2)}


def source_style_objective(model, batch, class_indices, *, backend):
    """sum-member mean CE + author R_sum; called only by an authorized fitter.

    Engineering runner never calls this labelled objective or takes SGD steps.
    """
    q, logits = selected_coefficients(model, batch, class_indices)
    r, details = evaluate_coefficients(q, batch, backend=backend)
    ce = sum(torch.nn.functional.cross_entropy(z, class_indices) for z in logits)
    return ce + r, details
