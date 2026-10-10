"""Inactive TRAIN-only context preparation, using the exact native label operators.

No torch/provider, dataset file, role loader or model is imported here. Root must
provide a custody-verified TRAIN-only bundle and native diagonal-removed operators.
"""
from dataclasses import dataclass
import hashlib
import json
import random

from .caps import CLOSED


LABEL_PATHS = (
    "MAM", "MAMAM", "MAMDM", "MAMKM", "MDM", "MDMAM", "MDMDM",
    "MDMKM", "MKM", "MKMAM", "MKMDM", "MKMKM",
)
LOCAL_PATHS = ("MAM", "MDM", "MKM")
CLASS_DIM = 5
MOVIE_NODES = 4932
FIELD_DIM = len(LOCAL_PATHS) * (CLASS_DIM + 1)


@dataclass(frozen=True)
class TrainOnly:
    ids: tuple
    targets: object  # [len(ids),5], never a full-y/VALID/TEST tensor.
    custody_sha256: str


@dataclass(frozen=True)
class HalfPlan:
    train_ids: tuple
    halves: tuple
    seed: int
    custody_sha256: str
    identity_sha256: str


@dataclass(frozen=True)
class HalfView:
    context_ids: tuple
    query_ids: tuple
    query_positions: tuple
    label_features: dict
    local_field: object  # [4932,18], same factual movie order as native channels.
    global_field: object  # [18], all-movie mean of the same local field.


@dataclass(frozen=True)
class PairedViews:
    plan: HalfPlan
    views: tuple
    support_audit: tuple
    native_paths: tuple = LABEL_PATHS
    local_paths: tuple = LOCAL_PATHS


def _fingerprint(train_ids, halves, seed, custody):
    payload = {"TRAIN": train_ids, "halves": halves, "seed": seed, "custody": custody}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_plan(bundle, seed, caps=CLOSED):
    caps.require("source_bound", "data", "runtime")
    ids = tuple(bundle.ids)
    if (not bundle.custody_sha256 or len(ids) < 2 or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < MOVIE_NODES for i in ids)):
        raise ValueError("Custody-bound unique complete TRAIN movie identities required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Prospectively fixed half seed required")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)  # Dedicated stream; no native RNG change.
    middle = len(shuffled) // 2
    halves = (tuple(sorted(shuffled[:middle])), tuple(sorted(shuffled[middle:])))
    if set(halves[0]) & set(halves[1]) or set(halves[0]) | set(halves[1]) != set(ids):
        raise ValueError("The two halves must partition complete TRAIN exactly")
    fingerprint = _fingerprint(ids, halves, seed, bundle.custody_sha256)
    return HalfPlan(ids, halves, seed, bundle.custody_sha256, fingerprint)


def prepare_views(torch, sparse_type, bundle, plan, operators, caps=CLOSED):
    caps.require("source_bound", "data", "runtime")
    if (tuple(bundle.ids) != plan.train_ids or bundle.custody_sha256 != plan.custody_sha256
            or set(operators) != set(LABEL_PATHS)):
        raise ValueError("Exact TRAIN custody/halves and all12 native label operators required")
    if (len(plan.halves) != 2 or any(not half or len(set(half)) != len(half) for half in plan.halves)
            or set(plan.halves[0]) & set(plan.halves[1])
            or set(plan.halves[0]) | set(plan.halves[1]) != set(plan.train_ids)
            or _fingerprint(plan.train_ids, plan.halves, plan.seed, plan.custody_sha256) != plan.identity_sha256):
        raise ValueError("Frozen halves must exactly partition full TRAIN and retain their fingerprint")
    y = bundle.targets
    if tuple(y.shape) != (len(plan.train_ids), CLASS_DIM) or not torch.isfinite(y).all().item():
        raise ValueError("Only complete finite TRAIN five-bit targets are accepted")
    if not ((y == 0) | (y == 1)).all().item() or y.requires_grad:
        raise ValueError("Native binary labels, independent of model gradients, required")
    for op in operators.values():
        if type(op) is not sparse_type or tuple(op.sparse_sizes()) != (MOVIE_NODES, MOVIE_NODES):
            raise ValueError("Literal full native sparse movie label operator required")
        row, col, value = op.coo()
        if (row == col).any().item():
            raise ValueError("Native label operators must already have their diagonal removed")
        if value is not None and (not torch.isfinite(value).all().item() or (value < 0).any().item()):
            raise ValueError("Native label operator weights must be finite/nonnegative")
        # The author removes the diagonal after normalized products, without
        # renormalization. This tolerance is a setup invariant, not quality evidence.
        if (op.sum(dim=-1) > 1 + 1e-6).any().item():
            raise ValueError("Preserve native substochastic diagonal-removed operators")
    positions = {node: p for p, node in enumerate(plan.train_ids)}
    views, audits = [], []
    for index in (0, 1):
        context_ids = plan.halves[index]
        query_ids = plan.halves[1-index]
        if set(context_ids) & set(query_ids):
            raise ValueError("Query labels must be excluded from context")
        context_positions = tuple(positions[i] for i in context_ids)
        source = torch.zeros((MOVIE_NODES, CLASS_DIM), dtype=y.dtype, device=y.device)
        known = torch.zeros((MOVIE_NODES, 1), dtype=y.dtype, device=y.device)
        ctx_ids = torch.tensor(context_ids, dtype=torch.long, device=y.device)
        ctx_positions = torch.tensor(context_positions, dtype=torch.long, device=y.device)
        source[ctx_ids] = y[ctx_positions]
        known[ctx_ids] = 1
        query_tensor = torch.tensor(query_ids, dtype=torch.long, device=y.device)
        if source[query_tensor].count_nonzero().item() or known[query_tensor].count_nonzero().item():
            raise ValueError("Query supervision entered a predictor input")
        # All12 native positive-label channels, same operators and key order.
        labels = {key: operators[key] @ source for key in LABEL_PATHS}
        columns = []
        for key in LOCAL_PATHS:
            mass = operators[key] @ known
            # Reuse the already computed native positive channel. This is the
            # linear image of2*source-known; no extra signed-source sparse pass.
            signed_mass = 2 * labels[key] - mass
            safe = torch.where(mass > 0, mass, torch.ones_like(mass))
            balance = torch.where(mass > 0, signed_mass / safe, torch.zeros_like(signed_mass))
            columns.extend((balance, mass))
        field = torch.cat(columns, dim=1)
        if tuple(field.shape) != (MOVIE_NODES, FIELD_DIM) or not torch.isfinite(field).all().item():
            raise ValueError("Complete finite signed evidence field required")
        if any(not torch.isfinite(tensor).all().item() for tensor in labels.values()):
            raise ValueError("Finite native half-context label channels required")
        views.append(HalfView(context_ids, query_ids, tuple(positions[i] for i in query_ids),
                              labels, field, field.mean(dim=0)))
        positive = y[ctx_positions].sum(dim=0)
        audits.append({"context": len(context_ids), "query": len(query_ids),
                       "positive_bits": positive.detach().cpu().tolist(),
                       "negative_bits": (len(context_ids)-positive).detach().cpu().tolist(),
                       "no_query_labels_in_context": True})
    return PairedViews(plan, tuple(views), tuple(audits))


def half_inputs(torch, view, ids, feature_dict, global_context=False, caps=CLOSED):
    caps.require("source_bound", "data", "runtime")
    index = torch.tensor(tuple(ids), dtype=torch.long, device=view.local_field.device)
    if global_context:
        field = view.global_field.unsqueeze(0).expand(len(ids), -1)
    else:
        field = view.local_field[index]
    labels = {key: value[index] for key, value in view.label_features.items()}
    features = {key: value[index.to(value.device)] for key, value in feature_dict.items()}
    return index, features, labels, field
