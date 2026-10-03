"""Native split-column and compact TRAIN-only role preparation; no data loaded."""
from dataclasses import dataclass
from hashlib import sha256
from collections import defaultdict
from core.groups import LabelRoles


@dataclass(frozen=True)
class NativeTrainIds:
    ids: tuple[int, ...]
    official_split_id: int
    release_and_mask_receipt: str
    source_mask_layout: str = "PyG2.7_processed_nodes_by_splits"


def select_native_train_column(train_mask, *, node_count: int, expected_split_count: int,
                               official_split_id: int, release_and_mask_receipt: str):
    """PyG2.7 native heterophilous masks use [:,split], equivalent to the
    author Amazon/standard branch heter_fixed_splits permute(1,0)[split].
    Row indexing belongs to the separate filtered chameleon/squirrel loader;
    this TRAIN-only interface does not accept that distinct mask layout.

    Accepts the native TRAIN mask only, not labels or validation/test arrays.
    Exact release/native mask hashes and expected split count are external gates.
    """
    import torch
    if (train_mask.dtype != torch.bool or tuple(train_mask.shape) != (node_count, expected_split_count) or
            not 0 <= official_split_id < expected_split_count or not release_and_mask_receipt):
        raise ValueError("Native [nodes,splits] TRAIN-mask layout/receipt invalid")
    ids = tuple(train_mask[:, official_split_id].nonzero(as_tuple=False).flatten().cpu().tolist())
    if not ids:
        raise ValueError("Empty native official TRAIN role")
    return NativeTrainIds(ids, official_split_id, release_and_mask_receipt)


def fixed_stratified_roles(native: NativeTrainIds, compact_train_labels: tuple[int, ...], *,
                           class_count: int, role_seed: str, explicit_role_split_receipt: str,
                           forbidden_node_ids: tuple[int, ...] = ()):
    """Proposed exact 80/20 rounding: floor(4*n_class/5) fit, remaining control.

    Fixed hash order per class before any outcomes. No class/target is dropped.
    Tiny classes may have zero fit or control targets; support/active screens stop
    where required. Do not silently force one per role, rebalance or retry seeds.
    This precise rounding/seed policy needs explicit prospective adoption.
    """
    if len(compact_train_labels) != len(native.ids) or not role_seed or not explicit_role_split_receipt:
        raise ValueError("Parallel compact official TRAIN labels and explicit split policy required")
    labels = tuple(zip(native.ids, compact_train_labels))
    by_class = defaultdict(list)
    for n, y in labels:
        if type(y) is not int or not 0 <= y < class_count:
            raise ValueError("Invalid compact TRAIN class")
        by_class[y].append(n)
    fit, control = set(), set()
    for y, ids in sorted(by_class.items()):
        order = sorted(ids, key=lambda n: (sha256(f"{role_seed}|fit_control|{y}|{n}".encode()).hexdigest(), n))
        count = 4 * len(ids) // 5
        fit.update(order[:count]); control.update(order[count:])
    return LabelRoles(tuple(n for n in native.ids if n in fit), tuple(n for n in native.ids if n in control),
                      labels, class_count, forbidden_node_ids)
