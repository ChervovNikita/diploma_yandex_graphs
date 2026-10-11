"""Fixed label-free walk metadata; stdlib only, no loader or numerical runtime.

Incoming nonself edge-record multiplicity is preserved. Every predecessor draw
is uniform with replacement; a self slot is used only at zero nonself indegree.
Sampling and swap-mask streams never touch the caller's global training RNG.
"""
from dataclasses import dataclass
import hashlib
import random
import struct

WALK_SEED = 11709
WALKS_PER_NODE = 8
EDGES_PER_WALK = 3
POSITIONS = 4
SWAP_DOMAIN = "ordered-path/intermediate-swap/11709"


@dataclass(frozen=True)
class WalkTable:
    node_count: int
    indices: tuple
    swap_mask: tuple
    counters: tuple
    sha256: str

    def receipt(self):
        return {"node_count": self.node_count, "seed": WALK_SEED,
                "walks_per_node": WALKS_PER_NODE, "positions": POSITIONS,
                "uniform_on": "incoming nonself edge records, with replacement",
                "multiplicity": "preserved; sorted by predecessor ID",
                "fallback": "self only when no incoming nonself record",
                "swap_domain": SWAP_DOMAIN, "sha256": self.sha256,
                "counters": dict(self.counters)}


def sample_walks(node_count, edge_records):
    """Receive only Python-int (source, target) records in original row IDs.

    The root-owned provider converts its authorized public graph explicitly.
    No feature, target, role, prediction or fitted representation is accepted.
    Returns immutable common tuples and an independent frozen Bernoulli mask.
    """
    if type(node_count) is not int or not 0 < node_count < 2**63:
        raise ValueError("Positive int64-compatible complete node count required")
    incoming = [[] for _ in range(node_count)]
    self_records = nonself_records = 0
    for source, target in edge_records:
        if type(source) is not int or type(target) is not int:
            raise TypeError("Provider must supply plain Python integer edge IDs")
        if not 0 <= source < node_count or not 0 <= target < node_count:
            raise ValueError("Edge record outside original complete node rows")
        if source == target:
            self_records += 1
        else:
            incoming[target].append(source)
            nonself_records += 1
    for predecessors in incoming:
        predecessors.sort()  # Canonical order, duplicates remain factual draws.
    walk_rng = random.Random(WALK_SEED)
    mask_seed = int.from_bytes(hashlib.sha256(SWAP_DOMAIN.encode()).digest(), "big")
    mask_rng = random.Random(mask_seed)
    all_walks, all_masks = [], []
    draws = fallbacks = swaps = id_changing_swaps = 0
    digest = hashlib.sha256(struct.pack("<qqq", node_count, WALKS_PER_NODE, POSITIONS))
    for root in range(node_count):
        node_walks, node_masks = [], []
        for _ in range(WALKS_PER_NODE):
            walk = [root]
            for _ in range(EDGES_PER_WALK):
                predecessors = incoming[walk[-1]]
                if predecessors:
                    walk.append(predecessors[walk_rng.randrange(len(predecessors))])
                    draws += 1
                else:
                    walk.append(walk[-1])
                    fallbacks += 1
            swap = bool(mask_rng.getrandbits(1))
            swaps += int(swap)
            id_changing_swaps += int(swap and walk[1] != walk[2])
            item = tuple(walk)
            node_walks.append(item)
            node_masks.append(swap)
            digest.update(struct.pack("<qqqq", *item))
            digest.update(bytes((int(swap),)))
        all_walks.append(tuple(node_walks))
        all_masks.append(tuple(node_masks))
    counters = {"input_self_records_excluded_from_sampler": self_records,
                "incoming_nonself_records": nonself_records,
                "empty_nonself_predecessor_rows": sum(not p for p in incoming),
                "walks": node_count * WALKS_PER_NODE,
                "predecessor_draws": draws, "self_fallback_steps": fallbacks,
                "frozen_swap_tuples": swaps,
                "swaps_changing_intermediate_node_ID_order": id_changing_swaps}
    return WalkTable(node_count, tuple(all_walks), tuple(all_masks),
                     tuple(sorted(counters.items())), digest.hexdigest())
