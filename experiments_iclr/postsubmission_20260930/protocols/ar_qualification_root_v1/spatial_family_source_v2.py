"""Pinned future complete-instance generator. No draws at module scope."""
from dataclasses import dataclass
from hashlib import sha256
from random import Random
import json


@dataclass(frozen=True)
class SpatialInstance:
    instance_id: str
    seed: int
    points: tuple[tuple[float, float], ...]
    edges: tuple[tuple[int, int], ...]
    order: tuple[int, ...]

    def record(self):
        data = {
            'instance_id': self.instance_id, 'seed': self.seed,
            'n_nodes': len(self.points),
            'transmitter_points': self.points,
            'transmitter_points_hex': [[x.hex(), y.hex()] for x, y in self.points],
            'edges': self.edges, 'order': self.order,
            'edge_rule': 'binary64 dx*dx + dy*dy <= binary64(2.4)*binary64(2.4), separately rounded scalar operations',
            'receiver_SINR_or_topology_filter': False,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        return dict(data, instance_sha256=sha256(canonical).hexdigest())


def make_complete_spatial_instance(*, instance_id: str, seed: int):
    """Fifteen point links, keep every topology and every node.

    This is a declared point-conflict adaptation, not the original SINR
    simulation. Coordinate RNG is Python Random.random binary64 scaled by 8.
    The order uses a separate fixed PRNG stream, independent of any model.
    """
    if not instance_id or type(seed) is not int or seed < 0:
        raise ValueError('Recorded identity and nonnegative seed required')
    n = 15
    rng = Random(seed)
    points = tuple((8.0 * rng.random(), 8.0 * rng.random()) for _ in range(n))
    radius = float(2.4)
    cutoff = radius * radius
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            dx = points[u][0] - points[v][0]
            dy = points[u][1] - points[v][1]
            dx_squared = dx * dx
            dy_squared = dy * dy
            distance_squared = dx_squared + dy_squared
            if distance_squared <= cutoff:
                edges.append((u, v))
    order = list(range(n))
    Random(seed ^ 0x68E31DA4).shuffle(order)
    return SpatialInstance(instance_id, seed, points, tuple(edges), tuple(order))


def complete_integer_target(instance: SpatialInstance, *, fugacity: int):
    """All 2**15 states examined. Integer weights and normalizer.

    For the selected homogeneous integer fugacities this specializes the
    rational oracle with no loss. Normalized/log values remain floating when
    converted later. No enumeration, table or RNG runs at import time.
    """
    if type(fugacity) is not int or fugacity not in (1, 4):
        raise ValueError('Only the prospectively selected fugacity regimes')
    n = len(instance.points)
    if n != 15:
        raise ValueError('Keep the complete fifteen-node instance')
    weighted = []
    partition = 0
    for mask in range(1 << n):
        if any((mask & (1 << u)) and (mask & (1 << v)) for u, v in instance.edges):
            continue
        weight = fugacity ** mask.bit_count()
        weighted.append((mask, weight))
        partition += weight
    if not weighted or partition <= 0:
        raise ArithmeticError('The inactive configuration must have weight one')
    return tuple(weighted), partition
