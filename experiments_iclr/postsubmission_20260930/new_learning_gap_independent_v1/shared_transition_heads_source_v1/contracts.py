"""Future package-relative reuse of the sealed AR contracts, without changes.

This source has not been imported. Importing must occur only in a prospectively
admitted environment with the parent directory available as a package root.
The packet manifest binds every reused module to its original source hash.
"""

# Admission must verify these dependency bytes before future package imports.
# This metadata is not an automatic runtime hash verifier.
REUSED_SOURCE_SHA256 = {
    "domain.py": "3a4f9e6b676b0c5ed2ef0030c4cbc4fa9b9e5b1f5f28f88a8ee2165b378b195b",
    "source_encoder.py": "5cc6d42c1a5f2cd7338f9295f6f4f69fc8bca16cc8c57af39df615f156cc4ca2",
    "mixture_objective.py": "01acec09972ec83f21591bd0aa58d25814688cb989a456ebeca2700989a090fd",
    "work_accounting.py": "420aa2c6a9df9e7543f7d723f673171639c614117909b236000b11cf46c792a3",
}

from ..ar_mixture_source_v1.domain import (
    FixedMixture, GeneratedSchedule, IndependentBaseline, ScheduleDomain,
)
from ..ar_mixture_source_v1.source_encoder import (
    EncodedGraph, make_static_source_encoder,
)
from ..ar_mixture_source_v1.mixture_objective import (
    ReverseKLResult, fixed_mixture_reverse_kl_surrogate,
)
from ..ar_mixture_source_v1.work_accounting import EncoderAudit
