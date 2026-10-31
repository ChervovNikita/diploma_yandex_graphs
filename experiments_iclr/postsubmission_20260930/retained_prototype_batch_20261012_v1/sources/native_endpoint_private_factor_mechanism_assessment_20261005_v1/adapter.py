"""Source-only outer endpoint extension of the pinned portable NCNC prototype.

Generic orthogonal-context/symmetric-bilinear ancestry; no novelty claim.
Factory inputs must be modules loaded from the two exact pinned source files.
This module itself imports only the standard library. It supplies no launcher.
"""
from hashlib import sha256
from pathlib import Path


PARENT_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "graph_ncNC_member_completion_qualification_preparation_20261003_v2"
)
PARENT_SOURCE_PINS = {
    "prototype.py": "1f954a340a65d02089661b4dab567cd0c2fa7efadcfa5b03cbe4f228c5e1bda0",
    "graph_ops.py": "37edca94c7980432c8b8c850a4e05fcdba855323136d996877d20e5d65713be5",
}


def verify_parent_sources():
    """Read source bytes only; reject a modified or absent parent."""
    for filename, expected in PARENT_SOURCE_PINS.items():
        path = PARENT_DIRECTORY / filename
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned parent source differs: {path}")
    return PARENT_DIRECTORY


def make_adapter_classes(prototype_module, graph_ops_module):
    """Return (decoder_class, twin_class) for exact caller-loaded parent modules.

    Load graph_ops.py as `graph_ops` before loading prototype.py, whose import
    contract uses that name. The factory verifies paths, bytes and graph-symbol
    identity. Parent loss/optimizer helpers accept the resulting Twin directly.
    """
    verify_parent_sources()
    for module, filename in (
        (prototype_module, "prototype.py"),
        (graph_ops_module, "graph_ops.py"),
    ):
        if Path(module.__file__).resolve() != (PARENT_DIRECTORY / filename).resolve():
            raise RuntimeError(f"Unexpected parent module path: {module.__file__}")
    for name in ("Graph", "enumerate_neighbors", "feature_sum"):
        if getattr(prototype_module, name) is not getattr(graph_ops_module, name):
            raise RuntimeError(f"Parent graph import was shadowed: {name}")

    torch, nn = prototype_module.torch, prototype_module.nn
    enumerate_neighbors = graph_ops_module.enumerate_neighbors
    feature_sum = graph_ops_module.feature_sum
    clamp_completion = prototype_module.clamp_completion
    route_weights = prototype_module.route_weights

    class EndpointReflectionDecoder(prototype_module.CompletionDecoder):
        def __init__(self, recipe):
            if recipe.member_count > recipe.hidden:
                raise ValueError("Axis initialization requires members <= hidden")
            super().__init__(recipe)
            reference = self.xijlin.ops["0"].weight
            # Exactly e_m, one distinct axis per member; no RNG call.
            self.v = nn.Parameter(torch.eye(
                self.members, recipe.hidden,
                dtype=reference.dtype, device=reference.device,
            ))

        def endpoint_product(self, h, queries, member):
            """Reflect both outer endpoints, then compress; no epsilon/clipping."""
            vector = self.v[member]
            q = torch.dot(vector, vector)
            if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(q)) or not bool(q > 0):
                raise FloatingPointError("Householder vector/norm must be finite and nonzero")
            left = h[queries[:, 0]]
            right = h[queries[:, 1]]
            left = left - 2 * (left @ vector).unsqueeze(-1) * vector / q
            right = right - 2 * (right @ vector).unsqueeze(-1) * vector / q
            return left * right

        def forward(self, h, graph, queries, mode, return_details=False):
            # Copied portable parent prototype.py:187-221, changing only the
            # products.append expression at its line198. No recursion override.
            neighbors = enumerate_neighbors(graph, queries)
            lq, ln = neighbors.left
            rq, rn = neighbors.right
            left_query = torch.stack((queries[lq, 1], ln), dim=1)
            right_query = torch.stack((queries[rq, 0], rn), dim=1)
            products, transformed, left, right, scores = [], [], [], [], []
            # Both twins use this identical phase-A member/candidate call schedule.
            # Per-member native xlin -> left recursion -> right recursion is kept;
            # only the cross-member final decode interleaving is standardized.
            for member in range(self.members):
                products.append(self.endpoint_product(h, queries, member))
                hm = h + self.xlin.forward_member(h, member)
                transformed.append(hm)
                ls = self.completion_scores(hm, graph, left_query, member)
                rs = self.completion_scores(hm, graph, right_query, member)
                scores.append((ls, rs))
                left.append(clamp_completion(ls, self.scale, self.offset, self.alpha, self.pt))
                right.append(clamp_completion(rs, self.scale, self.offset, self.alpha, self.pt))
            raw_left, raw_right = torch.stack(left), torch.stack(right)
            routed_left, routed_right = route_weights(raw_left, mode), route_weights(raw_right, mode)
            outputs = []
            # Native xlin/downstream feature graph was not detached with scores.
            for member in range(self.members):
                hm = transformed[member]
                common = feature_sum(hm, neighbors.common, len(queries))
                common = common + feature_sum(hm, neighbors.left, len(queries), routed_left[member])
                common = common + feature_sum(hm, neighbors.right, len(queries), routed_right[member])
                outputs.append(self.decode(products[member], common, member))
            logits = torch.stack(outputs, dim=1)
            if return_details:
                return logits, {"neighbors":neighbors,"transformed":transformed,
                                "scores":scores,"raw_left":raw_left,"raw_right":raw_right,
                                "routed_left":routed_left,"routed_right":routed_right}
            return logits

    class EndpointReflectionTwin(prototype_module.CompletionTwin):
        def __init__(self, recipe=None):
            # Same parent constructor order, substituting the decoder class.
            # Calling parent then replacing its decoder would consume RNG twice.
            nn.Module.__init__(self)
            self.recipe = recipe or prototype_module.Recipe()
            self.recipe.validate()
            self.encoder = prototype_module.SharedEncoder(self.recipe)
            self.decoder = EndpointReflectionDecoder(self.recipe)

    return EndpointReflectionDecoder, EndpointReflectionTwin
