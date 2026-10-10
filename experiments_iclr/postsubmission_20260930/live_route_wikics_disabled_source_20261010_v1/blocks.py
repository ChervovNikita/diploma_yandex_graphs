"""Fixed WikiCS blocks. Import is stdlib-only; construction is explicit.

Source-only and unadmitted. No CLI, model construction, numerical import,
training, teacher acquisition or provider access runs when this is imported.
The future reviewed caller supplies its already bound torch module.
"""
import math

WIDTH = 512
RANK = 16
MEMBERS = 4
EPS = 1e-5
KINDS = ("exchange", "separable")


def make_block(torch, kind, initializer_seed):
    """Create 32,768 shared own-risk weights using an isolated CPU generator.

    No nn.Linear constructor/default reset consumes the caller's RNG stream.
    Parameters are first allocated on CPU; .to(device) belongs to the fresh
    pre-Adam constructor. The final map is zero in both fixed constructions.
    """
    if kind not in KINDS or type(initializer_seed) is not int:
        raise ValueError("One fixed block kind and integer initializer seed")
    generator = torch.Generator(device="cpu").manual_seed(initializer_seed)

    def parameter(rows, columns, zero=False):
        value = torch.empty(rows, columns, dtype=torch.float32, device="cpu")
        if zero:
            value.zero_()
        else:
            bound = math.sqrt(6.0 / (rows + columns))
            value.uniform_(-bound, bound, generator=generator)
        return torch.nn.Parameter(value)

    class FixedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.initializer_seed = initializer_seed
            if kind == "exchange":
                self.Q = parameter(WIDTH, RANK)
                self.K = parameter(WIDTH, RANK)
                self.V = parameter(WIDTH, RANK)
                self.U = parameter(RANK, WIDTH, zero=True)
            else:
                self.B = parameter(WIDTH, 2 * RANK)
                self.C = parameter(2 * RANK, WIDTH, zero=True)

        def forward(self, raw):
            if raw.ndim != 3 or tuple(raw.shape[1:]) != (MEMBERS, WIDTH):
                raise ValueError("Complete node, four-route, width512 operands")
            if raw.dtype != torch.float32:
                raise ValueError("Bound native float32 operands; no AMP port")
            z = torch.nn.functional.layer_norm(raw, (WIDTH,), eps=EPS)
            if self.kind == "separable":
                return raw + torch.nn.functional.relu(z @ self.B) @ self.C
            q, k, v = z @ self.Q, z @ self.K, z @ self.V
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(RANK)
            diagonal = torch.eye(MEMBERS, dtype=torch.bool, device=raw.device)
            scores = scores.masked_fill(diagonal.unsqueeze(0), float("-inf"))
            attention = scores.softmax(dim=-1)
            return raw + (attention @ v) @ self.U

    return FixedBlock()
