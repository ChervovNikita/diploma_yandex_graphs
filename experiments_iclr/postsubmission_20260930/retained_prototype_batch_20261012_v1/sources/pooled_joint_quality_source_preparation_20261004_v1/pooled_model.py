"""Fresh J_P/F_P: unchanged F4 architecture, pooled clamped target weights.

NCNC recursive completion/readout, BatchEnsemble sharing and GRAN mixture
auxiliary ancestry are inherited and credited in README. No donor state.
"""
from hashlib import sha256
import torch
from torch import nn
from prototype import Recipe, SharedEncoder, CompletionTwin, CompletionDecoder, native_optimizer
from pattern_model import PatternDecoder
from pilot_model import seed_all
from count_density import require

MODEL_SEED = 610041


def domain_seed(domain):
    return int.from_bytes(sha256(("pooled-joint-quality-v1|seed=610041|" + domain).encode("ascii")).digest()[:8], "little") % (2**31 - 1)


class PooledDecoder(PatternDecoder):
    def pattern_forward(self, h, graph, queries, *, auxiliary_grad=True):
        require(getattr(self, "_pattern_capture", None) is None, "Nested scorer capture")
        capture = []
        self._pattern_capture, self._pattern_gradient = capture, auxiliary_grad
        try:
            logits, details = CompletionDecoder.forward(
                self, h, graph, queries, "pooled_after_clamp", True)
        finally:
            self._pattern_capture, self._pattern_gradient = None, False
        require(len(capture) == 8 and all(capture[2*m][0] == capture[2*m+1][0] == m for m in range(4)),
                "Native four-member left/right scorer schedule changed")
        score = torch.cat((torch.stack([capture[2*m][1] for m in range(4)]),
                           torch.stack([capture[2*m+1][1] for m in range(4)])), 1)
        t = self.scale * (score - self.offset) + torch.log(self.pt)
        return logits, {**details, "t": t,
                        "native_q": torch.cat((details["raw_left"], details["raw_right"]), 1) / self.alpha,
                        "raw_score_tensors": [part[1] for part in capture]}


class PooledTwin(CompletionTwin):
    def __init__(self):
        nn.Module.__init__(self)
        self.recipe = Recipe()
        self.recipe.validate()
        self.encoder, self.decoder = SharedEncoder(self.recipe), PooledDecoder(self.recipe)

    def encode(self, x, graph):
        return self.encoder(x, graph)

    def query_forward(self, h, graph, queries, *, auxiliary_grad=True):
        return self.decoder.pattern_forward(h, graph, queries, auxiliary_grad=auxiliary_grad)


def make_pooled(device):
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled()
            and not torch.is_autocast_enabled("cpu"), "Plain native FP32 profile required")
    seed_all(MODEL_SEED)
    model = PooledTwin().to(device)
    generator = torch.Generator(device="cpu").manual_seed(domain_seed("factor-signs"))
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if name.endswith(".r") or name.endswith(".s"):
                signs = 2 * torch.randint(0, 2, parameter.shape, generator=generator, dtype=torch.int64) - 1
                parameter.copy_(signs.to(device=parameter.device, dtype=parameter.dtype))
    require(sum(p.numel() for p in model.parameters()) == 43790, "Frozen pooled architecture differs")
    return model, native_optimizer(model)
