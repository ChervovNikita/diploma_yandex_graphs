"""NCNC native forward inherited verbatim; auxiliary-only scorer autograd."""
import torch
from torch import nn
from prototype import CompletionTwin, CompletionDecoder, SharedEncoder, Recipe
from graph_ops import feature_sum
from pilot_common import require
from pilot_model import seed_all


class PatternDecoder(CompletionDecoder):
    def diagnostic_routes(self, h, graph, queries):
        """One fixed scorer bank; recipient features/decoder held fixed."""
        require(not self.training, "Fixed-bank routes require evaluation mode")
        own, details = super().forward(h, graph, queries, "private", True)
        left, right = details["raw_left"], details["raw_right"]
        routes = {"own": own}
        alternatives = {"crossed_cyclic_"+str(shift):(left.roll(-shift,0),right.roll(-shift,0)) for shift in (1,2,3)}
        alternatives["pooled_clamped_weights"]=(left.mean(0,keepdim=True).expand_as(left),right.mean(0,keepdim=True).expand_as(right))
        for name,(lw,rw) in alternatives.items():
            outputs=[]
            for member in range(self.members):
                hm=details["transformed"][member]
                common=feature_sum(hm,details["neighbors"].common,len(queries))
                common=common+feature_sum(hm,details["neighbors"].left,len(queries),lw[member])
                common=common+feature_sum(hm,details["neighbors"].right,len(queries),rw[member])
                endpoint=h[queries[:,0]]*h[queries[:,1]]
                outputs.append(self.decode(endpoint,common,member))
            routes[name]=torch.stack(outputs,dim=1)
        return routes

    def completion_scores(self, transformed, graph, query, member):
        capture = getattr(self, "_pattern_capture", None)
        if capture is None:
            return super().completion_scores(transformed, graph, query, member)
        size = self.recipe.candidate_split_size
        context = torch.enable_grad() if self._pattern_gradient else torch.no_grad()
        with context:
            if size < 0:
                score = self.depth_zero(transformed, graph, query, member)
            else:
                chunks = [self.depth_zero(transformed, graph, query[start:start + size], member)
                          for start in range(0, len(query), size)]
                score = torch.cat(chunks) if chunks else transformed.new_empty(0)
        capture.append((member, score))
        # The inherited target forward sees only detached scores and performs
        # the exact existing clamp, private weighted sums and private decoding.
        return score.detach()

    def pattern_forward(self, h, graph, queries, *, auxiliary_grad=True):
        require(getattr(self, "_pattern_capture", None) is None, "Nested capture")
        capture = []
        self._pattern_capture = capture
        self._pattern_gradient = auxiliary_grad
        try:
            logits, details = super().forward(h, graph, queries, "private", True)
        finally:
            self._pattern_capture = None
            self._pattern_gradient = False
        require(len(capture) == 2 * self.members and all(capture[2*m][0] == capture[2*m+1][0] == m for m in range(self.members)), "Native member/left/right call order changed")
        left = torch.stack([capture[2*m][1] for m in range(self.members)])
        right = torch.stack([capture[2*m+1][1] for m in range(self.members)])
        raw_scores = torch.cat((left, right), dim=1)
        t = self.scale * (raw_scores - self.offset) + torch.log(self.pt)
        native_q = torch.cat((details["raw_left"], details["raw_right"]), dim=1) / self.alpha
        return logits, {"neighbors": details["neighbors"], "t": t, "native_q": native_q,
                        "raw_score_tensors": [x[1] for x in capture], "raw_left": details["raw_left"],
                        "raw_right": details["raw_right"]}


class PatternTwin(CompletionTwin):
    def __init__(self, recipe=None):
        # Same constructor order/draws/state keys as the qualified source.
        nn.Module.__init__(self)
        self.recipe = recipe or Recipe()
        self.recipe.validate()
        self.encoder = SharedEncoder(self.recipe)
        self.decoder = PatternDecoder(self.recipe)


def make_pattern(mods, seed, device, *, engineering_sign_seed=None):
    seed_all(seed)
    model = PatternTwin().to(device)
    sign_seed = mods["design"].factor_sign_seed(seed) if engineering_sign_seed is None else engineering_sign_seed
    generator = torch.Generator(device="cpu").manual_seed(sign_seed)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if name.endswith(".r") or name.endswith(".s"):
                signs = 2 * torch.randint(0, 2, parameter.shape, dtype=torch.int64, generator=generator, device="cpu") - 1
                parameter.copy_(signs.to(device=device, dtype=parameter.dtype))
    optimizer = mods["prototype"].native_optimizer(model)
    require(sum(p.numel() for p in model.parameters()) == 43790, "Qualified F4 parameter family differs")
    return model, optimizer
