"""HeaRT Citeseer NCN head adapters, using native donor initialization/operators.

Householder products and sole four-block initializer reuse the preceding frame
packets. No recursion exists in NCN. No Collab encoder or portable graph code.
Imports/constructs numerical modules only when make_predictor is called.
"""

def make_predictor(native, *, arm, members, width=256, axis_index=None, factor_seed=None):
    torch, nn, F = native.torch, native.nn, native.F
    valid = {"native_single": 1, "independent_member": 1, "independent_frame_member": 1,
             "unframed_f4": 4, "shared_frame_f4": 4, "private_frame_f4": 4,
             "same_four_frames_single": 1}
    if arm not in valid or type(members) is not int or members != valid[arm] or width != 256:
        raise ValueError("Explicit native member count/width does not match arm")
    ordinary = native.CNLinkPredictor(width, width, 1, 1, .3, edrop=0., ln=True,
                    cndeg=-1, use_xlin=True, tailact=True, twolayerlin=True, beta=1.)
    if arm in ("native_single", "independent_member"):
        return ordinary

    def reflect(x, vector):
        q = torch.dot(vector, vector)
        if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(q)) or not bool(q > 0):
            raise FloatingPointError("Householder vector/norm must be finite and nonzero")
        return x - 2 * (x @ vector).unsqueeze(-1) * vector / q

    class FactorLinear(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.weight, self.bias = original.weight, original.bias
            self.r = nn.Parameter(self.weight.new_ones((members, original.in_features)))
            self.s = nn.Parameter(self.weight.new_ones((members, original.out_features)))
        def forward_member(self, x, member):
            return F.linear(x * self.r[member], self.weight) * self.s[member] + self.bias

    class PrivateNorm(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.normalized_shape, self.eps = original.normalized_shape, original.eps
            self.weight = nn.Parameter(original.weight.detach().expand(members, -1).clone())
            self.bias = nn.Parameter(original.bias.detach().expand(members, -1).clone())
        def forward_member(self, x, member):
            return F.layer_norm(x, self.normalized_shape, self.weight[member], self.bias[member], self.eps)

    class MemberSequence(nn.Module):
        def __init__(self, sequence):
            super().__init__()
            self.ops = nn.ModuleList(FactorLinear(op) if isinstance(op, nn.Linear) else
                      PrivateNorm(op) if isinstance(op, nn.LayerNorm) else op for op in sequence)
        def forward_member(self, x, member):
            for op in self.ops:
                x = op.forward_member(x, member) if isinstance(op, (FactorLinear, PrivateNorm)) else op(x)
            return x

    class FourFrameLinear(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.in_features, self.out_features = 4 * width, width
            common = original.weight.detach() / 4
            self.weight = nn.Parameter(torch.cat((original.weight.detach(), common, common, -2 * common), 1))
            self.bias = original.bias
        def forward(self, x):
            if x.shape[-1] != self.in_features:
                raise ValueError("NCN four-frame target requires4d features")
            return F.linear(x, self.weight, self.bias)

    class FrameSingle(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.members = 1
            for name in ("xlin", "xcnlin", "xijlin", "lin", "beta", "dropadj"):
                setattr(self, name, getattr(original, name))
            nframes = 4 if arm == "same_four_frames_single" else 1
            self.v = nn.Parameter(self.xijlin[0].weight.new_zeros((nframes, width)))
            if nframes == 1:
                if type(axis_index) is not int or not 0 <= axis_index < 4:
                    raise ValueError("Independent framed member requires axis_index0..3")
                self.v.data[0, axis_index] = 1
            else:
                self.v.data[:, :4].copy_(torch.eye(4, dtype=self.v.dtype, device=self.v.device))
                old = self.xijlin[0]
                # Same four-block initializer as same-operation controls v2:
                # preserve native initial function in algebra, no extra RNG.
                self.xijlin[0] = FourFrameLinear(old)
        def forward(self, x, adj, edges):
            adj = self.dropadj(adj)
            left, right = x[edges[0]], x[edges[1]]
            transformed = x + self.xlin(x)
            cn = native.adjoverlap(adj, adj, edges, False, cnsampledeg=-1)
            common = native.spmm_add(cn, transformed)
            product = torch.cat([reflect(left, vector) * reflect(right, vector) for vector in self.v], -1)
            pair = self.xijlin(product)
            return self.lin(self.xcnlin(common) * self.beta + pair)

    class MemberBank(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.members = members
            self.dropadj = original.dropadj
            for name in ("xlin", "xcnlin", "xijlin", "lin"):
                setattr(self, name, MemberSequence(getattr(original, name)))
            self.beta = nn.Parameter(original.beta.detach().expand(members).clone())
            nframes = 1 if arm == "shared_frame_f4" else members if arm == "private_frame_f4" else 0
            if nframes:
                self.v = nn.Parameter(self.xijlin.ops[0].weight.new_zeros((nframes, width)))
                self.v.data[:, :nframes].copy_(torch.eye(nframes, dtype=self.v.dtype, device=self.v.device))
            if type(factor_seed) is not int or not 0 <= factor_seed < 2**63:
                raise ValueError("Bank requires an explicit prospectively fixed factor_seed")
            generator = torch.Generator(device="cpu").manual_seed(factor_seed)
            with torch.no_grad():
                for name, parameter in sorted(self.named_parameters()):
                    if name.endswith(".r") or name.endswith(".s"):
                        signs = 2 * torch.randint(0, 2, parameter.shape, dtype=torch.int64, generator=generator) - 1
                        parameter.copy_(signs.to(device=parameter.device, dtype=parameter.dtype))
        def forward(self, x, adj, edges):
            adj = self.dropadj(adj)
            left, right = x[edges[0]], x[edges[1]]
            cn = native.adjoverlap(adj, adj, edges, False, cnsampledeg=-1)
            outputs = []
            for member in range(self.members):
                transformed = x + self.xlin.forward_member(x, member)
                common = native.spmm_add(cn, transformed)
                product = left * right
                if hasattr(self, "v"):
                    vector = self.v[0 if len(self.v) == 1 else member]
                    product = reflect(left, vector) * reflect(right, vector)
                pair = self.xijlin.forward_member(product, member)
                context = self.xcnlin.forward_member(common, member)
                outputs.append(self.lin.forward_member(context * self.beta[member] + pair, member))
            return torch.cat(outputs, dim=-1)

    return FrameSingle(ordinary) if members == 1 else MemberBank(ordinary)
