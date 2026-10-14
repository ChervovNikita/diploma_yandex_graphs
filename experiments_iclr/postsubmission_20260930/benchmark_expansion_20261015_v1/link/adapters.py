"""Optional small cross-task hypotheses; no native session is edited here."""
import math


def make_hidden_residual(torch, width, members=4, rank=8, seed=0, nonlinear=True):
    """h + B_m GELU(A_m h), at the fixed native TASK prehead only.

    B starts at zero, preserving the existing wrapped function initially.
    This is a nonlinear hidden adapter, not a rank-8 scalar readout claim.
    With nonlinear=False plus a scalar affine head, it is an ordinary linear
    head-capacity control. A shared members=1/rank=32 adapter matches the
    parameter count of four rank-8 private adapters at the same hidden width.
    """
    if min(width, members, rank) < 1:
        raise ValueError('Positive adapter dimensions required')
    class HiddenResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            self.A = torch.nn.Parameter(torch.randn(members, rank, width, generator=generator) / math.sqrt(width))
            self.B = torch.nn.Parameter(torch.zeros(members, width, rank))
            self.metadata = {'law': 'h + B_m GELU(A_m h)' if nonlinear else 'h + B_m A_m h',
                'width': width, 'rank': rank, 'members': members, 'seed': int(seed),
                'zero_output_initialization': True, 'native_body_replaced': False,
                'parameter_count': 2*members*width*rank}
        def forward(self, hidden, member):
            member = 0 if members == 1 else int(member)
            value = hidden @ self.A[member].t()
            if nonlinear:
                value = torch.nn.functional.gelu(value)
            return hidden + value @ self.B[member].t()
    return HiddenResidual()


def install_task_prehead_adapter(native_leaf, adapter, route, task_gate):
    """Explicit optional integration point; source class/Parameters are retained.

    Require the caller's NCNC task-depth gate, never torch.is_grad_enabled().
    prepend=True puts the adapter BEFORE an existing output-boundary R hook.
    Caller must register adapter Parameters with its declared source LR group.
    """
    if task_gate is None:
        raise ValueError('An explicit native task-call gate is required')
    def pre(module, args):
        if not task_gate():
            return None
        return (adapter(args[0], route()),) + tuple(args[1:])
    return native_leaf.register_forward_pre_hook(pre, prepend=True)


def make_late_gate(torch, width, members=4, hidden=8, seed=0):
    """End-to-end TRAIN-fitted convex raw-logit gate, initially uniform.

    Only existing native preheads/logits are inputs; no endpoint labels,
    degree oracle, VALID fit, per-TEST optimization, or extra source forward.
    This is a known feature-conditioned aggregation hypothesis, not ELREA's
    gradient routing or a calibrated probability-pooling guarantee.
    """
    class LateGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            self.down = torch.nn.Parameter(torch.randn(hidden, width+2, generator=generator) / math.sqrt(width+2))
            self.out = torch.nn.Parameter(torch.zeros(hidden))
            self.route_bias = torch.nn.Parameter(torch.zeros(members))
            self.metadata = {'members': members, 'prehead_width': width, 'hidden': hidden,
                'uniform_initialization': True, 'pool_domain': 'native raw logits',
                'gate_inputs': 'normalized native prehead; mean and RMS route-logit deviation',
                'fit_role': 'TRAIN only; entire bank checkpoint selected on whole VALID'}
        def forward(self, logits, preheads):
            if logits.ndim == 2:
                logits = logits[..., None]
            if tuple(logits.shape[:2]) != tuple(preheads.shape[:2]) or logits.shape[0] != members:
                raise ValueError('Aligned [members,queries,...] logits/preheads required')
            normalized = torch.nn.functional.normalize(preheads, dim=-1, eps=1e-12)
            deviation = logits-logits.mean(0, keepdim=True)
            context = torch.cat((normalized, deviation.mean(-1, keepdim=True),
                                 deviation.square().mean(-1, keepdim=True).add(1e-12).sqrt()), -1)
            scores = torch.nn.functional.gelu(context @ self.down.t()) @ self.out
            scores = scores + self.route_bias[:, None]
            weights = torch.softmax(scores, dim=0)
            return (weights[..., None]*logits).sum(0), weights
    return LateGate()
