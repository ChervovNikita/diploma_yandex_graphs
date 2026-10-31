"""Frozen first-order affine scan backend; torch is injected after admission.

s[t]=a[t]*s[t-1]+q[t], component initial state0. Component starts mask ONLY
carry a; caller has already formed q=(1-a_original)*u with live gradients.
No numerical import/construction or execution happens at module import.
"""

def _prefix_value(torch,a,q):
    """Vectorized doubling under no_grad; O(L) live storage, O(L logL) work.

    Composition later o earlier = (a_later*a_earlier,
    q_later+a_later*q_earlier). Old level buffers are freed, not autograd saved.
    There is no per-token Python/CUDA loop, host copy, .item or extension.
    """
    n=a.shape[0];stride=1;prefix_a,prefix_q=a,q
    with torch.no_grad():
        while stride<n:
            next_a=torch.cat((prefix_a[:stride],prefix_a[stride:]*prefix_a[:-stride]),dim=0)
            next_q=torch.cat((prefix_q[:stride],prefix_q[stride:]+prefix_a[stride:]*prefix_q[:-stride]),dim=0)
            prefix_a,prefix_q=next_a,next_q;stride*=2
    return prefix_q


def make_segmented_scan(torch):
    """Bind one custom autograd Function. FIRST derivatives only.

    Input: original occurrence carry a[L,h], q[L,h], starts[L]bool.
    Forward saves effective carry a with zeros at starts, inclusive states,
    and the fixed starts mask; q is NOT recomputed using the masked carry.
    Reverse adjoint: lambda[t]=g[t]+a_eff[t+1]*lambda[t+1], last carry0.
    dq=lambda; da=lambda*s_previous with previous0 at component starts.
    Ordinary autograd outside this Function propagates q's original a/u path,
    node gathers, reverse-component permutation and occurrence means.
    """
    class SegmentedAffine(torch.autograd.Function):
        @staticmethod
        def forward(ctx,a,q,starts):
            if (a.ndim!=2 or a.shape!=q.shape or a.device!=q.device or a.dtype!=q.dtype
                    or not a.is_floating_point() or len(a)<1
                    or starts.shape!=(len(a),) or starts.dtype!=torch.bool or starts.device!=a.device):
                raise ValueError('Qualified floating [L,h] pairs and device-matched bool[L] starts')
            carry=a.masked_fill(starts[:,None],0)
            states=_prefix_value(torch,carry,q)
            ctx.save_for_backward(carry,states,starts)
            return states

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx,grad_states):
            carry,states,starts=ctx.saved_tensors
            # The coefficient connecting lambda[t+1] to lambda[t] is a[t+1],
            # not a[t]. A component-start zero cuts its predecessor's adjoint.
            next_carry=torch.cat((carry[1:],torch.zeros_like(carry[:1])),dim=0)
            adjoint=_prefix_value(torch,next_carry.flip(0),grad_states.flip(0)).flip(0)
            previous=torch.cat((torch.zeros_like(states[:1]),states[:-1]),dim=0)
            da=(adjoint*previous).masked_fill(starts[:,None],0)
            dq=adjoint
            return da,dq,None
    return SegmentedAffine.apply


def bidirectional_context(torch,scan,a_nodes,q_nodes,tour,reset_every_token=False):
    """Mean both directions at every occurrence, then all occurrences per node."""
    tokens,reverse=tour['tokens'],tour['reverse_index']
    token_a=a_nodes.index_select(0,tokens);token_q=q_nodes.index_select(0,tokens)
    if reset_every_token:
        # R4: reset carry, but preserve q=(1-a_original)*u and its derivatives.
        combined=token_q
    else:
        forward=scan(token_a,token_q,tour['starts'])
        backward_reversed=scan(token_a.index_select(0,reverse),token_q.index_select(0,reverse),tour['starts'])
        backward=backward_reversed.index_select(0,reverse)
        combined=(forward+backward)*.5
    total=torch.zeros_like(q_nodes).index_add(0,tokens,combined)
    return total/tour['occurrences'].to(q_nodes.dtype)[:,None]
