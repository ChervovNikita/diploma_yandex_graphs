"""Stateless five-step predicted-class decoder; caller owns Torch, B and P.

Only current native logits and predicted distributions are message values.
No parameter construction, label input, detached state, RNG or graph cache.
"""


def incoming_mean(torch, states, source, target, inverse_degree):
    """Incoming record mean; support and inverse degree are caller-owned.

    states is [N,C] or [M,N,C]. Repeated records retain their multiplicity.
    The caller's graphP support is off-diagonal factual records plus one
    identity record per node, yielding D^-1(A_off+I).
    """
    if states.ndim not in (2, 3) or inverse_degree.shape != (states.shape[-2],):
        raise ValueError('Full-node class states and one inverse degree per node')
    if (source.ndim != 1 or source.shape != target.shape
            or source.dtype != torch.long or target.dtype != torch.long
            or source.device != states.device or target.device != states.device
            or inverse_degree.device != states.device
            or inverse_degree.dtype != states.dtype):
        raise ValueError('Aligned caller-owned graph support, dtype and device')
    messages = states.index_select(-2, source)
    total = states.new_zeros(states.shape).index_add(-2, target, messages)
    shape = (1,) * (states.ndim - 2) + (states.shape[-2], 1)
    return total * inverse_degree.reshape(shape)


def decode(torch, native_logits, compatibility, *, support=None):
    """Exactly five q_next=softmax(Z+(Pq)B) updates; return final scores,q.

    support=None is the matched P=I control. Otherwise support is the explicit
    (source,target,inverse_degree) tuple for incoming_mean. B is [C,C] or
    [M,C,C], registered by the caller before its optimizer is constructed.
    A native binary logit is represented as [0,z]. Every gradient stays live.
    """
    if native_logits.ndim not in (2, 3) or native_logits.shape[-1] < 1:
        raise ValueError('Full native class logits [N,C] or [M,N,C]')
    logits = native_logits
    if logits.shape[-1] == 1:
        logits = torch.cat((torch.zeros_like(logits), logits), dim=-1)
    classes = logits.shape[-1]
    if compatibility.shape[-2:] != (classes, classes):
        raise ValueError('Compatibility matrix must use the native class coordinates')
    if compatibility.ndim not in (2, 3) or (compatibility.ndim == 3 and
            (logits.ndim != 3 or compatibility.shape[0] != logits.shape[0])):
        raise ValueError('One shared B or one B per live route')
    if compatibility.dtype != logits.dtype or compatibility.device != logits.device:
        raise ValueError('Compatibility and logits must share dtype and device')
    states = logits.softmax(-1)
    for _ in range(5):
        mean = states if support is None else incoming_mean(torch, states, *support)
        scores = logits + mean @ compatibility
        states = scores.softmax(-1)
    return scores, states
