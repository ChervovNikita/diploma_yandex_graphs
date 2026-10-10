"""Plain shared WikiCS F: one live two-view gradient accumulation, one Adam.

Stdlib-only import. The same coupled four-route graph is kept live within each
view. Only the two stochastic views are evaluated/backpropagated sequentially.
"""


def train_step(session, batch, labels):
    torch = session.torch
    if (session.task != 'wikics' or session.arm != 'be_unit' or session.model.contrastive
            or session.model.independent or session.model.members != 4 or len(session.optimizers) != 1):
        raise ValueError('Exactly plain shared four-route WikiCS F; no cross-view auxiliary or independent sum')
    if labels.device != session.device:
        raise ValueError('Move batch and labels to Session.device before updating')
    session.model.train()
    optimizer = session.optimizers[0]
    optimizer.zero_grad(set_to_none=True)
    selection = session.core['selection']; losses = session.core['objectives']
    session.live_last_update_events = []
    def event(name):
        session.live_last_update_events.append(name)
        observer = getattr(session, 'live_update_observer', None)
        if observer is not None:
            observer(name)
    event('zero_grad_completed')
    halves = []
    for view in ('A', 'B'):
        event(view+'_forward_attempt')
        logits, representation = session.forward(batch)
        event(view+'_forward_completed')
        selection.finite_predictions(logits, session.serving(logits))
        if not torch.isfinite(representation).all():
            raise FloatingPointError('TRAIN representation')
        own = losses.own_supervision(logits, labels, session.task)
        half = .5 * own.mean()
        if not torch.isfinite(half):
            raise FloatingPointError('TRAIN loss')
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(session.cuda_index).clone() if session.cuda_index is not None else None
        session.live_work['backward_attempts'] += 1
        event(view+'_backward_attempt')
        half.backward()
        session.live_work['backward_completions'] += 1
        event(view+'_backward_completed')
        if (not torch.equal(cpu_rng, torch.get_rng_state())
                or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(session.cuda_index)))):
            raise ValueError('Backward consumed default RNG; realized second view is unqualified')
        halves.append(half.detach())
        # Drop every output/loss reference to the completed view. The first
        # view graph is released before constructing the second view graph.
        del logits, representation, own, half, cpu_rng, cuda_rng
        event(view+'_graph_references_released')
    total = halves[0] + halves[1]
    if not torch.isfinite(total):
        raise FloatingPointError('TRAIN loss')
    active = [parameter for parameter in session.model.parameters() if parameter.grad is not None]
    if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
        raise FloatingPointError('TRAIN gradient')
    event('Adam_attempt_after_both_backwards')
    optimizer.step()
    event('Adam_completed')
    selection.finite_state(session.model, session.optimizers)
    session.steps += 1
    auxiliary = total * 0
    return {'loss':total.detach(), 'own_mean':total.detach(), 'auxiliary':auxiliary.detach()}
