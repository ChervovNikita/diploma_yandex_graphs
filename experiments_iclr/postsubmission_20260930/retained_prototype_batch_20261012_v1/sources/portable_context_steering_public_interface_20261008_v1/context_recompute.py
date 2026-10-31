"""Context-target WikiCS VJP replay; shared mean or explicit untied sum loss."""
from types import MethodType

MODE = 'context_target_detached_outputs_then_member_view_VJP_v1'


def clone_streams(streams):
    return [{key: value.clone() for key, value in stream.items()} for stream in streams]


def equal_streams(a, b, torch):
    return len(a) == len(b) and all(left.keys() == right.keys() and
        all(torch.equal(left[key], right[key]) for key in left) for left, right in zip(a, b))


def train_step(session, batch, labels):
    """dL/dtheta = sum_(view,member) J(output)^T dL/doutput; Adam after the sum."""
    torch = session.torch
    members = session.model.members
    optimizer_count = members if session.model.independent else 1
    if session.task != 'wikics' or members not in (1, 4) or len(session.optimizers) != optimizer_count:
        raise ValueError('Original WikiCS M1/shared-M4/untied-M4 optimizer bank required')
    if session.model.contrastive and session.config['contrastive']['residual_weight'] != 0.:
        raise ValueError('Only registered residual-free separable context objective')
    if labels.device != session.device or len(labels) != 580:
        raise ValueError('Complete580 TRAIN labels on Session.device required')
    session.model.train()
    for optimizer in session.optimizers:
        optimizer.zero_grad(set_to_none=True)
    parameter_versions = tuple(parameter._version for parameter in session.model.parameters())
    before = clone_streams(session.streams)
    with torch.no_grad():
        la, ha = session.forward(batch); session.execution_totals['shadow_member_forwards'] += members
        lb, hb = session.forward(batch); session.execution_totals['shadow_member_forwards'] += members
    shadow_end = clone_streams(session.streams)
    selection, losses = session.core['selection'], session.core['objectives']
    selection.finite_predictions(la, session.serving(la)); selection.finite_predictions(lb, session.serving(lb))
    if not torch.isfinite(ha).all() or not torch.isfinite(hb).all():
        raise FloatingPointError('TRAIN shadow representation')
    # These leaves retain only [4,580,10] logits and [4,580,512] representations,
    # not any fullgraph backbone tape. Explicit own/auxiliary scaling follows.
    la, ha, lb, hb = [value.detach().requires_grad_(True) for value in (la, ha, lb, hb)]
    size = min(len(labels), session.config['contrastive']['max_objects'])
    index = torch.linspace(0, len(labels)-1, steps=size, device=labels.device).long()
    own = .5 * (losses.own_supervision(la, labels, session.task) + losses.own_supervision(lb, labels, session.task))
    auxiliary = own.sum() * 0
    if session.model.contrastive:
        auxiliary = .05 * losses.alignment_loss(ha[:, index], hb[:, index], labels[index], session.task, identities=None)
        auxiliary = auxiliary + .05 * losses.residual_member_contrast(ha[:, index], hb[:, index], labels[index])
    total = own.sum() + members * auxiliary if session.model.independent else own.mean() + auxiliary
    if not torch.isfinite(total):
        raise FloatingPointError('TRAIN loss')
    result = {'loss': total.detach(), 'own_mean': own.mean().detach(), 'auxiliary': auxiliary.detach()}
    outputs = (la, ha, lb, hb)
    raw = torch.autograd.grad(total, outputs, allow_unused=True, create_graph=False)
    session.execution_totals['output_cotangent_collections'] += 1
    cotangents = tuple(torch.zeros_like(value) if grad is None else grad.detach() for value, grad in zip(outputs, raw))
    if any(not torch.isfinite(grad).all() for grad in cotangents):
        raise FloatingPointError('TRAIN output cotangent')
    del total, own, auxiliary, raw
    # Restoring the original starting streams reuses the same two stochastic
    # views. The replay's end states must equal the shadow's end states exactly.
    session.streams = clone_streams(before)
    maximum_logit_difference = maximum_representation_difference = 0.
    for view in range(2):
        for member, stream in enumerate(session.streams):
            with torch.random.fork_rng(devices=session.cuda_devices):
                torch.set_rng_state(stream['cpu'])
                if session.cuda_index is not None:
                    torch.cuda.set_rng_state(stream['cuda'], session.cuda_index)
                logits, representation = session.model.member_forward(batch, member)
                session.execution_totals['replay_member_forwards'] += 1
                stream['cpu'] = torch.get_rng_state()
                if session.cuda_index is not None:
                    stream['cuda'] = torch.cuda.get_rng_state(session.cuda_index)
            if not torch.isfinite(logits).all() or not torch.isfinite(representation).all():
                raise FloatingPointError('TRAIN replay output')
            if session.replay_prediction_diagnostics:
                with torch.no_grad():
                    maximum_logit_difference = max(maximum_logit_difference, float((logits - outputs[2*view][member]).abs().max()))
                    maximum_representation_difference = max(maximum_representation_difference, float((representation - outputs[2*view+1][member]).abs().max()))
            # Both outputs are roots of this same graph. Autograd sums their
            # paths, including logits' dependence on the representation.
            torch.autograd.backward((logits, representation),
                (cotangents[2*view][member], cotangents[2*view+1][member]))
            session.execution_totals['member_reverse_collections'] += 1
            del logits, representation
    if not equal_streams(session.streams, shadow_end, torch):
        raise RuntimeError('Replay member RNG endpoint differs from two original shadow views')
    if tuple(parameter._version for parameter in session.model.parameters()) != parameter_versions:
        raise RuntimeError('Parameters mutated before the complete member/view VJP accumulation')
    active = [parameter for parameter in session.model.parameters() if parameter.grad is not None]
    if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
        raise FloatingPointError('TRAIN accumulated gradient')
    # No optimizer step, parameter update or gradient clearing is permitted
    # between the complete member/view VJPs. Each original Adam optimizer steps once after complete accumulation.
    for optimizer in session.optimizers:
        optimizer.step(); session.execution_totals['optimizer_bank_updates'] += 1
    selection.finite_state(session.model, session.optimizers)
    session.steps += 1
    session.execution_totals['exact_member_RNG_endpoint_checks'] += 1
    session.last_replay_diagnostics = dict(exact_member_RNG_endpoint=True,
        parameter_versions_unchanged_before_Adam=True,
        replay_prediction_diagnostics=session.replay_prediction_diagnostics,
        maximum_absolute_logit_difference=maximum_logit_difference if session.replay_prediction_diagnostics else None,
        maximum_absolute_representation_difference=maximum_representation_difference if session.replay_prediction_diagnostics else None,
        prediction_difference_is_blocking=False, bitwise_author_parity_claimed=False)
    return result


def install(session, diagnostics=False):
    torch = session.torch
    if session.model.contrastive:
        from session_objectives_adapter import ContextObjectivesFacade
        if not isinstance(session.core['objectives'], ContextObjectivesFacade):
            raise ValueError('Legacy coupled contrast cannot use this separable context replay')
        if session.config['contrastive']['residual_weight'] != 0.:
            raise ValueError('Residual-free context objective required')
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in session.model.modules()) or list(session.model.buffers()):
        raise ValueError('Pinned stateless WikiCS model required; no BatchNorm or mutable buffers')
    session.replay_prediction_diagnostics = diagnostics
    session.execution_totals = dict(shadow_member_forwards=0, replay_member_forwards=0,
        output_cotangent_collections=0, member_reverse_collections=0, optimizer_bank_updates=0,
        exact_member_RNG_endpoint_checks=0)
    session.last_replay_diagnostics = None
    session.train_step = MethodType(train_step, session)
    return session
