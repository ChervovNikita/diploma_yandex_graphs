"""Selection/transition operators shared by runner and synthetic tests."""
import torch


def local_transition(model, optimizers, load, ordinary_independent):
    """Preserve live end-local RNG; restore own or complete joint state exactly.

    Coupled packed untied members are one trajectory and must never be spliced
    from independently selected epochs, including their optimizer histories.
    """
    if ordinary_independent:
        if model.arm != 'independent4' or not model.independent or len(optimizers) != 4:
            raise ValueError('Only ordinary independently selected four-member control')
        for m, body in enumerate(model.models):
            state = load('own_local_'+str(m)+'.pt')
            body.load_state_dict(state['model'])
            optimizers[m].load_state_dict(state['optimizer'])
    else:
        state = load('selected_local.pt')
        if len(state['optimizers']) != len(optimizers):
            raise ValueError('Joint optimizer bank differs')
        model.load_state_dict(state['model'])
        for optimizer, saved in zip(optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)
    model.set_global(True)


def finite_predictions(member_logits, pooled):
    if not torch.isfinite(member_logits).all() or not torch.isfinite(pooled).all():
        raise FloatingPointError('Nonfinite member or pooled prediction before metric/selection')


def finite_state(model, optimizers):
    for name, value in model.named_parameters():
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite parameter: '+name)
    for index, optimizer in enumerate(optimizers):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise FloatingPointError('Nonfinite optimizer '+str(index)+' '+key)
