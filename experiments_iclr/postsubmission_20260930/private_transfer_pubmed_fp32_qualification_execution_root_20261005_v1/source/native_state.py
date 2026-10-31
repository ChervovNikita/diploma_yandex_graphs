"""Unchanged saved native complete-state/RNG helper bodies; no numerical imports."""
import copy
import random

def require(value, message):
    if not value: raise ValueError(message)

def clone(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone(child, torch) for key, child in value.items()}
    if isinstance(value, list):
        return [clone(child, torch) for child in value]
    if isinstance(value, tuple):
        return tuple(clone(child, torch) for child in value)
    return copy.deepcopy(value)

def rng(torch, np, gpu):
    nr = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (nr[0], torch.tensor(nr[1].astype('int64')), nr[2], nr[3], nr[4]),
            'torch_cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if gpu else []}

def restore_rng(state, torch, np, gpu):
    random.setstate(state['python'])
    nr = state['numpy']
    np.random.set_state((nr[0], nr[1].numpy().astype('uint32'), nr[2], nr[3], nr[4]))
    torch.set_rng_state(state['torch_cpu'])
    if gpu:
        torch.cuda.set_rng_state_all(state['cuda'])

def capture(unit, torch, np):
    model, predictor, optimizer, _, _, _ = unit
    return clone({'encoder': model.state_dict(), 'predictor': predictor.state_dict(), 'Adam': optimizer.state_dict(),
                  'gradients': [{name: parameter.grad for name, parameter in root.named_parameters()} for root in (model, predictor)],
                  'flags': [{name: module.training for name, module in root.named_modules()} for root in (model, predictor)],
                  'invest': [getattr(module, 'invest', None) for module in (model, predictor)], 'RNG': rng(torch, np, True)}, torch)

def restore(unit, state, torch, np):
    model, predictor, optimizer, _, _, _ = unit
    model.load_state_dict(state['encoder'], strict=True)
    predictor.load_state_dict(state['predictor'], strict=True)
    optimizer.load_state_dict(state['Adam'])
    for index, root in enumerate((model, predictor)):
        require(set(dict(root.named_parameters())) == set(state['gradients'][index]), 'Serialized gradient names differ')
        for name, parameter in root.named_parameters():
            grad = state['gradients'][index][name]
            parameter.grad = None if grad is None else grad.to(parameter.device).clone()
        require(set(dict(root.named_modules())) == set(state['flags'][index]), 'Serialized module flags differ')
        for name, module in root.named_modules():
            module.training = state['flags'][index][name]
        if state['invest'][index] is not None:
            root.invest = state['invest'][index]
    restore_rng(state['RNG'], torch, np, True)

def equal(left, right, torch, label):
    require(type(left) is type(right), label + ': type differs')
    if isinstance(left, torch.Tensor):
        require(left.shape == right.shape and left.dtype == right.dtype and left.layout == right.layout and torch.equal(left, right), label + ': tensor differs')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), label + ': keys differ')
        for key in left:
            equal(left[key], right[key], torch, label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), label + ': length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, torch, label + '/' + str(index))
    else:
        require(left == right, label + ': value differs')
