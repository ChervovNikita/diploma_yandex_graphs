"""Exact native/family training, stage state, safe CPU images and portable replay."""
from copy import deepcopy
import os
import random
import resource
import struct
import sys
import time
import common as c

_neural = None


def neural():
    global _neural
    if _neural is None:
        _neural = (c.imported_source('reused_models/native_polynormer.py', '_amazon_native_polynormer_v2'),
                   c.imported_source('reused_models/backbone_boundary_adapter.py', '_amazon_boundary_v2'))
    return _neural


def rng(device):
    import numpy as np
    import torch
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'name': n[0], 'keys': n[1].tolist(),
            'position': int(n[2]), 'has_gauss': int(n[3]), 'cached_gaussian': float(n[4])},
            'torch_cpu': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(torch.device(device)).clone() if device.startswith('cuda:') else None}


def restore_rng(value, device):
    import numpy as np
    import torch
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32), n['position'],
                         n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(value['torch_cpu'])
    if device.startswith('cuda:'):
        c.require(value['cuda'] is not None, 'CUDA RNG missing')
        torch.cuda.set_rng_state(value['cuda'], torch.device(device))
    else:
        c.require(value['cuda'] is None, 'CPU image has unexpected CUDA RNG')


def seed_all(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed_all(seed)


def build(row, device):
    import torch
    native, boundary = neural()
    seed_all(row['seed'], device)
    construction = {'seed': row['seed'], 'after_seed': rng(device)}
    model = native.Polynormer(300, 256, 5, local_layers=10, global_layers=1,
            in_dropout=0.2, dropout=0.3, global_dropout=0.3, heads=2, beta=-1, pre_ln=False)
    construction['after_constructor_cpu'] = rng(device)
    model.to(device)
    model.reset_parameters()
    model._global = False
    construction['after_device_reset'] = rng(device)
    if row['kind'] == 'gnnm_boundary_4':
        model = boundary.PolynormerBoundaryFamily(model, members=4)
    else:
        c.require(row['kind'] == 'native_independent', 'Unregistered model kind')
    construction['after_optional_factor_wrap'] = rng(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    optimizer_names(model, optimizer)
    construction['after_Adam_constructor'] = rng(device)
    return model, optimizer, construction


def core(model):
    return model.core if hasattr(model, 'core') else model


def stage(model):
    return bool(core(model)._global)


def set_stage(model, enabled):
    if hasattr(model, 'set_global_stage'):
        model.set_global_stage(enabled)
    else:
        model._global = bool(enabled)
    c.require(stage(model) is bool(enabled), 'Stage flag failed explicit assignment')


def primitives(model):
    q = core(model)
    return {'family': type(model).__name__, 'members': int(getattr(model, 'members', 1)),
            'local_layers': len(q.local_convs), 'global_layers': q.global_attn.num_layers,
            'hidden_per_head': q.global_attn.hidden_channels, 'heads': q.global_attn.heads,
            'beta': q.beta, 'in_dropout': q.in_drop, 'local_dropout': q.dropout,
            'global_dropout': q.global_attn.dropout, 'qk_shared': q.global_attn.qk_shared,
            'pre_ln': q.pre_ln, 'features': 300, 'classes': 5}


def optimizer_names(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    c.require(len(optimizer.param_groups) == 1 and len(parameters) == len(names) and
              len({id(p) for p in parameters}) == len(parameters) and
              {id(p) for p in parameters} == set(names) and all(p.requires_grad for p in parameters),
              'Adam must own every trainable parameter exactly once in one group')
    c.require(optimizer.param_groups[0]['lr'] == 0.001 and
              optimizer.param_groups[0]['weight_decay'] == 0.0 and
              optimizer.param_groups[0]['betas'] == (0.9, 0.999) and
              optimizer.param_groups[0]['eps'] == 1e-8 and not optimizer.param_groups[0]['amsgrad'],
              'Source Adam defaults/recipe changed')
    return [names[id(p)] for p in parameters]


def inactive(name, model):
    family = hasattr(model, 'core')
    if stage(model):
        return name.startswith('local_head.' if family else 'pred_local.')
    prefix = 'core.' if family else ''
    return name.startswith((prefix + 'global_attn.', prefix + 'ln.',
                            'global_head.' if family else 'pred_global.'))


def finite(value, message):
    import torch
    c.require(torch.isfinite(value).all().item(), message)


def check_gradients(model):
    for name, p in model.named_parameters():
        if inactive(name, model):
            c.require(p.grad is None, 'Stage-inactive gradient must be None: ' + name)
        else:
            c.require(p.grad is not None, 'Disconnected active parameter: ' + name)
            finite(p.grad, 'Nonfinite active gradient: ' + name)
    if hasattr(model, 'core'):
        for owner in (model.stem, model.global_head if stage(model) else model.local_head):
            for factor in (owner.R, owner.S, owner.B):
                c.require(factor.grad is not None and factor.grad.shape[0] == 4,
                          'Each complete member must be connected to FIT CE')
                for member in range(4):
                    finite(factor.grad[member], 'Member-specific active gradient invalid')


def forward(model, x, edge):
    import torch
    z = model(x, edge)
    if not hasattr(model, 'core'):
        z = z.unsqueeze(0)
    c.require(z.dtype == torch.float32 and tuple(z.shape) ==
              (int(getattr(model, 'members', 1)), 24492, 5), 'Full FP32 member logits required')
    finite(z, 'Nonfinite full-graph logits')
    return z


def loss_for(z, roles):
    import torch
    index = torch.tensor(roles['fit'], dtype=torch.long, device=z.device)
    labels = torch.tensor(roles['fit_labels'], dtype=torch.long, device=z.device)
    losses = [torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(z[m], dim=1).index_select(0, index),
                                          labels) for m in range(z.shape[0])]
    return losses[0] if len(losses) == 1 else torch.stack(losses).mean()


def train_step(model, optimizer, x, edge, roles):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z = forward(model, x, edge)
    loss = loss_for(z, roles)
    finite(loss, 'Nonfinite FIT CE')
    loss.backward()
    check_gradients(model)
    optimizer.step()
    for name, p in model.named_parameters():
        finite(p, 'Nonfinite post-update parameter: ' + name)
    for entry in optimizer.state.values():
        for v in entry.values():
            if hasattr(v, 'dtype'):
                finite(v, 'Nonfinite Adam state')
    return loss.detach().cpu().clone()


def eval_logits(model, x, edge):
    import torch
    model.eval()
    with torch.no_grad():
        return forward(model, x, edge)


def correct_count(z, roles):
    import torch
    ids = torch.tensor(roles['val'], dtype=torch.long, device=z.device)
    labels = torch.tensor(roles['val_labels'], dtype=torch.long, device=z.device)
    prediction = (z[0].argmax(-1) if z.shape[0] == 1 else
                  torch.softmax(z, dim=-1).mean(dim=0).argmax(-1))
    return int((prediction.index_select(0, ids) == labels).sum().item())


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    c.require(value is None or type(value) in (str, bool, int, float), 'Unsafe checkpoint leaf')
    return value


def state(model, optimizer, device):
    return {'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
            'optimizer_parameter_names': optimizer_names(model, optimizer), 'rng': cpu_tree(rng(device)),
            'global_flags': {n: bool(m._global) for n, m in model.named_modules() if hasattr(m, '_global')},
            'modes': {n: m.training for n, m in model.named_modules()},
            'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()},
            'gradients': {n: cpu_tree(p.grad) for n, p in model.named_parameters()},
            'primitives': primitives(model)}


def restore_model_adam(model, optimizer, s):
    c.require(primitives(model) == s['primitives'] and
              optimizer_names(model, optimizer) == s['optimizer_parameter_names'], 'Model/Adam ownership differs')
    model.load_state_dict(s['model'], strict=True)
    # Adam can retain CPU step tensors from its input state by reference.
    # Give the live optimizer its own tensors, preserving the saved image.
    optimizer.load_state_dict(cpu_tree(s['optimizer']))
    c.require(optimizer_names(model, optimizer) == s['optimizer_parameter_names'], 'Restored Adam ownership differs')


def restore(model, optimizer, image, device):
    s = image['state']
    restore_model_adam(model, optimizer, s)
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    c.require(set(modules) == set(s['modes']) and set(params) == set(s['requires_grad']) == set(s['gradients']) and
              {n for n, m in modules.items() if hasattr(m, '_global')} == set(s['global_flags']),
              'Registered state inventory differs')
    for name, value in s['global_flags'].items():
        c.require(type(value) is bool, 'Primitive stage Boolean required')
        modules[name]._global = value
    c.require(all(value == stage(model) for value in s['global_flags'].values()), 'Conflicting stage flags')
    c.require(stage(model) == image['selection']['global'], 'Image selection/stage flags conflict')
    for name, value in s['modes'].items():
        c.require(type(value) is bool, 'Primitive mode Boolean required')
        modules[name].training = value
    for name, p in params.items():
        p.requires_grad_(s['requires_grad'][name])
        p.grad = None if s['gradients'][name] is None else s['gradients'][name].to(p.device).clone()
    optimizer_names(model, optimizer)
    restore_rng(s['rng'], device)
    c.require(exact(state(model, optimizer, device), s), 'Fresh full-state restore changed logical bytes or metadata')


def tensor_bytes(value):
    import torch
    # Byte view preserves signed zero, NaN representation, dtype and logical order.
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(order='C')


def exact(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and tensor_bytes(a) == tensor_bytes(b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    if type(a) is float:
        return struct.pack('!d', a) == struct.pack('!d', b)
    return a == b


def save_image(path, image):
    import torch
    p = c.confined(path)
    c.require(not p.exists(), 'Immutable image path required')
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(cpu_tree(image), stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Exclusive link prevents accidental replacement of a previously selected image.
    os.link(temporary, p)
    temporary.unlink()
    return c.record(p)


def load_image(row, expected_bindings=None):
    import torch
    image = torch.load(c.verify(row), map_location='cpu', weights_only=True)
    c.require(image['schema'] == 'amazon_polynormer_full_state_v2', 'Unknown portable image')
    if expected_bindings is not None:
        c.require(image['bindings'] == expected_bindings, 'Checkpoint identity differs')
    return image


def snapshot(model, optimizer, bindings, selection, construction, logits, device):
    return {'schema': 'amazon_polynormer_full_state_v2', 'bindings': bindings, 'selection': selection,
            'construction_rng': cpu_tree(construction), 'state': state(model, optimizer, device),
            'selected_raw_logits': cpu_tree(logits)}


def snapshot_write(path, model, optimizer, bindings, selection, construction, logits, device):
    synchronize(device)
    before = time.perf_counter()
    image = snapshot(model, optimizer, bindings, selection, construction, logits, device)
    synchronize(device)
    copied = time.perf_counter()
    copy_memory = memory(device)
    row = save_image(path, image)
    written = time.perf_counter()
    return row, {'CPU_snapshot_seconds': copied - before, 'safe_write_seconds': written - copied,
                 'serialized_bytes': row['bytes'], 'snapshot_memory': copy_memory,
                 'post_write_memory': memory(device)}


def logit_gate(a, b):
    import torch
    c.require(a.dtype == b.dtype == torch.float32 and a.shape == b.shape and
              torch.isfinite(a).all().item() and torch.isfinite(b).all().item() and
              torch.allclose(a, b, rtol=1e-5, atol=1e-6), 'Raw logit replay/parity failed fixed tolerance')
    return float((a - b).abs().max().item())


def portable_replay(model, optimizer, checkpoint, row, x, edge, roles, device, bindings):
    """Isolated clones; canonical RNG/model/Adam are preserved even on failure."""
    started = time.perf_counter()
    caller_rng = rng(device)
    reference_model, reference_optimizer = deepcopy((model, optimizer))
    # torch.nn.Parameter.__deepcopy__ copies parameter data and requires_grad,
    # but does not retain .grad. Preserve the actual live gradient tensors too.
    originals = dict(model.named_parameters())
    references = dict(reference_model.named_parameters())
    c.require(set(originals) == set(references), 'Joint clone parameter inventory changed')
    for name, parameter in originals.items():
        references[name].grad = None if parameter.grad is None else parameter.grad.detach().clone()
    cloned = time.perf_counter()
    image = load_image(checkpoint, bindings)
    loaded = time.perf_counter()
    image_state_before = cpu_tree(image['state'])
    clone_memory = memory(device)
    try:
        c.require(exact(state(model, optimizer, device), image['state']),
                  'Original live state differs from serialized selected image')
        c.require(exact(state(reference_model, reference_optimizer, device), image['state']),
                  'Full live clone differs from serialized selected image')
        reference_logits = eval_logits(reference_model, x, edge)
        reopened_model, reopened_optimizer, _ = build(row, device)
        restore(reopened_model, reopened_optimizer, image, device)
        reopened_logits = eval_logits(reopened_model, x, edge)
        maximum = logit_gate(reference_logits, reopened_logits)
        logit_gate(image['selected_raw_logits'], reopened_logits.cpu())
        c.require(correct_count(reopened_logits, roles) == image['selection']['val_correct'], 'Selected VAL replay differs')
        del reference_logits, reopened_logits
        restore(reference_model, reference_optimizer, image, device)
        loss_reference = train_step(reference_model, reference_optimizer, x, edge, roles)
        after_reference = state(reference_model, reference_optimizer, device)
        c.require(exact(image['state'], image_state_before),
                  'Reference replay mutated saved image state')
        restore(reopened_model, reopened_optimizer, image, device)
        loss_reopened = train_step(reopened_model, reopened_optimizer, x, edge, roles)
        after_reopened = state(reopened_model, reopened_optimizer, device)
        c.require(exact(image['state'], image_state_before),
                  'Reopened replay mutated saved image state')
        c.require(exact(loss_reference, loss_reopened) and exact(after_reference, after_reopened),
                  'Portable next-step full-state bitwise logical-byte equality failed')
        return {'stage': 'global' if image['selection']['global'] else 'local',
                'raw_logit_max_absolute': maximum, 'bitwise_full_next_step': True,
                'extra_updates': 2, 'extra_complete_eval_forwards': 2,
                'joint_live_clone_seconds': cloned - started, 'safe_read_seconds': loaded - cloned,
                'clone_and_reopened_image_memory': clone_memory,
                'seconds': time.perf_counter() - started, 'memory': memory(device)}
    finally:
        restore_rng(caller_rng, device)


def transition(model, optimizer, image, device):
    live_rng = rng(device)
    c.require(not stage(model) and image['selection']['global'] is False, 'Local selected transition required')
    restore_model_adam(model, optimizer, image['state'])
    c.require(exact(cpu_tree(model.state_dict()), image['state']['model']) and
              exact(cpu_tree(optimizer.state_dict()), image['state']['optimizer']), 'Local model/Adam restore differs')
    set_stage(model, True)
    c.require(exact(live_rng, rng(device)), 'Transition rewound/consumed live end-of-local RNG')
    return {'from_selected': image['selection'], 'restored_model_Adam_bitwise': True,
            'live_end_local_RNG_preserved_bitwise': True, 'global': True}


def synchronize(device):
    if device.startswith('cuda:'):
        import torch
        torch.cuda.synchronize(torch.device(device))


def memory(device):
    units = 1 if sys.platform == 'darwin' else 1024
    value = {'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units}
    p = c.Path('/proc/self/status')
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, number, _ = line.split()
                value[key.rstrip(':') + '_bytes'] = int(number) * 1024
    if device.startswith('cuda:'):
        import torch
        value.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                     cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return value


def state_bytes(model, optimizer):
    import torch
    def count(v):
        if isinstance(v, torch.Tensor):
            return v.numel() * v.element_size()
        if isinstance(v, dict):
            return sum(count(w) for w in v.values())
        if isinstance(v, (tuple, list)):
            return sum(count(w) for w in v)
        return 0
    return {'trainable_parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad),
            'stored_model_bytes': count(model.state_dict()), 'optimizer_tensor_bytes': count(optimizer.state_dict()),
            'gradient_tensor_bytes': sum(count(p.grad) for p in model.parameters()),
            'portable_state_tensor_bytes': count(state(model, optimizer, str(next(model.parameters()).device)))}
