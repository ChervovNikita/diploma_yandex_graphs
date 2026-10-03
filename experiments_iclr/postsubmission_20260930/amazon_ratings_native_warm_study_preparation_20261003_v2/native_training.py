"""Native-single warm primitives, exact portable state and actual replay checks."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
import random
import resource
import time
import common as c
from byte_identity import tensor_equal, array_equal, float_equal


def seed_all(seed, device):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda:'):
        torch.cuda.manual_seed(seed)


def build(recipe, feature_count, seed, device):
    import torch
    adapter, _ = c.import_native()
    spec = adapter.NativeSpec(dataset='amazon-ratings', num_features=feature_count, num_classes=5,
                              recipe_receipt='bounded_amazon_native_single_v1:'+c.object_sha(recipe),
                              **{k: recipe[k] for k in ('hidden','d_ffn','K','nlayer','n_head','q','multi','dropout','dprate','base')})
    model = adapter.build_all_layer_polyformer(spec, seed=seed, members=1,
                composition_receipt='qualified_v4_native_single_all_one_frozen_factors')
    # M1 parity preserves the native function. These are allocation-only adapter
    # factors; they are not learned parameters in either native-single recipe.
    for name in model.site_names:
        factor = model.core.get_submodule(name)
        for p in (factor.R, factor.S):
            c.require(torch.equal(p.detach(), torch.ones_like(p)), 'Native-single factor must remain1')
            p.requires_grad_(False); p.grad = None
    model.to(device)
    groups, names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        attention = 'attnmodule' in name
        groups.append({'params': p, 'lr': recipe['attn_lr'] if attention else recipe['lr'],
                       'weight_decay': recipe['attn_wd'] if attention else recipe['weight_decay']})
        names.append(name)
    # Same source Adam rule and one native group per common parameter. No
    # scheduler, clipping, mixed precision, AdamW or factor optimization is added.
    optimizer = torch.optim.Adam(groups)
    return model, optimizer, tuple(names), asdict(spec)


def assert_native_single(model):
    import torch
    factor_ids = set()
    for name in model.site_names:
        factor = model.core.get_submodule(name)
        for p in (factor.R, factor.S):
            factor_ids.add(id(p))
            c.require(torch.equal(p.detach(), torch.ones_like(p)) and not p.requires_grad and p.grad is None,
                      'Restored native-single factor must be fixed at1')
    c.require(all(p.requires_grad for p in model.parameters() if id(p) not in factor_ids),
              'All restored native common parameters must retain training permission')


def adam_constructor_options(optimizer):
    import torch
    c.require(type(optimizer) is torch.optim.Adam, 'Admitted default Adam class required')
    return {'constructor_call': 'torch.optim.Adam(groups)',
            'class': type(optimizer).__module__+'.'+type(optimizer).__qualname__,
            'defaults': c.plain(optimizer.defaults)}


def tokens(x, edge, recipe, data_record, runtime_record, *, qualify_native=False):
    import torch
    _, module = c.import_native()
    from native_source.native_preprocess import mono_base
    identity = module.CacheIdentity(dataset_release_sha256=data_record['raw_release']['sha256'],
        features_sha256=data_record['public_graph']['sha256'],
        native_edge_records_sha256=data_record['public_graph']['sha256'], edge_attributes_sha256=c.object_sha(None),
        removed_units_sha256=c.object_sha([]), fit_role_labels_sha256='native_tokens_do_not_use_labels',
        view='native', K=recipe['K'], base='mono',
        normalization_source_sha256=c.record(c.V4/'native_source/native_preprocess.py')['sha256'],
        runtime_precision_receipt=runtime_record['sha256'])
    bank = module.build_uncached_mono_tokens(x, edge, None, identity)
    if qualify_native:
        reference = mono_base(recipe['K'], x, edge, None)
        c.require(len(reference) == len(bank.tokens) and
                  all(tensor_equal(a, b) for a, b in zip(reference, bank.tokens)),
                  'Actual-data uncached tokens differ from saved exact native mono body')
        del reference
    packed = torch.stack(bank.tokens, dim=1)
    return packed, {'identity': identity.fingerprint, 'shape': list(packed.shape),
                    'dtype': str(packed.dtype), 'native_actual_data_bitwise_qualified': qualify_native}


def train_step(model, optimizer, tokens, roles):
    import torch
    model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(tokens)[0]
    ids = torch.tensor(roles.fit, dtype=torch.long, device=logits.device)
    labels = torch.tensor(roles.labels_for('fit'), dtype=torch.long, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits.index_select(0, ids), labels)
    c.require(torch.isfinite(loss).item(), 'Nonfinite native fit loss')
    loss.backward()
    c.require(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                  for p in model.parameters() if p.requires_grad), 'Missing/nonfinite native gradient')
    optimizer.step()
    value = loss.detach().cpu().clone()
    del logits, loss
    if tokens.device.type == 'cuda':
        torch.cuda.empty_cache()  # Retained source behavior, charged as real work.
    return value


def forward_eval(model, tokens):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(tokens)[0]
    c.require(torch.isfinite(logits).all().item(), 'Nonfinite complete native logits')
    return logits


def score(logits, ids, labels):
    import torch
    index = torch.tensor(ids, dtype=torch.long, device=logits.device)
    y = torch.tensor(labels, dtype=torch.long, device=logits.device)
    z = logits.index_select(0, index); prediction = z.argmax(-1)
    accuracy = float((prediction == y).to(torch.float64).mean().cpu())
    nll = float(torch.nn.functional.cross_entropy(z, y).cpu())
    f1 = []
    for klass in range(5):
        tp = ((prediction == klass) & (y == klass)).sum().item()
        fp = ((prediction == klass) & (y != klass)).sum().item()
        fn = ((prediction != klass) & (y == klass)).sum().item()
        denominator = 2*tp+fp+fn
        f1.append(0.0 if denominator == 0 else 2*tp/denominator)
    return {'accuracy': accuracy, 'NLL': nll, 'macro_F1_fixed5': sum(f1)/5}


def rng(device):
    import numpy as np
    import torch
    state = np.random.get_state()
    value = {'torch_cpu': torch.get_rng_state().clone(), 'python': random.getstate(),
             'numpy': {'name': state[0], 'keys': state[1].tolist(), 'position': state[2],
                       'has_gauss': state[3], 'cached_gaussian': state[4]}, 'cuda': None}
    if device.startswith('cuda:'):
        value['cuda'] = torch.cuda.get_rng_state(torch.device(device)).clone()
    return value


def restore_rng(state, device):
    import numpy as np
    import torch
    torch.set_rng_state(state['torch_cpu'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32), n['position'],
                         n['has_gauss'], n['cached_gaussian']))
    if device.startswith('cuda:'):
        c.require(state['cuda'] is not None, 'Missing selected-device RNG')
        torch.cuda.set_rng_state(state['cuda'], torch.device(device))
    else:
        c.require(state['cuda'] is None, 'CPU checkpoint has unexpected CUDA state')


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    c.require(value is None or type(value) in (str,int,float,bool), 'Unsupported checkpoint state leaf')
    return value


def state(model, optimizer, device):
    return {'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
            'optimizer_constructor_options': adam_constructor_options(optimizer),
            'rng': rng(device), 'modes': {n: m.training for n,m in model.named_modules()},
            'requires_grad': {n: p.requires_grad for n,p in model.named_parameters()},
            'gradients': {n: cpu_tree(p.grad) for n,p in model.named_parameters()},
            'native_primitives': c.plain(model._primitive_state())}


def save_checkpoint(path, model, optimizer, bindings, selection, device):
    import torch
    p = c.confined(path)
    c.require(not p.exists(), 'Immutable checkpoint path required')
    image = {'schema': 'amazon_native_single_full_state_v1', 'bindings': bindings,
             'selection': selection, 'state': state(model, optimizer, device)}
    temporary = p.with_name(p.name+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(image, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, p)
    return c.record(p)


def restore(model, optimizer, image, device):
    import torch
    s = image['state']
    c.require(adam_constructor_options(optimizer) == s['optimizer_constructor_options'] ==
              image['bindings']['optimizer_constructor_options'], 'Bound default-Adam constructor options differ')
    c.require(c.plain(model._primitive_state()) == s['native_primitives'], 'Native primitive/config source changed')
    c.require(set(s['modes']) == dict(model.named_modules()).keys() and
              set(s['requires_grad']) == dict(model.named_parameters()).keys(), 'Native registration changed')
    model.load_state_dict(s['model'], strict=True)
    optimizer.load_state_dict(s['optimizer'])
    for name, m in model.named_modules():
        m.training = s['modes'][name]
    for name, p in model.named_parameters():
        p.requires_grad_(s['requires_grad'][name])
        p.grad = None if s['gradients'][name] is None else s['gradients'][name].to(device).clone()
    assert_native_single(model)
    restore_rng(s['rng'], device)


def exact(a, b):
    import torch
    import numpy as np
    if isinstance(a, torch.Tensor):
        return tensor_equal(a, b)
    if isinstance(a, np.ndarray):
        return array_equal(a, b)
    if type(a) is float:
        return float_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a == b


def snapshot_live(model, optimizer, device):
    # A joint deepcopy keeps the copied optimizer attached to copied parameters.
    pair = deepcopy((model, optimizer))
    return pair, rng(device)


def replay_selected(reference, checkpoint, recipe, features, seed, device, tokens, roles):
    import torch
    started = time.perf_counter(); caller_rng = rng(device)
    image = torch.load(c.verify(checkpoint), map_location='cpu', weights_only=True)
    c.require(image['schema'] == 'amazon_native_single_full_state_v1', 'Unknown checkpoint')
    (live_model, live_optimizer), live_rng = reference
    try:
        restore_rng(live_rng, device)
        live_logits = forward_eval(live_model, tokens)
        replay_model, replay_optimizer, _, _ = build(recipe, features, seed, device)
        restore(replay_model, replay_optimizer, image, device)
        replay_logits = forward_eval(replay_model, tokens)
        c.require(tensor_equal(live_logits, replay_logits), 'Saved selected full native logits fail bitwise replay')
        del live_logits, replay_logits
        restore_rng(live_rng, device)
        loss_live = train_step(live_model, live_optimizer, tokens, roles)
        live_after = state(live_model, live_optimizer, device)
        restore(replay_model, replay_optimizer, image, device)
        loss_replay = train_step(replay_model, replay_optimizer, tokens, roles)
        c.require(exact(loss_live, loss_replay) and exact(live_after, state(replay_model,replay_optimizer,device)),
                  'Portable model/Adam/RNG/next-step state replay failed')
        del replay_model, replay_optimizer
        return {'selected_logits_bitwise_replayed': True, 'default_Adam_state_dict_RNG_next_step_bitwise_replayed': True,
                'optimizer_state_scope': 'admitted default Adam state_dict plus bound constructor options; no arbitrary hooks or optimizer.__dict__',
                'extra_training_steps': 2, 'extra_complete_eval_forwards': 2,
                'elapsed_seconds': time.perf_counter()-started}
    finally:
        restore_rng(caller_rng, device)


def synchronize(device):
    if device.startswith('cuda:'):
        import torch
        torch.cuda.synchronize(torch.device(device))


def memory(device):
    value = {'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    status = c.Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, number, _ = line.split(); value[key.rstrip(':')+'_bytes'] = int(number)*1024
    if device.startswith('cuda:'):
        import torch
        value.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                     cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return value
