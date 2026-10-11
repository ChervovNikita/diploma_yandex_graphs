"""Two inactive, separate representative-full-graph engineering entries."""
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time


def clone(t, v):
    if t.is_tensor(v): return v.detach().cpu().clone()
    if isinstance(v, dict): return {k: clone(t, x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return type(v)(clone(t, x) for x in v)
    return v


def equal(t, a, b, atol=0., rtol=0.):
    if t.is_tensor(a):
        return t.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and (
            t.isfinite(a).all().item() and t.isfinite(b).all().item() and t.allclose(a, b.cpu(), atol=atol, rtol=rtol)
            if a.is_floating_point() else t.equal(a, b.cpu()))
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(t, a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(t, x, y, atol, rtol) for x, y in zip(a, b))
    return a == b


def state(f, m, o, streams):
    s = {'model': clone(f.torch, m.state_dict()), 'optimizer': clone(f.torch, o.state_dict()),
         'streams': clone(f.torch, streams), 'cpu_rng': f.torch.get_rng_state().clone()}
    if f.device.type == 'cuda': s['cuda_rng'] = f.torch.cuda.get_rng_state(f.device).cpu().clone()
    return s


def restore(f, m, o, s):
    m.load_state_dict(s['model'], strict=True)
    o.load_state_dict(clone(f.torch, s['optimizer']))  # Preserve CPU Adam step snapshots.
    f.torch.set_rng_state(s['cpu_rng'])
    if f.device.type == 'cuda': f.torch.cuda.set_rng_state(s['cuda_rng'], f.device)
    return clone(f.torch, s['streams'])


def restore_probe(f, m, o, streams, path):
    saved = state(f, m, o, streams); f.torch.save(saved, path)
    with f.torch.no_grad():
        for p in m.parameters(): p.add_(1)
        for values in o.state.values():
            for v in values.values():
                if f.torch.is_tensor(v): v.add_(1)
    streams = []; f.torch.manual_seed(997)
    loaded = f.torch.load(path, map_location='cpu', weights_only=True)
    streams = restore(f, m, o, loaded)
    if not equal(f.torch, saved, state(f, m, o, streams)): raise ValueError('Strict poisoned full-state restore failed')
    return streams


def paths(f, m, o, step, second=False):
    grads = [(n, p.grad) for n, p in m.named_parameters()]
    if any(g is None or not f.torch.isfinite(g).all().item() for n, g in grads): raise ValueError('Registered gradient missing/nonfinite')
    if not any(g.ne(0).any().item() for n, g in grads if 'value_' not in n): raise ValueError('Native gradient inactive')
    added = [(n, g) for n, g in grads if 'value_' in n]
    if added and (not any(g.ne(0).any().item() for n, g in added if n.rsplit('.', 1)[-1] in ('weight', 'bias'))
                  or second and not any(g.ne(0).any().item() for n, g in added if n.rsplit('.', 1)[-1] in ('r', 's'))): raise ValueError('Builder gradient inactive')
    if not o.state or any(v['step'].item() != step for v in o.state.values()): raise ValueError('One AdamW advancement required')


@contextmanager
def session(factory, output, dataset, release, limits):
    action = 'ROOT_ENGINE_EQUIVALENCE_ADMITTED' if dataset == 'WikiCS' else 'ROOT_ARXIV_RESOURCE_QUALIFICATION_ADMITTED'
    if release.get('status') != action or set(limits) != {'wall_seconds', 'cuda_bytes', 'rss_bytes'} or any(type(v) is not int or v <= 0 for v in limits.values()): raise ValueError('Inactive source: finite root qualification release required')
    if dataset == 'ogbn-arxiv' and release.get('engine_equivalence') != 'ROOT_VERIFIED_PASS': raise ValueError('Bounded engine risk must be resolved first')
    output = Path(output); output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    r = {'status': 'RUNNING', 'dataset': dataset, 'units': [], 'limits': limits, 'fits': 0,
         'SELECT_CAL_TEST_label_payload_or_quality_access': False, 'automatic_fit_release': False,
         'actual_arxiv_resource_certified': False, 'TRAIN_trajectories': 0, 'backwards': 0, 'AdamW_steps': 0,
         'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0}
    def budget(f=None):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        if time.perf_counter() - started > limits['wall_seconds'] or rss > limits['rss_bytes']: raise RuntimeError('Qualification wall/RSS cap exceeded')
        if f is not None and f.device.type == 'cuda':
            r['cuda_peak_allocated_bytes'] = max(r['cuda_peak_allocated_bytes'], f.torch.cuda.max_memory_allocated(f.device))
            r['cuda_peak_reserved_bytes'] = max(r['cuda_peak_reserved_bytes'], f.torch.cuda.max_memory_reserved(f.device))
            if r['cuda_peak_allocated_bytes'] > limits['cuda_bytes']: raise RuntimeError('Actual full-graph CUDA cap exceeded')
    try:
        budget(); f = factory(output); budget(f)
        if not f.training_only or f.cfg['profile']['dataset'] != dataset or f.valid_ids.numel() or f.valid_y.numel(): raise ValueError('Complete TRAIN-only admitted factory required')
        r['profile'] = f.cfg['profile']; r['port_receipt'] = f.port_receipt
        spec = importlib.util.spec_from_file_location('qualified_own_ce', Path(__file__).with_name('own_ce.py'))
        own = importlib.util.module_from_spec(spec); spec.loader.exec_module(own)
        yield f, own, r, budget, output
        budget(f); r['status'] = 'PASS'; r['actual_arxiv_resource_certified'] = dataset == 'ogbn-arxiv'
    except BaseException as e:
        r['status'] = 'FAIL'; r['failure'] = {'type': type(e).__name__, 'message': str(e)}; raise
    finally:
        r['elapsed_seconds'] = time.perf_counter() - started
        r['process_lifetime_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        filename = 'ENGINE_EQUIVALENCE.json' if dataset == 'WikiCS' else 'ARXIV_RESOURCE_QUALIFICATION.json'
        (output / filename).write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')


def equivalence(factory, output, *, seed, release, limits, tolerances):
    if set(tolerances) != {'gradient_atol', 'gradient_rtol', 'update_atol', 'update_rtol'} or any(not math.isfinite(v) or v < 0 for v in tolerances.values()): raise ValueError('Root must freeze engineering tolerances')
    with session(factory, output, 'WikiCS', release, limits) as (f, own, r, budget, output):
        if seed not in f.cfg['seeds']: raise ValueError('Frozen seed required')
        r['tolerances'] = tolerances; r['scope'] = 'full existing WikiCS TRAIN; E4 two common-state pairs and BASE one'
        for arm, operation, pairs in (('E4', 'endpoint', 2), ('BASE', 'none', 1)):
            f._operation = operation; m, o, streams = f.make(seed, 0, 'baseline', True, 4, 'shared'); rows = []
            for step in range(1, pairs + 1):
                common = state(f, m, o, streams); ends, grads = {}, {}
                for mode in ('sequential', 'simultaneous'):
                    streams = restore(f, m, o, common); f.synchronize(); tick = time.perf_counter()
                    result = own.update(f, m, o, streams, True, mode=mode, qualification=True, capture_gradients=True)
                    f.synchronize(); budget(f); paths(f, m, o, step, step == 2)
                    grads[mode] = result.pop('gradients'); ends[mode] = state(f, m, o, streams)
                    if equal(f.torch, common['model'], ends[mode]['model']): raise ValueError('AdamW state unchanged')
                    rows.append(dict(result, step=step, seconds=time.perf_counter() - tick))
                    for k, v in (('TRAIN_trajectories', 'native_trajectories'), ('backwards', 'backwards'), ('AdamW_steps', 'adam_steps')): r[k] += result[v]
                if not equal(f.torch, grads['sequential'], grads['simultaneous'], tolerances['gradient_atol'], tolerances['gradient_rtol']): raise ValueError('Accumulated gradient mismatch')
                for k in ('model', 'optimizer'):
                    if not equal(f.torch, ends['sequential'][k], ends['simultaneous'][k], tolerances['update_atol'], tolerances['update_rtol']): raise ValueError('One-AdamW state mismatch')
                for k in ('streams', 'cpu_rng') + (('cuda_rng',) if f.device.type == 'cuda' else ()):
                    if not equal(f.torch, ends['sequential'][k], ends['simultaneous'][k]): raise ValueError('Persistent/global RNG mismatch')
                streams = restore(f, m, o, ends['sequential']); del common, ends, grads
            streams = restore_probe(f, m, o, streams, output / (arm + '_discarded.pt')); budget(f)
            r['units'].append({'arm': arm, 'common_state_pairs': pairs, 'checks': rows, 'strict_restore': True})
            f._fit_model = f._single_adapter = None; del m, o, streams
        if (r['TRAIN_trajectories'], r['backwards'], r['AdamW_steps']) != (24, 15, 6): raise ValueError('Bounded engine check incomplete')
    return r


def arxiv_resources(factory, selector_ids, output, *, seed, release, limits):
    with session(factory, output, 'ogbn-arxiv', release, limits) as (f, own, r, budget, output):
        t, p = f.torch, f.cfg['profile']; r['scope'] = 'actual full arxiv sequential TRAIN/integration and label-free SELECT serving only'
        if seed not in f.cfg['seeds'] or selector_ids.dtype != t.long or selector_ids.device != f.device or selector_ids.shape != (p['select_rows'],) or selector_ids.unique().numel() != selector_ids.numel() or selector_ids.min().item() < 0 or selector_ids.max().item() >= p['nodes'] or t.isin(selector_ids, f.train_ids).any().item(): raise ValueError('Root-bound complete label-free SELECT IDs required')
        for arm, operation, members in (('E4', 'endpoint', 4), ('N4', 'node', 4), ('BASE', 'none', 4), ('equipped_M1', 'endpoint', 1)):
            f._operation = operation; m, o, streams = f.make(seed, 0, 'baseline', True, members, 'shared' if members == 4 else 'single')
            before = state(f, m, o, streams)
            if f.device.type == 'cuda': t.cuda.empty_cache(); t.cuda.reset_peak_memory_stats(f.device)
            f.synchronize(); tick = time.perf_counter()
            result = own.update(f, m, o, streams, members == 4, qualification=True)
            f.synchronize(); budget(f); paths(f, m, o, 1)
            if equal(t, before['model'], clone(t, m.state_dict())): raise ValueError('Sequential AdamW state unchanged')
            result.pop('gradients'); result['TRAIN_seconds'] = time.perf_counter() - tick
            result['TRAIN_cuda_peak_allocated_bytes'] = t.cuda.max_memory_allocated(f.device) if f.device.type == 'cuda' else 0
            for k, v in (('TRAIN_trajectories', 'native_trajectories'), ('backwards', 'backwards'), ('AdamW_steps', 'adam_steps')): r[k] += result[v]
            for replay in range(2):
                if replay: streams = restore_probe(f, m, o, streams, output / (arm + '_discarded.pt'))
                m.eval(); f.synchronize(); tick = time.perf_counter()
                with t.no_grad():
                    logits = f.logits(m, streams, members == 4, selector_ids); pool = logits.softmax(-1).mean(0)
                    if logits.shape != (members, p['select_rows'], p['classes']) or not t.isfinite(pool).all().item(): raise ValueError('Complete label-free serving shape/finiteness differs')
                f.synchronize(); budget(f); result['serving_seconds_' + str(replay)] = time.perf_counter() - tick
                del logits, pool
            r['units'].append(dict(result, arm=arm, strict_restore=True, SELECT_trajectories=2 * members))
            f._fit_model = f._single_adapter = None; del before, m, o, streams
        f._operation = 'endpoint'; bodies = [f.make(seed, i, 'baseline', True, 1, 'single') for i in (0, 1)]
        params = [x for m, o, streams in bodies for x in m.parameters()]
        if len({id(o) for m, o, streams in bodies}) != 2 or len({id(x) for x in params}) != len(params) or len({x.untyped_storage().data_ptr() for x in params}) != len(params): raise ValueError('Independent equipped bodies/builders/optimizers alias')
        for m, o, streams in bodies:
            ids = [id(x) for g in o.param_groups for x in g['params']]
            if len(ids) != len(set(ids)) or set(ids) != {id(x) for x in m.parameters() if x.requires_grad}: raise ValueError('Independent optimizer ownership differs')
        r['I4_ownership'] = {'separately_constructed_M1_bodies': 2, 'native_seeds': [f.seeds(seed, i)[0] for i in (0, 1)], 'disjoint_parameters_storage_optimizers': True, 'TRAIN_updates': 0}
        if (r['TRAIN_trajectories'], r['backwards'], r['AdamW_steps']) != (13, 13, 4): raise ValueError('Bounded actual-graph integration incomplete')
        r['SELECT_trajectories'] = 26; budget(f)
    return r


if __name__ == '__main__':
    raise SystemExit('Inactive source: root supplies separately admitted complete-graph factories, budgets and membership IDs.')
