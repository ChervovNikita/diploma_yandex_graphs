"""Inactive fixed author-SAGE M1 acquisition; root owns admission and watchdog.

No stock OGB provider/test(), calibrator, truth scorer, qualifier or retry grid.
"""
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import resource
import sys
import time


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fingerprint(torch, value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().contiguous(); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v: visit(x)
        else: h.update(repr(v).encode())
    visit(value); return h.hexdigest()


def rng(torch, np, device):
    s = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (s[0], s[1].tolist(), s[2], s[3], s[4]),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def restore_rng(torch, np, device, value):
    random.setstate(value['python']); s = value['numpy']
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(value['cpu'].cpu()); torch.cuda.set_rng_state(value['cuda'].cpu(), device)


def run(cfg, output, root_validate):
    """Callable integration entry. The root callback validates every input hash.

    cfg: execution_enabled, seeds, point, author_source, device, limits,
    profile{edges,self_edges}, paths{train,select,cal_ids,confirm_ids}, and
    runtime_sources{sage_conv,sparse_tensor}: exact path/bytes/sha256 receipts.
    Additional provider/source/official-role custody fields are bound by root.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); current = {'seed': None, 'update': 0}
    result = {'terminal': False, 'complete': False, 'records': [], 'current': current,
              'CAL_CONFIRM_truth_access': False, 'automatic_calibration_or_TEST_release': False,
              'unused_confirmation_claim': False, 'scope': 'encountered arxiv fixed public-author context'}
    try:
        spec = importlib.util.spec_from_file_location('fixed_author_model', Path(__file__).with_name('author_model.py'))
        author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
        limits = cfg.get('limits', {})
        if (cfg.get('execution_enabled') is not True or cfg.get('seeds') != list(author.SEEDS)
                or cfg.get('point') != author.POINT or not callable(root_validate)
                or set(limits) != {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
                or any(type(v) is not int or v <= 0 for v in limits.values())
                or set(cfg.get('paths', {})) != {'train', 'select', 'cal_ids', 'confirm_ids'}
                or cfg.get('support_policy') != 'symmetric_coalesced_no_added_loops'):
            raise ValueError('Inactive source: exact fixed point, roster and root admission required')
        source_receipts = {f: receipt(Path(__file__).with_name(f)) for f in ('consumer.py', 'author_model.py')}
        custody = root_validate(cfg, action='fixed_author_SAGE_context_fits', source_receipts=source_receipts)
        if not isinstance(custody, dict) or custody.get('status') != 'ROOT_AUTHOR_CONTEXT_FITS_ADMITTED':
            raise ValueError('Root must bind provider, official memberships, runtime/resource and prospective role admission')
        json.dumps(custody, allow_nan=False)  # Checkpoint custody is primitive metadata.
        config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        result.update(custody=custody, config_sha256=config_sha, fixed_point=author.POINT, fixed_seeds=list(author.SEEDS))
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        from torch_sparse import SparseTensor
        if torch.get_default_dtype() != torch.float32 or cfg['device'] != 'cuda:0':
            raise ValueError('Admitted sole-device native float32 runtime required')
        device = torch.device(cfg['device']); torch.cuda.set_device(device)
        for key, obj in (('sage_conv', SAGEConv), ('sparse_tensor', SparseTensor)):
            actual = receipt(inspect.getfile(obj))
            if actual != cfg['runtime_sources'][key]: raise ValueError('Imported runtime source differs: ' + key)
        SAGE = author.model_class(torch, F, SAGEConv, cfg['author_source'])
        def synchronize(): torch.cuda.synchronize(device)
        def budget():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if time.perf_counter() - started > limits['wall_seconds'] or rss > limits['rss_bytes'] or torch.cuda.max_memory_allocated(device) > limits['cuda_bytes']:
                raise RuntimeError('Finite entire-context wall/RSS/CUDA cap exceeded')
        budget()
        with np.load(cfg['paths']['train'], allow_pickle=False) as a:
            if set(a.files) != {'x', 'edge_index', 'ids', 'y'}: raise ValueError('Exact safe TRAIN fields')
            x, edges, train_ids, train_y = [a[k].copy() for k in ('x', 'edge_index', 'ids', 'y')]
        with np.load(cfg['paths']['select'], allow_pickle=False) as a:
            if set(a.files) != {'ids', 'y'}: raise ValueError('SELECT-only selector fields')
            select_ids, select_y = a['ids'].copy(), a['y'].copy()
        cal_ids = np.load(cfg['paths']['cal_ids'], allow_pickle=False)
        confirm_ids = np.load(cfg['paths']['confirm_ids'], allow_pickle=False)
        roles = {'TRAIN': train_ids, 'SELECT': select_ids, 'CAL': cal_ids, 'CONFIRM': confirm_ids}
        if (x.shape != (169343, 128) or x.dtype != np.float32 or not np.isfinite(x).all()
                or edges.dtype != np.int64 or edges.shape != (2, cfg['profile']['edges'])
                or edges.size == 0 or edges.min() < 0 or edges.max() >= 169343
                or int((edges[0] == edges[1]).sum()) != cfg['profile']['self_edges']):
            raise ValueError('Complete frozen original features/support required')
        for role, count in (('TRAIN', 90941), ('SELECT', 14900), ('CAL', 14899), ('CONFIRM', 48603)):
            if roles[role].dtype != np.int64 or roles[role].shape != (count,): raise ValueError('Official role-count preparation discrepancy: ' + role)
        all_ids = np.concatenate(list(roles.values()))
        if not np.array_equal(np.sort(all_ids), np.arange(169343, dtype=np.int64)):
            raise ValueError('Complete unique disjoint official role membership required')
        for y, ids in ((train_y, train_ids), (select_y, select_ids)):
            if y.dtype != np.int64 or y.shape != ids.shape or y.min() < 0 or y.max() >= 40: raise ValueError('Safe role-label shape/range differs')
        tensor = lambda a: torch.from_numpy(a).to(device)
        features, train_idx, train_targets, select_idx, select_targets = map(tensor, (x, train_ids, train_y, select_ids, select_y))
        # Same published support. COO source->target becomes adj_t row=target.
        adj_t = SparseTensor(row=torch.from_numpy(edges[1]), col=torch.from_numpy(edges[0]),
                             sparse_sizes=(169343, 169343)).coalesce().to(device)
        if adj_t.nnz() != edges.shape[1]: raise ValueError('Prepared coalesced support changed')
        export_ids = {k: ids for k, ids in roles.items() if k != 'TRAIN'}
        export_idx = {k: tensor(ids) for k, ids in export_ids.items()}
        synchronize(); budget()
        result['preparation_seconds'] = time.perf_counter() - started
        result['graph'] = {'nodes': 169343, 'edges': int(edges.shape[1]), 'self_edges': cfg['profile']['self_edges'],
                           'adjacency': 'torch_sparse.SparseTensor adj_t; no loops or symmetry added by consumer'}
        del x, edges, train_ids, train_y, select_ids, select_y, all_ids, cal_ids, confirm_ids, roles
        for seed in author.SEEDS:
            current.update(seed=seed, update=0); budget(); folder = output / f'author_SAGE_M1_seed{seed}'; folder.mkdir()
            synchronize(); tick = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            counts = {'TRAIN_routes': 0, 'SELECT_routes': 0, 'backwards': 0, 'Adam_steps': 0, 'selected_routes': 0, 'checkpoint_writes': 0}
            record = {'arm': 'author_SAGE_M1', 'seed': seed, 'status': 'RUNNING', 'operation_counts': counts}
            result['records'].append(record)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = SAGE(128, 256, 40, 3, .5).to(device); model.reset_parameters()
            construction = author.inspect_model(torch, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.)
            owned = [id(p) for g in optimizer.param_groups for p in g['params']]
            if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in model.parameters() if p.requires_grad}: raise ValueError('Independent M1 optimizer ownership differs')
            best_correct, selected, saves = -1, 0, 0; phase_seconds = {'TRAIN': 0., 'SELECT': 0., 'checkpoint_save': 0.}
            record['construction'] = construction
            with (folder / 'trace.jsonl').open('x') as trace:
                for update in range(1, 501):
                    current['update'] = update; budget(); synchronize(); phase = time.perf_counter()
                    model.train(); optimizer.zero_grad()
                    log_probs = model(features, adj_t); counts['TRAIN_routes'] += 1
                    loss = F.nll_loss(log_probs[train_idx], train_targets)
                    if not math.isfinite(loss.item()): raise ValueError('Nonfinite fixed full-TRAIN NLL')
                    loss.backward(); counts['backwards'] += 1
                    optimizer.step(); counts['Adam_steps'] += 1; train_loss = loss.item(); del log_probs, loss
                    synchronize(); phase_seconds['TRAIN'] += time.perf_counter() - phase
                    model.eval(); synchronize(); phase = time.perf_counter()
                    with torch.no_grad(): selected_logp = model(features, adj_t)[select_idx]
                    counts['SELECT_routes'] += 1
                    if selected_logp.shape != (14900, 40) or not torch.isfinite(selected_logp).all().item(): raise ValueError('SELECT output dimensions/finiteness differ')
                    correct = int(selected_logp.argmax(-1).eq(select_targets).sum().item()); del selected_logp
                    synchronize(); phase_seconds['SELECT'] += time.perf_counter() - phase
                    improved = correct > best_correct
                    if improved:
                        best_correct, selected = correct, update
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'global_rng': rng(torch, np, device),
                                 'seed': seed, 'selected_step': selected, 'SELECT_correct': correct, 'SELECT_rows': 14900,
                                 'custody': custody, 'config_sha256': config_sha,
                                 'model_sha256': fingerprint(torch, model.state_dict()), 'optimizer_sha256': fingerprint(torch, optimizer.state_dict())}
                        synchronize(); phase = time.perf_counter(); torch.save(state, folder / 'selected.pt'); synchronize()
                        phase_seconds['checkpoint_save'] += time.perf_counter() - phase; saves += 1; del state
                        counts['checkpoint_writes'] += 1
                    trace.write(json.dumps({'update': update, 'TRAIN_NLL': train_loss, 'SELECT_correct': correct, 'SELECT_rows': 14900,
                                            'strict_improvement': improved, 'selected_step': selected}, allow_nan=False) + '\n'); trace.flush()
                    write(folder / 'PROGRESS.json', dict(current, selected_step=selected)); budget()
            synchronize(); restore_start = time.perf_counter()
            state = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=True)
            if state['custody'] != custody or state['config_sha256'] != config_sha or state['seed'] != seed: raise ValueError('Selected-state custody differs')
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer']); restore_rng(torch, np, device, state['global_rng'])
            if (fingerprint(torch, model.state_dict()) != state['model_sha256'] or fingerprint(torch, optimizer.state_dict()) != state['optimizer_sha256']
                    or fingerprint(torch, rng(torch, np, device)) != fingerprint(torch, state['global_rng'])
                    or any(int(b.num_batches_tracked.item()) != selected for b in model.bns)):
                raise ValueError('Exact selected parameters/BN buffers/Adam/global-RNG restoration failed')
            synchronize(); phase_seconds['restoration'] = time.perf_counter() - restore_start
            model.eval(); synchronize(); phase = time.perf_counter()
            with torch.no_grad(): final = model(features, adj_t)
            counts['selected_routes'] += 1
            if final.shape != (169343, 40) or not torch.isfinite(final).all().item(): raise ValueError('Selected complete graph output differs')
            if int(final[select_idx].argmax(-1).eq(select_targets).sum().item()) != state['SELECT_correct']: raise ValueError('Restored SELECT criterion differs')
            archives = {}
            for role, idx in export_idx.items():
                logp = final[idx]; path = folder / ('selected_' + role + '.npz')
                np.savez(path, ids=export_ids[role], member_log_probs=logp.unsqueeze(0).cpu().numpy(),
                         native_member_argmax=logp.argmax(-1).unsqueeze(0).cpu().numpy())
                archives[role] = receipt(path); del logp
            synchronize(); phase_seconds['selected_forward_and_archives'] = time.perf_counter() - phase
            if any(counts[k] != 500 for k in ('TRAIN_routes', 'SELECT_routes', 'backwards', 'Adam_steps')) or counts['selected_routes'] != 1:
                raise ValueError('Exact fixed 500-update/native-route loop did not close')
            counts['SAGE_conv_calls'] = 3 * (counts['TRAIN_routes'] + counts['SELECT_routes'] + counts['selected_routes'])
            record.update(status='SELECTED_COMPLETE', completed_updates=500, selected_step=selected, checkpoint=receipt(folder / 'selected.pt'),
                          archives=archives, representation='member_log_probs_float32', members=1, serving='probability_mean',
                          selected_restore_verified=True,
                          costs={**phase_seconds, 'fit_wall_seconds': time.perf_counter() - tick, 'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                          'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                          'process_lifetime_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                          'optimizer_tensor_state_bytes': sum(v.numel() * v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))})
            write(folder / 'RESULT.json', record); del model, optimizer, state, final; torch.cuda.empty_cache(); budget()
        if len(result['records']) != 10 or any(r['status'] != 'SELECTED_COMPLETE' for r in result['records']): raise ValueError('Entire ten-context roster required')
        result.update(terminal=True, complete=True, fits=10, selected_states=10, TRAIN_routes=5000, SELECT_routes=5000,
                      backwards=5000, Adam_steps=5000, selected_routes=10, SAGE_conv_calls=30030)
    except BaseException as e:
        if result['records'] and result['records'][-1]['status'] == 'RUNNING': result['records'][-1].update(status='FAILED_RETAINED', failure={'type': type(e).__name__, 'message': str(e)}, completed_updates=result['records'][-1]['operation_counts']['Adam_steps'])
        result.update(terminal=True, failure={'type': type(e).__name__, 'message': str(e)}, retry=False, missing_seed_substitution=False)
        result['entire_context_wall_seconds'] = time.perf_counter() - started; write(output / 'COMPLETE_AUTHOR_CONTEXT.json', result); raise
    result['entire_context_wall_seconds'] = time.perf_counter() - started
    write(output / 'COMPLETE_AUTHOR_CONTEXT.json', result); return result


if __name__ == '__main__':
    raise SystemExit('Inactive source. Root supplies fixed cfg/admission, provider custody, runtime qualification and finite owner.')
