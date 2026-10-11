"""Inactive fixed published Physics meanSAGE acquisition and selected export.

Root owns source/provider/runtime/resource/admission and external watchdog.
No stock TensorFlow test driver, CAL/CONFIRM truth, scorer or retry grid.
"""
import hashlib
import importlib.util
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
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def fingerprint(torch, value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().contiguous()
            h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v: visit(x)
        else: h.update(repr(v).encode())
    visit(value); return h.hexdigest()


def rng(torch, np, device):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def restore_rng(torch, np, device, value):
    random.setstate(value['python']); state = value['numpy']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    torch.set_rng_state(value['cpu'].cpu()); torch.cuda.set_rng_state(value['cuda'].cpu(), device)


def run(cfg, output, root_validate):
    """Root callable: ten published-recipe singles on ten paired masks.

    cfg exact controls: execution_enabled=True, seeds, point=author.POINT,
    device='cuda:0', limits{wall_seconds,cuda_bytes,rss_bytes}, profile,
    role_blocks{seedstr:{train,select,cal_ids,confirm_ids}:pathstrings}.
    Root callback receives action='fixed_Physics_author_meanSAGE_context_fits'
    and source_receipts; must return ROOT_AUTHOR_CONTEXT_FITS_ADMITTED custody,
    binding provider, all ten NPZ/ID/source/runtime hashes, common graph,
    prospective roles, resource qualification/caps and all scientific decisions.
    CAL/CONFIRM memberships are label-free; corresponding truth paths are absent.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); current = {'seed': None, 'update': 0}
    result = {'terminal': False, 'complete': False, 'records': [], 'current': current,
              'CAL_CONFIRM_truth_access': False, 'automatic_calibration_or_CONFIRM_release': False,
              'published_score_replication_claim': False,
              'scope': 'published TensorFlow author meanSAGE source-derived Torch port on root custom Physics masks'}
    try:
        spec = importlib.util.spec_from_file_location('fixed_Physics_author_meanSAGE', Path(__file__).with_name('author_model.py'))
        author = importlib.util.module_from_spec(spec); sys.modules[spec.name] = author; spec.loader.exec_module(author)
        limits = cfg.get('limits', {}); blocks = cfg.get('role_blocks', {})
        expected_profile = {'dataset': 'CoauthorPhysics', 'nodes': 34493, 'input_dim': 8415, 'classes': 5,
                            'edges': 530417, 'self_edges': 34493,
                            'train_rows': 100, 'select_rows': 150, 'cal_rows': 150, 'confirm_rows': 34093}
        if (cfg.get('execution_enabled') is not True or cfg.get('seeds') != list(author.SEEDS)
                or cfg.get('point') != author.POINT or not callable(root_validate)
                or set(limits) != {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
                or any(type(v) is not int or v <= 0 for v in limits.values())
                or set(blocks) != {str(s) for s in author.SEEDS}
                or any(set(b) != {'train', 'select', 'cal_ids', 'confirm_ids'} for b in blocks.values())
                or any(cfg.get('profile', {}).get(k) != v for k, v in expected_profile.items())):
            raise ValueError('Inactive source: exact published point, complete paired masks and root admission required')
        source_receipts = {f: receipt(Path(__file__).with_name(f)) for f in ('consumer.py', 'author_model.py')}
        custody = root_validate(cfg, action='fixed_Physics_author_meanSAGE_context_fits', source_receipts=source_receipts)
        if not isinstance(custody, dict) or custody.get('status') != 'ROOT_AUTHOR_CONTEXT_FITS_ADMITTED':
            raise ValueError('Root must bind provider, custom memberships, source/runtime/resource and prospective role admission')
        json.dumps(custody, allow_nan=False)
        config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        result.update(custody=custody, config_sha256=config_sha, fixed_point=author.POINT,
                      fixed_seeds=list(author.SEEDS), profile=expected_profile)
        import numpy as np
        import torch
        import torch.nn.functional as F
        if torch.get_default_dtype() != torch.float32 or cfg.get('device') != 'cuda:0':
            raise ValueError('Admitted sole-device float32 runtime required')
        device = torch.device(cfg['device']); torch.cuda.set_device(device)
        MeanSAGE, TF1Adam = author.model_class(torch, F), author.optimizer_class(torch)
        def synchronize(): torch.cuda.synchronize(device)
        def budget():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if time.perf_counter() - started > limits['wall_seconds'] or rss > limits['rss_bytes'] or torch.cuda.max_memory_allocated(device) > limits['cuda_bytes']:
                raise RuntimeError('Finite entire native-context wall/RSS/CUDA cap exceeded')
        budget()
        # Read the complete public graph only once. Root binds identical X/support
        # in all safe block TRAIN NPZs; later blocks deserialize ids/y alone.
        with np.load(blocks[str(author.SEEDS[0])]['train'], allow_pickle=False) as a:
            if set(a.files) != {'x', 'edge_index', 'ids', 'y'}: raise ValueError('Exact safe TRAIN fields')
            x, edges = a['x'].copy(), a['edge_index'].copy()
        if (x.shape != (34493, 8415) or x.dtype != np.float32 or not np.isfinite(x).all()
                or not np.logical_or(x == 0., x == 1.).all()
                or edges.dtype != np.int64 or edges.shape != (2, 530417)
                or edges.min() < 0 or edges.max() >= 34493
                or int((edges[0] == edges[1]).sum()) != 34493):
            raise ValueError('Complete raw binary Physics input and already-looped common support required')
        tensor = lambda a: torch.from_numpy(a).to(device)
        # Canonical row-major stored values correspond to scipy COO raw X.
        features = torch.from_numpy(x).to_sparse_coo().coalesce().to(device)
        # Provider edge source->target becomes adjacency row=target,col=source.
        edge_tensor = torch.from_numpy(edges)
        adjacency = torch.sparse_coo_tensor(edge_tensor.flip(0), torch.ones(530417), (34493, 34493)).coalesce().to(device)
        if adjacency._nnz() != 530417: raise ValueError('Prepared coalesced support changed')
        degrees = torch.bincount(adjacency.indices()[0], minlength=34493).to(torch.float32).unsqueeze(1)
        if degrees.min().item() < 1.: raise ValueError('Common explicit-self support must have positive degrees')
        synchronize(); budget()
        result['preparation_seconds'] = time.perf_counter() - started
        result['graph'] = {**expected_profile, 'input_stored_values': features._nnz(),
                           'adjacency': 'torch sparse COO, no added loops or symmetry',
                           'normalize_features': False}
        del x, edges, edge_tensor
        for seed in author.SEEDS:
            current.update(seed=seed, update=0); budget()
            block = blocks[str(seed)]
            with np.load(block['train'], allow_pickle=False) as a:
                if set(a.files) != {'x', 'edge_index', 'ids', 'y'}: raise ValueError('Exact safe TRAIN block fields')
                train_ids, train_y = a['ids'].copy(), a['y'].copy()
            with np.load(block['select'], allow_pickle=False) as a:
                if set(a.files) != {'ids', 'y'}: raise ValueError('SELECT-only selector fields')
                select_ids, select_y = a['ids'].copy(), a['y'].copy()
            cal_ids = np.load(block['cal_ids'], allow_pickle=False)
            confirm_ids = np.load(block['confirm_ids'], allow_pickle=False)
            roles = {'TRAIN': train_ids, 'SELECT': select_ids, 'CAL': cal_ids, 'CONFIRM': confirm_ids}
            for role, count in (('TRAIN', 100), ('SELECT', 150), ('CAL', 150), ('CONFIRM', 34093)):
                if roles[role].dtype != np.int64 or roles[role].shape != (count,): raise ValueError('Custom Physics role-count preparation differs: ' + role)
            all_ids = np.concatenate(list(roles.values()))
            if not np.array_equal(np.sort(all_ids), np.arange(34493, dtype=np.int64)):
                raise ValueError('Complete unique disjoint custom role membership required')
            for y, ids in ((train_y, train_ids), (select_y, select_ids)):
                if y.dtype != np.int64 or y.shape != ids.shape or y.min() < 0 or y.max() >= 5:
                    raise ValueError('Safe role-label shape/range differs')
            train_idx, train_targets, select_idx, select_targets = map(tensor, (train_ids, train_y, select_ids, select_y))
            export_ids = {k: ids for k, ids in roles.items() if k != 'TRAIN'}
            export_idx = {k: tensor(ids) for k, ids in export_ids.items()}
            folder = output / f'author_meanSAGE_M1_seed{seed}'; folder.mkdir()
            synchronize(); tick = time.perf_counter(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            counts = {'TRAIN_routes': 0, 'SELECT_routes': 0, 'backwards': 0,
                      'Adam_steps': 0, 'selected_routes': 0, 'checkpoint_writes': 0}
            record = {'block': seed, 'seed': seed, 'arm': 'author_meanSAGE_M1', 'members': 1,
                      'status': 'RUNNING', 'operation_counts': counts}
            result['records'].append(record)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = MeanSAGE().to(device); record['construction'] = author.inspect_model(model)
            optimizer = TF1Adam(model.parameters())
            owned = [id(p) for g in optimizer.param_groups for p in g['params']]
            if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in model.parameters() if p.requires_grad}:
                raise ValueError('Independent native M1 optimizer ownership differs')
            best, selected, stale = math.inf, 0, 0
            phase_seconds = {'TRAIN': 0., 'SELECT': 0., 'checkpoint_save': 0.}
            with (folder / 'trace.jsonl').open('x') as trace:
                for update in range(1, 100001):
                    current['update'] = update; budget(); synchronize(); phase = time.perf_counter()
                    model.train(); optimizer.zero_grad()
                    logits = model(features, adjacency, degrees); counts['TRAIN_routes'] += 1
                    ce = F.cross_entropy(logits[train_idx], train_targets)
                    penalty = model.regularizer(); loss = ce + penalty
                    train_ce, train_penalty, train_total = ce.item(), penalty.item(), loss.item()
                    if not math.isfinite(train_total): raise ValueError('Nonfinite full-TRAIN published CE+L2')
                    loss.backward(); counts['backwards'] += 1
                    optimizer.step(); counts['Adam_steps'] += 1
                    del logits, ce, penalty, loss
                    synchronize(); phase_seconds['TRAIN'] += time.perf_counter() - phase
                    model.eval(); synchronize(); phase = time.perf_counter()
                    with torch.no_grad():
                        select_logits = model(features, adjacency, degrees)[select_idx]
                        select_ce_tensor = F.cross_entropy(select_logits, select_targets)
                        select_penalty_tensor = model.regularizer()
                        select_total = (select_ce_tensor + select_penalty_tensor).item()
                        select_ce, select_penalty = select_ce_tensor.item(), select_penalty_tensor.item()
                        correct = int(select_logits.argmax(-1).eq(select_targets).sum().item())
                    counts['SELECT_routes'] += 1
                    if select_logits.shape != (150, 5) or not torch.isfinite(select_logits).all().item() or not math.isfinite(select_total):
                        raise ValueError('Full SELECT output/total loss differs')
                    del select_logits, select_ce_tensor, select_penalty_tensor
                    synchronize(); phase_seconds['SELECT'] += time.perf_counter() - phase
                    accepted = select_total <= best
                    if accepted:
                        best, selected, stale = select_total, update, 0
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'global_rng': rng(torch, np, device), 'seed': seed, 'selected_step': selected,
                                 'SELECT_total_loss': select_total, 'SELECT_CE': select_ce, 'SELECT_L2': select_penalty,
                                 'SELECT_correct': correct, 'SELECT_rows': 150, 'custody': custody, 'config_sha256': config_sha,
                                 'model_sha256': fingerprint(torch, model.state_dict()),
                                 'optimizer_sha256': fingerprint(torch, optimizer.state_dict())}
                        synchronize(); phase = time.perf_counter(); torch.save(state, folder / 'selected.pt'); synchronize()
                        phase_seconds['checkpoint_save'] += time.perf_counter() - phase
                        counts['checkpoint_writes'] += 1; del state
                    else: stale += 1
                    trace.write(json.dumps({'update': update, 'TRAIN_CE': train_ce, 'TRAIN_L2': train_penalty,
                                            'TRAIN_total': train_total, 'SELECT_CE': select_ce, 'SELECT_L2': select_penalty,
                                            'SELECT_total': select_total, 'SELECT_correct': correct, 'SELECT_rows': 150,
                                            'accepted_minimum_including_ties': accepted,
                                            'selected_step': selected, 'stale_updates': stale}, allow_nan=False) + '\n'); trace.flush()
                    write(folder / 'PROGRESS.json', dict(current, selected_step=selected, stale_updates=stale)); budget()
                    if stale >= 50: break
            synchronize(); phase_seconds['acquisition'] = time.perf_counter() - tick
            restore_start = time.perf_counter()
            state = torch.load(folder / 'selected.pt', map_location='cpu', weights_only=True)
            if state['custody'] != custody or state['config_sha256'] != config_sha or state['seed'] != seed:
                raise ValueError('Selected-state custody differs')
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
            restore_rng(torch, np, device, state['global_rng'])
            if (fingerprint(torch, model.state_dict()) != state['model_sha256']
                    or fingerprint(torch, optimizer.state_dict()) != state['optimizer_sha256']
                    or fingerprint(torch, rng(torch, np, device)) != fingerprint(torch, state['global_rng'])
                    or optimizer.state['tf1_global']['updates'] != selected):
                raise ValueError('Exact selected parameters/Adam/global-RNG restoration failed')
            synchronize(); phase_seconds['restoration'] = time.perf_counter() - restore_start
            model.eval(); synchronize(); phase = time.perf_counter()
            with torch.no_grad():
                final = model(features, adjacency, degrees)
                restored_total = (F.cross_entropy(final[select_idx], select_targets) + model.regularizer()).item()
            counts['selected_routes'] += 1
            if final.shape != (34493, 5) or not torch.isfinite(final).all().item() or restored_total != state['SELECT_total_loss']:
                raise ValueError('Selected complete output or restored total SELECT criterion differs')
            archives = {}
            for role, idx in export_idx.items():
                logits = final[idx]; path = folder / ('selected_' + role + '.npz')
                np.savez(path, ids=export_ids[role], member_log_probs=logits.log_softmax(-1).unsqueeze(0).cpu().numpy(),
                         native_member_argmax=logits.argmax(-1).unsqueeze(0).cpu().numpy(),
                         native_pool_argmax=logits.softmax(-1).argmax(-1).cpu().numpy())
                archives[role] = receipt(path); del logits
            synchronize(); phase_seconds['selected_forward_and_archives'] = time.perf_counter() - phase
            if any(counts[k] != update for k in ('TRAIN_routes', 'SELECT_routes', 'backwards', 'Adam_steps')) or counts['selected_routes'] != 1:
                raise ValueError('Published update/native-route count did not close')
            counts['meanSAGE_layer_calls'] = 2 * (counts['TRAIN_routes'] + counts['SELECT_routes'] + counts['selected_routes'])
            costs = {**phase_seconds, 'fit_wall_seconds': time.perf_counter() - tick,
                     'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                     'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                     'process_lifetime_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                     'optimizer_tensor_state_bytes': sum(v.numel() * v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))}
            selected_receipt = receipt(folder / 'selected.pt')
            record.update(status='SELECTED_COMPLETE', completed_updates=update, selected_step=selected,
                          stopping='patience50_consecutive_worse' if stale >= 50 else 'max100000',
                          checkpoint=selected_receipt, selected_state_receipt=selected_receipt,
                          archives=archives, selected_CAL_archive=archives['CAL'], selected_CONFIRM_archive=archives['CONFIRM'],
                          fits=[{'status': 'SELECTED_COMPLETE', 'selected_state_receipt': selected_receipt,
                                 'costs': costs, 'selected_step': selected, 'completed_updates': update}],
                          representation='member_log_probs_float32', serving='probability_mean',
                          selected_restore_verified=True, selected_SELECT_total_loss=state['SELECT_total_loss'], costs=costs)
            write(folder / 'RESULT.json', record)
            del model, optimizer, state, final, train_idx, train_targets, select_idx, select_targets
            del train_ids, train_y, select_ids, select_y, cal_ids, confirm_ids, roles, all_ids, export_ids, export_idx
            torch.cuda.empty_cache(); budget()
        if len(result['records']) != 10 or any(r['status'] != 'SELECTED_COMPLETE' for r in result['records']):
            raise ValueError('Entire ten-context roster required')
        result.update(terminal=True, complete=True, banks=10, fits=10, selected_states=10)
        result['operation_counts'] = {k: sum(r['operation_counts'][k] for r in result['records']) for k in result['records'][0]['operation_counts']}
    except BaseException as e:
        if result['records'] and result['records'][-1]['status'] == 'RUNNING':
            result['records'][-1].update(status='FAILED_RETAINED', failure={'type': type(e).__name__, 'message': str(e)},
                                         completed_updates=result['records'][-1]['operation_counts']['Adam_steps'])
        result.update(terminal=True, failure={'type': type(e).__name__, 'message': str(e)}, retry=False, missing_seed_substitution=False)
        result['entire_context_wall_seconds'] = time.perf_counter() - started
        write(output / 'COMPLETE_AUTHOR_CONTEXT.json', result); raise
    result['entire_context_wall_seconds'] = time.perf_counter() - started
    write(output / 'COMPLETE_AUTHOR_CONTEXT.json', result)
    return result


if __name__ == '__main__':
    raise SystemExit('Inactive source. Root supplies exact cfg/admission, provider/runtime/resource custody and finite owner.')
