"""Three native fits and declared competence controls; validation only."""
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import resource
import time
from contextlib import contextmanager
from pathlib import Path

from common import (PACKET, empty_output, metrics, read_public, sha256,
                    verify_runtime, write_json, write_predictions)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def sampling_namespace(seed):
    """Reset validation sampling while preserving every training RNG stream."""
    import numpy as np
    import torch
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_seed(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def sync(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def make_loader(data, table, shuffle=False, batch_size=512):
    from types import SimpleNamespace
    from relbench.base import TaskType
    from relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader
    task = SimpleNamespace(entity_col='driverId', entity_table='drivers', time_col='date',
                           target_col='position', task_type=TaskType.REGRESSION)
    inp = get_node_train_table_input(table=table, task=task)
    return NeighborLoader(data, num_neighbors=[128, 64], time_attr='time',
                          input_nodes=inp.nodes, input_time=inp.time,
                          transform=inp.transform, batch_size=batch_size,
                          temporal_strategy='uniform', subgraph_type='directional',
                          disjoint=True, shuffle=shuffle, num_workers=0,
                          persistent_workers=False)


def audit_batch(batch, table):
    """Audit temporal cutoff, forecast identity, and disjoint root components."""
    import numpy as np
    import torch
    from relbench.modeling.utils import to_unix_time
    root = batch['drivers']
    ids = root.input_id.cpu().numpy()
    n = len(ids)
    if len(np.unique(ids)) != n or (ids < 0).any() or (ids >= len(table)).any():
        raise RuntimeError('Invalid forecast input_id in sampled batch')
    expected = table.df.iloc[ids]
    if not np.array_equal(root.n_id[:n].cpu().numpy(), expected.driverId.astype(int).to_numpy()):
        raise RuntimeError('Sampler root IDs do not match forecast rows')
    if not np.array_equal(root.seed_time.cpu().numpy(), to_unix_time(expected.date)):
        raise RuntimeError('Sampler seed times do not match forecast rows')
    if not torch.equal(root.batch[:n].cpu(), torch.arange(n)):
        raise RuntimeError('Repeated roots must belong to distinct disjoint components')
    if 'position' in expected and not np.array_equal(root.y.cpu().numpy(), expected.position.to_numpy(dtype=float)):
        raise RuntimeError('Target attachment did not use forecast input_id')
    digest = hashlib.sha256()
    for name in sorted(batch.node_types):
        store = batch[name]
        if 'time' in store:
            if 'batch' not in store or (store.time > root.seed_time[store.batch]).any():
                raise RuntimeError(f'Sampled future timestamp in node type {name}')
        for attr in ['n_id', 'batch']:
            if attr in store:
                digest.update(name.encode() + attr.encode() + store[attr].cpu().numpy().tobytes())
    for edge in sorted(batch.edge_types):
        digest.update(str(edge).encode() + batch[edge].edge_index.cpu().numpy().tobytes())
    digest.update(ids.tobytes())
    return ids, digest.hexdigest()


def repeated_driver_probe(data, table):
    from relbench.base import Table
    df = table.df
    pair = None
    for _, group in df.groupby('driverId', sort=True):
        if group.date.nunique() > 1:
            first = group.iloc[0]
            second = group[group.date != first.date].iloc[0]
            pair = df.loc[[first.name, second.name]].reset_index(drop=True)
            break
    if pair is None:
        raise RuntimeError('Expected repeated driver with distinct forecast dates')
    probe = Table(pair, {'driverId': 'drivers'}, time_col='date')
    with sampling_namespace(900042):
        batches = list(make_loader(data, probe, batch_size=2))
        if len(batches) != 1:
            raise RuntimeError('Repeated-driver probe was not one two-root batch')
        ids, digest = audit_batch(batches[0], probe)
        if ids.tolist() != [0, 1]:
            raise RuntimeError('Repeated forecast rows were collapsed')
    return {'driverId': int(pair.driverId.iloc[0]), 'original_row_ids': pair.row_id.tolist(),
            'dates': [x.isoformat() for x in pair.date], 'distinct_input_ids': ids.tolist(),
            'sampling_digest': digest, 'passed': True}


def validate(model, loader, table, device, clamp, seed):
    import numpy as np
    import torch
    pred, seen, digests = np.empty(len(table)), np.zeros(len(table), dtype=int), []
    started = time.perf_counter()
    with sampling_namespace(100000 + seed), torch.no_grad():
        model.eval()
        for batch in loader:
            ids, digest = audit_batch(batch, table)
            batch = batch.to(device)
            values = model(batch, 'drivers')
            if values.shape != (len(ids), 1) or not torch.isfinite(values).all():
                raise RuntimeError('Expected finite scalar native forward')
            values = torch.clamp(values.flatten(), *clamp).cpu().numpy()
            pred[ids] = values
            seen[ids] += 1
            digests.append(digest)
        sync(device)
    if not (seen == 1).all():
        raise RuntimeError('Validation predictions must cover every forecast exactly once')
    return pred, digests, time.perf_counter() - started


def native_fit(data, stats, rows, device, seed, out):
    import numpy as np
    import torch
    set_seed(seed)
    spec = importlib.util.spec_from_file_location('pinned_native_model', PACKET / 'sources/legacy_examples_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model(data=data, col_stats_dict=stats, num_layers=2, channels=128,
                         out_channels=1, aggr='sum', norm='batch_norm').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=0)
    train_loader, val_loader = make_loader(data, rows['train'], True), make_loader(data, rows['val'])
    clamp = np.percentile(rows['train'].df.position.to_numpy(), [2, 98]).tolist()
    if len(train_loader) != 15 or len(val_loader) != 1:
        raise RuntimeError('Native profile expects 15 train batches and one validation batch')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started, best, selected, selected_epoch = time.perf_counter(), float('inf'), None, None
    history, sampling_digest = [], None
    updates, batches_seconds = 0, []
    for epoch in range(1, 11):
        model.train()
        seen = np.zeros(len(rows['train']), dtype=int)
        total_loss = 0.
        iterator = iter(train_loader)
        for _ in range(len(train_loader)):
            sync(device)
            batch_start = time.perf_counter()
            batch = next(iterator)
            ids, _ = audit_batch(batch, rows['train'])
            seen[ids] += 1
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, 'drivers')
            if pred.shape != (len(ids), 1) or not torch.isfinite(pred).all():
                raise RuntimeError('Expected finite scalar native forward')
            loss = torch.nn.functional.l1_loss(pred.flatten().float(), batch['drivers'].y.float())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite native loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite native backward gradient')
            optimizer.step()
            sync(device)
            batches_seconds.append(time.perf_counter() - batch_start)
            total_loss += float(loss.detach()) * len(ids)
            updates += 1
        if not (seen == 1).all():
            raise RuntimeError('Training epoch must cover every forecast exactly once')
        pred, digest, val_seconds = validate(model, val_loader, rows['val'], device, clamp, seed)
        if sampling_digest is None:
            sampling_digest = digest
        elif digest != sampling_digest:
            raise RuntimeError('Validation sampling changed between checkpoint comparisons')
        score = metrics(rows['val'].df.position, pred)
        if score['mae'] < best:  # strict; equal MAE retains earlier checkpoint
            best, selected, selected_epoch = score['mae'], copy.deepcopy(model.state_dict()), epoch
        history.append({'epoch': epoch, 'train_l1': total_loss / len(rows['train']),
                        'validation': score, 'validation_seconds': val_seconds})
        write_json(out / f'native_seed{seed}_epochs.json', history)
        print(f'seed={seed} epoch={epoch} val_mae={score["mae"]:.6f}', flush=True)
    if updates != 150 or selected is None:
        raise RuntimeError('Unexpected native update/checkpoint count')
    model.load_state_dict(selected)
    pred, digest, restored_seconds = validate(model, val_loader, rows['val'], device, clamp, seed)
    score = metrics(rows['val'].df.position, pred)
    if digest != sampling_digest or not np.isclose(score['mae'], best, rtol=1e-6, atol=1e-7):
        raise RuntimeError('Restored checkpoint did not reproduce fixed-sampling validation')
    torch.save({'state_dict': selected, 'seed': seed, 'epoch': selected_epoch,
                'clamp_train_percentiles_2_98': clamp, 'profile_sha256': sha256(PACKET / 'PROFILE.json'),
                'public_manifest_sha256': sha256(out / 'PUBLIC_INPUTS_COPY.json')},
               out / f'native_seed{seed}_best.pt')
    write_predictions(out / f'native_seed{seed}_val.csv', rows['val'], pred)
    report = {'seed': seed, 'best_epoch': selected_epoch, 'validation': score,
              'updates': updates, 'fit_seconds': time.perf_counter() - started,
              'train_batch_seconds_including_sampling': batches_seconds,
              'validation_sampling_seed': 100000 + seed, 'sampling_digest': digest,
              'restored_mae_minus_selected_mae': score['mae'] - best,
              'restored_metric_tolerance': {'relative': 1e-6, 'absolute': 1e-7},
              'restored_validation_seconds': restored_seconds,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    write_json(out / f'native_seed{seed}_summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--glove-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    a = parser.parse_args()
    out = empty_output(a.output_dir)
    # Offline settings precede all scientific imports.
    import os
    os.environ['HF_HUB_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'] = '1', '1'
    runtime = verify_runtime()
    import torch
    import torch_geometric.typing as pyg_typing
    from torch_frame.config import TextEmbedderConfig
    from features import LocalGlove, chronological_graph
    from controls import run_controls
    if not pyg_typing.WITH_PYG_LIB or not hasattr(torch.ops.pyg, 'hetero_neighbor_sample'):
        raise RuntimeError('Compiled pyg-lib heterogeneous temporal sampler is required')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    if device.type == 'cuda':
        torch.set_num_threads(1)
    runtime.update(device=str(device), torch_cuda=torch.version.cuda,
                   gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    write_json(out / 'RUNTIME.json', runtime)
    started = time.perf_counter()
    manifest, db, rows = read_public(a.input_root)
    write_json(out / 'PUBLIC_INPUTS_COPY.json', manifest)
    profile = json.loads((PACKET / 'PROFILE.json').read_text())
    write_json(out / 'PROFILE_COPY.json', profile)
    set_seed(42)
    embedder = LocalGlove(a.glove_dir, device, out)
    cfg = TextEmbedderConfig(text_embedder=embedder, batch_size=256)
    prep_start = time.perf_counter()
    data, stats, stypes = chronological_graph(db, cfg, out)
    probe = repeated_driver_probe(data, rows['train'])
    write_json(out / 'SAMPLER_RUNTIME_QA.json', probe)
    prep_seconds = time.perf_counter() - prep_start
    fits = [native_fit(data, stats, rows, device, seed, out) for seed in [42, 43, 44]]
    controls = run_controls(db, rows, stypes, cfg, out)
    mean_mae = sum(f['validation']['mae'] for f in fits) / 3
    competence = (mean_mae < controls['median']['validation']['mae'] and
                  mean_mae <= controls['raw_gbdt']['validation']['mae'])
    write_json(out / 'STAGE1_SUMMARY.json', {
        'native_fits': fits, 'controls': controls, 'graph_mean_validation_mae': mean_mae,
        'competence_passed': competence, 'competence_rule': profile['competence_rule'],
        'graph_preparation_seconds': prep_seconds, 'total_seconds': time.perf_counter() - started,
        'process_max_rss_platform_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'test_evaluations': 0, 'test_predictions': 0, 'native_updates_total': 450,
        'user_default_members': 4, 'initializer_ensemble_decisions': 'unadopted'})
    print(f'Stage 1 complete; competence_passed={competence}; outputs={out}', flush=True)


if __name__ == '__main__':
    main()
