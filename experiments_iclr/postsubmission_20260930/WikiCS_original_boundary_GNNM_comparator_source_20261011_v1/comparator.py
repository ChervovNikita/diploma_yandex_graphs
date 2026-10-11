"""Inactive original-boundary TABMModel WikiCS comparator, separate roster.

Native multiclass train_step_tabm, global sequential dropout, raw-logit fusion.
No numerical imports until an explicit new root amendment/admission is supplied.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (7301, 7403, 7507)
ARM = 'original_boundary_shared4'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def file_receipt(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def checked(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}
            and file_receipt(item['path']) == item, 'Exact original/native/input/amendment receipt')
    return Path(item['path']).resolve()


def write(path, value):
    require(not path.exists(), 'Fresh record; no overwrite or retry')
    temp = path.with_suffix(path.suffix + '.tmp'); require(not temp.exists(), 'Retain partial write')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def global_rng(t, np, device):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]),
                cpu=t.get_rng_state(), cuda=t.cuda.get_rng_state(device))


def restore_rng(t, np, device, value):
    state = value['numpy']; random.setstate(value['python'])
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    t.set_rng_state(value['cpu'].cpu()); t.cuda.set_rng_state(value['cuda'].cpu(), device)


def fingerprint(t, value):
    h = hashlib.sha256()
    def visit(v):
        if t.is_tensor(v):
            a = v.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str): h.update(str(key).encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v: visit(item)
        else: h.update(repr(v).encode())
    visit(value); return h.hexdigest()


def acquire_comparator(spec, root_guard):
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Correct host/repository before project metadata')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID],
            'Sole authorized GPU allocation')
    require(callable(root_guard), 'New explicit root comparator amendment/admission callback')
    here = Path(__file__).resolve().parent
    source_paths = [here / name for name in ('comparator.py', 'PLAN.json', 'SOURCE_BINDINGS.json')]
    proof = root_guard(copy.deepcopy(spec), action='WikiCS_original_boundary_GNNM_comparator', source_paths=[str(p) for p in source_paths])
    require(isinstance(proof, dict) and proof.get('admitted') is True
            and proof.get('action') == 'WikiCS_original_boundary_GNNM_comparator'
            and proof.get('source_sha256') == {str(p): sha(p) for p in source_paths}
            and proof.get('future_roster_amendment_verified') is True and proof.get('native_runtime_and_full_inputs_verified') is True
            and proof.get('finite_external_owner_bound') is True and proof.get('TEST_access') is False
            and proof.get('current_Physics_Penn_access') is False
            and type(proof.get('external_owner_wall_seconds')) is int and proof['external_owner_wall_seconds'] > 0,
            'Actual separate amendment/source/runtime/full-input/finite-owner admission')
    checked(spec['roster_amendment'])  # Root decides actual future roster/reader/gates.
    limits = spec['limits']
    require(set(limits) == {'wall_seconds', 'cuda_bytes', 'rss_bytes'} and all(type(v) is int and v > 0 for v in limits.values())
            and proof.get('limits') == limits, 'Root-chosen positive finite limits, no default admission')
    bindings = json.loads((here / 'SOURCE_BINDINGS.json').read_text())
    require(set(spec['source_receipts']) == set(bindings['runtime_sources']), 'Exact saved native source set')
    for name, item in spec['source_receipts'].items():
        checked(item); fixed = bindings['runtime_sources'][name]
        require(item['bytes'] == fixed['bytes'] and item['sha256'] == fixed['sha256'], 'Original native source unchanged')
    require(set(spec['roles']) == {'train', 'valid'}, 'Same WikiCS TRAIN/VALID only')
    for name, item in spec['roles'].items():
        checked(item); fixed = bindings['WikiCS_roles'][name]
        require(item['bytes'] == fixed['bytes'] and item['sha256'] == fixed['sha256'], 'Original E paired WikiCS masks/public graph')
    output = Path(spec['output']).resolve(); require(not output.exists() and output.is_relative_to(R), 'Fresh separate comparator output')
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Dedicated externally bounded process')
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame): raise TimeoutError('Finite comparator owner cap; no retry or partial continuation')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(limits['wall_seconds']); output.mkdir()
    started = time.monotonic(); records = [dict(seed=s, arm=ARM, status='unstarted_retained') for s in SEEDS]
    peak_cuda = 0
    try:
        import numpy as np
        import torch as t
        require(t.get_default_dtype() == t.float32, 'Original float32 native constructor')
        device = t.device('cuda:0'); t.cuda.set_device(device); t.set_num_threads(2)
        models = load('WikiCS_original_TABM_native_models', checked(spec['source_receipts']['native_models']))
        common = load('WikiCS_original_TABM_native_common', checked(spec['source_receipts']['native_common']))
        original = load('WikiCS_original_TABM_train_step', checked(spec['source_receipts']['native_tabm_training']))
        original.F, original.TABM_INITS = t.nn.functional, 4
        def guard():
            nonlocal peak_cuda
            peak_cuda = max(peak_cuda, t.cuda.max_memory_allocated(device))
            require(time.monotonic() - started < limits['wall_seconds'] and peak_cuda <= limits['cuda_bytes']
                    and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits['rss_bytes'], 'Finite actual whole action resources')
        with np.load(checked(spec['roles']['train']), allow_pickle=False) as a:
            require(set(a.files) == {'x', 'edge_index', 'ids', 'y'}, 'Safe full graph/TRAIN schema')
            x, edges, train_ids, train_y = (a[k].copy() for k in ('x', 'edge_index', 'ids', 'y'))
        with np.load(checked(spec['roles']['valid']), allow_pickle=False) as a:
            require(set(a.files) == {'ids', 'y'}, 'Separate complete original VALID role')
            valid_ids, valid_y = a['ids'].copy(), a['y'].copy()
        require(x.dtype == np.float32 and x.shape == (11701, 300) and np.isfinite(x).all()
                and edges.dtype == np.int64 and edges.shape == (2, 442907) and edges.min() >= 0 and edges.max() < 11701
                and int((edges[0] == edges[1]).sum()) == 11701, 'Original full factual WikiCS graph')
        require(all(a.dtype == np.int64 for a in (train_ids, train_y, valid_ids, valid_y))
                and train_ids.shape == train_y.shape == (580,) and valid_ids.shape == valid_y.shape == (5274,)
                and len(np.unique(train_ids)) == 580 and len(np.unique(valid_ids)) == 5274
                and not np.intersect1d(train_ids, valid_ids).size
                and train_ids.min() >= 0 and valid_ids.min() >= 0 and train_ids.max() < 11701 and valid_ids.max() < 11701
                and np.isin(train_y, np.arange(10)).all() and np.isin(valid_y, np.arange(10)).all(), 'Original ten-class proper roles')
        tensor = lambda a: t.from_numpy(a).to(device)
        ids, vid, vy = tensor(train_ids), tensor(valid_ids), tensor(valid_y)
        labels = t.zeros(11701, dtype=t.long, device=device); labels[ids] = tensor(train_y)
        mask = t.zeros(11701, dtype=t.bool, device=device); mask[ids] = True
        data = SimpleNamespace(x=tensor(x), edge_index=tensor(edges), y=labels, train_mask=mask)
        del x, edges, train_y
        def logits(model):
            value = t.stack([model(data, data.x, tabm_seed=m) for m in range(4)])
            require(value.shape == (4, 11701, 10) and value.dtype == t.float32 and bool(t.isfinite(value).all()), 'Four complete native multiclass routes')
            return value
        for record in records:
            guard(); record['status'] = 'running'; seed = record['seed']; folder = output / (ARM + '_seed' + str(seed)); folder.mkdir()
            common.set_seed(seed)  # One paired WikiCS base seed before the original constructor.
            model = models.TABMModel('SAGE', 2, 300, 128, 10, 1, 8, 'LayerNorm', .2, tabm_inits=4, device=device).to(device)
            require(model.input_be_block.W.bias is None and model.output_be_block.W.bias is None
                    and bool(((model.input_be_block.R == -1) | (model.input_be_block.R == 1)).all())
                    and bool((model.output_be_block.R == 1).all())
                    and all(bool((b.S == 1).all()) and bool((b.B == 0).all()) for b in (model.input_be_block, model.output_be_block)),
                    'Native boundary Rademacher/Xavier/no-W-bias/private-B initializer law')
            optimizer = t.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            best, stale, selected, saves = -math.inf, 0, 0, 0
            checkpoint = folder / 'selected.pt'; versions = data.x._version, data.edge_index._version
            with (folder / 'trace.jsonl').open('x') as trace:
                for step in range(1, 5001):
                    guard(); loss = original.train_step_tabm(model, data, optimizer, False)
                    require(math.isfinite(loss) and versions == (data.x._version, data.edge_index._version), 'Finite original multiclass TRAIN/no graph mutation')
                    model.eval()
                    with t.no_grad(): raw = logits(model)[:, vid]; decision = raw.mean(0).argmax(-1)
                    correct = int(decision.eq(vy).sum()); accuracy = correct / 5274
                    improving = accuracy > best
                    stale = 0 if improving else stale + 1
                    saved = improving and step % 10 == 0
                    if saved:
                        best, selected = accuracy, step
                        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), global_rng=global_rng(t, np, device),
                                     seed=seed, arm=ARM, selected_step=step, selection_correct=correct, selection_nodes=5274,
                                     model_sha256=fingerprint(t, model.state_dict()), optimizer_sha256=fingerprint(t, optimizer.state_dict()),
                                     source_sha256=proof['source_sha256'], roster_amendment=spec['roster_amendment'])
                        t.save(state, checkpoint); saves += 1
                    trace.write(json.dumps(dict(update=step, TRAIN_own_CE=loss, raw_logit_VALID_correct=correct,
                         strict_above_saved_best=improving, divisible_by_ten=step % 10 == 0, checkpoint_written=saved,
                         selected_step=selected, stale=stale), allow_nan=False) + '\n'); trace.flush()
                    del raw, decision
                    if stale >= 300: break
            require(selected > 0, 'Original per-ten cadence must select a checkpoint')
            state = t.load(checkpoint, map_location='cpu', weights_only=True)
            require(state['seed'] == seed and state['arm'] == ARM and state['source_sha256'] == proof['source_sha256']
                    and state['roster_amendment'] == spec['roster_amendment'], 'Strict selected comparator identity/custody')
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
            restore_rng(t, np, device, state['global_rng'])
            require(fingerprint(t, model.state_dict()) == state['model_sha256'] and fingerprint(t, optimizer.state_dict()) == state['optimizer_sha256']
                    and fingerprint(t, global_rng(t, np, device)) == fingerprint(t, state['global_rng']), 'Exact model/optimizer/global RNG restoration')
            model.eval()
            with t.no_grad():
                raw = logits(model)[:, vid]; pool = raw.mean(0); served = pool.argmax(-1)
                archive = folder / 'selected_VALID.npz'
                np.savez(archive, ids=valid_ids, y=valid_y, raw_logits=raw.cpu().numpy(), raw_logit_mean=pool.cpu().numpy(),
                         raw_logit_mean_probabilities=pool.softmax(-1).cpu().numpy(), raw_logit_decision=served.cpu().numpy(),
                         member_decision=raw.argmax(-1).cpu().numpy())
            guard(); record.update(status='selected_complete', optimizer_fits=1, selected_states=1, members=4,
                  selected_step=selected, completed_updates=step, checkpoint_writes=saves,
                  selected_state_receipt=file_receipt(checkpoint), selected_VALID_archive=file_receipt(archive),
                  selector_and_serving='raw_logit_mean_argmax', parameters=sum(p.numel() for p in model.parameters()),
                  TRAIN_native_routes=4 * step, VALID_selector_native_routes=4 * step, selected_native_routes=4,
                  backward_calls=4 * step, optimizer_steps=step, dropout='original_global_sequential')
            write(folder / 'RESULT.json', record)
            del model, optimizer, state, raw, pool, served
            t.cuda.empty_cache()
        require(len(records) == 3 and all(r['status'] == 'selected_complete' for r in records), 'Whole separately amended comparator panel')
        for item in (*spec['source_receipts'].values(), *spec['roles'].values(), spec['roster_amendment']): checked(item)
        for path, digest in proof['source_sha256'].items(): require(sha(path) == digest, 'Source/plan/bindings unchanged')
        guard()
        result = dict(complete=True, arm=ARM, banks=3, optimizer_fits=3, selected_states=3, selected_native_routes=12, records=records,
                      same_WikiCS_graph_roles_seeds=True, architecture_and_recipe_port_not_isolated_placement=True,
                      historical_scores_rerun_or_replaced=False, seven_arm_21_39_105_roster_modified=False,
                      reader_adaptation_performed=False, automatic_roster_merge_or_followup=False, TEST_access=False,
                      current_Physics_Penn_access=False, admission=proof, seconds=time.monotonic() - started,
                      peak_CUDA_allocated_bytes=peak_cuda, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write(output / 'COMPLETE_ORIGINAL_BOUNDARY_COMPARATOR.json', result); return result
    except BaseException as error:
        for record in records:
            if record['status'] == 'running': record['status'] = 'failed_retained'
        write(output / 'FAILURE.json', dict(complete=False, records=records, type=type(error).__name__, message=str(error),
               traceback=traceback.format_exc(), retry=False, survivor_mean=False, automatic_continuation=False))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    raise SystemExit('Inactive separate adapter; root must amend future roster/reader and explicitly admit a finite owner.')
