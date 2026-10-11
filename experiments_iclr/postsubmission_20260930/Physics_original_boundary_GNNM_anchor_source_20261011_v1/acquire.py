"""Source-only original GNNM Physics anchor. Root owns admission and dispatch.

Reuse the pinned original comparator helpers and native multiclass training.
No numerical imports, input decoding or execution occurs on module import.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
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
BLOCKS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
ARM = 'original_boundary_shared4'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def full_input_readiness(spec, root_guard):
    """Exactly one fresh full-input TRAIN update, save and strict restore."""
    return _run(spec, root_guard, training_only=True)


def acquire(spec, root_guard):
    """One fresh complete ten-bank panel; TRAIN/SELECT only, no readout join."""
    return _run(spec, root_guard, training_only=False)


def _run(spec, root_guard, training_only):
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R,
            'Correct host/repository before project metadata')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=15).splitlines() == [UUID],
            'Exact sole authorized GPU')
    action = 'Physics_original_boundary_readiness' if training_only else 'Physics_original_boundary_acquisition'
    here = Path(__file__).resolve().parent
    own = [here / name for name in ('acquire.py', 'PLAN.json', 'INTERFACE.json', 'SOURCE_BINDINGS.json')]
    require(callable(root_guard), 'Root supplies new decision and real finite external owner')
    proof = root_guard(copy.deepcopy(spec), action=action, source_paths=[str(p) for p in own])
    require(isinstance(proof, dict) and proof.get('admitted') is True and proof.get('action') == action
            and proof.get('source_sha256') == {str(p): sha(p) for p in own}
            and proof.get('native_runtime_and_full_inputs_verified') is True
            and proof.get('finite_external_owner_bound') is True
            and proof.get('CAL_TEST_access') is False
            and type(proof.get('external_owner_wall_seconds')) is int
            and proof['external_owner_wall_seconds'] > 0,
            'Actual new exact-source/runtime/input/finite-owner admission')
    limits = spec['limits']
    require(set(limits) == {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
            and all(type(v) is int and v > 0 for v in limits.values())
            and proof.get('limits') == limits
            and proof['external_owner_wall_seconds'] >= limits['wall_seconds'],
            'Root chooses positive finite limits; no source budgets')
    expected_keys = {'output', 'limits', 'decision', 'source_receipts', 'roles'}
    if not training_only:
        expected_keys.add('readiness')
    require(set(spec) == expected_keys, 'Simple fixed action spec; no candidate or CAL/TEST roles')
    bindings = json.loads((here / 'SOURCE_BINDINGS.json').read_text())
    require(set(spec['source_receipts']) == set(bindings['runtime_sources']), 'Exact reused native/helper sources')
    for name, item in spec['source_receipts'].items():
        fixed = bindings['runtime_sources'][name]
        require(set(item) == {'path', 'bytes', 'sha256'} and item['bytes'] == fixed['bytes']
                and item['sha256'] == fixed['sha256'] and sha(item['path']) == item['sha256']
                and Path(item['path']).stat().st_size == item['bytes'], 'Reused source bytes differ')
    helper_path = spec['source_receipts']['original_comparator']['path']
    module_spec = importlib.util.spec_from_file_location('Physics_original_comparator_helpers', helper_path)
    h = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = h
    module_spec.loader.exec_module(h)  # Pinned comparator is stdlib-only on import.
    h.checked(spec['decision'])
    require(proof.get('decision_receipt') == spec['decision'], 'Actual root scientific decision bound')
    require(spec['roles'] == bindings['Physics_roles'], 'All ten existing paired TRAIN/SELECT receipt bytes')
    # Readiness verifies/decodes only block8101 TRAIN, never SELECT payloads.
    for block in ((BLOCKS[0],) if training_only else BLOCKS):
        for role in (('train',) if training_only else ('train', 'select')):
            h.checked(spec['roles'][str(block)][role])
    if not training_only:
        require(set(spec['readiness']) == {'result', 'owner_end'}, 'One completed actual check and direct owner receipt')
        ready = json.loads(h.checked(spec['readiness']['result']).read_text())
        owner = json.loads(h.checked(spec['readiness']['owner_end']).read_text())
        require(proof.get('readiness_verified_by_root') is True and ready.get('complete') is True
                and ready.get('training_updates') == 1 and ready.get('block') == BLOCKS[0]
                and ready.get('source_sha256') == proof['source_sha256']
                and ready.get('TRAIN_role_receipt') == spec['roles'][str(BLOCKS[0])]['train']
                and ready.get('restoration_exact') is True and ready.get('SELECT_array_read') is False
                and ready.get('CAL_TEST_access') is False
                and owner.get('status') == 'COMPLETE_EXIT_0' and owner.get('exit_code') == 0
                and owner.get('direct_child_wait') is True and owner.get('child_pid_absent') is True,
                'Root verifies same-source full-input check and actual external exit/reap before acquisition')
    output = Path(spec['output']).resolve()
    require(output.is_relative_to(R) and not output.exists(), 'One fresh authorized output; no overwrite/retry')
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Dedicated externally bounded child')
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise TimeoutError('Finite original GNNM action cap; retain all partial/unstarted records')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limits['wall_seconds'])
    output.mkdir()
    started = time.monotonic()
    records = [dict(block=b, arm=ARM, status='unstarted_retained')
               for b in ((BLOCKS[0],) if training_only else BLOCKS)]
    peak_cuda = 0
    try:
        import numpy as np
        import torch as t
        import inspect
        from torch_geometric.nn.conv import MessagePassing
        require(t.get_default_dtype() == t.float32, 'Original float32 constructor')
        device = t.device('cuda:0')
        t.cuda.set_device(device)
        t.set_num_threads(2)
        t.backends.cuda.matmul.allow_tf32 = False
        t.set_float32_matmul_precision('highest')
        t.cuda.reset_peak_memory_stats(device)
        models = h.load('Physics_original_TABM_native_models', h.checked(spec['source_receipts']['native_models']))
        common = h.load('Physics_original_TABM_native_common', h.checked(spec['source_receipts']['native_common']))
        original = h.load('Physics_original_TABM_train_step', h.checked(spec['source_receipts']['native_tabm_training']))
        require(Path(inspect.getfile(models.SAGEConv)).resolve() == h.checked(spec['source_receipts']['native_sage_conv'])
                and Path(inspect.getfile(MessagePassing)).resolve() == h.checked(spec['source_receipts']['native_message_passing']),
                'Existing Physics native graph runtime files')
        original.F, original.TABM_INITS = t.nn.functional, 4
        def guard():
            nonlocal peak_cuda
            peak_cuda = max(peak_cuda, t.cuda.max_memory_allocated(device))
            require(time.monotonic() - started < limits['wall_seconds'] and peak_cuda <= limits['cuda_bytes']
                    and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits['rss_bytes'],
                    'Actual finite wall/CUDA/RSS limits')
        with np.load(h.checked(spec['roles'][str(BLOCKS[0])]['train']), allow_pickle=False) as a:
            require(set(a.files) == {'x', 'edge_index', 'ids', 'y'}, 'Existing safe full-graph TRAIN schema')
            x, edges = a['x'].copy(), a['edge_index'].copy()
        require(x.dtype == np.float32 and x.shape == (34493, 8415) and np.isfinite(x).all()
                and edges.dtype == np.int64 and edges.shape == (2, 530417)
                and edges.min() >= 0 and edges.max() < 34493
                and int((edges[0] == edges[1]).sum()) == 34493,
                'Complete raw Physics features/support; no normalization or reconstruction')
        tensor = lambda a: t.from_numpy(a).to(device)
        data = SimpleNamespace(x=tensor(x), edge_index=tensor(edges),
                               y=t.zeros(34493, dtype=t.long, device=device),
                               train_mask=t.zeros(34493, dtype=t.bool, device=device))
        del x, edges
        versions = data.x._version, data.edge_index._version
        guard()
        custody = dict(source_sha256=proof['source_sha256'], decision_receipt=spec['decision'],
                       runtime_sources=spec['source_receipts'], role_receipts=spec['roles'])
        def checkpoint(model, optimizer, block, step, path, correct=None):
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         global_rng=h.global_rng(t, np, device), block=block, arm=ARM,
                         selected_step=step, SELECT_correct=correct, custody=custody,
                         model_sha256=h.fingerprint(t, model.state_dict()),
                         optimizer_sha256=h.fingerprint(t, optimizer.state_dict()))
            t.save(state, path)
            return state
        def restore(model, optimizer, block, path):
            state = t.load(path, map_location='cpu', weights_only=True)
            require(state['block'] == block and state['arm'] == ARM and state['custody'] == custody,
                    'Strict original state/source/input/decision identity')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            h.restore_rng(t, np, device, state['global_rng'])
            require(h.fingerprint(t, model.state_dict()) == state['model_sha256']
                    and h.fingerprint(t, optimizer.state_dict()) == state['optimizer_sha256']
                    and h.fingerprint(t, h.global_rng(t, np, device)) == h.fingerprint(t, state['global_rng']),
                    'Exact complete model/optimizer/global Python/NumPy/CPU/CUDA RNG restoration')
            return state
        def logits(model):
            value = t.stack([model(data, data.x, tabm_seed=m) for m in range(4)])
            require(value.shape == (4, 34493, 5) and value.dtype == t.float32 and bool(t.isfinite(value).all()),
                    'Four complete native multiclass routes')
            return value
        for record in records:
            guard()
            record['status'] = 'running'
            block = record['block']
            folder = output / (ARM + '_block' + str(block))
            folder.mkdir()
            with np.load(h.checked(spec['roles'][str(block)]['train']), allow_pickle=False) as a:
                require(set(a.files) == {'x', 'edge_index', 'ids', 'y'}, 'Safe per-block TRAIN only')
                train_ids, train_y = a['ids'].copy(), a['y'].copy()
            require(train_ids.dtype == train_y.dtype == np.int64 and train_ids.shape == train_y.shape == (100,)
                    and len(np.unique(train_ids)) == 100 and train_ids.min() >= 0 and train_ids.max() < 34493
                    and np.isin(train_y, np.arange(5)).all()
                    and bool((np.bincount(train_y, minlength=5) == 20).all()), 'Exact 100 proper TRAIN rows')
            data.y.zero_()
            data.train_mask.zero_()
            ids = tensor(train_ids)
            data.y[ids] = tensor(train_y)
            data.train_mask[ids] = True
            if not training_only:
                with np.load(h.checked(spec['roles'][str(block)]['select']), allow_pickle=False) as a:
                    require(set(a.files) == {'ids', 'y'}, 'Separate SELECT-only schema')
                    select_ids, select_y = a['ids'].copy(), a['y'].copy()
                require(select_ids.dtype == select_y.dtype == np.int64
                        and select_ids.shape == select_y.shape == (150,) and len(np.unique(select_ids)) == 150
                        and select_ids.min() >= 0 and select_ids.max() < 34493
                        and not np.intersect1d(train_ids, select_ids).size
                        and np.isin(select_y, np.arange(5)).all()
                        and bool((np.bincount(select_y, minlength=5) == 30).all()), 'Exact separate 150 SELECT rows')
                sid, sy = tensor(select_ids), tensor(select_y)
            common.set_seed(block)  # One original native paired seed; no route scopes or block offsets.
            model = models.TABMModel('SAGE', 2, 8415, 128, 5, 1, 8, 'LayerNorm', .2,
                                     tabm_inits=4, device=device).to(device)
            require(model.input_be_block.W.bias is None and model.output_be_block.W.bias is None
                    and bool(((model.input_be_block.R == -1) | (model.input_be_block.R == 1)).all())
                    and bool((model.output_be_block.R == 1).all())
                    and all(bool((b.S == 1).all()) and bool((b.B == 0).all())
                            for b in (model.input_be_block, model.output_be_block)),
                    'Original boundary initialization and native constructor order')
            optimizer = t.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            fit_started = time.monotonic()
            state_path = folder / ('readiness.pt' if training_only else 'selected.pt')
            if training_only:
                loss = original.train_step_tabm(model, data, optimizer, False)
                require(math.isfinite(loss) and versions == (data.x._version, data.edge_index._version),
                        'One finite full-input original own-CE update with unchanged graph')
                checkpoint(model, optimizer, block, 1, state_path)
                state = restore(model, optimizer, block, state_path)
                guard()
                result = dict(complete=True, block=block, arm=ARM, training_updates=1,
                              backward_calls=4, TRAIN_native_routes=4, optimizer_steps=1,
                              restoration_exact=True, SELECT_array_read=False, CAL_TEST_access=False,
                              new_fit_selection_or_quality_claim=False, source_sha256=proof['source_sha256'],
                              TRAIN_role_receipt=spec['roles'][str(block)]['train'],
                              state_receipt=h.file_receipt(state_path), decision_receipt=spec['decision'],
                              parameters=sum(p.numel() for p in model.parameters()),
                              seconds=time.monotonic() - started, peak_CUDA_allocated_bytes=peak_cuda,
                              peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                h.write(output / 'FULL_INPUT_READINESS_COMPLETE.json', result)
                return result
            best, stale, selected, saves = -math.inf, 0, 0, 0
            with (folder / 'trace.jsonl').open('x') as trace:
                for step in range(1, 5001):
                    guard()
                    loss = original.train_step_tabm(model, data, optimizer, False)
                    require(math.isfinite(loss) and versions == (data.x._version, data.edge_index._version),
                            'Finite original TRAIN update and no full-graph mutation')
                    model.eval()
                    with t.no_grad():
                        raw = logits(model)[:, sid]
                        decision = raw.mean(0).argmax(-1)
                    correct = int(decision.eq(sy).sum())
                    accuracy = correct / 150
                    improving = accuracy > best
                    stale = 0 if improving else stale + 1
                    saved = improving and step % 10 == 0
                    if saved:
                        best, selected = accuracy, step
                        checkpoint(model, optimizer, block, step, state_path, correct)
                        saves += 1
                    trace.write(json.dumps(dict(update=step, TRAIN_own_CE=loss,
                               raw_logit_SELECT_correct=correct, strict_above_saved_best=improving,
                               divisible_by_ten=step % 10 == 0, checkpoint_written=saved,
                               selected_step=selected, stale=stale), allow_nan=False) + '\n')
                    trace.flush()
                    del raw, decision
                    if stale >= 300:
                        break
            require(selected > 0, 'Original per-ten selector must save a complete checkpoint')
            state = restore(model, optimizer, block, state_path)
            model.eval()
            with t.no_grad():
                raw = logits(model)[:, sid]
                pool = raw.mean(0)
                served = pool.argmax(-1)
                archive = folder / 'selected_SELECT.npz'
                np.savez(archive, ids=select_ids, y=select_y, raw_logits=raw.cpu().numpy(),
                         raw_logit_mean=pool.cpu().numpy(), raw_logit_mean_probabilities=pool.softmax(-1).cpu().numpy(),
                         raw_logit_decision=served.cpu().numpy(), member_decision=raw.argmax(-1).cpu().numpy())
            guard()
            record.update(status='selected_complete', optimizer_fits=1, selected_states=1, members=4,
                          selected_step=selected, completed_updates=step, checkpoint_writes=saves,
                          selected_state_receipt=h.file_receipt(state_path), selected_SELECT_archive=h.file_receipt(archive),
                          selected_SELECT_correct=int(served.eq(sy).sum()), SELECT_nodes=150,
                          selector_and_serving='raw_logit_mean_argmax', dropout='original_global_sequential',
                          parameters=sum(p.numel() for p in model.parameters()),
                          TRAIN_native_routes=4 * step, SELECT_selector_native_routes=4 * step,
                          selected_native_routes=4, backward_calls=4 * step, optimizer_steps=step,
                          acquisition_restore_export_seconds=time.monotonic() - fit_started)
            h.write(folder / 'RESULT.json', record)
            del model, optimizer, state, raw, pool, served, sid, sy
            t.cuda.empty_cache()
        require(len(records) == 10 and all(r['status'] == 'selected_complete' for r in records),
                'Whole ten-bank/ten-fit/four-route original panel; no subset or retry')
        for item in spec['source_receipts'].values():
            h.checked(item)
        for block in BLOCKS:
            for item in spec['roles'][str(block)].values():
                h.checked(item)
        h.checked(spec['decision'])
        for path, digest in proof['source_sha256'].items():
            require(sha(path) == digest, 'Own source and plan unchanged')
        guard()
        result = dict(complete=True, arm=ARM, banks=10, optimizer_fits=10, selected_states=10,
                      selected_native_routes=40, blocks=list(BLOCKS), records=records,
                      encountered_graph_exploration=True, closed_E_confirmation_amended=False,
                      architecture_and_recipe_port_not_isolated_placement=True,
                      CAL_TEST_access=False, calibration_or_TEST_readout=False,
                      existing_logs_opened_or_paired_comparison_performed=False,
                      SELECT_interpretation_only_after_whole_closure=True, automatic_followup=False,
                      admission=proof, seconds=time.monotonic() - started,
                      peak_CUDA_allocated_bytes=peak_cuda,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        h.write(output / 'COMPLETE_ORIGINAL_PHYSICS_ANCHOR.json', result)
        return result
    except BaseException as error:
        for record in records:
            if record['status'] == 'running':
                record['status'] = 'failed_retained'
        h.write(output / 'FAILURE.json', dict(complete=False, records=records,
                type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                retry=False, survivor_mean=False, automatic_continuation=False, CAL_TEST_access=False))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    raise SystemExit('Source-only entries; root supplies exact admission and an existing finite-owner dispatch.')
