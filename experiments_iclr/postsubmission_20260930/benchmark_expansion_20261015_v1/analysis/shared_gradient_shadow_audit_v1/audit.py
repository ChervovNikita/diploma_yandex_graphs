"""Three disposable native TRAIN epochs; no fitted-model or heldout export."""
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (101, 202, 303)
BOUNDS = {'construction': 1, 'selected_load': 1, 'epoch': 1,
          'optimizer': 38, 'native_reverse': 38, 'route_reverse': 152,
          'native_forward': 152, 'B_forward': 152, 'D_forward': 152,
          'O_forward': 4}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path outside the allocation project: ' + str(path))
    return path


def verify(pin, root):
    path = inside(pin['path'], root)
    actual = receipt(path)
    if any(actual[k] != pin[k] for k in ('sha256', 'bytes') if k in pin):
        raise ValueError('Saved source/evidence differs: ' + str(path))
    return path


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def allocation():
    if socket.gethostname() != 'anogena-2-0':
        raise RuntimeError('Authorized allocation required before project reads')
    found = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                    text=True, timeout=10).splitlines()
    if found != [UUID]:
        raise RuntimeError('Wrong allocation GPU route')


def partition(named, shared_named, private, optimizer_parameters):
    """Identity partition; deliberately usable in stdlib source checks."""
    all_ids = [id(p) for _, p in named]
    shared_ids = [id(p) for _, p in shared_named]
    private_ids = [id(p) for p in private]
    owners = [id(p) for p in optimizer_parameters]
    if any(len(v) != len(set(v)) for v in (all_ids, shared_ids, private_ids, owners)):
        raise ValueError('Aliased/duplicate parameter or optimizer owner')
    if set(shared_ids) & set(private_ids) or set(shared_ids) | set(private_ids) != set(all_ids):
        raise ValueError('Shared/private partition is not disjoint and exhaustive')
    if set(owners) != set(all_ids):
        raise ValueError('Native optimizer must own every parameter exactly once')
    return ([i for i, (_, p) in enumerate(named) if id(p) in set(shared_ids)],
            [i for i, (_, p) in enumerate(named) if id(p) in set(private_ids)])


class Ledger:
    def __init__(self, event_path=None):
        self.counts = {key: {'attempts': 0, 'completed': 0} for key in BOUNDS}
        self.event_path = event_path

    def emit(self, event):
        if self.event_path:
            with Path(self.event_path).open('a') as stream:
                stream.write(json.dumps({'unix': time.time(), **event}, allow_nan=False) + '\n')

    def start(self, key):
        row = self.counts[key]
        if row['attempts'] >= BOUNDS[key]:
            raise RuntimeError('Frozen operation bound exhausted: ' + key)
        row['attempts'] += 1
        self.emit({'operation': key, 'event': 'attempt', **row})

    def finish(self, key):
        row = self.counts[key]
        if row['completed'] >= row['attempts']:
            raise RuntimeError('Completion without attempted operation: ' + key)
        row['completed'] += 1
        self.emit({'operation': key, 'event': 'complete', **row})

    def complete(self):
        return all(self.counts[k] == {'attempts': n, 'completed': n} for k, n in BOUNDS.items())


def gate(rows):
    if len(rows) != 152 or sorted((r['batch'], r['member']) for r in rows) != [
            (b, m) for b in range(1, 39) for m in range(4)]:
        raise ValueError('Gate requires all 38 batches and four routes')
    count = sum(r['predicted_joint_change'] > r['tau']
                and r['finite_shared_after_private_change'] > r['tau']
                and r['finite_joint_change'] > r['tau'] for r in rows)
    compensated = sum(r['predicted_shared_change'] > r['tau']
                      and r['predicted_joint_change'] <= r['tau'] for r in rows)
    return {'qualifying_harm_events': count, 'required_harm_events': 16,
            'seed_gate_open': count >= 16, 'compensated_shared_ascent_events': compensated,
            'route_batch_denominator': 152, 'independent_seed_n': 1}


def tree_hash(torch, value):
    def identity(v):
        if torch.is_tensor(v):
            x = v.detach().cpu().contiguous()
            return {'shape': list(x.shape), 'dtype': str(x.dtype),
                    'hash': hashlib.sha256(x.numpy().tobytes()).hexdigest()}
        if type(v).__module__.startswith('numpy') and hasattr(v, 'tobytes'):
            return {'shape': list(v.shape), 'dtype': str(v.dtype),
                    'hash': hashlib.sha256(v.tobytes()).hexdigest()}
        if isinstance(v, dict):
            return {str(k): identity(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [identity(x) for x in v]
        if hasattr(v, 'item'):
            return v.item()
        return v
    return digest(identity(value))


def objects(model):
    return [(kind, name, id(x), x.data_ptr(), str(x.device), str(x.dtype), list(x.shape))
            for kind, iterator in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
            for name, x in iterator]


class NativeAudit:
    """Observe the original session.train_update and original adapter.train_epoch."""
    def __init__(self, torch, common, adapter, ledger, output):
        self.torch, self.common, self.adapter = torch, common, adapter
        self.session, self.model = adapter.session, adapter.model
        self.ledger, self.output, self.frame = ledger, output, None
        self.phase, self.rows, self.batch_records = None, [], []
        self.named = list(self.model.named_parameters())
        self.params = tuple(p for _, p in self.named)
        owners = [p for group in self.session.optimizer.param_groups for p in group['params']]
        self.shared, self.private = partition(self.named, list(self.session.source_parameter_references.items()),
                                               self.model.private_parameters(), owners)
        if not self.shared or not self.private or any(not p.requires_grad for p in self.params):
            raise ValueError('Live native and private trainable blocks required')
        if any(self.params[i].ndim < 1 or self.params[i].shape[0] != 4 for i in self.private):
            raise ValueError('Original four-row private boundary banks required')
        self.identity = objects(self.model)
        self.buffers = tuple(self.model.buffers())
        self.original_next = self.session._next_train_batch
        self.original_member = self.model.run_member
        self.original_step = self.session.optimizer.step
        self.original_backward = torch.autograd.backward
        self.original_validate = self.adapter.validate
        self.original_session_validate = self.session.validate
        self.original_diagnostic = self.adapter.train_diagnostic
        self.original_valid_loader = self.session.valid_loader
        self.hooks = []

    def clone(self, tensors):
        return tuple(x.detach().clone() for x in tensors)

    def put(self, tensors, values):
        with self.torch.no_grad():
            for x, value in zip(tensors, values):
                x.copy_(value)

    def loader_state(self):
        return {'train_generator': self.adapter.train_generator.get_state().clone(),
                'valid_generator': self.session.valid_generator.get_state().clone(),
                'original_sampler': self.session.sampler.state()}

    def next_batch(self):
        if self.frame is not None:
            raise RuntimeError('Prior native update did not finish')
        batch, positions, skipped = self.original_next()
        batch_number = len(self.batch_records) + 1
        if batch_number > 38 or skipped:
            raise RuntimeError('Exact 38 eligible native TRAIN batches with no skips required')
        self.phase = 'native'
        self.frame = {'batch': batch, 'positions': positions.detach().cpu().clone(),
                      'number': batch_number, 'routes': [],
                      'old': self.clone(self.params), 'pre_buffers': self.clone(self.buffers),
                      'pre_rng': self.common.global_rng(self.torch),
                      'update_before': self.session.updates}
        return batch, positions, skipped

    def member(self, member, batch):
        frame = self.frame
        if self.phase != 'native' or frame is None or batch is not frame['batch'] or member != len(frame['routes']):
            raise RuntimeError('Original sequential native route calls required')
        result = self.original_member(member, batch)
        frame['routes'].append(result)
        if member == 3:
            losses = [self.session._native_loss(route[0], batch) for route in frame['routes']]
            frame['losses'] = [float(loss.detach()) for loss in losses]
            if not all(math.isfinite(v) for v in frame['losses']):
                raise FloatingPointError('Nonfinite original own TRAIN losses')
            gradients = []
            for route, loss in enumerate(losses):
                self.ledger.start('route_reverse')
                grad = self.torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=False)
                self.ledger.finish('route_reverse')
                if any(not self.torch.isfinite(g).all().item() for g in grad):
                    raise FloatingPointError('Nonfinite own supervised route gradient')
                for i in self.private:
                    others = [j for j in range(4) if j != route]
                    if self.torch.count_nonzero(grad[i][others]).item():
                        raise ValueError('Own prediction depends on another route private row')
                gradients.append(tuple(g.detach() for g in grad))
            frame['gradients'] = gradients
        return result

    def backward(self, *args, **kwargs):
        if self.phase != 'native' or self.frame is None or 'gradients' not in self.frame:
            raise RuntimeError('Only the original mean-own reverse collection is allowed')
        expected = len(self.batch_records) + 1
        if self.ledger.counts['native_reverse']['attempts'] != expected - 1:
            raise RuntimeError('Exactly one native mean-own backward per update required')
        self.ledger.start('native_reverse')
        result = self.original_backward(*args, **kwargs)
        self.ledger.finish('native_reverse')
        return result

    def attempted_forward(self, module, args):
        key = self.phase + '_forward' if self.phase else ''
        if key not in BOUNDS:
            raise RuntimeError('Unexpected model forward outside the frozen TRAIN audit')
        self.ledger.start(key)

    def completed_forward(self, module, args, output):
        self.ledger.finish(self.phase + '_forward')

    def replay(self, label, values, frame):
        torch = self.torch
        self.phase = label
        self.put(self.params, values)
        self.put(self.buffers, frame['pre_buffers'])
        self.common.restore_rng(torch, frame['pre_rng'])
        if tree_hash(torch, self.common.global_rng(torch)) != tree_hash(torch, frame['pre_rng']):
            raise ValueError('Fixed-view replay RNG restoration differs')
        with torch.no_grad():
            losses = [float(self.session._native_loss(self.original_member(m, frame['batch'])[0], frame['batch']))
                      for m in range(4)]
        if not all(math.isfinite(v) for v in losses):
            raise FloatingPointError('Nonfinite TRAIN replay loss')
        if tree_hash(torch, self.common.global_rng(torch)) != tree_hash(torch, frame['post_rng']):
            raise ValueError('Replay did not consume the original native dropout RNG progression')
        self.model.route_bn_state.assert_native_identity()
        return losses

    def step(self, *args, **kwargs):
        torch, frame = self.torch, self.frame
        if frame is None or self.phase != 'native' or 'gradients' not in frame:
            raise RuntimeError('Native proposal requires all four supervised gradients')
        if self.ledger.counts['native_reverse']['completed'] != frame['number']:
            raise RuntimeError('Native source backward was not completed exactly once')
        max_residual = 0.0
        for i, parameter in enumerate(self.params):
            expected = sum(g[i] for g in frame['gradients']) / 4
            if parameter.grad is None or not torch.allclose(parameter.grad, expected, rtol=1e-4, atol=1e-6):
                raise ValueError('Observed native mean-own gradient differs from route decomposition')
            max_residual = max(max_residual, float((parameter.grad - expected).abs().max()))
        self.ledger.start('optimizer')
        value = self.original_step(*args, **kwargs)
        self.ledger.finish('optimizer')
        post = self.clone(self.params)
        post_buffers = self.clone(self.buffers)
        post_rng = self.common.global_rng(torch)
        frame['post_rng'] = post_rng
        post_loader_hash = tree_hash(torch, self.loader_state())
        post_optimizer_hash = tree_hash(torch, self.session.optimizer.state_dict())
        if any(not torch.isfinite(p).all().item() for p in post):
            raise FloatingPointError('Nonfinite native Adam proposal')
        def vector(tensors, indices):
            return torch.cat([tensors[i].reshape(-1) for i in indices]).double()
        shared = torch.stack([vector(g, self.shared) for g in frame['gradients']])
        private = torch.stack([vector(g, self.private) for g in frame['gradients']])
        delta = tuple(new - old for new, old in zip(post, frame['old']))
        ds, dp = vector(delta, self.shared), vector(delta, self.private)
        s = (shared @ ds).cpu().tolist()
        p = (private @ dp).cpu().tolist()
        gram = (shared @ shared.T).cpu().tolist()
        private_only = tuple(frame['old'][i] if i in self.shared else post[i] for i in range(len(post)))
        reconstruction = None
        try:
            if frame['number'] == 1:
                reconstruction = self.replay('O', frame['old'], frame)
                if any(abs(x - y) > 1e-4 + 1e-4 * abs(y) for x, y in zip(reconstruction, frame['losses'])):
                    raise ValueError('Original TRAIN forward/dropout reconstruction exceeds fixed tolerance')
            b = self.replay('B', private_only, frame)
            d = self.replay('D', post, frame)
        finally:
            self.put(self.params, post)
            self.put(self.buffers, post_buffers)
            self.common.restore_rng(torch, post_rng)
            self.phase = 'native'
            if objects(self.model) != self.identity:
                raise ValueError('Model/private-BN object or pointer identity changed')
            if any(not torch.equal(a, z) for a, z in zip(self.params, post)) or any(
                    not torch.equal(a, z) for a, z in zip(self.buffers, post_buffers)):
                raise ValueError('Finite replay did not restore the actual native post-state')
            if tree_hash(torch, self.common.global_rng(torch)) != tree_hash(torch, post_rng):
                raise ValueError('Finite replay advanced native RNG')
            if tree_hash(torch, self.loader_state()) != post_loader_hash:
                raise ValueError('Finite replay changed private loader/sampler RNG state')
            if tree_hash(torch, self.session.optimizer.state_dict()) != post_optimizer_hash:
                raise ValueError('Finite replay changed native optimizer state')
            self.model.route_bn_state.assert_native_identity()
        for m in range(4):
            old = frame['losses'][m]
            row = {'batch': frame['number'], 'member': m, 'old_loss': old, 'private_only_loss': b[m],
                   'joint_loss': d[m], 'predicted_shared_change': s[m], 'predicted_private_change': p[m],
                   'predicted_joint_change': s[m] + p[m], 'finite_private_change': b[m] - old,
                   'finite_shared_after_private_change': d[m] - b[m], 'finite_joint_change': d[m] - old,
                   'tau': 1e-4 + 1e-4 * abs(old)}
            self.rows.append(row)
            with (self.output / 'ROUTES.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        pairs = []
        for a in range(4):
            for b_index in range(a + 1, 4):
                denominator = math.sqrt(max(0., gram[a][a]) * max(0., gram[b_index][b_index]))
                pairs.append({'a': a, 'b': b_index, 'dot': gram[a][b_index],
                              'cosine': gram[a][b_index] / denominator if denominator else None})
        record = {'batch': frame['number'], 'native_update_before': frame['update_before'],
                  'TRAIN_graph_ids': self.session.train_ids[frame['positions']].tolist(),
                  'TRAIN_positions': frame['positions'].tolist(), 'native_gradient_max_absolute_residual': max_residual,
                  'shared_Gram': gram, 'shared_pairs': pairs, 'shared_delta_norm': float(ds.norm()),
                  'private_delta_norm': float(dp.norm()), 'old_reconstruction': reconstruction,
                  'pre_native_RNG': tree_hash(torch, frame['pre_rng']), 'post_native_RNG': tree_hash(torch, post_rng),
                  'post_native_loader_RNG': post_loader_hash, 'post_native_optimizer': post_optimizer_hash,
                  'post_native_state_and_RNG_restored_exactly': True}
        self.batch_records.append(record)
        with (self.output / 'BATCHES.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.frame = None
        return value

    def blocked(self, *args, **kwargs):
        self.ledger.emit({'event': 'forbidden_heldout_or_extra_diagnostic_attempt'})
        raise RuntimeError('No VALID/TEST or shuffled TRAIN-eval diagnostic in this audit')

    def install(self):
        self.session._next_train_batch = self.next_batch
        self.model.run_member = self.member
        self.session.optimizer.step = self.step
        self.torch.autograd.backward = self.backward
        self.adapter.validate = self.blocked
        self.session.validate = self.blocked
        self.adapter.train_diagnostic = self.blocked
        self.session.valid_loader = ()
        self.hooks = [self.model.native.register_forward_pre_hook(self.attempted_forward),
                      self.model.native.register_forward_hook(self.completed_forward)]

    def close(self):
        self.session._next_train_batch = self.original_next
        self.model.run_member = self.original_member
        self.session.optimizer.step = self.original_step
        self.torch.autograd.backward = self.original_backward
        self.adapter.validate = self.original_validate
        self.session.validate = self.original_session_validate
        self.adapter.train_diagnostic = self.original_diagnostic
        self.session.valid_loader = self.original_valid_loader
        for handle in self.hooks:
            handle.remove()


def preflight(root, output):
    allocation()
    root = Path(root).resolve()
    inside(HERE, root)
    output = inside(output, root)
    if output != root / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/results/shared_gradient_shadow_audit_v1':
        raise ValueError('The one reviewed fresh output directory is required')
    if output.exists():
        raise ValueError('Preserve previous attempt; overwrite/retry is forbidden')
    inputs = json.loads((HERE / 'INPUTS.json').read_text())
    if [c['seed'] for c in inputs['cases']] != list(SEEDS):
        raise ValueError('All three original seeds required')
    bound = []
    for case in inputs['cases']:
        fit = json.loads(verify(case['fit_RESULT'], root).read_text())
        job_path = inside(Path(case['fit_RESULT']['path']).with_name('JOB.json'), root)
        job = json.loads(job_path.read_text())
        if fit['status'] != 'complete' or digest(job) != case['job_hash'] or fit['job_hash'] != case['job_hash']:
            raise ValueError('Genuine completed own JOB/RESULT identity differs')
        if job['seed'] != case['seed'] or job['procedure'] != 'gnnm_private_bn_state_v3' or job['family'] != 'ogb' or job['recipe']['dataset'] != 'ogbg-molbace' or job['TEST_access'] is not False or fit['TEST_accessed'] is not False:
            raise ValueError('Only the original three MolBACE private-BN fits are admitted')
        if fit['selected_epoch'] != case['selected_epoch'] or fit['runtime_sources'] != inputs['genuine_runtime_sources']:
            raise ValueError('Original selected epoch or authentic runtime differs')
        for pin in fit['runtime_sources'].values():
            verify(pin, root)
        binding = fit['runtime_binding']
        if binding['runtime_sha256'] != {name: pin['sha256'] for name, pin in inputs['genuine_runtime_sources'].items()}:
            raise ValueError('Original producer runtime binding differs')
        for path, sha in binding['dependency_sha256'].items():
            verify({'path': path, 'sha256': sha}, root)
        verify({'path': job['callback_module'], 'sha256': binding['callback_sha256']}, root)
        verify({'path': job['session_transform']['path'], 'sha256': binding['extension_sha256']}, root)
        verify({'path': str(Path(job['session_transform']['path']).with_name('banks.py')),
                'sha256': job['method_settings']['banks_sha256']}, root)
        verify(fit['selected_checkpoint'], root)
        metadata = fit['source_metadata']['native_session_metadata']
        for role, pin in metadata['scaffold_role_pins'].items():
            verify(pin, root)
            path = Path(pin['path'])
            if path.name.endswith('.csv.gz') and path.with_name(role + '.pt').exists():
                raise ValueError('Original role-loader precedence changed')
        for pin in metadata['encoding_dependencies'].values():
            verify(pin, root)
        bound.append((case, job, fit))
    return root, output, inputs, bound


def execute_case(torch, common, factory, policy_class, root, output, case, job, fit):
    output.mkdir()
    ledger = Ledger(output / 'OPERATIONS.jsonl')
    result = {'seed': case['seed'], 'status': 'running', 'selected_epoch': case['selected_epoch'],
              'fit_RESULT': case['fit_RESULT'], 'checkpoint': fit['selected_checkpoint'],
              'job_hash': case['job_hash'], 'role': 'TRAIN', 'VALID_calls': 0, 'TEST_calls': 0,
              'retained_trained_models': 0, 'automatic_retries': 0, 'work': ledger.counts}
    write_json(output / 'START.json', result)
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    adapter = audit = checkpoint = None
    outside = common.global_rng(torch)
    torch.cuda.reset_peak_memory_stats()
    def timed_out(signum, frame):
        raise TimeoutError('Fixed 600-second per-seed audit bound reached')
    signal.signal(signal.SIGALRM, timed_out)
    signal.signal(signal.SIGTERM, timed_out)
    signal.alarm(600)
    try:
        checkpoint = torch.load(verify(fit['selected_checkpoint'], root), map_location='cpu', weights_only=False)
        if checkpoint['schema'] != 'native-full-epoch-recovery-v1' or checkpoint['job_hash'] != case['job_hash'] or checkpoint['runtime_binding'] != fit['runtime_binding'] or checkpoint['TEST_accessed'] is not False or checkpoint['selector']['selected_epoch'] != case['selected_epoch'] or checkpoint['selector']['completed_epoch'] != case['selected_epoch']:
            raise ValueError('Genuine own selected checkpoint differs')
        expected_state = tree_hash(torch, checkpoint['adapter'])
        expected_rng = tree_hash(torch, checkpoint['global_rng'])
        common.seed_all(torch, job['seed'])
        ledger.start('construction')
        adapter = factory.create_adapter(torch, job, policy_class.from_dict(job['policy'], 'ogb'), root)
        ledger.finish('construction')
        binding = checkpoint['runtime_binding']
        if adapter.callback_receipt['sha256'] != binding['callback_sha256'] or adapter.extension_receipt['sha256'] != binding['extension_sha256'] or {pin['path']: pin['sha256'] for pin in adapter.dependency_receipts} != binding['dependency_sha256']:
            raise ValueError('Authentic source construction differs')
        if adapter.session.split_pins != fit['source_metadata']['native_session_metadata']['scaffold_role_pins']:
            raise ValueError('Original scaffold provider authority differs')
        ledger.start('selected_load')
        adapter.load_state(copy.deepcopy(checkpoint['adapter']))
        common.restore_rng(torch, checkpoint['global_rng'])
        ledger.finish('selected_load')
        if tree_hash(torch, adapter.state()) != expected_state or tree_hash(torch, common.global_rng(torch)) != expected_rng:
            raise ValueError('Exact selected optimizer/loader/model/RNG restoration failed')
        if len(adapter.session.train_ids) != 1210 or adapter.session.train_view.role != 'TRAIN' or str(adapter.session.device) != 'cuda:0':
            raise ValueError('Original complete TRAIN population and CUDA device required')
        audit = NativeAudit(torch, common, adapter, ledger, output)
        result['partition'] = {'shared_parameters': [audit.named[i][0] for i in audit.shared],
                               'private_parameters': [audit.named[i][0] for i in audit.private],
                               'shared_coordinates': sum(audit.params[i].numel() for i in audit.shared),
                               'private_coordinates': sum(audit.params[i].numel() for i in audit.private),
                               'native_BN_affine_is_shared': True, 'private_BN_is_buffer_state': True}
        initial_updates = adapter.session.updates
        audit.install()
        ledger.start('epoch')
        epoch = adapter.train_epoch()
        ledger.finish('epoch')
        if not ledger.complete() or len(audit.rows) != 152 or adapter.session.updates != initial_updates + 38 or adapter.completed_epochs != case['selected_epoch'] + 1:
            raise ValueError('Full native epoch/update accounting differs')
        if len(adapter._seen) != 1210 or sorted(adapter._seen) != list(range(1210)) or adapter._raw_batches != 38 or adapter._skipped != 0:
            raise ValueError('Every official TRAIN graph must appear once, with original skip law')
        result.update(status='complete', native_epoch=epoch, gate=gate(audit.rows),
                      state_range='own selected state and next complete native TRAIN epoch',
                      temporary_native_Adam_maps=38, shared_gradient_pair_dots_are_not_gate=True)
    except BaseException as error:
        result.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                               'traceback': traceback.format_exc(limit=10)})
    finally:
        signal.alarm(0)
        if audit is not None:
            try:
                audit.close()
                audit.frame = None
            except BaseException as error:
                result['status'] = 'failed'
                result['instrumentation_close_error'] = str(error)
        if adapter is not None and checkpoint is not None:
            try:
                adapter.load_state(copy.deepcopy(checkpoint['adapter']))
                common.restore_rng(torch, checkpoint['global_rng'])
                adapter.session.optimizer.zero_grad(set_to_none=True)
                result['selected_state_restored_after_discard'] = tree_hash(torch, adapter.state()) == tree_hash(torch, checkpoint['adapter'])
                result['selected_RNG_restored_after_discard'] = tree_hash(torch, common.global_rng(torch)) == tree_hash(torch, checkpoint['global_rng'])
                result['model_object_identity_restored'] = audit is None or objects(adapter.model) == audit.identity
                if not result['selected_state_restored_after_discard'] or not result['selected_RNG_restored_after_discard']:
                    raise ValueError('Discard did not restore exact original selected state/RNG')
                if not result['model_object_identity_restored']:
                    raise ValueError('Discard changed native/private bank objects or live pointers')
            except BaseException as error:
                result['status'] = 'failed'
                result['restoration_error'] = str(error)
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as error:
                result['status'] = 'failed'
                result['close_error'] = str(error)
        try:
            common.restore_rng(torch, outside)
            torch.cuda.synchronize()
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as error:
            result['status'] = 'failed'
            result['CUDA_cleanup_or_resource_error'] = str(error)
        result.update(wall_seconds=time.perf_counter() - started,
                      lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      case_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime - usage_before.ru_utime,
                      case_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime - usage_before.ru_stime)
        write_json(output / 'RESULT.json', result)
    return result


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output, inputs, bound = preflight(args.root, args.output)
    if 'torch' in sys.modules:
        raise RuntimeError('Fresh process required before Torch/device setup')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one authorized CUDA device required')
    runtime = Path(inputs['genuine_runtime_sources']['adapters.py']['path']).parent
    if any(name in sys.modules for name in ('common', 'adapters', 'bookkeeping')):
        raise RuntimeError('Authentic runtime module namespace must be fresh')
    sys.path.insert(0, str(runtime))
    common = importlib.import_module('common')
    factory = importlib.import_module('adapters')
    policy_class = importlib.import_module('bookkeeping').Policy
    common.require_allocation()
    output.mkdir(parents=True)
    results = []
    for case, job, fit in bound:
        results.append(execute_case(torch, common, factory, policy_class, root, output / str(case['seed']), case, job, fit))
        gc.collect()
        torch.cuda.empty_cache()
    complete = all(r['status'] == 'complete' for r in results)
    summary = {'schema': 'private-BN-native-shared-gradient-shadow-audit-v1',
               'status': 'complete' if complete else 'failed', 'seeds': list(SEEDS), 'results': results,
               'interference_gate': {'open': sum(r['gate']['seed_gate_open'] for r in results) >= 2,
                                     'required_seed_count': 2, 'seed_count': 3} if complete else None,
               'VALID_calls': 0, 'TEST_calls': 0, 'retained_trained_models': 0,
               'wall_seconds': time.perf_counter() - start,
               'automatic_intervention_launch': False, 'automatic_retries': 0}
    write_json(output / 'SUMMARY.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'seeds', 'interference_gate', 'wall_seconds')}))
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
