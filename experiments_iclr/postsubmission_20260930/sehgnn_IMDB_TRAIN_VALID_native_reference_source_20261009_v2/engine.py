"""Literal native IMDB computation; runtime/providers supplied after root gate."""
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from state_helpers import capture_rng, restore_rng, cpu_tree, exact, parameter_counts


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


class Costs:
    def __init__(self, folder, torch=None, device=None):
        self.folder, self.torch, self.device, self.rows = folder, torch, device, []

    @contextmanager
    def measure(self, scope, gpu=False):
        start, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        first = last = None
        row = {'scope': scope, 'status': 'started'}
        try:
            if gpu:
                self.torch.cuda.synchronize(self.device)
                first, last = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
                first.record()
            yield row
            row['status'] = 'complete'
        except BaseException as error:
            row.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            timing_failure = None
            if first is not None:
                try:
                    last.record()
                    last.synchronize()
                    row['CUDA_stream_ms'] = float(first.elapsed_time(last))
                except BaseException as error:
                    timing_failure = error
                    row['CUDA_timing_failure'] = {'type': type(error).__name__, 'message': str(error)}
                    if row['status'] != 'failed':
                        row['status'] = 'failed'
            end = resource.getrusage(resource.RUSAGE_SELF)
            row.update(seconds=time.perf_counter() - start, CPU_user_seconds=end.ru_utime - usage.ru_utime,
                       CPU_system_seconds=end.ru_stime - usage.ru_stime)
            self.rows.append(row)
            with (self.folder / 'COST_EVENTS.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            if timing_failure is not None and 'failure' not in row:
                raise timing_failure


@dataclass
class StaticContext:
    data: object
    raw_feats: dict
    adjs: dict
    data_size: dict
    feature_shapes: dict


@dataclass
class SeedContext:
    data: object
    seed: int
    feats: dict
    label_feats: dict
    data_size: dict
    targets: object
    targets_cuda: object
    train_index: object
    valid_index: object
    train_loader: object
    eval_batches: list
    train_count: int
    valid_count: int


def prepare_static(rt, data, costs):
    """Exact IMDB typed graph convention, native full feature propagation."""
    torch, np, dgl, SparseTensor, native = (rt[k] for k in ('torch', 'numpy', 'dgl', 'SparseTensor', 'native'))
    symbols = ('M', 'D', 'A', 'K')
    adjs, graph_edges, features = {}, {}, {}
    with costs.measure('native_CPU_features_and_typed_graph') as row:
        for type_id, symbol in enumerate(symbols):
            count = data.counts[type_id]
            if data.features[type_id] is None:
                features[symbol] = torch.eye(count)  # Preserve the original dense keyword identity.
            else:
                view = np.frombuffer(data.features[type_id], dtype=np.float32).reshape(count, -1)
                features[symbol] = torch.FloatTensor(view)
        for meta in data.relation_rows:
            head_type, tail_type = meta['raw_head_type'], meta['raw_tail_type']
            name = symbols[head_type] + symbols[tail_type]
            pairs = data.canonical_relation_pairs[meta['raw_relation_id']]
            rows = torch.LongTensor([head - data.offsets[head_type] for head, _ in pairs])
            columns = torch.LongTensor([tail - data.offsets[tail_type] for _, tail in pairs])
            adj = SparseTensor(row=rows, col=columns, sparse_sizes=(data.counts[head_type], data.counts[tail_type]))
            adjs[name] = adj
            # Native CSR row=head/destination; column=tail/message source.
            graph_edges[(symbols[tail_type], symbols[tail_type] + '-' + symbols[head_type], symbols[head_type])] = (columns.numpy(), rows.numpy())
        graph = dgl.heterograph(graph_edges)
        require(all(graph.num_nodes(symbol) == data.counts[i] for i, symbol in enumerate(symbols)), 'Literal native inferred typed graph retains complete node populations')
        for symbol, feature in features.items():
            graph.nodes[symbol].data[symbol] = feature
        row['raw_input_feature_bytes'] = sum(value.numel() * value.element_size() for value in features.values())
        for key in adjs:
            adjs[key].storage._value = None
            adjs[key].storage._value = torch.ones(adjs[key].nnz()) / adjs[key].sum(dim=-1)[adjs[key].storage.row()]
    with costs.measure('native_full_CPU_feature_propagation'):
        graph = native.hg_propagate_feat_dgl(graph, 'M', 4, 5, [], echo=True)
        raw = {key: graph.nodes['M'].data.pop(key) for key in list(graph.nodes['M'].data.keys())}
        expected = rt['protocol']['feature_paths']
        require(sorted(raw) == sorted(expected) and len(raw) == 25, 'All original four-hop feature channels')
        shapes = {key: list(value.shape) for key, value in raw.items()}
        require(all(value.shape[0] == data.counts[0] and torch.isfinite(value).all().item() for value in raw.values()), 'Finite complete native movie features')
        data_size = {key: value.size(-1) for key, value in raw.items()}
    del graph, features
    gc.collect()
    return StaticContext(data, raw, adjs, data_size, shapes)


def prepare_seed(rt, static, frozen_role, seed, costs):
    torch, np, native = rt['torch'], rt['numpy'], rt['native']
    data = static.data.with_roles(frozen_role)
    native.set_random_seed(seed)
    shuffled = np.asarray(sorted(data.train_ids + data.valid_ids), dtype=np.int64)
    np.random.shuffle(shuffled)  # Consume exactly the original native global shuffle.
    boundary = int(shuffled.shape[0] * 0.2)
    train_ids, valid_ids = np.sort(shuffled[boundary:]), np.sort(shuffled[:boundary])
    require(train_ids.tolist() == list(data.train_ids) and valid_ids.tolist() == list(data.valid_ids), 'Frozen roles match literal native RNG split')
    require(0 < len(train_ids) <= 10000, 'Literal IMDB one native TRAIN minibatch per epoch')
    with costs.measure('native_per_seed_clone_and_TRAIN_label_propagation') as row:
        feats = {key: value.detach().clone() for key, value in static.raw_feats.items()}
        # This is a masked target interface, not a full ground-truth matrix.
        targets = torch.full((data.counts[0], data.class_dim), float('nan'))
        train_index, valid_index = torch.LongTensor(train_ids), torch.LongTensor(valid_ids)
        targets[train_index] = torch.FloatTensor(data.train_labels)
        targets[valid_index] = torch.FloatTensor(data.valid_labels)
        require(torch.isfinite(targets[train_index]).all().item() and torch.isfinite(targets[valid_index]).all().item(), 'Only explicit known targets are finite')
        label_source = torch.zeros((data.counts[0], data.class_dim))
        label_source[train_index] = targets[train_index].float()
        meta = native.hg_propagate_sparse_pyg(static.adjs, 'M', 4, 5, [], prop_feats=False, echo=True, prop_device='cpu')
        label_feats = {key: rt['remove_diag'](value) @ label_source for key, value in meta.items()}
        require(sorted(label_feats) == sorted(rt['protocol']['label_paths']) and len(label_feats) == 12, 'All original diagonal-removed TRAIN-label channels')
        require(all(torch.isfinite(value).all().item() for value in label_feats.values()), 'Finite native TRAIN-only label propagation')
        require(all(torch.count_nonzero(label_source[index]).item() == 0 for index in (valid_index, torch.LongTensor(data.unassigned_movie_ids))), 'No non-TRAIN label input')
        targets_cuda = targets.to(rt['device'])
        # Native generator=None and construction before model initialization are preserved.
        train_loader = torch.utils.data.DataLoader(train_ids, batch_size=10000, shuffle=True, drop_last=False)
        known = np.concatenate((train_ids, valid_ids))
        batches = []
        for start in range(0, len(known), 20000):
            batch = torch.LongTensor(known[start:start + 20000])
            batches.append((batch, {key: value[batch] for key, value in feats.items()},
                            {key: value[batch] for key, value in label_feats.items()}, None))
        row.update(cloned_feature_bytes=sum(value.numel() * value.element_size() for value in feats.values()),
                   label_feature_bytes=sum(value.numel() * value.element_size() for value in label_feats.values()),
                   prebuilt_evaluation_feature_bytes=sum(value.numel() * value.element_size() for _, values, _, _ in batches for value in values.values()),
                   full_movie_targets_are_NaN_outside_known_roles=True, TRAIN_label_source_only=True)
    del meta, label_source
    gc.collect()
    return SeedContext(data, seed, feats, label_feats, static.data_size, targets, targets_cuda,
                       train_index, valid_index, train_loader, batches, len(train_ids), len(valid_ids))


def make_model(rt, ctx, costs):
    """Native CPU construction, full architecture, ordinary original placement."""
    with costs.measure('original_CPU_model_constructor_and_placement', gpu=True):
        rt['torch'].cuda.empty_cache()
        gc.collect()
        model = rt['model_class']('IMDB', 512, 512, 5, ctx.feats.keys(), ctx.label_feats.keys(), 'M',
                                0.5, 0.0, 0.0, 2, 4, 'none', False, data_size=ctx.data_size)
        require(all(value.device.type == 'cpu' for value in model.parameters()), 'Original CPU initialization')
        model = model.to(rt['device'])
    return model


@contextmanager
def observe_update(rt, model, optimizer, counters):
    """Original helpers/step delegate once; counters and gross finite observations."""
    torch = rt['torch']
    step = optimizer.step
    def forward(module, inputs, output):
        require(output.ndim == 2 and output.shape[1] == 5 and torch.isfinite(output).all().item(), 'Finite native five-logit TRAIN output')
        counters['TRAIN_member_forwards'] += 1
    def backward(module, grad_input, grad_output):
        counters['TRAIN_backwards'] += 1
    def observed_step(*args, **kwargs):
        require(all(value.grad is None or torch.isfinite(value.grad).all().item() for value in model.parameters()), 'Finite unscaled native gradients')
        value = step(*args, **kwargs)
        counters['actual_Adam_steps'] += 1
        return value
    handles = [model.register_forward_hook(forward), model.register_full_backward_hook(backward)]
    optimizer.step = observed_step
    try:
        yield
    finally:
        optimizer.step = step
        for handle in reversed(handles):
            handle.remove()


def evaluate(rt, model, ctx, counters, costs, scope):
    torch, np = rt['torch'], rt['numpy']
    before = capture_rng(np, torch)
    try:
        with costs.measure(scope, gpu=True), torch.no_grad():
            model.eval()
            values = []
            for batch, features, labels, mask in ctx.eval_batches:
                output = model(batch.to(rt['device']), {key: value.to(rt['device']) for key, value in features.items()},
                               {key: value.to(rt['device']) for key, value in labels.items()}, mask)
                require(torch.isfinite(output).all().item(), 'Finite complete known-role output')
                values.append(output.cpu())
                counters[scope + '_forward_calls'] = counters.get(scope + '_forward_calls', 0) + 1
            logits = torch.cat(values, dim=0)
            require(logits.shape == (ctx.train_count + ctx.valid_count, 5), 'Complete TRAIN/VALID output order')
            scores, role_logits = {}, {}
            for name, selected, start, end in (('TRAIN', ctx.train_index, 0, ctx.train_count),
                                              ('VALID', ctx.valid_index, ctx.train_count, ctx.train_count + ctx.valid_count)):
                current, truth = logits[start:end], ctx.targets[selected]
                bce = torch.nn.functional.binary_cross_entropy_with_logits(current, truth).item()
                micro, macro = rt['native'].evaluator(truth, (current > 0.).int())
                served_micro, served_macro = rt['native'].evaluator(truth, (torch.sigmoid(current) > 0.5).int())
                scores[name] = dict(BCE=float(bce), native_logit_micro_F1=float(micro), native_logit_macro_F1=float(macro),
                                    served_micro_F1=float(served_micro), served_macro_F1=float(served_macro))
                role_logits[name] = current.clone()
            return scores, role_logits
    finally:
        restore_rng(np, torch, before)
        require(exact(torch, capture_rng(np, torch), before), 'Evaluation preserves original stream position')


def snapshot(rt, model, optimizer, scalar, ctx, epoch, scores, logits, identity):
    torch = rt['torch']
    return dict(schema='owned-literal-SeHGNN-IMDB-selected-state-v1', identity=identity, seed=ctx.seed, epoch=epoch,
                model_state=cpu_tree(torch, model.state_dict()), optimizer_state=cpu_tree(torch, optimizer.state_dict()),
                scaler_state=scalar.state_dict(), training_rng=capture_rng(rt['numpy'], torch),
                scores=scores, role_logits=logits, known_roles_only=True)


def replay(rt, ctx, checkpoint, counters, costs, identity):
    torch, np = rt['torch'], rt['numpy']
    caller_rng = capture_rng(np, torch)
    model = optimizer = scalar = saved = None
    try:
        with costs.measure('fresh_selected_reconstruction', gpu=True):
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(saved['identity'] == identity and saved['seed'] == ctx.seed, 'Owned selected checkpoint identity')
            model = make_model(rt, ctx, costs)
            model.load_state_dict(saved['model_state'], strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
            optimizer.load_state_dict(saved['optimizer_state'])
            scalar = torch.cuda.amp.GradScaler()
            scalar.load_state_dict(saved['scaler_state'])
            require(exact(torch, cpu_tree(torch, model.state_dict()), saved['model_state'])
                    and exact(torch, cpu_tree(torch, optimizer.state_dict()), saved['optimizer_state'])
                    and exact(torch, scalar.state_dict(), saved['scaler_state']), 'Exact model/buffer/Adam/AMP reconstruction')
            restore_rng(np, torch, saved['training_rng'])
            require(exact(torch, capture_rng(np, torch), saved['training_rng']), 'Exact selected owned streams')
        scores, logits = evaluate(rt, model, ctx, counters, costs, 'selected_serving')
        diagnostics = {name: dict(max_abs_logit=float((logits[name] - saved['role_logits'][name]).abs().max().item()),
                                 max_abs_probability=float((torch.sigmoid(logits[name]) - torch.sigmoid(saved['role_logits'][name])).abs().max().item()),
                                 prediction_changes=int(((torch.sigmoid(logits[name]) > .5) != (torch.sigmoid(saved['role_logits'][name]) > .5)).sum().item()))
                       for name in ('TRAIN', 'VALID')}
        return dict(epoch=saved['epoch'], fresh_scores=scores, selected_recorded_scores=saved['scores'],
                    replay_diagnostics=diagnostics, model_optimizer_scaler_and_streams_exact=True,
                    output_bitwise_gate=False)
    finally:
        model = optimizer = scalar = saved = None
        gc.collect()
        torch.cuda.empty_cache()
        restore_rng(np, torch, caller_rng)
        require(exact(torch, capture_rng(np, torch), caller_rng), 'Fresh reconstruction preserves caller stream position')


def run_one(rt, static, role, seed, folder, costs, scalar, identity, qualify=False):
    torch = rt['torch']
    folder.mkdir(exist_ok=False)
    local = Costs(folder, torch, rt['device'])
    started = time.perf_counter()
    initial_usage = resource.getrusage(resource.RUSAGE_SELF)
    torch.cuda.reset_peak_memory_stats(rt['device'])
    record = dict(status='started', seed=seed, qualification=qualify, TEST_truth=False, TEST_membership_known=False,
                  complete=False, identity=identity, native_scaler_lifecycle='one master across source seed loop')
    counters = {'epochs_completed': 0, 'TRAIN_member_forwards': 0, 'TRAIN_backwards': 0, 'actual_Adam_steps': 0}
    model = optimizer = ctx = None
    def persist():
        write(folder / 'RESULT.json', dict(record, counters=counters))
    persist()
    try:
        ctx = prepare_seed(rt, static, role, seed, local)
        model = make_model(rt, ctx, local)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        record.update(parameters=parameter_counts(model), native_parameter_count=rt['native'].get_n_params(model),
                      feature_shapes={key: list(value.shape) for key, value in ctx.feats.items()},
                      label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()},
                      TRAIN_count=ctx.train_count, VALID_count=ctx.valid_count, unassigned_count=len(ctx.data.unassigned_movie_ids))
        require(record['native_parameter_count'] == rt['protocol']['predicted_native_model_parameters']
                == record['parameters']['total'] == record['parameters']['active'], 'Measured complete literal native parameter count')
        initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()} if qualify else None
        best_epoch, best_loss = -1, 1000000.0
        checkpoint = folder / 'SELECTED_STATE.pt'
        for epoch in range(1 if qualify else 200):
            with local.measure('native_pre_epoch_GC'):
                gc.collect()
            with observe_update(rt, model, optimizer, counters), local.measure('native_TRAIN_epoch', gpu=True):
                loss, accuracy = rt['native'].train(model, ctx.feats, ctx.label_feats, ctx.targets_cuda,
                                                  torch.nn.BCEWithLogitsLoss(), optimizer, ctx.train_loader,
                                                  rt['native'].evaluator, scalar=scalar)
                require(math.isfinite(loss), 'Finite native TRAIN BCE')
            with local.measure('native_post_TRAIN_cache_cleanup'):
                torch.cuda.empty_cache()
            counters['epochs_completed'] += 1
            scores, logits = evaluate(rt, model, ctx, counters, local, 'VALID_selection')
            selected = scores['VALID']['BCE'] < best_loss
            if selected:
                best_epoch, best_loss = epoch, scores['VALID']['BCE']
                with local.measure('selected_checkpoint_CPU_copy_write', gpu=True):
                    saved = snapshot(rt, model, optimizer, scalar, ctx, epoch, scores, logits, identity)
                    temporary = checkpoint.with_suffix('.tmp')
                    torch.save(saved, temporary)
                    temporary.replace(checkpoint)
                    del saved
            with (folder / 'HISTORY.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch, native_TRAIN_BCE=float(loss), native_TRAIN_F1=[float(v) for v in accuracy],
                                             scores=scores, selected=selected, counters=dict(counters)), allow_nan=False) + '\n')
            record.update(last_epoch=epoch, selected_epoch=best_epoch, selected_VALID_BCE=best_loss)
            persist()
            if epoch - best_epoch > 50:
                break
        if qualify:
            changed = sum(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters())
            require(changed > 0 and counters['epochs_completed'] == counters['TRAIN_member_forwards']
                    == counters['TRAIN_backwards'] == counters['actual_Adam_steps'] == 1, 'Exactly one full native TRAIN/backward/actual Adam update')
            record['changed_parameter_objects'] = changed
        initial = model = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        master_scaler_state = scalar.state_dict()
        fresh = replay(rt, ctx, checkpoint, counters, local, identity)
        require(exact(torch, scalar.state_dict(), master_scaler_state), 'Original master AMP state survives selected replay')
        if qualify:
            require(all(row['max_abs_logit'] <= .001 and row['max_abs_probability'] <= .001
                        for row in fresh['replay_diagnostics'].values()), 'Practical gross selected reconstruction drift')
        record.update(status='complete', complete=True, qualification_passed=qualify, fresh_selected=fresh,
                      selected_checkpoint_sha256=sha(checkpoint), master_scaler_state_preserved=True,
                      native_full_VALID_BCE_selector=True, native_patience_rule=True)
        return record
    except BaseException as error:
        record.update(status='failed', complete=False, qualification_passed=False,
                      failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        try:
            torch.cuda.synchronize(rt['device'])
            record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rt['device']),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(rt['device']))
            model = optimizer = ctx = None
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as error:
            record.update(status='failed', complete=False, qualification_passed=False,
                          cleanup_or_cost_failure={'type': type(error).__name__, 'message': str(error)})
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter() - started, CPU_user_seconds=usage.ru_utime-initial_usage.ru_utime,
                      CPU_system_seconds=usage.ru_stime-initial_usage.ru_stime, cumulative_RSS_peak_bytes=int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                      complete_native_preprocessing_training_validation_checkpoint_replay_costs_charged=True)
        persist()
        write(folder / 'COMPLETE.json', dict(complete=record['complete'], status=record['status'], counters=counters,
                                           qualification_passed=record.get('qualification_passed', False), TEST_truth=False))
