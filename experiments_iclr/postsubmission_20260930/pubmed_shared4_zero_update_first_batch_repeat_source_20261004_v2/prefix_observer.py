"""One unchanged first-batch forward/backward prefix, stopped before Adam."""
from types import FunctionType
from common import require
from diagnostic_helpers import exact
from storage_aliases import canonical_storage_aliases


class PrefixComplete(BaseException):
    pass


def module_cache_state(unit, native, torch):
    """Inspect all nonregistered named-module attributes, including caches.

    Registered parameters/buffers and module flags are in native.capture.
    All buffer registries are also captured here, including nonpersistent
    buffers, together with parameter/cache alias and layout relations.
    Observer hook registries are excluded explicitly. An opaque object with
    no inspectable fields makes the snapshot incomplete and nonmatching.
    Object aliases/cycles use traversal references. Distinct tensor views
    also retain canonical backing-storage alias classes; local raw storage
    pointers/handles never enter the compared snapshot. Uninspectable backing
    storage makes the snapshot incomplete and nonmatching.
    """
    excluded = {'_parameters', '_buffers', '_modules', '_backward_hooks',
                '_backward_pre_hooks', '_forward_hooks', '_forward_pre_hooks',
                '_forward_hooks_with_kwargs', '_forward_hooks_always_called',
                '_forward_pre_hooks_with_kwargs', '_state_dict_hooks',
                '_state_dict_pre_hooks', '_load_state_dict_pre_hooks',
                '_load_state_dict_post_hooks'}
    memo, unsupported, storage_rows, tensor_records = {}, [], [], {}
    def visit(value, path):
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return {'immutable_torch_descriptor': str(value)}
        key = id(value)
        if key in memo:
            return {'reference': memo[key]}
        memo[key] = path
        if isinstance(value, torch.Tensor):
            try:
                storage = value.untyped_storage()
                require(str(storage.device) == str(value.device), 'Storage/tensor device differs')
                storage_rows.append({'path': path, 'device': str(storage.device),
                                     'handle': storage._cdata, 'start': storage.data_ptr(), 'bytes': storage.nbytes()})
            except Exception as error:
                unsupported.append({'path': path, 'type': 'opaque_uninspected_backing_storage',
                                    'issue': type(error).__name__})
            record = {'tensor': native.clone(value, torch), 'device': str(value.device),
                    'requires_grad': value.requires_grad,
                    'layout': str(value.layout), 'stride': list(value.stride()) if value.layout == torch.strided else None,
                    'storage_offset': value.storage_offset() if value.layout == torch.strided else None}
            tensor_records[path] = record
            return record
        if isinstance(value, dict):
            require(all(type(k) in (str, int) for k in value), 'Cache dictionary key unsupported')
            return {k: visit(v, path + '/' + str(k)) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (tuple, list)):
            return [visit(v, path + '/' + str(i)) for i, v in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            require(all(type(v) in (bool, int, float, str) for v in value), 'Cache set member unsupported')
            return {'set': sorted(value, key=lambda v: (type(v).__name__, repr(v)))}
        fields = dict(vars(value)) if hasattr(value, '__dict__') else {}
        for klass in type(value).__mro__:
            slots = getattr(klass, '__slots__', ())
            slots = (slots,) if isinstance(slots, str) else slots
            for name in slots:
                if name not in ('__dict__', '__weakref__') and hasattr(value, name):
                    fields.setdefault(name, getattr(value, name))
        if fields:
            return {'type': type(value).__module__ + '.' + type(value).__qualname__,
                    'fields': {name: visit(v, path + '/' + name) for name, v in sorted(fields.items())}}
        unsupported.append({'path': path, 'type': type(value).__module__ + '.' + type(value).__qualname__})
        return {'opaque_uninspected_type': unsupported[-1]['type']}
    result = {}
    for number, root in enumerate(unit[:2]):
        for name, module in root.named_modules():
            path = str(number) + '/' + name
            result[path] = {
                'attributes': {key: visit(value, path + '/' + key) for key, value in sorted(vars(module).items())
                               if key not in excluded},
                'registered_parameters_and_aliases': visit(module._parameters, path + '/_parameters'),
                'all_registered_buffers_including_nonpersistent': visit(module._buffers, path + '/_buffers')}
    storage_aliases = canonical_storage_aliases(storage_rows)
    unsupported.extend(storage_aliases['unsupported'])
    for path, record in tensor_records.items():
        record['backing_storage'] = storage_aliases['classes'].get(path, {'opaque_uninspected_storage': True})
    return {'attributes': result, 'fully_inspected': not unsupported, 'unsupported': unsupported,
            'excluded_hook_and_registered_containers': sorted(excluded)}


def full_state(unit, native, torch, np):
    value = native.capture(unit, torch, np)
    value['unregistered_module_state'] = module_cache_state(unit, native, torch)
    return value


def prefix(native, bodies, observed, unit, train, torch, np, progress):
    model, predictor, optimizer, _, data, _ = unit
    original = bodies.candidate_ncnc_train
    sampler, iterator = original.__globals__['negative_sampling'], original.__globals__['PermIterator']
    require(sampler is bodies.negative_sampling and iterator is bodies.PermIterator, 'Original native globals differ')
    receipt = {'negative_calls': [], 'iterator_calls': [], 'batches': [], 'decoder_inputs': [],
               'RNG_neutral_observations': 0, 'backward_calls': 0, 'backward_completed': 0}
    channels, prestate = {}, []
    def observe(callback):
        before = native.clone(native.rng(torch, np, True), torch)
        value = callback()
        exact(before, native.rng(torch, np, True), torch, 'prefix_observer_RNG_neutral')
        receipt['RNG_neutral_observations'] += 1
        return value
    start = native.clone(native.rng(torch, np, True), torch)
    def sample(*args, **kwargs):
        require(len(args) == 2 and not kwargs and args[1] == 19717 and args[0] is data.edge_index,
                'Original negative sampler invocation differs')
        value = sampler(*args, **kwargs)
        receipt['negative_calls'].append(observe(lambda: observed.tensor_identity(value)))
        return value
    class FirstIterator:
        def __init__(self, *args, **kwargs):
            require(len(args) == 3 and not kwargs and args[1:] == (37676, 1024), 'Native iterator invocation differs')
            self.original = iterator(*args, **kwargs)
            receipt['iterator_calls'].append(observe(lambda: {
                'full_order': observed.tensor_identity(self.original.idx),
                'dropped_tail': observed.tensor_identity(self.original.idx[36864:]),
                'native_length': len(self.original), 'training': self.original.training, 'batch_size': self.original.bs}))
        def __iter__(self):
            iter(self.original)
            return self
        def __next__(self):
            require(not receipt['batches'], 'Second native batch is forbidden')
            value = next(self.original)
            require(value.shape == (1024,), 'First native batch shape differs')
            receipt['batches'].append(observe(lambda: observed.tensor_identity(value)))
            return value
    def before_encoder(module, args):
        require(not prestate, 'One encoder call per prefix required')
        prestate.append(observe(lambda: full_state(unit, native, torch, np)))
        rowptr, col, value = args[1].csr()
        receipt['masked_encoder_input'] = observe(lambda: {
            'x': observed.tensor_identity(args[0]), 'rowptr': observed.tensor_identity(rowptr),
            'col': observed.tensor_identity(col), 'value': None if value is None else observed.tensor_identity(value)})
        progress.add(encoder_started=1)
    def after_encoder(module, args, value):
        channels['encoder_output'] = observe(lambda: native.clone(value, torch))
        progress.add(encoder_completed=1)
    def before_decoder(module, args):
        require(len(receipt['decoder_inputs']) < 2, 'Third root decoder call forbidden')
        progress.add(root_decoder_started=1, root_query_rows_started=int(args[2].shape[0]))
    def after_decoder(module, args, value):
        index = len(receipt['decoder_inputs'])
        require(index < 2, 'One positive and one negative native decoder call required')
        receipt['decoder_inputs'].append(observe(lambda: observed.tensor_identity(args[2])))
        channels['positive_raw_logits' if index == 0 else 'negative_raw_logits'] = observe(lambda: native.clone(value, torch))
        progress.add(root_decoder_completed=1, root_query_rows_completed=int(args[2].shape[0]))
    original_backward = torch.Tensor.backward
    def backward(tensor, *args, **kwargs):
        require(not args and not kwargs and receipt['backward_calls'] == 0 and len(prestate) == 1
                and len(receipt['decoder_inputs']) == 2 and tensor.numel() == 1, 'Original backward invocation differs')
        channels['loss'] = observe(lambda: native.clone(tensor, torch))
        receipt['backward_calls'] += 1
        progress.add(gradient_backward_calls=1)
        original_backward(tensor, *args, **kwargs)
        receipt['backward_completed'] += 1
        channels['named_preAdam_gradients'] = observe(lambda: native.capture(unit, torch, np)['gradients'])
        raise PrefixComplete()
    handles = [model.register_forward_pre_hook(before_encoder), model.register_forward_hook(after_encoder),
               predictor.decoder.register_forward_pre_hook(before_decoder), predictor.decoder.register_forward_hook(after_decoder)]
    scope = dict(original.__globals__)
    scope.update(negative_sampling=sample, PermIterator=FirstIterator)
    measured = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    require(measured.__code__ is original.__code__, 'Original training code object changed')
    require(progress.snapshot()['prefixes_started'] < 4, 'Frozen four-prefix allowance exhausted')
    progress.add(prefixes_started=1)
    try:
        torch.Tensor.backward = backward
        try:
            measured(model, predictor, data, {'train': {'edge': train}}, optimizer, 1024, True, [], None)
            raise RuntimeError('Native body returned without prospective backward sentinel')
        except PrefixComplete:
            require(receipt['backward_calls'] == receipt['backward_completed'] == 1, 'Sentinel without completed original backward')
    finally:
        require(torch.Tensor.backward is backward, 'Backward observer unexpectedly replaced')
        torch.Tensor.backward = original_backward
        for handle in handles:
            handle.remove()
    progress.add(prefixes_completed=1)
    require(len(receipt['negative_calls']) == len(receipt['iterator_calls']) == len(receipt['batches']) == 1,
            'Exactly one first native batch/draw/iterator required')
    order = receipt['iterator_calls'][0]
    require(order['native_length'] == 36 and order['training'] is True and order['batch_size'] == 1024
            and order['full_order']['shape'] == [37676] and order['dropped_tail']['shape'] == [812], 'Original order/tail changed')
    receipt.update(start_RNG_sha256=observed.state_identity(start, torch),
                   end_RNG_sha256=observed.state_identity(native.rng(torch, np, True), torch),
                   original_body_reused=True, native_sampler_defaults_unchanged=True,
                   full_batches=1, complete_native_epoch=False, optimizer_step_calls=0,
                   stopped_after_original_backward_before_optimizer_call=True)
    return channels, prestate[0], receipt
