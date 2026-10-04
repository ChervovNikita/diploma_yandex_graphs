"""Reused complete preforward cache/storage inspection; no prefix runner."""
from common import require
from storage_aliases import canonical_storage_aliases


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
