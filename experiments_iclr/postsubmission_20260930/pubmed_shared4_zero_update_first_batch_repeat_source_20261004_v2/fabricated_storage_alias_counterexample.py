"""Stdlib metadata fixture for the exact omitted-storage counterexample.

No Torch, model, feature, scientific array, server or process probe is used.
Only the pure canonical_storage_aliases metadata helper is exercised.
"""
import json
from storage_aliases import canonical_storage_aliases


def run():
    # Two different tensor objects have identical values and tensor schema.
    # This is precisely the v1 tensor-object/value/layout representation:
    # neither b is the same Python object as a in either fixture.
    value = {'values': [1.0, 2.0, 3.0, 4.0], 'shape': [4], 'dtype': 'float32',
             'stride': [1], 'storage_offset': 0, 'requires_grad': False}
    shared_v1 = {'a': dict(value), 'b': dict(value)}
    independent_v1 = {'a': dict(value), 'b': dict(value)}
    assert shared_v1['a'] is not shared_v1['b'] and independent_v1['a'] is not independent_v1['b']
    assert shared_v1 == independent_v1  # v1 cannot distinguish these states.
    def row(path, handle, start, size=16, device='cpu'):
        return dict(path=path, handle=handle, start=start, bytes=size, device=device)
    shared = canonical_storage_aliases([row('a', 17, 4096), row('b', 17, 4096)])
    independent = canonical_storage_aliases([row('a', 26, 8192), row('b', 27, 12288)])
    assert shared['fully_inspected'] and independent['fully_inspected']
    assert shared['classes']['a']['storage_alias_class'] == shared['classes']['b']['storage_alias_class']
    assert independent['classes']['a']['storage_alias_class'] != independent['classes']['b']['storage_alias_class']
    assert shared['classes'] != independent['classes']  # v2 detects the omitted relation.
    moved_shared = canonical_storage_aliases([row('b', 901, 1048576), row('a', 901, 1048576)])
    moved_independent = canonical_storage_aliases([row('b', 993, 2097216), row('a', 992, 2097152)])
    assert moved_shared == shared and moved_independent == independent
    # Different StorageImpl wrappers can still overlap the same backing bytes.
    overlapping = canonical_storage_aliases([row('a', 31, 16384), row('b', 32, 16392)])
    assert overlapping['classes']['a']['storage_alias_class'] == overlapping['classes']['b']['storage_alias_class']
    assert overlapping['classes']['b']['storage_start_relative_to_class_bytes'] == 8
    empty_shared = canonical_storage_aliases([row('a', 40, 0, 0), row('b', 40, 0, 0)])
    empty_independent = canonical_storage_aliases([row('a', 41, 0, 0), row('b', 42, 0, 0)])
    assert empty_shared['classes'] != empty_independent['classes']
    opaque = canonical_storage_aliases([dict(path='a', device='cpu', handle=None, start=None, bytes=16)])
    assert not opaque['fully_inspected'] and opaque['unsupported'] and not opaque['classes']
    inconsistent = canonical_storage_aliases([row('a', 51, 32768), row('b', 51, 32776)])
    assert not inconsistent['fully_inspected'] and len(inconsistent['unsupported']) == 2 and not inconsistent['classes']
    return {'status': 'PASS_PURE_STDLIB_FABRICATED_STORAGE_METADATA_ONLY',
            'v1_equal_value_shape_stride_offset_states_indistinguishable': True,
            'distinct_tensor_objects_shared_vs_independent_storage_detected': True,
            'canonical_output_invariant_to_raw_pointer_handle_relocation_and_row_order': True,
            'overlapping_distinct_storage_wrappers_detected': True,
            'empty_views_use_storage_identity_without_conflating_null_pointers': True,
            'opaque_or_inconsistent_storage_cannot_qualify': True,
            'shared_storage_canonical_classes': shared['classes'],
            'independent_storage_canonical_classes': independent['classes'],
            'numerical_library_imported': False, 'numerical_scientific_execution': False,
            'worker_prefix_or_target_dependency_import_compile_execution': False}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
