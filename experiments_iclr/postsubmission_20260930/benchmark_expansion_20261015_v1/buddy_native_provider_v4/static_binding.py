"""Exact integer lineage and numerical validation of native HLL cardinalities.

The historical seed0 card hash remains provenance. The local reference uses
the original source algorithm and authenticated integer registers. No tensor
is substituted, and validation does not perform model or optimizer work.
"""
import copy
import hashlib
from pathlib import Path

PHASE = Path(__file__).resolve().parents[2]
SOURCE = PHASE / 'reference_wrapper_redesign_20261013_v1/native_sources/buddy/buddy_src_hashing.py'
SOURCE_SHA256 = 'fa6a7283ff50c042a8c23dd84e8f7aafdbed6e0364987046f5b1481f5426c134'
RTOL = 1e-6
ATOL = 1e-6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def validate_static_inputs(torch, native, observed, expected):
    require(set(observed) == set(expected) == {'features', 'degrees', 'cards', 'source_positive_queries', 'hash_tables'},
            'Exact static input roles required')
    require({key: value for key, value in observed.items() if key != 'cards'} ==
            {key: value for key, value in expected.items() if key != 'cards'},
            'Native features/degrees/queries/integer hash lineage differs')
    actual = native.cards
    require(observed['cards'] == tensor(actual) and
            {key: observed['cards'][key] for key in ('shape', 'dtype')} ==
            {key: expected['cards'][key] for key in ('shape', 'dtype')} ==
            {'shape': [235868, 2], 'dtype': 'torch.float32'}, 'Native cardinality shape/dtype/receipt differs')
    require(bool(torch.isfinite(actual).all()), 'Nonfinite native cardinalities')
    function = getattr(native.hashes.hll_count, '__func__', None)
    require(function is not None and Path(function.__code__.co_filename).resolve() == SOURCE,
            'Use the original pinned native HLL cardinality function')
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, 'Native HLL source differs')
    require(native.hashes.p == 8 and native.hashes.max_hops == 2, 'Frozen HLL parameters required')
    with torch.no_grad():
        reference = torch.stack([native.hashes.hll_count(native.hash_tables[hop]['hll']) for hop in (1, 2)], dim=1)
    require(reference.shape == actual.shape and reference.dtype == actual.dtype and
            bool(torch.isfinite(reference).all()), 'Finite native reference shape/dtype required')
    require(bool(torch.allclose(actual, reference, rtol=RTOL, atol=ATOL)),
            'Native cardinalities differ numerically from their source/register reference')
    difference = (actual.double() - reference.double()).abs()
    return {'schema': 'BUDDY-native-floating-cardinality-validation-v1',
            'canonical_cards_provenance': copy.deepcopy(expected['cards']),
            'actual_cards': copy.deepcopy(observed['cards']), 'local_native_reference': tensor(reference),
            'reference_source': {'path': str(SOURCE), 'bytes': len(raw), 'sha256': SOURCE_SHA256},
            'integer_hash_inputs': copy.deepcopy(observed['hash_tables']),
            'rtol': RTOL, 'atol': ATOL, 'maximum_absolute_difference': float(difference.max()),
            'maximum_relative_difference': float((difference / reference.double().abs().clamp_min(1e-30)).max()),
            'cross_host_float_bitwise_equality_required': False, 'native_card_tensor_replaced': False,
            'model_forwards': 0, 'optimizer_updates': 0, 'TEST_accessed': False}
