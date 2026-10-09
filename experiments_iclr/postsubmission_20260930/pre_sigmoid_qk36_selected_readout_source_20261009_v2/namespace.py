"""Explicit POSIX origin namespace to serving-repository mapping; stdlib only.

Original paths are compared lexically against the immutable route catalog.
They are never resolved/accessed on a different host or rewritten in receipts.
Only the separately specified phase-relative artifact is read locally.
"""
from pathlib import PurePosixPath


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError('Explicit phase-relative artifact path required')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) in ('', '.') or str(path) != value:
        raise ValueError('Canonical confined phase-relative artifact required')
    return path


def origin_path(pins, route_id, relative):
    phase = PurePosixPath(pins['routes'][route_id]['phase'])
    if not phase.is_absolute() or '..' in phase.parts:
        raise ValueError('Immutable absolute original route phase required')
    return phase / relative_path(relative)


def mapped(g, pins, route_id, recorded_absolute, expected_relative):
    expected = origin_path(pins, route_id, expected_relative)
    if not isinstance(recorded_absolute, str) or recorded_absolute != expected.as_posix():
        raise ValueError('Original route-qualified namespace mismatch: ' + route_id + ' / ' + expected_relative)
    # Do not Path(recorded_absolute).resolve(): it belongs to its original host.
    return g.inside(expected_relative)


def expected_namespaces(pins, serving_route_id):
    return dict(serving_route_id=serving_route_id, serving_phase=pins['routes'][serving_route_id]['phase'],
        routes={route_id: dict(original_phase=route['phase'], original_hostname=route['hostname'],
            original_GPU_uuid=route['GPU_uuid'], original_output_prefix=str(origin_path(pins, route_id,
                pins['plan']['fresh_output_relative'] + '/' + route_id)),
            original_release_path=str(origin_path(pins, route_id, pins['training_releases'][route_id]['path'])),
            staged_output_relative=pins['plan']['fresh_output_relative'] + '/' + route_id,
            staged_release=pins['training_releases'][route_id]) for route_id, route in pins['routes'].items()})


def verify_custody(g, pins, cfg):
    row = g.read(g.bound(cfg['namespace_custody'])); expected = expected_namespaces(pins, cfg['readout_route_id'])
    g.require(row['schema'] == 'qk36-original-route-namespace-custody-v2' and row['complete'] is True
        and row['root_verified_original_namespaces'] is True and row['plan'] == pins['global_plan']
        and row['namespace_mapping'] == expected and row['original_receipt_bytes_preserved'] is True
        and row['source_files_or_receipts_rewritten'] is False and row['mounts_or_symlinks_created_by_mapping'] is False,
        'Explicit verified original route namespaces; exact byte-preserved central staging')
    return row
