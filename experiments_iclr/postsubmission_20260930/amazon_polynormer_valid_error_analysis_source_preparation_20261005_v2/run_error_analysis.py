"""Root-only future descriptive selected-VALID entry point. This source grants no execution."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    body = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


def authenticate(base, row):
    p = (base/row['path']).resolve()
    require(p.is_relative_to(base) and not any(q.is_symlink() for q in (base/row['path'], *(base/row['path']).parents) if q.is_relative_to(base)), 'Source confinement/symlink violation')
    require(descriptor(p) == {'sha256': row['sha256'], 'bytes': row['bytes']}, 'Changed source: '+row['path'])
    return p


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--closure-freeze', required=True)
    parser.add_argument('--closure-sha256', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--source-seal-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute-authorized-valid-error-analysis', action='store_true')
    args = parser.parse_args()
    require(args.execute_authorized_valid_error_analysis, 'Root admission after source review is required')
    phase = Path(args.phase).resolve()
    packet = Path(__file__).resolve().parent
    require(packet.is_relative_to(phase), 'Source packet must be phase-confined')
    require(descriptor(packet/'SEAL.json')['sha256'] == args.source_seal_sha256, 'Root-bound source seal differs')
    seal = json.loads((packet/'SEAL.json').read_text())
    own_manifest = authenticate(packet, seal['manifest'])
    for row in json.loads(own_manifest.read_text())['payload']:
        authenticate(packet, row)
    bindings = json.loads((packet/'SOURCE_BINDINGS.json').read_text())
    v2 = phase/bindings['v2_source_directory']
    require(v2.is_relative_to(phase) and not v2.is_symlink(), 'V2 source must be phase-confined')
    for row in bindings['v2_source_authentication']:
        authenticate(phase, row)
    # Only stdlib custody is imported before prepare authenticates scientific inputs.
    custody = module('_amazon_valid_error_custody_v2', v2/'custody.py')
    closure = custody.record(phase, args.closure_freeze)
    protocol = custody.record(phase, args.protocol)
    require(closure['sha256'] == args.closure_sha256, 'Root-bound complete closure differs')
    require(protocol['sha256'] == args.protocol_sha256 and protocol == bindings['protocol'], 'Exact reviewed V2 protocol differs')
    output = custody.confined(phase, args.output)
    require(not output.exists(), 'Fresh phase-local output required')
    output.mkdir()
    try:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
            os.environ[key] = '1'
        started = time.perf_counter()
        state = custody.prepare(phase, closure, protocol)
        state['custody_cost'] = {'wall_seconds': time.perf_counter()-started,
                                'bound_payload_bytes_hashed': state['input_bytes']}
        with (output/'RUN_BINDINGS.json').open('x') as stream:
            json.dump({'source_seal_sha256': args.source_seal_sha256,
                       'v2_source_authentication': bindings['v2_source_authentication'],
                       'closure': closure, 'protocol': protocol,
                       'retrospective_descriptive_VALID_only': True}, stream, indent=2, allow_nan=False)
            stream.write('\n')
        metrics = module('_amazon_valid_error_metrics', packet/'metrics.py')
        metrics.run(state, output, custody)
    except Exception as exc:
        with (output/'FAILURE.json').open('x') as stream:
            json.dump({'status': 'failed_incomplete', 'exception_type': type(exc).__name__,
                       'message': str(exc), 'complete_analysis': False}, stream, indent=2, allow_nan=False)
            stream.write('\n')
        raise


if __name__ == '__main__':
    main()
