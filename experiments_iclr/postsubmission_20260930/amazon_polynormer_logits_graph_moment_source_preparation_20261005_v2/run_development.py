"""Explicit entry point for a later source-reviewed authorized development run.

This source packet itself grants no execution or final-label access. No refit or
TEST CLI exists. Metadata/array custody is checked before numerical imports.
"""
import os
import time
import argparse
from pathlib import Path
from custody import require, record, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--closure-freeze', required=True)
    parser.add_argument('--closure-sha256', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute-authorized-development', action='store_true')
    args = parser.parse_args()
    require(args.execute_authorized_development, 'Source review and external root admission must precede execution')
    phase = Path(args.phase).resolve()
    closure = record(phase, args.closure_freeze)
    protocol = record(phase, args.protocol)
    require(closure['sha256'] == args.closure_sha256 and protocol['sha256'] == args.protocol_sha256,
            'Exact root-bound closure/amended protocol hashes required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    before = time.perf_counter()
    state = prepare(phase, closure, protocol)
    state['custody_cost'] = {'wall_seconds': time.perf_counter()-before,
                             'bound_payload_bytes_hashed':state['input_bytes']}
    output = Path(args.output)
    output = output if output.is_absolute() else phase/output
    require(output.is_relative_to(phase) and not output.exists(), 'Fresh phase-local output directory required')
    from study import run
    run(state, output)


if __name__ == '__main__':
    main()
