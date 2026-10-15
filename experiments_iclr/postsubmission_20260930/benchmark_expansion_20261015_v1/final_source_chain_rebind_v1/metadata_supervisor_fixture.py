"""Engineering subprocess only; the production CLI never accepts this entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True); parser.add_argument('--run-spec', required=True)
    parser.add_argument('--run-spec-sha256', required=True)
    args = parser.parse_args()
    raw = Path(args.run_spec).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.run_spec_sha256:
        raise ValueError('Exact engineering fixture spec required')
    spec = json.loads(raw)
    if spec['schema'] != 'metadata-only-supervisor-engineering-fixture-v1' or args.worker != 'allocation/gpu0':
        raise ValueError('Not a scientific training/final supervisor')
    output, root = Path(spec['output']).resolve(), Path(spec['root']).resolve()
    if not output.is_relative_to(root) or spec['case'] not in ('success', 'failure', 'briefly_live'):
        raise ValueError('Scoped fixed engineering fixture required')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'ENGINEERING_CHILD.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'case': spec['case'], 'scientific_fits': 0, 'TEST_accessed': False}, stream)
    if spec['case'] == 'briefly_live':
        time.sleep(2)
    print(json.dumps({'engineering_case': spec['case'], 'models': 0, 'optimizer_updates': 0}))
    raise SystemExit(2 if spec['case'] == 'failure' else 0)


if __name__ == '__main__':
    main()
