"""Bounded proof that task custody locks are visible across the shared disk."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from host_extension import bind

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    parser.add_argument('--hold', type=int)
    args = parser.parse_args()
    bind(args.worker)
    with (HERE / 'SHARED_FLOCK_PROBE.lock').open('a+') as handle:
        if args.hold:
            assert 1 <= args.hold <= 45
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('SHARED_FLOCK_HELD', flush=True)
            time.sleep(args.hold)
        else:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result = {'status': 'PASS', 'worker': args.worker,
                          'cross_host_existing_lock_observed': True, 'time': time.time()}
                (HERE / 'SHARED_FLOCK_PASS.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(result)); return
            raise ValueError('Cross-host lock was not visible; do not launch shared queues')


if __name__ == '__main__':
    main()
