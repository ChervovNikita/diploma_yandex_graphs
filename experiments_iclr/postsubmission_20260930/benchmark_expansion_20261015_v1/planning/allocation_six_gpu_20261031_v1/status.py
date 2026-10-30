"""Read all six shared allocation shards without interpreting foreign PIDs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from host_extension import bind, SHARDS
from partition import original_plan, validate_partition

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, choices=SHARDS)
    args = parser.parse_args()
    bind(args.worker)
    plan = original_plan()
    if not (HERE / 'PARTITION.json').exists():
        print(json.dumps({'phase': 'preparing_immutable_partition'})); return
    partition = validate_partition(json.loads((HERE / 'PARTITION.json').read_text()), plan)
    import hashlib
    root = Path(plan['state_root'])
    workers = {}
    combined = Counter()
    for shard, ids in partition['assignment'].items():
        counts = Counter(); active = []
        for key in ids:
            path = root / 'full' / hashlib.sha256(key.encode()).hexdigest() / 'STATE.json'
            state = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
            counts[state['status']] += 1
            if state['status'] == 'running':
                active.append(key)
        combined.update(counts)
        control = HERE / 'runs' / shard.replace('/', '_') / 'STATUS.json'
        owner = json.loads(control.read_text()) if control.exists() else None
        workers[shard] = {'counts': dict(counts), 'active_fit_ids': active,
            'phase': owner['phase'] if owner else 'waiting_for_custody_or_not_started',
            'last_status_age_seconds': time.time() - owner['updated'] if owner else None}
    print(json.dumps({'total_tasks': len(plan['tasks']), 'counts': dict(combined), 'workers': workers,
                      'original_owner_status_is_historical': True, 'server51_touched': False}, indent=2))


if __name__ == '__main__':
    main()
