"""Small binding to the existing directly waited finite owner; disabled plan."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    script = Path(__file__).resolve().parent/'run.py'
    loader = importlib.util.spec_from_file_location('_private_hop_qualification_entry',script)
    entry = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = entry
    loader.loader.exec_module(entry)
    entry.route(owner=True)
    sha,HERE,PHASE,RECORD = entry.sha,entry.HERE,entry.PHASE,entry.RECORD
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args = parser.parse_args()
    path = args.plan.resolve(strict=True)
    if not path.is_relative_to(HERE) or sha(path) != args.plan_sha256:
        raise ValueError('Exact root-reviewed disabled/adopted owner plan required')
    plan = json.loads(path.read_text())
    if (plan.get('enabled') is not True or plan.get('automatic_retry') is not False or
            plan.get('purpose') != 'engineering' or plan.get('owner_sha256') != sha(__file__)):
        raise ValueError('Inactive owner admission')
    if len(plan['records']) != 1 or plan['records'][0]['record_id'] != RECORD:
        raise ValueError('One finite five-condition qualification worker required')
    row = plan['records'][0]
    if row['entry_args'] != ['--mode','engineering']:
        raise ValueError('No scientific mode in the qualification entry')
    owner = entry.load('_private_hop_existing_directly_waited_owner',PHASE/'pubmed_factor1_controls_source_20261010_v1/queue.py')
    if sha(owner.__file__) != plan['reused_owner_source_sha256']:
        raise ValueError('Existing finite owner source changed')
    owner.HERE = HERE  # Only root confinement changes; lifecycle/caps stay exact.
    if not owner.own_one(row,plan):
        raise RuntimeError('Qualification owner or child failed; no automatic retry')


if __name__ == '__main__':
    main()
