"""Fixed18 wrapper around the existing immutable directly waited owner."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    script=Path(__file__).resolve().parent/'run.py'
    loader=importlib.util.spec_from_file_location('_fixed18_entry',script)
    entry=importlib.util.module_from_spec(loader);sys.modules[loader.name]=entry;loader.loader.exec_module(entry)
    route=entry.qualification_surface();route.route(owner=True)
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();path=args.plan.resolve(strict=True)
    if not path.is_relative_to(entry.HERE) or entry.sha(path)!=args.plan_sha256:raise ValueError('Exact new owner plan required')
    plan=json.loads(path.read_text())
    if (plan.get('enabled') is not True or plan.get('automatic_retry') is not False or
            plan.get('purpose')!='science' or plan.get('owner_sha256')!=entry.sha(__file__)):
        raise ValueError('Disabled full-study owner plan')
    expected=[f'seed{s}__{c}' for s in entry.SEEDS for c in entry.CONDITIONS]
    if [row['record_id'] for row in plan['records']]!=expected:raise ValueError('All fixed18 records required in declared order')
    if plan['source_manifest_sha256']!=entry.sha(entry.HERE/'SOURCE_MANIFEST.json'):
        raise ValueError('Exact current full-study source required')
    for row in plan['records']:
        if (row['owner_id']!='science__'+row['record_id'] or
                row['entrypoint']!=dict(path=entry.HERE.name+'/run.py',sha256=entry.sha(script)) or
                row['entry_args']!=['--mode','science']):
            raise ValueError('Exact fixed full-fit worker and owner identity required')
    owner=entry.load('_fixed18_existing_owner',entry.PHASE/'pubmed_factor1_controls_source_20261010_v1/queue.py')
    if entry.sha(owner.__file__)!=plan['reused_owner_source_sha256']:raise ValueError('Existing owner source changed')
    owner.HERE=entry.HERE;plan['_completed']=[];family=entry.HERE/'science';family.mkdir(exist_ok=True)
    for row in plan['records']:
        if row['entry_args']!=['--mode','science']:raise ValueError('Full science mode only')
        if not owner.own_one(row,plan):
            owner.write(family/'FAMILY_FAILURE.json',dict(complete=False,failed_record=row['record_id'],completed=plan['_completed'],automatic_retry=False))
            raise RuntimeError('Retained full-fit/owner failure; no retry or partial comparison')
        plan['_completed'].append(row['record_id'])
    owner.write(family/'FAMILY_COMPLETE.json',dict(complete=True,records=plan['_completed'],source_manifest_sha256=plan['source_manifest_sha256'],
                all_records_directly_waited=True,TEST_access=False,partial_family_comparison_allowed=False))


if __name__ == '__main__':
    main()
