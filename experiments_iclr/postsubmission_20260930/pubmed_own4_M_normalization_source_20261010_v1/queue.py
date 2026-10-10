"""Small fixed-stage wrapper; the original own_one owns every process/cap/wait."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    here=Path(__file__).resolve().parent;loader=importlib.util.spec_from_file_location('_normalization_entry',here/'run.py')
    entry=importlib.util.module_from_spec(loader);sys.modules[loader.name]=entry;loader.loader.exec_module(entry)
    full=entry.surface();full.qualification_surface().route(owner=True)
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);args=p.parse_args()
    path=args.plan.resolve(strict=True)
    if not path.is_relative_to(here) or full.sha(path)!=args.plan_sha256:raise ValueError('Exact new root owner plan required')
    plan=json.loads(path.read_text());mode=plan['purpose']
    if mode not in ('qualification','science','comparison') or plan.get('enabled') is not True or plan.get('automatic_retry') is not False or plan.get('owner_sha256')!=full.sha(__file__) or plan.get('source_manifest_sha256')!=full.sha(here/'SOURCE_MANIFEST.json') or [r['record_id'] for r in plan['records']]!=entry.record_ids(mode):
        raise ValueError('Disabled or changed fixed-stage owner plan')
    owner=full.load('_normalization_original_owner',here.parent/'pubmed_factor1_controls_source_20261010_v1/queue.py')
    if full.sha(owner.__file__)!=plan['reused_owner_source_sha256']:raise ValueError('Original finite owner changed')
    owner.HERE=here;plan['_completed']=[];family=here/mode;family.mkdir(exist_ok=True)
    for row in plan['records']:
        if row['owner_id']!=mode+'__'+row['record_id'] or row['entry_args']!=['--mode',mode] or row['entrypoint']!=dict(path=here.name+'/run.py',sha256=full.sha(here/'run.py')):
            raise ValueError('Exact worker and directly waited owner identity required')
        if not owner.own_one(row,plan):
            owner.write(family/'FAMILY_FAILURE.json',dict(complete=False,failed_record=row['record_id'],completed=plan['_completed'],automatic_retry=False));raise RuntimeError('Retained owner/worker failure; no retry')
        plan['_completed'].append(row['record_id'])
    owner.write(family/'FAMILY_COMPLETE.json',dict(complete=True,records=plan['_completed'],all_records_directly_waited=True,source_manifest_sha256=plan['source_manifest_sha256'],automatic_retry=False,partial_comparison_allowed=False))


if __name__=='__main__':main()
