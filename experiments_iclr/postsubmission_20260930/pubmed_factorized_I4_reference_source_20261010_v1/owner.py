"""Stage adapter around the byte-identical M1 own_one finite owner."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE,PHASE,LIMITS,bind,descriptor,frozen,inside,load,raw_wait,read,records,route,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);args=p.parse_args()
    route();path=inside(args.plan)
    if path.parent!=HERE or sha(path)!=args.plan_sha256:raise ValueError('Exact root-owned stage plan required')
    plan=read(dict(path=str(path),sha256=args.plan_sha256));b,data,roster,reuse,manifest_sha=frozen();stage=plan.get('purpose')
    if stage not in LIMITS or path.name!=stage.upper()+'_OWNER_PLAN.json':raise ValueError('Fixed five-stage plan required')
    expected=[r['record_id'] for r in records(stage,roster)]
    if plan.get('enabled') is not True or plan.get('automatic_retry') is not False or plan.get('owner_sha256')!=sha(HERE/'owner.py') or plan.get('source_manifest_sha256')!=manifest_sha or plan.get('roster_sha256')!=sha(HERE/'ROSTER.json') or [r['record_id'] for r in plan['records']]!=expected:raise ValueError('Fixed entire stage authority required')
    review=read(plan['source_review']);oreview=read(plan['owner_review'])
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=manifest_sha or oreview.get('approved') is not True or oreview.get('source_manifest_sha256')!=manifest_sha or oreview.get('owner_sha256')!=plan['owner_sha256'] or oreview.get('existing_owner_source_sha256')!=b['M1_owner_source']['sha256']:raise ValueError('Root source and unchanged owner approval required')
    bind(plan['readiness'])
    for row in plan['records']:
        if row['owner_id']!=stage+'__'+row['record_id'] or row['entry_args']!=['--mode',stage] or row['limits']!=LIMITS[stage] or bind(row['entrypoint'])!=HERE/'run.py':raise ValueError('Fixed worker command required')
        release=read(dict(path=str(PHASE/row['release']),sha256=row['release_sha256']))
        if release.get('purpose')!=stage or release.get('record_id')!=row['record_id'] or release.get('output')!=str(HERE/stage/'cells'/row['record_id']):raise ValueError('Fixed stage output required')
    old=load('_factorized_I4_unchanged_M1_owner',bind(b['M1_owner_source']))
    # The original owner function, route constants, process/cap/wait/cleanup
    # implementation and worker argv remain unchanged. Only its output root moves.
    old.HERE=HERE
    if old.PHASE!=PHASE:raise ValueError('Original allocation owner phase required')
    family=HERE/stage;family.mkdir(exist_ok=True)
    if any((family/n).exists() for n in ('FAMILY_START.json','FAMILY_COMPLETE.json','FAMILY_FAILURE.json')):raise ValueError('No rerun, retry or survivor continuation')
    write(family/'FAMILY_START.json',dict(owner=old.identity(os.getpid()),plan_sha256=args.plan_sha256,source_manifest_sha256=manifest_sha,UTC=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    plan['_completed']=[]
    for row in plan['records']:
        try:
            if not old.own_one(row,plan):raise RuntimeError('Finite owner did not close successfully')
            raw_wait(descriptor(HERE/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json'),row['release_sha256'])
        except BaseException as error:
            write(family/'FAMILY_FAILURE.json',dict(failed_record=row['record_id'],completed=plan['_completed'],error_type=type(error).__name__,error=str(error),plan_sha256=args.plan_sha256,source_manifest_sha256=manifest_sha,automatic_retry=False,no_partial_comparison=True,fresh12_fallback=False,quality_fields_opened=False))
            raise
        plan['_completed'].append(row['record_id'])
    write(family/'FAMILY_COMPLETE.json',dict(complete=True,records=plan['_completed'],all_records_directly_waited=True,purpose=stage,plan_sha256=args.plan_sha256,source_manifest_sha256=manifest_sha,automatic_retry=False,quality_fields_opened=False,comparison_pending=stage!='comparison'))


if __name__=='__main__':main()
