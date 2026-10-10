"""Disabled minimal rendering for the existing CMCL complete-task API."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from run import HERE,PHASE,REPO,SOURCE,SOURCE_SHA,GPU,CONDITIONS,LIMITS,ENGINEERING_LIMITS,sha,bind,inside,require_terminal


def descriptor(path):
    path=inside(path);return dict(path=str(path),sha256=sha(path))


def save(path,value):
    path=inside(path,existing=False)
    if not path.is_relative_to(HERE) or path.exists():raise ValueError('Fresh root-confined rendered artifact only')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def context(args):
    if socket.gethostname()!='anogena-2-0' or Path.cwd().resolve()!=REPO or not HERE.is_relative_to(PHASE):raise ValueError('Exact allocation cwd required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:raise ValueError('Exact sole allocation GPU required')
    review=descriptor(args.source_review);v=json.loads(bind(review).read_text());adapter_sha=sha(HERE/'SOURCE_MANIFEST.json')
    if v.get('approved') is not True or v.get('source_delta_assessment_approved') is not True or v.get('source_manifest_sha256')!=SOURCE_SHA or v.get('adapter_manifest_sha256')!=adapter_sha:
        raise ValueError('Root must approve exact source and API adapter')
    owner=descriptor(args.owner_review);v=json.loads(bind(owner).read_text())
    if v.get('approved') is not True or v.get('owner_sha256')!=sha(HERE/'queue.py'):raise ValueError('Exact owner successor review required')
    ready=descriptor(args.readiness);v=json.loads(bind(ready).read_text());inventory=v.get('GPU_inventory','').split(',')
    if v.get('normal_host_execution') is not True or len(inventory)<2 or inventory[0].strip()!=GPU or int(inventory[1])<32768:raise ValueError('Actual normal-host32GiB readiness required')
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    if len(free)!=1 or int(free[0])<32768:raise ValueError('Current GPU readiness below frozen capacity')
    return review,owner,ready,adapter_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--purpose',choices=('engineering','science'),required=True)
    p.add_argument('--source-review',type=Path,default=HERE/'ROOT_SOURCE_REVIEW.json')
    p.add_argument('--owner-review',type=Path,default=HERE/'OWNER_REVIEW.json')
    p.add_argument('--readiness',type=Path,default=HERE/'READINESS.json')
    args=p.parse_args();review,owner_review,readiness,adapter_sha=context(args)
    adoption=None
    if args.purpose=='science':
        owner_plan=json.loads((HERE/'ENGINEERING_OWNER_PLAN.json').read_text())
        if owner_plan.get('source_manifest_sha256')!=SOURCE_SHA or owner_plan.get('owner_sha256')!=sha(HERE/'queue.py') or owner_plan.get('purpose')!='engineering' or [r['record_id'] for r in owner_plan['records']]!=['seed9101__'+c for c in CONDITIONS]:
            raise ValueError('Exact six-condition engineering owner plan required')
        qualifications={}
        for condition,row in zip(CONDITIONS,owner_plan['records']):
            raw=descriptor(HERE/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json');require_terminal(raw,row['release_sha256'])
            qualifications[condition]=dict(raw_terminal=raw)
        for condition,row in zip(CONDITIONS,owner_plan['records']):
            complete=descriptor(HERE/'engineering'/'cells'/row['record_id']/'COMPLETE.json');result=json.loads(bind(complete).read_text())
            if result.get('schema')!='private-CMCL-engineering-complete-v1' or result.get('complete') is not True or result.get('condition')!=condition or result.get('source_manifest_sha256')!=SOURCE_SHA or result.get('adapter_manifest_sha256')!=adapter_sha or result.get('release_sha256')!=row['release_sha256'] or result.get('updates')!=1 or result.get('fresh_reconstruction_verified') is not True or result.get('VALID_access') is not True or result.get('science_enabled') is not False:
                raise ValueError('Every actual full-input update/reconstruction must pass')
            qualifications[condition]['complete']=complete
        value=dict(schema='private-CMCL-full18-root-adoption-v1',enabled=True,source_manifest_sha256=SOURCE_SHA,
                   adapter_manifest_sha256=adapter_sha,roster_sha256=sha(HERE/'ROSTER.json'),
                   source_review=review,source_delta_assessment=review,qualifications=qualifications,
                   post_screen_exploration=True,CORE=False,further18_activated=False,TEST_access=False,paper_score_recalculation=False)
        path=HERE/'ROOT_ADOPTION.json';save(path,value);adoption=descriptor(path)
    frozen=json.loads((HERE/'ROSTER.json').read_text())
    selected=[r for r in frozen if args.purpose=='science' or r['seed']==9101]
    records=[]
    for row in selected:
        record=row['record_id'];owner_id=args.purpose+'__'+record
        spec=json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
        spec.update(row,purpose=args.purpose,schema='private-CMCL-'+args.purpose+'-release-v1',enabled=True,
            root_authorized=True,source_review_approved=True,source_delta_assessment_approved=True,
            complete_input_custody_verified=True,VALID_custody_verified=True,external_hard_bound_confirmed=True,
            fresh_resource_readiness_confirmed=True,ordinary_runtime_confirmed=True,
            science_enabled=args.purpose=='science',adapter_manifest_sha256=adapter_sha,
            source_review=review,source_delta_assessment=review,resource_readiness_evidence=readiness,
            limits=ENGINEERING_LIMITS if args.purpose=='engineering' else LIMITS,
            output=str(HERE/args.purpose/'cells'/record))
        if adoption is not None:spec['root_admission']=adoption
        contract=dict(schema='private-CMCL-finite-owner-v1',enabled=True,record_id=record,limits=spec['limits'],
                      separate_process_group=True,direct_wait_required=True,resource_caps_enforced=True,
                      output_and_log_caps_enforced=True,automatic_retry=False,owner_source=descriptor(HERE/'queue.py'),owner_review=owner_review)
        path=HERE/'contracts'/(owner_id+'.json');save(path,contract);spec['external_owner_release']=descriptor(path)
        path=HERE/'releases'/(owner_id+'.json');save(path,spec)
        records.append(dict(record_id=record,owner_id=owner_id,release=str(path.relative_to(PHASE)),release_sha256=sha(path),
                            limits=spec['limits'],entrypoint=descriptor(HERE/'run.py'),entry_args=['--mode',args.purpose]))
    value=dict(schema='private-CMCL-finite-owner-plan-v1',enabled=True,purpose=args.purpose,automatic_retry=False,
               owner_sha256=sha(HERE/'queue.py'),source_manifest_sha256=SOURCE_SHA,adapter_manifest_sha256=adapter_sha,
               roster_sha256=sha(HERE/'ROSTER.json'),source_review=review,owner_review=owner_review,
               readiness=readiness,records=records,CORE=False,further18_activated=False,TEST_access=False,paper_score_recalculation=False)
    path=HERE/(args.purpose.upper()+'_OWNER_PLAN.json');save(path,value)
    print(json.dumps(dict(owner_plan=descriptor(path),rendered_cells=len(records),nothing_started=True)))


if __name__=='__main__':main()
