"""Minimal disabled rendering: two TRAIN qualifiers, then six M1 full fits."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
from run import HERE,PHASE,REPO,GPU,CONDITIONS,LIMITS,ENGINEERING_LIMITS,sha,bind,inside,terminal


def descriptor(path):
    path=inside(path);return dict(path=str(path),sha256=sha(path))


def save(path,value):
    path=inside(path,existing=False)
    if not path.is_relative_to(HERE) or path.exists():raise ValueError('Fresh root-confined artifact required')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--purpose',choices=('engineering','science'),required=True)
    p.add_argument('--source-review',type=Path,default=HERE/'ROOT_SOURCE_REVIEW.json')
    p.add_argument('--owner-review',type=Path,default=HERE/'OWNER_REVIEW.json')
    p.add_argument('--readiness',type=Path,default=HERE/'READINESS.json')
    args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or Path.cwd().resolve()!=REPO or not HERE.is_relative_to(PHASE):raise ValueError('Exact allocation route/cwd required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:raise ValueError('Exact sole GPU required')
    source_sha=sha(HERE/'SOURCE_MANIFEST.json');review=descriptor(args.source_review);v=json.loads(bind(review).read_text())
    if v.get('approved') is not True or v.get('source_delta_assessment_approved') is not True or v.get('source_manifest_sha256')!=source_sha:raise ValueError('Root exact-source review required')
    owner_review=descriptor(args.owner_review);v=json.loads(bind(owner_review).read_text())
    if v.get('approved') is not True or v.get('owner_sha256')!=sha(HERE/'queue.py'):raise ValueError('Root exact-owner review required')
    readiness=descriptor(args.readiness);v=json.loads(bind(readiness).read_text());inventory=v.get('GPU_inventory','').split(',')
    if v.get('normal_host_execution') is not True or len(inventory)<2 or inventory[0].strip()!=GPU or int(inventory[1])<32768:raise ValueError('Actual32GiB free normal-host readiness required')
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    if len(free)!=1 or int(free[0])<32768:raise ValueError('Current free memory below frozen capacity')
    adoption=None
    if args.purpose=='science':
        plan=json.loads((HERE/'ENGINEERING_OWNER_PLAN.json').read_text())
        if plan.get('source_manifest_sha256')!=source_sha or plan.get('owner_sha256')!=sha(HERE/'queue.py') or plan.get('purpose')!='engineering' or [r['record_id'] for r in plan['records']]!=['seed9101__'+c for c in CONDITIONS]:raise ValueError('Exact two-interface qualification plan required')
        qualifications={}
        for c,row in zip(CONDITIONS,plan['records']):
            raw=descriptor(HERE/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json');terminal(raw,row['release_sha256']);qualifications[c]=dict(raw_terminal=raw)
        for c,row in zip(CONDITIONS,plan['records']):
            complete=descriptor(HERE/'engineering'/'cells'/row['record_id']/'COMPLETE.json');result=json.loads(bind(complete).read_text())
            if result.get('schema')!='PubMed-factor1-engineering-complete-v1' or result.get('complete') is not True or result.get('condition')!=c or result.get('source_manifest_sha256')!=source_sha or result.get('release_sha256')!=row['release_sha256'] or result.get('updates')!=1 or result.get('fresh_weight_restore_verified') is not True or result.get('VALID_access') is not False:raise ValueError('Both actual whole-TRAIN updates/fresh restores must pass')
            qualifications[c]['complete']=complete
        root=dict(schema='PubMed-factor1-root-adoption-v1',enabled=True,source_manifest_sha256=source_sha,roster_sha256=sha(HERE/'ROSTER.json'),qualifications=qualifications,
                  source_review=review,source_delta_assessment=review,post_screen_exploration=True,confirmation_claim=False,pure_parameter_match=False,CORE=False,further18_activated=False,TEST_access=False,paper_score_recalculation=False)
        path=HERE/'ROOT_ADOPTION.json';save(path,root);adoption=descriptor(path)
    data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text());rows=json.loads((HERE/'ROSTER.json').read_text());records=[]
    for row in rows:
        if args.purpose=='engineering' and row['seed']!=9101:continue
        record=row['record_id'];owner_id=args.purpose+'__'+record
        spec=json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
        spec.update(row,schema='PubMed-factor1-'+args.purpose+'-release-v1',purpose=args.purpose,enabled=True,
            root_authorized=True,source_review_approved=True,source_delta_assessment_approved=True,complete_input_custody_verified=True,
            external_hard_bound_confirmed=True,fresh_resource_readiness_confirmed=True,ordinary_runtime_confirmed=True,
            VALID_access=args.purpose=='science',science_enabled=args.purpose=='science',source_manifest_sha256=source_sha,
            roster_sha256=sha(HERE/'ROSTER.json'),source_review=review,source_delta_assessment=review,resource_readiness_evidence=readiness,
            limits=ENGINEERING_LIMITS if args.purpose=='engineering' else LIMITS,output=str(HERE/args.purpose/'cells'/record))
        if args.purpose=='science':spec.update(VALID_custody_verified=True,valid_bundle=data['valid_bundle'],validation_custody=data['validation_custody'],root_admission=adoption)
        contract=dict(schema='PubMed-factor1-finite-owner-v1',enabled=True,record_id=record,limits=spec['limits'],separate_process_group=True,direct_wait_required=True,
                      resource_caps_enforced=True,output_and_log_caps_enforced=True,automatic_retry=False,owner_source=descriptor(HERE/'queue.py'),owner_review=owner_review)
        path=HERE/'contracts'/(owner_id+'.json');save(path,contract);spec['external_owner_release']=descriptor(path)
        path=HERE/'releases'/(owner_id+'.json');save(path,spec)
        records.append(dict(record_id=record,owner_id=owner_id,release=str(path.relative_to(PHASE)),release_sha256=sha(path),limits=spec['limits'],entrypoint=descriptor(HERE/'run.py'),entry_args=['--mode',args.purpose]))
    plan=dict(schema='PubMed-factor1-finite-owner-plan-v1',enabled=True,purpose=args.purpose,automatic_retry=False,owner_sha256=sha(HERE/'queue.py'),
              source_manifest_sha256=source_sha,roster_sha256=sha(HERE/'ROSTER.json'),source_review=review,owner_review=owner_review,readiness=readiness,records=records,
              post_screen_exploration=True,confirmation_claim=False,CORE=False,further18_activated=False,TEST_access=False,paper_score_recalculation=False)
    path=HERE/(args.purpose.upper()+'_OWNER_PLAN.json');save(path,plan)
    print(json.dumps(dict(owner_plan=descriptor(path),rendered_cells=len(records),nothing_started=True)))


if __name__=='__main__':main()
