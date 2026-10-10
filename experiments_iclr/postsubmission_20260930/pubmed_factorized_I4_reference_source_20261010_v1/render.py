"""Root-reviewed fixed stage rendering; this script starts no process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE,PHASE,GPU,LIMITS,anchor_static,bind,closed_stage,descriptor,frozen,inside,read,records,root_admission,route,sha,stage_result


def save(path,value):
    path=inside(path,existing=False)
    if not path.is_relative_to(HERE) or path.exists():raise ValueError('Fresh exact root-confined artifact required')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--purpose',choices=tuple(LIMITS),required=True)
    p.add_argument('--source-review',type=Path,default=HERE/'ROOT_SOURCE_REVIEW.json');p.add_argument('--owner-review',type=Path,default=HERE/'OWNER_REVIEW.json');p.add_argument('--readiness',type=Path,default=HERE/'READINESS.json');args=p.parse_args()
    route();b,data,roster,reuse,manifest_sha=frozen();stage=args.purpose
    review=descriptor(args.source_review);rv=read(review);oreview=descriptor(args.owner_review);ov=read(oreview)
    if rv.get('approved') is not True or rv.get('source_manifest_sha256')!=manifest_sha or any(rv.get(k) is not True for k in ('all3_anchors_before_science_required','all9_new_fits_before_assembly_required','all3_banks_before_comparison_required','new_four_body_assembly_and_replay_approved','unchanged_M1_native_fit_approved')) or rv.get('standalone_body1_requalification') is not False or rv.get('TEST_access') is not False:raise ValueError('Root exact-source and narrowed-qualification review required')
    if ov.get('approved') is not True or ov.get('source_manifest_sha256')!=manifest_sha or ov.get('owner_sha256')!=sha(HERE/'owner.py') or ov.get('existing_owner_source_sha256')!=b['M1_owner_source']['sha256'] or ov.get('automatic_retry') is not False:raise ValueError('Root unchanged finite-owner review required')
    readiness=descriptor(args.readiness);v=read(readiness);inventory=v.get('GPU_inventory','').split(',');need=LIMITS[stage]['GPU_bytes']//1024**2
    if v.get('normal_host_execution') is not True or len(inventory)<2 or inventory[0].strip()!=GPU or int(inventory[1])<need:raise ValueError('Actual frozen-capacity normal-host readiness required')
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    if len(free)!=1 or int(free[0])<need:raise ValueError('Current free GPU memory below frozen readiness')
    adoption=None
    if stage=='qualification':closed_stage('admission',roster,manifest_sha)
    if stage=='science':
        closed_stage('admission',roster,manifest_sha);closed_stage('qualification',roster,manifest_sha)
        anchor_static(b,data,reuse)
        admission=descriptor(stage_result('admission','factorized_I4_anchor_admission'));qualification=descriptor(stage_result('qualification','factorized_I4_assembly_qualification'))
        a,q=read(admission),read(qualification)
        if a.get('complete') is not True or a.get('all_three_admitted') is not True or a.get('fresh12_fallback') is not False or a.get('source_manifest_sha256')!=manifest_sha:raise ValueError('Owned all-three exact anchor replay admission required')
        if q.get('complete') is not True or q.get('new_four_body_assembly_restore_verified') is not True or q.get('VALID_access') is not False or q.get('source_manifest_sha256')!=manifest_sha:raise ValueError('Owned new four-body TRAIN-only assembly/restore required')
        root=dict(schema='PubMed-factorized-I4-root-adoption-v1',enabled=True,source_manifest_sha256=manifest_sha,roster_sha256=sha(HERE/'ROSTER.json'),admission=admission,qualification=qualification,source_review=review,owner_review=oreview,all_three_original_body0_required=True,nine_new_full_fits=True,fresh12_fallback=False,standalone_body1_requalification=False,TEST_access=False,novelty='none',paper_score_recalculation=False)
        path=HERE/'ROOT_ADOPTION.json';save(path,root);adoption=descriptor(path)
    elif stage in ('assembly','comparison'):
        adoption=descriptor(HERE/'ROOT_ADOPTION.json');root_admission(dict(root_admission=adoption),b,data,roster,reuse,manifest_sha)
        # Entire raw science/assembly closure precedes any new quality read.
        closed_stage('science',roster,manifest_sha)
        if stage=='comparison':closed_stage('assembly',roster,manifest_sha)
    rendered=[]
    for row in records(stage,roster):
        record=row['record_id'];owner_id=stage+'__'+record;spec=json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
        spec.update(row,purpose=stage,enabled=True,root_authorized=True,source_review_approved=True,complete_roster_frozen=True,complete_input_custody_verified=True,external_hard_bound_confirmed=True,fresh_resource_readiness_confirmed=True,ordinary_runtime_confirmed=True,
                    VALID_access=stage!='qualification',source_manifest_sha256=manifest_sha,roster_sha256=sha(HERE/'ROSTER.json'),source_review=review,owner_review=oreview,resource_readiness_evidence=readiness,limits=LIMITS[stage],output=str(HERE/stage/'cells'/record))
        if stage!='qualification':spec.update(valid_bundle=data['valid_bundle'],validation_custody=data['validation_custody'])
        if adoption:spec['root_admission']=adoption
        contract=dict(schema='PubMed-factorized-I4-finite-owner-v1',enabled=True,record_id=record,limits=LIMITS[stage],separate_process_group=True,direct_wait_required=True,resource_caps_enforced=True,output_and_log_caps_enforced=True,automatic_retry=False,owner_source=descriptor(HERE/'owner.py'),owner_review=oreview,unchanged_M1_owner_source=b['M1_owner_source'])
        path=HERE/'contracts'/(owner_id+'.json');save(path,contract);spec['external_owner_release']=descriptor(path)
        path=HERE/'releases'/(owner_id+'.json');save(path,spec)
        rendered.append(dict(record_id=record,owner_id=owner_id,release=str(path.relative_to(PHASE)),release_sha256=sha(path),limits=LIMITS[stage],entrypoint=descriptor(HERE/'run.py'),entry_args=['--mode',stage]))
    plan=dict(schema='PubMed-factorized-I4-finite-owner-plan-v1',enabled=True,purpose=stage,automatic_retry=False,owner_sha256=sha(HERE/'owner.py'),source_manifest_sha256=manifest_sha,roster_sha256=sha(HERE/'ROSTER.json'),source_review=review,owner_review=oreview,readiness=readiness,records=rendered,novelty='none',TEST_access=False,paper_score_recalculation=False,fresh12_fallback=False,no_partial_comparison=True)
    path=HERE/(stage.upper()+'_OWNER_PLAN.json');save(path,plan);print(json.dumps(dict(owner_plan=descriptor(path),rendered_records=len(rendered),nothing_started=True)))


if __name__=='__main__':main()
