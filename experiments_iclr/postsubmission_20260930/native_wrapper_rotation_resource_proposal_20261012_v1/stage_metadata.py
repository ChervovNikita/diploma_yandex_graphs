"""Existing admission shape, specialized to ten node source cases; metadata only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess


def local(root,relative):
    value=Path(relative);path=root/value
    if value.is_absolute() or '..' in value.parts or path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Confined project path required')
    return path
def receipt(root,path):
    raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
    path=local(root,row['path'])
    if receipt(root,path)!=row:raise ValueError('Exact source/helper/metadata binding changed')
    return path
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def stage(root,source,output,reviewed=False,cases=()):
    if output.exists():raise ValueError('Fresh output required; preserve previous attempts')
    proposal=json.loads((source/'PROPOSAL.json').read_text())
    if proposal['schema']!='wrapper-rotation-ten-case-resource-proposal-v1' or len(proposal['requested_cases'])!=10 or proposal['launch_enabled'] is not False:
        raise ValueError('Exact disabled ten-case proposal required')
    all_ids={c['case_id'] for c in proposal['requested_cases']};selected=list(cases) or [c['case_id'] for c in proposal['requested_cases']]
    if len(set(selected))!=len(selected) or not set(selected)<=all_ids:raise ValueError('Exact finite case subset required')
    imports={};studies={};shapes={}
    for c in proposal['requested_cases']:
        path=bound(root,c['study_receipt']);study=json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(study)!=c['study_spec_hash']:raise ValueError('Frozen study identity differs')
        run=next(r for r in study['runs'] if r['run_id']==c['source_run_id'])
        if fingerprint(run)!=c['source_run_hash'] or run['adapter']!=c['factory_entrypoint']:raise ValueError('Exact factory/run changed')
        studies[c['case_id']]=c['study_receipt']
        for r in c['source_import_receipts']:bound(root,r);imports[r['path']]=r
        custody=json.loads(bound(root,c['provider_custody']).read_text())
        if sorted({s['role_counts']['train'] for s in custody['splits']})!=c['all_mask_TRAIN_shape_set']:raise ValueError('Maximum TRAIN geometry changed')
        # Same bytes under common copied custody / original prepared custody;
        # one dataset entry suffices for source shape-only dummy banks.
        if c['dataset'] not in shapes:
            shapes[c['dataset']]={role:max(s['role_counts'][role] for s in custody['splits']) for role in ('cal','evaluate')}
            shapes[c['dataset']]['custody_receipt']=c['provider_custody']
    for r in proposal['runtime_dependency_receipts']:bound(root,r)
    for r in proposal['inherited_helpers'].values():bound(root,r)
    output.mkdir(parents=True);owned=output/'source';owned.mkdir()
    for name in ('PROPOSAL.json','qualify_wrapper_rotation.py','PROBE_REUSE_PROOF.json','OBSERVATIONS_SCHEMA.json','README.md'):
        shutil.copyfile(source/name,owned/name)
    helper=receipt(root,owned/'qualify_wrapper_rotation.py')
    if helper['sha256']!=proposal['probe_receipt']['sha256']:raise ValueError('Probe differs from proposal')
    admission={'schema':'root-approved-zero-update-resource-admission-v1','root_source_review_confirmed':bool(reviewed),'helper_sha256':helper['sha256'],
      'proposal_receipt':receipt(root,owned/'PROPOSAL.json'),'scientific_GO':False,'max_actual_optimizer_updates':0,
      'approved_case_ids':selected if reviewed else [],'reviewed_proposed_case_ids':selected,'case_study_receipts':studies,
      **proposal['inherited_helpers'],'runtime_package':proposal['runtime_package'],'runtime_dependency_receipts':proposal['runtime_dependency_receipts'],
      'source_dependency_receipts':list(imports.values()),'node_bank_role_shapes':shapes,'two_process_pairs_approved':False,'no_automatic_retry':True,
      'scope':'ten full native random-state source/cost/storage/restore cases;root engineering approval only;no scientificGO'}
    (output/'ENGINEERING_ADMISSION.json').write_text(json.dumps(admission,indent=2,sort_keys=True)+'\n')
    (output/'STAGE_MANIFEST.json').write_text(json.dumps({'schema':'wrapper-rotation-metadata-stage-v1','source_files':[receipt(root,p) for p in owned.iterdir()],
      'admission':receipt(root,output/'ENGINEERING_ADMISSION.json'),'source_studies':studies,'source_imports':list(imports.values()),
      'models_created':0,'CUDA_executions':0,'scientific_fits':0,'GO':False,'root_review_confirmed':bool(reviewed)},indent=2)+'\n')
    return {'stage':str(output.relative_to(root)),'proposed_cases':len(selected),'approved_cases':len(admission['approved_case_ids']),'CUDA':0,'fits':0,'GO':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project-root','source-directory','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--root-reviewed',action='store_true');p.add_argument('--case-id',action='append',default=[]);a=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation hostname/sole UUID before project operations')
    root=Path(a.project_root).resolve();print(json.dumps(stage(root,local(root,a.source_directory),local(root,a.output),a.root_reviewed,a.case_id)))
