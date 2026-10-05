"""Bind the reviewed, finite source check. No numerical execution or SSH."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REMOTE_PHASE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
SOURCE='shared_backbone_private_transfer_training_source_20261005_v2'
REVIEWS=['shared_backbone_private_transfer_source_independent_review_20261005_v2','shared_backbone_private_transfer_source_independent_review_20261005_v3']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    source=PHASE/SOURCE
    assert sha(source/'SOURCE_MANIFEST.json')=='db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f'
    manifest=json.loads((source/'SOURCE_MANIFEST.json').read_text())
    for r in manifest['files']:
        p=source/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
        if p.suffix=='.py':ast.parse(p.read_text())
    for name,expected in zip(REVIEWS,['c8b471f1cff1e7462de768f99564695cb15c7c6bdd8402237697870bea54b5c3','ade060b5ad4d380d4b026d9970f9194f0285997e96a4a1b548573f991e21e05f']):
        root=PHASE/name;assert sha(root/'REVIEW_MANIFEST.json')==expected
        for r in json.loads((root/'REVIEW_MANIFEST.json').read_text())['files']:assert sha(root/r['path'])==r['sha256']
    donor=json.loads((PHASE/'citeseer_endpoint_frame_paired_development_20261005_v1/b0_native_single_seed0.job.json').read_text())
    job=json.loads((source/'QUALIFICATION_JOB_TEMPLATE.json').read_text())
    job.update(schema='root_released_discarded_FP32_training_step_v1',source_review_approved=True,
        available_manifest_relative=donor['available_manifest_relative'],available_manifest_sha256=donor['available_manifest_sha256'],
        constant_adjacency_recursive_adjoint_authorized=True,external_hard_bound_confirmed=True,
        soft_seconds=600,source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),
        output_directory=str(REMOTE_PHASE/HERE.name/'result'))
    def evidence(relative):
        p=PHASE/relative;return {'path':relative,'sha256':sha(p)}
    for key in ('feature_authority','negative_pool_authority'):
        job[key]={'approved':True,'evidence':donor[key]['evidence']}
        for r in job[key]['evidence']:assert sha(PHASE/r['path'])==r['sha256']
    job['runtime_qualification']={'approved':True,'evidence':[evidence('citeseer_ncn_native_runtime_qualification_20261005_v2/RESULT.json')]}
    job['source_review']={'approved':True,'evidence':[evidence(name+'/REVIEW_MANIFEST.json') for name in REVIEWS]+[evidence(HERE.name+'/ROOT_REVIEW.md')]}
    write(HERE/'JOB.json',job)
    release={'UTC':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':job['source_manifest_sha256'],
        'job_sha256':sha(HERE/'JOB.json'),'root_review_sha256':sha(HERE/'ROOT_REVIEW.md'),
        'supervisor_sha256':sha(HERE/'stage_and_execute_once.py'),'source_review_manifests':[r for r in job['source_review']['evidence'] if r['path'].endswith('REVIEW_MANIFEST.json')],
        'numerical_launches':0,'fits':0,'VALID_TEST_values_access':False,'attempts_authorized':1}
    write(HERE/'ROOT_RELEASE.json',release)
    paths=[]
    for name in [SOURCE,'endpoint_episode_geometry_preparation_20261005_v1',*REVIEWS]:
        paths.extend(p for p in (PHASE/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(HERE/name for name in ['JOB.json','ROOT_REVIEW.md','ROOT_RELEASE.json','prepare_job.py','stage_and_execute_once.py','physical_preflight.py','PREFLIGHT.json'])
    files=[]
    for p in paths:
        assert p.resolve().is_relative_to(PHASE.resolve()) and not p.is_symlink() and p.stat().st_size<2_000_000
        b=p.read_bytes();files.append({'path':str(p.relative_to(PHASE)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
    write(HERE/'STAGING_PAYLOAD.json',{'remote_execution_root':str(REMOTE_PHASE/HERE.name),'training_source_name':SOURCE,'files':files})
    print(json.dumps({'source_v2_verified':True,'reviews_verified':True,'files':len(files),'job_sha256':sha(HERE/'JOB.json'),'numerical_launches':0}))

if __name__=='__main__':main()
