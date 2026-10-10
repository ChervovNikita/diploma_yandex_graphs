"""Minimal exact-source/root-review/readiness binding shared by both renderers."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=Path(__file__).resolve().parent
SOURCE=PHASE/'combination_pubmed_strong_reference_source_20261010_v2'
SOURCE_SHA='a695504452e82203671e11f40c417024184a6c61e4bbe7f107cb5d1b2702e32a'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path,root=PHASE,existing=True):
    path=Path(path).resolve(strict=existing)
    if path==root or not path.is_relative_to(root):raise ValueError('Bound project path required')
    return path


def bound(path):
    path=inside(path)
    return dict(path=str(path),sha256=sha(path))


def write(path,value):
    path=inside(path,root=HERE,existing=False)
    if path.exists():raise ValueError('Never replace a rendered root artifact: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def read_bound(row):
    path=inside(row['path'])
    if sha(path)!=row['sha256']:raise ValueError('Exact artifact binding changed')
    return json.loads(path.read_text())


def arguments(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--source-review',type=Path,default=HERE/'ROOT_SOURCE_REVIEW.json')
    p.add_argument('--source-delta-assessment',type=Path)
    p.add_argument('--owner-review',type=Path,default=HERE/'OWNER_REVIEW.json')
    p.add_argument('--readiness',type=Path,default=HERE/'READINESS.json')
    return p.parse_args()


def context(args):
    if socket.gethostname()!='anogena-2-0' or Path.cwd().resolve()!=REPO or not HERE.is_relative_to(PHASE):
        raise ValueError('Exact allocation route/project/cwd required before project operations')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:
        raise ValueError('Exact sole allocation GPU required')
    if sha(SOURCE/'SOURCE_MANIFEST.json')!=SOURCE_SHA:raise ValueError('Exact sealed source V2 required')
    manifest=json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        file=(SOURCE/row['path']).resolve(strict=True)
        if not file.is_relative_to(SOURCE) or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:
            raise ValueError('Sealed source payload changed')
    review=bound(args.source_review);value=read_bound(review)
    if not (value.get('source_review_approved') is True or value.get('approved') is True) or value.get('source_manifest_sha256')!=SOURCE_SHA:
        raise ValueError('Root must approve the exact source V2')
    if value.get('source_delta_assessment_approved') is not True:raise ValueError('Root must approve this source delta')
    delta=bound(args.source_delta_assessment or args.source_review)
    owner_review=bound(args.owner_review);value=read_bound(owner_review)
    if value.get('approved') is not True or value.get('owner_sha256')!=sha(HERE/'queue.py'):
        raise ValueError('Root must approve this owner successor')
    readiness=bound(args.readiness);value=read_bound(readiness)
    if value.get('normal_host_execution') is not True:raise ValueError('Actual normal-host readiness required')
    inventory=value.get('GPU_inventory','').split(',')
    if len(inventory)<2 or inventory[0].strip()!=GPU or int(inventory[1])<32768:
        raise ValueError('Bound readiness must show the exact GPU with32GiB free')
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    if len(free)!=1 or int(free[0])<32768:raise ValueError('Current GPU readiness below frozen capacity')
    return dict(review=review,delta=delta,owner_review=owner_review,readiness=readiness)


def release(template,purpose,ctx,adoption=None):
    spec=json.loads((SOURCE/'releases_disabled'/template).read_text())
    for flag in ('enabled','root_authorized','source_review_approved','source_delta_assessment_approved',
                 'external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed',
                 'complete_graph_data_custody_verified'):
        spec[flag]=True
    record=spec['record_id'];owner_id=purpose+'__'+record
    contract=json.loads((SOURCE/'FINITE_OWNER_CONTRACT_TEMPLATE_DISABLED.json').read_text())
    contract.update(enabled=True,record_id=record,limits=spec['limits'],separate_process_group=True,
                    direct_wait_required=True,resource_caps_enforced=True,output_and_log_caps_enforced=True,
                    owner_source=bound(HERE/'queue.py'),owner_review=ctx['owner_review'])
    contract_file=HERE/'contracts'/(owner_id+'.json');write(contract_file,contract)
    spec.update(source_manifest_sha256=SOURCE_SHA,source_review=ctx['review'],source_delta_assessment=ctx['delta'],
                external_owner_release=bound(contract_file),resource_readiness_evidence=ctx['readiness'],
                output=str(HERE/purpose/'cells'/record))
    if purpose=='science':
        spec.update(VALID_custody_verified=True,root_admission=adoption)
    release_file=HERE/'releases'/(owner_id+'.json');write(release_file,spec)
    entrypoint=SOURCE/('qualify.py' if purpose=='engineering' else 'train.py')
    return dict(record_id=record,owner_id=owner_id,release=str(release_file.relative_to(PHASE)),
                release_sha256=sha(release_file),limits=spec['limits'],entrypoint=bound(entrypoint),
                entry_args=[] if purpose=='engineering' else ['--mode','run'])


def plan(purpose,records,ctx):
    value=dict(schema='PubMed-strong-reference-finite-owner-plan-v1',enabled=True,purpose=purpose,
               automatic_retry=False,owner_sha256=sha(HERE/'queue.py'),source_manifest_sha256=SOURCE_SHA,
               roster_sha256=sha(SOURCE/'ROSTER.json'),anchors_sha256=sha(SOURCE/'ANCHORS.json'),
               source_review=ctx['review'],owner_review=ctx['owner_review'],readiness=ctx['readiness'],records=records,
               root_advertised_prior_HEAD='d0c006db9346a3c5e1fb03f6f99554bea51381ee',
               original_scores_unchanged=True,CORE_continuation_eligible=False,further18_activated=False)
    path=HERE/(purpose.upper()+'_OWNER_PLAN.json');write(path,value)
    return bound(path)
