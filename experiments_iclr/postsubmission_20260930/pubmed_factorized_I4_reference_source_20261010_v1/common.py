"""Closed fixed9+3 reference authority; standard library only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
SERVER_PHASE=REPO/'experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS=(9101,9203,9307)
SELECTOR='first strict maximum own raw-logit max(1)[1] VALID accuracy'
BASE_LIMITS=dict(GPU_bytes=32*1024**3,RSS_bytes=16*1024**3,output_bytes=512*1024**2,log_bytes=8*1024**2,
                 external_active_seconds=9000,external_cleanup_seconds=10,automatic_retry=False)
LIMITS={'admission':dict(BASE_LIMITS,external_active_seconds=600),'qualification':dict(BASE_LIMITS,external_active_seconds=600),
        'science':BASE_LIMITS,'assembly':dict(BASE_LIMITS,external_active_seconds=120),
        'comparison':dict(BASE_LIMITS,GPU_bytes=2*1024**3,RSS_bytes=2*1024**3,output_bytes=16*1024**2,log_bytes=1024**2,external_active_seconds=120)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path,existing=True):
    path=Path(path).resolve(strict=existing)
    if path==PHASE or not path.is_relative_to(PHASE):raise ValueError('Explicit owned phase path required')
    return path


def bind(row):
    path=inside(row['path'])
    if sha(path)!=row['sha256'] or ('bytes' in row and path.stat().st_size!=row['bytes']):raise ValueError('Bound artifact changed')
    return path


def read(row):return json.loads(bind(row).read_text())


def descriptor(path):
    path=inside(path);return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def write(path,value):
    path=inside(path,existing=False);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def route():
    if socket.gethostname()!='anogena-2-0' or PHASE!=SERVER_PHASE or Path.cwd().resolve()!=REPO:raise ValueError('Allocation-compatible route/cwd only')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:raise ValueError('Exact sole allocation GPU required')


def verify(folder,digest,name='SOURCE_MANIFEST.json'):
    path=folder/name
    if sha(path)!=digest:raise ValueError('Exact sealed source required')
    for row in json.loads(path.read_text())['files']:
        file=(folder/row['path']).resolve(strict=True)
        if not file.is_relative_to(folder) or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:raise ValueError('Sealed source payload changed')


def frozen():
    manifest_sha=sha(HERE/'SOURCE_MANIFEST.json');verify(HERE,manifest_sha)
    b=json.loads((HERE/'BINDINGS.json').read_text());data=json.loads((HERE/'DATA_AND_RUNTIME.json').read_text());roster=json.loads((HERE/'ROSTER.json').read_text())
    for key in ('M1_source_manifest','reference_source_manifest'):
        row=b[key];verify(bind(row).parent,row['sha256'])
    bind(b['M1_owner_source'])
    seal=read(b['history_seal']);plan=read(b['history_plan']);reuse=read(b['history_reuse'])
    for key,name in (('history_plan','PLAN_DISABLED.json'),('history_reuse','REUSE_BINDINGS.json')):
        if seal['files'][name]!=b[key]['sha256']:raise ValueError('Exact sealed history plan required')
    if [{k:r[k] for k in r if k!='body_index'} for r in roster]!=plan['roster']:raise ValueError('Fixed original12 position roster required')
    if len(roster)!=12 or any(r['initialization_seed']!=r['seed_block']+1009*r['body_index'] or r['factual_dropout_seeds']!=[r['initialization_seed']+300001] for r in roster):raise ValueError('Fixed native/body/dropout seeds required')
    return b,data,roster,reuse,manifest_sha


def records(mode,roster):
    if mode=='admission':return [dict(record_id='factorized_I4_anchor_admission')]
    if mode=='qualification':return [dict(record_id='factorized_I4_assembly_qualification',seed_block=9101)]
    if mode=='science':return [r for r in roster if r['body_index']>0]
    if mode=='assembly':return [dict(record_id=f'seed{s}__factorized_I4_native__committee',seed_block=s) for s in SEEDS]
    if mode=='comparison':return [dict(record_id='factorized_I4_complete_comparison')]
    raise ValueError('Fixed stage required')


def raw_wait(row,release_sha):
    value=read(row);absence=value.get('owned_absence',{})
    if value.get('complete') is not True or value.get('directly_waited') is not True or value.get('child_exit_code')!=0 or value.get('cap_or_owner_failure') is not None or value.get('automatic_retry') is not False:
        raise ValueError('Successful actual direct wait required')
    argv=value.get('argv',[])
    if argv[-2:]!=['--release-sha256',release_sha]:raise ValueError('Exact release absent from waited command')
    if absence.get('owned_process_absence_verified') is not True or absence.get('owned_CUDA_absence_verified') is not True:raise ValueError('Owned PID/CUDA absence required')
    return value


def closed_stage(stage,roster,manifest_sha):
    plan_path=HERE/(stage.upper()+'_OWNER_PLAN.json');plan=json.loads(plan_path.read_text())
    family=json.loads((HERE/stage/'FAMILY_COMPLETE.json').read_text());expected=[r['record_id'] for r in records(stage,roster)]
    if plan.get('enabled') is not True or plan.get('purpose')!=stage or plan.get('source_manifest_sha256')!=manifest_sha or plan.get('owner_sha256')!=sha(HERE/'owner.py') or plan.get('roster_sha256')!=sha(HERE/'ROSTER.json') or plan.get('automatic_retry') is not False or [r['record_id'] for r in plan['records']]!=expected:raise ValueError('Exact whole-stage owner plan required')
    if family.get('complete') is not True or family.get('records')!=expected or family.get('all_records_directly_waited') is not True or family.get('quality_fields_opened') is not False or family.get('plan_sha256')!=sha(plan_path) or family.get('source_manifest_sha256')!=manifest_sha or family.get('automatic_retry') is not False:raise ValueError('Entire stage must close before quality opening')
    owners={}
    # Every raw owner check precedes any stage COMPLETE quality read.
    for row in plan['records']:
        if row['owner_id']!=stage+'__'+row['record_id'] or row['limits']!=LIMITS[stage] or row['entry_args']!=['--mode',stage]:raise ValueError('Fixed stage owner identity required')
        release=read(dict(path=str(PHASE/row['release']),sha256=row['release_sha256']));bind(row['entrypoint'])
        argv=[release['runtime']['python']['path'],'-B','-P',str(PHASE/row['entrypoint']['path']),*row['entry_args'],'--release',str(PHASE/row['release']),'--release-sha256',row['release_sha256']]
        raw=descriptor(HERE/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json');owner=raw_wait(raw,row['release_sha256'])
        custody=read(descriptor(HERE/'owners'/row['owner_id']/'TERMINAL_CUSTODY.json'))
        if owner['argv']!=argv or custody.get('raw_owner_terminal')!={k:raw[k] for k in ('path','sha256')} or custody.get('complete') is not True or custody.get('source_manifest_sha256')!=manifest_sha or custody.get('release_sha256')!=row['release_sha256'] or custody.get('complete_sha256')!=sha(stage_result(stage,row['record_id'])):raise ValueError('Hash-bound waited release/output custody required')
        owners[row['record_id']]=owner
    return plan,owners


def stage_result(stage,record):return HERE/stage/'cells'/record/'COMPLETE.json'


def anchor_static(b,data,reuse):
    values={}
    # Hash every selected weight/output artifact before any model or new fit.
    for r in reuse['reused_records']:
        bind(r['complete_custody']['complete']);bind(r['complete_custody']['raw_terminal']);bind(r['selected_predictor']);bind(r['complete_saved_member_logits'])
    for r in reuse['reused_records']:
        value=read(r['complete_custody']['complete']);raw_wait(r['complete_custody']['raw_terminal'],value['release_sha256'])
        body=value['body'];seed=r['seed_block']
        if value.get('schema')!='PubMed-factor1-science-complete-v1' or value.get('complete') is not True or value.get('condition')!='factor1_native' or value.get('record_id')!=r['record_id'] or value.get('seed')!=seed or value.get('source_manifest_sha256')!=b['M1_source_manifest']['sha256'] or value.get('providers')!=data['frozen_providers']:raise ValueError('Exact whole M1-native anchor required')
        if (value['predictor_parameters'],value['native_slow_parameters'],value['extra_factor_coordinates'],value['factor_sites'],value['factor_members'])!=(2083818,2069875,13943,23,1):raise ValueError('Exact M1 reparameterization required')
        if value['own_native_selector']!=SELECTOR or value['max_epochs']!=2000 or value['patience']!=250 or value['internal_reference_alias']!='single_native' or value['TEST_access'] is not False:raise ValueError('Frozen native selector/task required')
        if body['initialization_seed']!=seed or body['factual_dropout_seeds']!=[seed+300001] or body['body_id']!='body0' or body['own_selector'] is not True or body['common_bank_selector'] is not False or body['epochs_executed']!=r['epochs_executed'] or body['selected_epoch']!=r['selected_epoch']:raise ValueError('Original body0 initialization/horizon identity required')
        if value['selected_VALID']!=r['selected_VALID'] or value['selected_prediction_signatures']!=r['selected_prediction_signatures'] or value['complete_saved_member_logits']!=r['complete_saved_member_logits']:raise ValueError('Immutable anchor selected output/scalars changed')
        checkpoint=dict(body['selected_predictor']);checkpoint['path']=str(bind(r['complete_custody']['complete']).parent/checkpoint['path'])
        if checkpoint!=r['selected_predictor']:raise ValueError('Original selected weight binding changed')
        release_path=bind(b['M1_source_manifest']).parent/'releases'/('science__'+r['record_id']+'.json')
        release=read(dict(path=str(release_path),sha256=value['release_sha256']))
        for k in ('train_bundle','valid_bundle','split_custody','validation_custody','runtime','frozen_providers'):
            if release[k]!=data[k]:raise ValueError('Original exact task/provider/release required')
        values[seed]=value
    if set(values)!=set(SEEDS):raise ValueError('All three exact body0 anchors required')
    return values


def root_admission(spec,b,data,roster,reuse,manifest_sha):
    # This never replays, substitutes or refits an anchor after admission failure.
    _,aowners=closed_stage('admission',roster,manifest_sha);_,qowners=closed_stage('qualification',roster,manifest_sha)
    root=read(spec['root_admission'])
    if root.get('enabled') is not True or root.get('source_manifest_sha256')!=manifest_sha or root.get('roster_sha256')!=sha(HERE/'ROSTER.json'):raise ValueError('Root exact9+3 adoption required')
    if bind(root['admission'])!=stage_result('admission','factorized_I4_anchor_admission') or bind(root['qualification'])!=stage_result('qualification','factorized_I4_assembly_qualification'):raise ValueError('Exact owned engineering outputs required')
    a=read(root['admission']);q=read(root['qualification'])
    if a.get('complete') is not True or a.get('all_three_admitted') is not True or a.get('fresh12_fallback') is not False or a.get('source_manifest_sha256')!=manifest_sha:raise ValueError('All anchors must pass owned exact selected replay before fits')
    if q.get('complete') is not True or q.get('new_four_body_assembly_restore_verified') is not True or q.get('VALID_access') is not False or q.get('source_manifest_sha256')!=manifest_sha:raise ValueError('Actual new four-body TRAIN-only qualification required')
    anchor_static(b,data,reuse)
    return root,a,q,aowners,qowners


def admit(args):
    route()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Exact visible allocation GPU required')
    path=inside(args.release)
    if not path.is_relative_to(HERE) or sha(path)!=args.release_sha256:raise ValueError('Exact root release required')
    spec=json.loads(path.read_text());b,data,roster,reuse,manifest_sha=frozen();mode=args.mode
    if spec.get('schema')!='PubMed-factorized-I4-release-v1' or spec.get('purpose')!=mode or spec.get('source_manifest_sha256')!=manifest_sha or spec.get('roster_sha256')!=sha(HERE/'ROSTER.json'):raise ValueError('Exact purpose/source/roster release required')
    for k in ('enabled','root_authorized','source_review_approved','complete_roster_frozen','complete_input_custody_verified','external_hard_bound_confirmed','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','post_screen_exploration'):
        if spec.get(k) is not True:raise ValueError('Inactive admission: '+k)
    for k in ('TEST_access','CORE','masking','HPO','automatic_retry','resume','fresh12_fallback','paper_score_recalculation','confirmation_claim','pure_parameter_match'):
        if spec.get(k) is not False:raise ValueError('Frozen closed scope: '+k)
    if any(k in spec for k in ('test_bundle','full_y','test_y','test_ids')):raise ValueError('No TEST interface')
    review=read(spec['source_review']);owner_review=read(spec['owner_review'])
    if review.get('approved') is not True or review.get('source_manifest_sha256')!=manifest_sha or any(review.get(k) is not True for k in ('all3_anchors_before_science_required','all9_new_fits_before_assembly_required','all3_banks_before_comparison_required','new_four_body_assembly_and_replay_approved','unchanged_M1_native_fit_approved')) or owner_review.get('approved') is not True or owner_review.get('source_manifest_sha256')!=manifest_sha or owner_review.get('owner_sha256')!=sha(HERE/'owner.py') or owner_review.get('existing_owner_source_sha256')!=b['M1_owner_source']['sha256']:raise ValueError('Exact source and reused-owner approval required')
    bind(spec['resource_readiness_evidence'])
    declared=next((r for r in records(mode,roster) if r['record_id']==spec.get('record_id')),None)
    if declared is None or any(spec.get(k)!=v for k,v in declared.items() if k!='enabled'):raise ValueError('Fixed body/stage identity required')
    for k in ('train_bundle','split_custody','runtime','frozen_providers'):
        if spec.get(k)!=data[k]:raise ValueError('Existing exact data/runtime required')
    bind(spec['train_bundle']);spec['_train_custody']=read(spec['split_custody'])
    valid=mode!='qualification'
    if spec.get('VALID_access') is not valid:raise ValueError('Declared VALID scope required')
    if valid:
        for k in ('valid_bundle','validation_custody'):
            if spec.get(k)!=data[k]:raise ValueError('Exact full VALID projection required')
        bind(spec['valid_bundle']);spec['_validation_custody']=read(spec['validation_custody'])
    elif any(k in spec for k in ('valid_bundle','validation_custody')):raise ValueError('TRAIN-only new assembly qualification required')
    if spec.get('selector')!=SELECTOR or spec.get('max_epochs')!=2000 or spec.get('patience')!=250 or spec.get('limits')!=LIMITS[mode]:raise ValueError('Unchanged selectors and finite bounds required')
    contract=read(spec['external_owner_release'])
    if contract.get('enabled') is not True or contract.get('record_id')!=spec['record_id'] or contract.get('limits')!=LIMITS[mode] or contract.get('automatic_retry') is not False or any(contract.get(k) is not True for k in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced')):raise ValueError('Exact finite owned child required')
    bind(contract['owner_source']);bind(contract['owner_review'])
    python=Path(os.path.abspath(data['runtime']['python']['path']))
    if sha(python)!=data['runtime']['python']['sha256'] or python.resolve()!=Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','')!=data['runtime']['PYTHONPATH']:raise ValueError('Exact qualified interpreter/provider path required')
    if mode in ('science','assembly','comparison'):root_admission(spec,b,data,roster,reuse,manifest_sha)
    if mode=='qualification':closed_stage('admission',roster,manifest_sha)
    if mode in ('assembly','comparison'):closed_stage('science',roster,manifest_sha)
    if mode=='comparison':closed_stage('assembly',roster,manifest_sha)
    output=inside(spec['output'],existing=False)
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh owned output required')
    spec['_release_sha256']=args.release_sha256
    return spec,output,b,data,roster,reuse
