"""Disabled CPU-only update diagnostic of two discarded LOCAL qualifier states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PUBLIC = PHASE/'portable_internal_be_public_interface_20261007_v2'
BRANCH = PHASE/'contrastive_BE_steering_continuing_research_20261007_v1'
INTEGRATION = BRANCH/'integration_successor_v2'
SEED = 901337


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(row):
    rel=Path(row['path'])
    require(not rel.is_absolute() and '..' not in rel.parts,'Phase-relative input custody')
    path=(PHASE/rel).resolve()
    require(path.is_relative_to(PHASE) and path.is_file() and sha(path)==row['sha256'],'Exact bound evidence/source')
    return path


def canonical_config(value):
    # Only the fixed route assignment/method identity differs between these cases.
    value=dict(value);value.pop('arms');method=dict(value.pop('context_target_method'))
    for key in ('method','mode'):
        method.pop(key)
    value['context_target_method']=method
    return value


def snapshots(torch,public,dispatch,cfg):
    result={}
    for condition in ('shared_common','shared_route'):
        row=cfg['inputs'][condition]
        path=bound(row);state=torch.load(path,map_location='cpu',weights_only=False)
        require(state.get('schema')=='portable-internal-be-continuation-v1'
            and state.get('task')=='wikics' and state.get('arm')=='be_unit_contrastive'
            and state.get('seed')==SEED and state.get('backend')=='cuda'
            and state.get('steps')==1 and state.get('epoch')==0
            and state.get('body_global')==[False]
            and state.get('config')==dispatch.configured_recipe(public,condition),
            'Only the discarded first LOCAL native qualifier update')
        require(state['checkpoint_selection'].startswith('none;'),'No scored/selected checkpoint')
        result[condition]=state
    a,b=result['shared_common'],result['shared_route']
    live_core={name:sha(public.ROOT/'core'/name)
               for name in ('factors.py','models.py','objectives.py','selection.py')}
    require(canonical_config(a['config'])==canonical_config(b['config'])
        and a['core_provenance']==b['core_provenance']==live_core
        and a['native_provenance']==b['native_provenance']=={
            'polynormer_model_sha256':cfg['native_polynormer']['sha256']},
        'Matched model/optimizer recipe and native/core provenance')
    require(len(a['streams'])==len(b['streams'])==4,'Complete four-member native streams')
    for x,y in zip(a['streams'],b['streams']):
        require(set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in x),
                'Identical member RNG endpoint after the same two stochastic views')
    return result


def parameter_groups(model,factors):
    private={id(p) for module in model.modules() if isinstance(module,factors.FactorLinear)
             for p in (module.r,module.s)}
    groups={}
    for name,p in model.named_parameters(remove_duplicate=False):
        row=groups.setdefault(id(p),dict(names=[],parameter=p,private=id(p) in private))
        row['names'].append(name)
    return list(groups.values())


def storage_aliases(groups):
    """Report exact overlapping storage without merging distinct optimizer objects."""
    by_storage={}
    for group in groups:
        p=group['parameter']
        key=(p.untyped_storage().data_ptr(),p.storage_offset(),tuple(p.shape),tuple(p.stride()),str(p.dtype))
        by_storage.setdefault(key,[]).append(group['names'][0])
    return [names for names in by_storage.values() if len(names)>1]


def scalar_stats(torch,values):
    count=sum(value.numel() for value in values)
    square=sum(float(value.double().square().sum()) for value in values)
    return dict(unique_entries=count,nonzero_entries=sum(int(torch.count_nonzero(value)) for value in values),
        L2=square**.5,RMS=(square/count)**.5 if count else None,
        maximum_absolute=max((float(value.abs().max()) for value in values),default=0.))


def compare(torch,model,groups,states):
    initial=model.state_dict();common=states['shared_common']['model'];route=states['shared_route']['model']
    names={name for group in groups for name in group['names']}
    require(set(initial)==set(common)==set(route)==names,'No unaccounted buffers/state slots')
    accum={key:[] for key in ('private_common_update','private_route_update','private_route_minus_common',
                             'shared_common_update','shared_route_update','shared_route_minus_common')}
    rows=[];inactive_count=0;private_common=[];private_route=[]
    for group in groups:
        name=group['names'][0];reference=initial[name].detach()
        for state in (initial,common,route):
            require(all(torch.equal(state[name],state[alias]) for alias in group['names']),
                    'Aliased native parameter slots disagree')
            require(state[name].shape==reference.shape and state[name].dtype==torch.float32
                and torch.isfinite(state[name]).all(),'Finite original float32 parameter state')
        dc=common[name]-reference;dr=route[name]-reference;difference=route[name]-common[name]
        kind='private' if group['private'] else 'shared'
        accum[kind+'_common_update'].append(dc);accum[kind+'_route_update'].append(dr)
        accum[kind+'_route_minus_common'].append(difference)
        native=name.split('.body.',1)[1]
        inactive=native.startswith(('ln.','global_attn.','pred_global.'))
        if inactive:
            require(torch.equal(reference,common[name]) and torch.equal(reference,route[name]),
                    'Declared unused LOCAL module differs from reconstructed matched start')
            inactive_count+=1
        if group['private']:
            require(reference.ndim==2 and reference.shape[0]==4 and torch.equal(reference,torch.ones_like(reference)),
                    'Original unit four-member private factors required')
            private_common.append(dc.reshape(4,-1));private_route.append(dr.reshape(4,-1))
        rows.append(dict(canonical_name=name,alias_names=group['names'],kind=kind,
            entries=reference.numel(),declared_LOCAL_inactive=inactive,
            common_update=scalar_stats(torch,[dc]),route_update=scalar_stats(torch,[dr]),
            route_minus_common=scalar_stats(torch,[difference])))
    require(inactive_count>0 and private_common,'Native inactive-start and factor accounting')
    by_member=[]
    pc=torch.cat(private_common,dim=1);pr=torch.cat(private_route,dim=1)
    for member in range(4):
        by_member.append(dict(member_index0=member,common_update=scalar_stats(torch,[pc[member]]),
            route_update=scalar_stats(torch,[pr[member]]),route_minus_common=scalar_stats(torch,[pr[member]-pc[member]])))
    return dict(unique_parameter_objects=len(groups),state_slots=len(names),
        aliases_deduplicated_by_live_parameter_identity=True,
        parameter_alias_groups=[row['names'] for row in groups if len(row['names'])>1],
        distinct_parameter_objects_with_exact_shared_storage=storage_aliases(groups),
        reconstructed_inactive_LOCAL_parameters_exactly_unchanged=inactive_count,
        aggregate={key:scalar_stats(torch,value) for key,value in accum.items()},
        private_by_member=by_member,parameters=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();require(sha(args.release)==args.release_sha256,'Exact separate root release')
    cfg=read(args.release)
    require(cfg.get('enabled') is True and cfg.get('root_CPU_execution_authorized') is True
        and cfg.get('source_review_approved') is True and cfg.get('scientific_fit') is False
        and cfg.get('model_forward') is False and cfg.get('GPU_used') is False
        and cfg.get('automatic_retry') is False and cfg.get('schema')=='native-first-update-CPU-diagnostic-release-v1',
        'CPU diagnostic disabled pending root admission')
    require(socket.gethostname()=='anogena-2-0' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Authorized allocation CPU-only execution')
    require(cfg['source_manifest_sha256']==sha(HERE/'MANIFEST.json'),'Exact source snapshot')
    for row in read(HERE/'MANIFEST.json')['files']:
        require(sha(HERE/row['path'])==row['sha256'],'Sealed diagnostic source changed')
    pins=read(HERE/'SOURCE_BINDINGS.json')
    for row in pins['files']:
        bound(row)
    require(set(cfg['inputs'])=={'shared_common','shared_route','qualification'},'Exactly two LOCAL states and qualification receipt')
    qualification=read(bound(cfg['inputs']['qualification']))
    require(qualification.get('complete') is True and qualification.get('discarded_native_updates')==10
        and qualification.get('scientific_fit') is False and qualification.get('VALID_labels_access') is False
        and qualification.get('TEST_access') is False,'Actual complete discarded native qualifier')
    for condition in ('shared_common','shared_route'):
        cell=next(row for row in qualification['rows'] if row['method']==condition)
        local=next(row for row in cell['rows'] if row['global_mode'] is False)
        require(cfg['inputs'][condition]['sha256']==local['snapshot_sha256'],'Actual qualifier LOCAL snapshot identity')
    output=(PHASE/Path(cfg['output'])).resolve()
    require(output.is_relative_to(PHASE) and output!=PHASE and not output.exists(),'Fresh research output')
    output.mkdir(parents=True);began=time.monotonic()
    try:
        for path in (PUBLIC,BRANCH,INTEGRATION):
            sys.path.insert(0,str(path))
        import torch
        import portable as public
        import context_dispatch as dispatch
        torch.set_num_threads(2)
        require(str(torch.__version__)=='2.1.2+cu118' and torch.get_default_dtype()==torch.float32,'Original CPU construction runtime/dtype')
        states=snapshots(torch,public,dispatch,cfg)
        core=public._core();native,_=public.native_sources('wikics',bound(cfg['native_polynormer']))
        model=core['models'].Ensemble('wikics','be_unit_contrastive',SEED,
            public.recipe('wikics')['model'],native)
        require(all(p.device.type=='cpu' for p in model.parameters()),'Only one reconstructed CPU model; no forward')
        result=compare(torch,model,parameter_groups(model,core['factors']),states)
        result.update(schema='native-first-update-CPU-diagnostic-v1',complete=True,matched_constructor_seed=SEED,
            matched_constructor_core_and_native_source=True,discarded_updates_per_snapshot=1,
            scientific_fits=0,model_constructions=1,optimizer_constructions=0,model_forward_calls=0,GPU_used=False,
            input_snapshots=cfg['inputs'],seconds=time.monotonic()-began,
            interpretation='First-update parameter diagnostic at this initialization; no member competence, predictive diversity, accuracy, novelty or causal-gradient guarantee. Native floating execution is not bitwise certified.')
        (output/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    except BaseException as error:
        (output/'FAILURE.json').write_text(json.dumps(dict(complete=False,error_type=type(error).__name__,
            error=str(error),scientific_fits=0,model_forward_calls=0,GPU_used=False,automatic_retry=False,
            seconds=time.monotonic()-began),indent=2)+'\n');raise
    print(json.dumps(dict(complete=True,scientific_fits=0,model_forward_calls=0,output=str(output))))


if __name__=='__main__':
    main()
