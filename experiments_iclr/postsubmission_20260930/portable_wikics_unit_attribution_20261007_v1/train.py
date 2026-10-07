"""Public WikiCS unit attribution: fixed four objectives, recomputation, full1100epochs."""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
try:
    import resource
except ImportError:
    resource=None

HERE=Path(__file__).resolve().parent
CONDITIONS={'plain':(0.,0.),'alignment_only':(.05,0.),'residual_only':(0.,.05),'combined':(.05,.05)}
PUBLIC_MANIFEST_SHA='190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
RECOMPUTE_SHA='1554e9c6b181d68be937596530e809d1133ea6a682faa04c155a533856156d73'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value); return value


def public_dependency(root):
    root=Path(root).resolve(); manifest=root/'MANIFEST.json'
    require(sha(manifest)==PUBLIC_MANIFEST_SHA,'Use the sealed portable public v2 code dependency')
    for row in json.loads(manifest.read_text())['files']:
        file=root/row['path']; require(file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'Public v2 code bytes changed')
    require(sha(HERE/'recompute.py')==RECOMPUTE_SHA,'Bundled v2 recomputation bytes changed')
    return root


class ObjectivesFacade:
    """Original functions; no new loss or extra forward/backward/label opportunity."""
    def __init__(self, original, condition):
        self.original, self.weights = original, CONDITIONS[condition]
        self.own_supervision = original.own_supervision
        self.alignment_source_calls = self.residual_source_calls = 0

    def alignment_loss(self, a, b, labels, task, temperature=.2, identities=None):
        require(task == 'wikics' and temperature == .2, 'Fixed WikiCS supervised alignment')
        if self.weights[0] == 0:
            return a.sum() * 0
        self.alignment_source_calls += 1
        return self.original.alignment_loss(a, b, labels, task, temperature, identities)

    def residual_member_contrast(self, a, b, labels, temperature=.2):
        require(temperature == .2, 'Fixed residual route contrast temperature')
        if self.weights[1] == 0:
            return a.sum() * 0
        self.residual_source_calls += 1
        return self.original.residual_member_contrast(a, b, labels, temperature)


def identity(condition,recompute):
    weights=CONDITIONS[condition]
    return dict(condition=condition,method_identity='be_unit__public_attribution_'+condition,
        underlying_session_arm='be_unit' if condition=='plain' else 'be_unit_contrastive',
        model='shared four-member Polynormer; native reset; unit factors',alignment_weight=weights[0],residual_weight=weights[1],
        temperature=.2,max_objects=512,own_views=2,members=4,epochs=1100,local_epochs=100,
        execution_mode=recompute.MODE,training_member_forwards_per_update=16,shadow_member_forwards_per_update=8,
        replay_member_forwards_per_update=8,member_reverse_collections_per_update=8,output_cotangent_collections_per_update=1,
        optimizer_bank_updates_per_update=1,original_stochastic_views=2,
        public_wrapper_sha256=sha(__file__),recompute_sha256=RECOMPUTE_SHA,public_interface_manifest_sha256=PUBLIC_MANIFEST_SHA,
        execution_profile='public caller environment; separate from registered author Wiki12',
        author_execution_equivalence_claimed=False,bitwise_author_parity_claimed=False)


def recipe(public,specification):
    config=copy.deepcopy(public.recipe('wikics')); config['arms']=[specification['method_identity']]
    config['contrastive'].update(alignment_weight=specification['alignment_weight'],residual_weight=specification['residual_weight'])
    config['metric']='complete_development_union_split0_validation_and_stopping_accuracy'
    config['public_attribution']=specification; return config


def make_session(public,recompute,specification,seed,device,polynormer):
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    session=public.Session('wikics',specification['underlying_session_arm'],seed,device,polynormer)
    require(session.model.members==4 and not session.model.independent and len(session.optimizers)==1,'Shared unit-factor/Adam trajectory')
    facade=ObjectivesFacade(session.core['objectives'],specification['condition'])
    session.core=dict(session.core,objectives=facade); session.config=recipe(public,specification); recompute.install(session)
    session.public_cost=dict(training_member_calls_attempted=0,training_member_calls_returned=0,
        development_member_calls_attempted=0,development_member_calls_returned=0,update_calls_attempted=0,
        update_calls_returned=0,update_wall_seconds=0.)
    original_member=session.model.member_forward
    def counted_member(batch,member):
        phase='training' if session.model.training else 'development'
        session.public_cost[phase+'_member_calls_attempted']+=1
        result=original_member(batch,member)
        session.public_cost[phase+'_member_calls_returned']+=1; return result
    session.model.member_forward=counted_member
    original_update=session.train_step
    def timed_update(batch,labels):
        started=time.monotonic(); session.public_cost['update_calls_attempted']+=1
        try:
            result=original_update(batch,labels); session.public_cost['update_calls_returned']+=1; return result
        finally:
            session.public_cost['update_wall_seconds']+=time.monotonic()-started
    session.train_step=timed_update
    session.public_versions={'torch':str(torch.__version__),'numpy':session.np.__version__}
    for name in ('torch-geometric','torch-scatter','torch-sparse','ogb'):
        try: session.public_versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: session.public_versions[name]=None
    return session,facade


def run(args,public_root,recompute):
    public=load('_public_wiki_attribution_library',public_root/'portable.py')
    previous={name:sys.modules.get(name) for name in ('portable','data_interface')}; old_argv=sys.argv
    sys.modules['portable']=public; box={}; started=time.monotonic(); usage_start=resource.getrusage(resource.RUSAGE_SELF) if resource else None
    specification=identity(args.condition,recompute); completed=False; own_output=False
    try:
        data=load('_public_wiki_attribution_data',public_root/'data_interface.py'); sys.modules['data_interface']=data
        driver=load('_public_wiki_attribution_full_driver',public_root/'train.py')
        original_write,original_snapshot=driver.json_write,driver.joint_snapshot
        def config(task):
            require(task=='wikics','WikiCS attribution only'); return recipe(public,specification)
        def factory(task,arm,seed,device,polynormer,ncn_model,ncn_utils):
            require(task=='wikics' and arm==specification['method_identity'] and seed==args.seed,'One explicit public condition/seed')
            session,facade=make_session(public,recompute,specification,seed,device,polynormer)
            box.update(session=session,facade=facade); return session
        def annotate(value):
            session,facade=box.get('session'),box.get('facade')
            return {**value,'public_attribution':specification,'method_identity':specification['method_identity'],
                'execution_accounting':dict(session.execution_totals) if session else None,
                'public_execution_cost':dict(session.public_cost) if session else None,
                'actual_provider_versions':session.public_versions if session else None,
                'active_original_loss_calls':dict(alignment=facade.alignment_source_calls,residual=facade.residual_source_calls) if facade else None,
                'author_execution_equivalence_claimed':False,'bitwise_author_parity_claimed':False,
                'part_of_registered_author_Wiki12':False}
        def write(path,value):
            nonlocal own_output
            if isinstance(value,dict) and Path(path).name in ('RUN.json','COMPLETE.json','FAILURE.json','PROGRESS.json'):
                own_output=True; value=annotate(value)
                if Path(path).name=='COMPLETE.json':
                    weights=CONDITIONS[args.condition]
                    require(value['epochs']==value['steps']==1100 and value['active_original_loss_calls']==dict(alignment=1100 if weights[0] else 0,residual=1100 if weights[1] else 0),'Full original1100 updates and active losses')
                    require(value['execution_accounting']==dict(shadow_member_forwards=8800,replay_member_forwards=8800,
                        output_cotangent_collections=1100,member_reverse_collections=8800,optimizer_bank_updates=1100,exact_member_RNG_endpoint_checks=1100),'Full recomputation costs')
                    cost=value['public_execution_cost']
                    require(cost['training_member_calls_attempted']==cost['training_member_calls_returned']==17600
                        and cost['development_member_calls_attempted']==cost['development_member_calls_returned']==4400
                        and cost['update_calls_attempted']==cost['update_calls_returned']==1100,'All actual original fullgraph training/serving calls charged')
            original_write(path,value)
        def snapshot(session,epoch,metric,per,run_record):
            value=original_snapshot(session,epoch,metric,per,annotate(run_record)); value['public_attribution']=specification; return value
        driver.Session,driver.recipe,driver.json_write,driver.joint_snapshot=factory,config,write,snapshot
        sys.argv=['public-wikics-attribution','--task','wikics','--arm',specification['method_identity'],'--seed',str(args.seed),
            '--device',args.device,'--polynormer',str(args.polynormer),'--train',str(args.train),'--valid',str(args.valid),'--output',str(args.output)]
        try:
            driver.main()  # Unchanged1100epoch/two-own-view/local-transition/full-development selection driver.
            completed=True
        finally:
            if own_output:
                usage=resource.getrusage(resource.RUSAGE_SELF) if resource else None
                session=box.get('session')
                original_write(args.output/'PUBLIC_COST.json',annotate(dict(complete=completed,inclusive_wrapper_wall_seconds=time.monotonic()-started,
                    CPU_user_seconds=usage.ru_utime-usage_start.ru_utime if usage else None,CPU_system_seconds=usage.ru_stime-usage_start.ru_stime if usage else None,
                    peak_RSS_bytes=int(usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024)) if usage else None,
                    peak_CUDA_allocated_bytes=int(session.torch.cuda.max_memory_allocated(session.cuda_index)) if session and session.cuda_index is not None else None,
                    peak_CUDA_reserved_bytes=int(session.torch.cuda.max_memory_reserved(session.cuda_index)) if session and session.cuda_index is not None else None,
                    TEST_scoring=False,automatic_retry=False,exact_resume_supported=False)))
    finally:
        sys.argv=old_argv
        for name,value in previous.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=tuple(CONDITIONS),required=True)
    parser.add_argument('--seed',type=int,default=6101)
    parser.add_argument('--device',default='cpu',help='cpu or cuda[:index] in your own environment')
    parser.add_argument('--train',type=Path,required=True,help='Complete TRAIN safe NPZ')
    parser.add_argument('--valid',type=Path,required=True,help='Complete development safe NPZ')
    parser.add_argument('--polynormer',type=Path,required=True,help='Pinned upstream model.py supplied by caller')
    parser.add_argument('--output',type=Path,required=True,help='Fresh caller output directory')
    parser.add_argument('--public-interface',type=Path,default=HERE.parent/'portable_internal_be_public_interface_20261007_v2',help='Sealed public v2 source directory; any local path')
    args=parser.parse_args()
    for name in ('train','valid','polynormer','output'): setattr(args,name,getattr(args,name).resolve())
    require(not args.output.exists(),'Fresh caller output required')
    root=public_dependency(args.public_interface); recompute=load('_public_wikics_v2_recompute',HERE/'recompute.py')
    run(args,root,recompute)


if __name__=='__main__': main()
