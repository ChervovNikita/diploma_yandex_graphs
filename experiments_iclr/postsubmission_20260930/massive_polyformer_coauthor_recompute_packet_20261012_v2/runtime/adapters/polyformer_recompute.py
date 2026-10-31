"""Optional source PolyFormer TRAIN route replay with separately charged work.

The delegated adapter owns objectives, backward, optimizer, selection and state.
This wrapper records execution work without changing its logical counters.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .polyformer import make_adapter as source_make_adapter
from .route_recompute import route_recompute

UTILITY_SHA256='90100bbcb992bb416e2bfca539bbe56f79159b176c90a16d091560505383adfa'
METHOD_FORWARDS={'polyformer_shared4_paired_class_full':8,'polyformer_shared4_gncl_supcon':4}
COUNT_KEYS=('logical_checkpoint_route_calls','completed_logical_route_forwards',
            'recomputed_route_forward_attempts','completed_recomputed_route_forwards',
            'bypassed_route_calls','failed_route_invocations')
HOST_KEYS=('logical_route_calls','recomputed_route_calls','bypassed_route_calls')


def execution_option(method,*,enabled=True):
    if method not in METHOD_FORWARDS or type(enabled) is not bool:
        raise ValueError('One declared source callback and explicit boolean required')
    count=METHOD_FORWARDS[method]
    return {'schema':'polyformer-route-recompute-execution-v1','enabled':enabled,
            'utility_sha256':UTILITY_SHA256,'logical_route_forwards_per_train_unit':count,
            'planned_full_replay_forwards_per_train_unit':count if enabled else 0,
            'training_only':True,'source_objective_changed':False}


def validate_option(option,method):
    if option!=execution_option(method,enabled=option.get('enabled')):
        raise ValueError('Exact fixed execution option required; not a search coordinate')
    return option


def compact_report(report):
    counts=report['counts']
    if set(counts)!=set(COUNT_KEYS) or any(type(v) is not int or v<0 for v in counts.values()):
        raise ValueError('Exact nonnegative source work counters required')
    return {'counts':dict(counts),'RNG_snapshot_bytes':sum(r['RNG_snapshot_bytes'] for r in report['records']),
            'host_route_call_seconds':dict(report['host_route_call_seconds']),
            'token_bank_copies_created':report['token_bank_copies_created'],
            'time_scope':'unsynchronized route host-call wall time; source TRAIN unit elapsed time belongs to engine'}


class PaidReplayCosts:
    """Compact paid-work sidecar; checkpoint restore never rewinds these costs."""
    def __init__(self,identity,path=None):
        self.path=None if path is None else Path(path)
        self.identity=dict(identity)
        self.value={'schema':'polyformer-paid-route-recompute-costs-v1','identity':self.identity,
          'train_units_attempted':0,'source_train_units_completed':0,'train_units_failed':0,
          'counts':{k:0 for k in COUNT_KEYS},'RNG_snapshot_bytes':0,
          'host_route_call_seconds':{k:0. for k in HOST_KEYS},'last_unit':None,
          'persistence_boundary':'Each complete or Python-exception TRAIN callback; hard-killed active unit may be unknown',
          'partial_optimizer_updates':'Source exception uses unchanged engine unknown-partial-cost policy',
          'replay_is_additional_source_forward_work':True,'model_or_optimizer_state':False}
        if self.path is not None and self.path.exists():
            if self.path.is_symlink():raise ValueError('Replay sidecar may not be a symlink')
            previous=json.loads(self.path.read_text())
            if previous.get('schema')!=self.value['schema'] or previous.get('identity')!=self.identity:
                raise ValueError('Replay paid-work sidecar belongs to another execution contract')
            for key in ('train_units_attempted','source_train_units_completed','train_units_failed','RNG_snapshot_bytes'):
                if type(previous.get(key)) is not int or previous[key]<0:raise ValueError('Invalid prior paid work')
            if set(previous.get('counts',{}))!=set(COUNT_KEYS) or any(type(v) is not int or v<0 for v in previous['counts'].values()):
                raise ValueError('Invalid prior replay counters')
            if set(previous.get('host_route_call_seconds',{}))!=set(HOST_KEYS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in previous['host_route_call_seconds'].values()):
                raise ValueError('Invalid prior route-call cost')
            self.value=previous

    def record(self,step,report,*,source_completed,error):
        work=compact_report(report);value=self.value
        value['train_units_attempted']+=1
        value['source_train_units_completed']+=int(source_completed)
        value['train_units_failed']+=int(error is not None)
        for key in COUNT_KEYS:value['counts'][key]+=work['counts'][key]
        for key in HOST_KEYS:value['host_route_call_seconds'][key]+=work['host_route_call_seconds'][key]
        value['RNG_snapshot_bytes']+=work['RNG_snapshot_bytes']
        value['last_unit']={'step':step,'source_train_unit_completed':source_completed,
                            'error_type':None if error is None else type(error).__name__,'work':work}
        if self.path is not None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_name(self.path.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp')
            with temporary.open('x') as stream:
                json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
            temporary.replace(self.path)


def install(adapter,option,costs,*,checkpoint_factory=route_recompute):
    """Wrap the exact delegated TRAIN method; ordinary eval/capture/restore stay source owned."""
    if not option['enabled']:return adapter
    expected=option['logical_route_forwards_per_train_unit']
    original=adapter.train_step
    if getattr(original,'_route_recompute_train_wrapper',False):raise ValueError('No nested TRAIN wrapper')
    def train_step(step):
        handle=checkpoint_factory(adapter.t,adapter.model);error=None;source_completed=False
        try:
            with handle:raw=original(step)
            source_completed=True
            report=handle.report();c=report['counts']
            if (c['logical_checkpoint_route_calls']!=expected or c['completed_logical_route_forwards']!=expected
                    or c['recomputed_route_forward_attempts']!=expected or c['completed_recomputed_route_forwards']!=expected
                    or c['bypassed_route_calls'] or c['failed_route_invocations']):
                raise ValueError('Actual complete source route/replay work differs from fixed execution contract')
            result=dict(raw);native=dict(raw['operation_counts'])
            if native['model_forwards']!=expected:raise ValueError('Delegated logical source work differs')
            native.update(route_recompute_logical_forward_attempts=c['logical_checkpoint_route_calls'],
                          route_recompute_logical_forwards=c['completed_logical_route_forwards'],
                          route_recompute_replay_attempts=c['recomputed_route_forward_attempts'],
                          route_recompute_full_replay_forwards=c['completed_recomputed_route_forwards'],
                          TRAIN_route_forwards_including_replay=c['completed_logical_route_forwards']+c['completed_recomputed_route_forwards'])
            result['operation_counts']=native;result['route_recompute_work']=compact_report(report)
            return result
        except BaseException as caught:
            error=caught;raise  # Preserve source type and unknown partial optimizer counts.
        finally:
            report=handle.report();adapter.last_route_recompute_work=compact_report(report)
            try:costs.record(step,report,source_completed=source_completed,error=error)
            except BaseException as storage_error:
                if error is None:raise
                if hasattr(error,'add_note'):error.add_note('Replay cost sidecar write failed: '+str(storage_error))
    train_step._route_recompute_train_wrapper=True
    adapter.train_step=train_step;adapter.paid_route_recompute_costs=costs
    return adapter


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    option=validate_option(run['config']['execution_engineering'],run['method'])
    if not option['enabled']:
        return source_make_adapter(torch=torch,project_root=project_root,study=study,run=run,device=device)
    if study['dataset']!='coauthor-physics' or run['config']['members']!=4 or run['config']['polyformer_mode']!='polyformer_internal_be':
        raise ValueError('Successor admits only its two four-route Physics callbacks')
    utility=Path(__file__).with_name('route_recompute.py')
    if hashlib.sha256(utility.read_bytes()).hexdigest()!=UTILITY_SHA256:raise ValueError('Frozen recomputation utility changed')
    if execution_context is None:
        # Unchanged export code reconstructs a selected learner without a
        # training context. Keep its exact enabled configuration/provenance;
        # serving and checkpoint restore are eager, while unaudited TRAIN is denied.
        adapter=source_make_adapter(torch=torch,project_root=project_root,study=study,run=run,device=device)
        def denied_train(*unused,**keywords):
            raise RuntimeError('Source controller execution context required before TRAIN replay work')
        adapter.train_step=denied_train
        return adapter
    if not isinstance(execution_context,dict) or 'output_root' not in execution_context:
        raise ValueError('Source controller execution context required for paid replay custody')
    root=Path(project_root).resolve();output=Path(execution_context['output_root']).resolve()
    if not output.is_relative_to(root):raise ValueError('Replay costs must stay inside admitted project output')
    for value in (study['study_id'],run['run_id']):
        if not isinstance(value,str) or not value or value in ('.','..') or '/' in value or '\\' in value:
            raise ValueError('Safe source study/run identity required')
    adapter=source_make_adapter(torch=torch,project_root=project_root,study=study,run=run,device=device)
    identity={'study_id':study['study_id'],'run_id':run['run_id'],'method':run['method'],'execution_option':option}
    costs=PaidReplayCosts(identity,output/study['study_id']/run['run_id']/'route_recompute_paid_costs.json')
    return install(adapter,option,costs)
