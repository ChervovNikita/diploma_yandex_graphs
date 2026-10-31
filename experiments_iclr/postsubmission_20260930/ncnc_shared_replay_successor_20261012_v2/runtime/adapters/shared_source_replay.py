"""Immutable shared integer tapes for source LinkPrototype learners only.

No sampling law or learning work changes. A separately admitted physical
runtime/device is required before any prospective scientific store is opened.
All source failures, tapes, leases and consumer costs are retained.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
try:
    from ..shared_replay_store import SharedReplayStore, atomic_json, digest, file_pin, stream_key
except ImportError:
    from shared_replay_store import SharedReplayStore, atomic_json, digest, file_pin, stream_key
from .common_native_wrapper import bound


def confined(root, relative):
    value=Path(relative);path=(root/value).resolve()
    if value.is_absolute() or '..' in value.parts or not path.is_relative_to(root):
        raise ValueError('Shared replay authority must remain inside the project')
    return path


def physical_binding(torch, device):
    if device.type!='cuda':raise ValueError('Scientific source replay requires an admitted CUDA device')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible:raise ValueError('Explicit physical CUDA device binding required')
    tokens=visible.split(',');logical=torch.cuda.current_device() if device.index is None else device.index
    if not 0<=logical<len(tokens):raise ValueError('CUDA visibility differs from the source factory device')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10)
    devices=dict(tuple(v.strip() for v in row.split(',')) for row in rows.splitlines())
    token=tokens[logical];gpu=devices.get(token,token if token in devices.values() else None)
    if gpu is None:raise ValueError('Unknown physical source CUDA device')
    return {'hostname':socket.gethostname(),'gpu_uuid':gpu,'CUDA_VISIBLE_DEVICES':visible,'logical_device':logical}


def admitted_runtime(torch, device, root, policy, data):
    # The fixture branch cannot enter the actual make_adapter factory, which
    # requires the full235868-node collab schema and scientific policy.
    if policy.get('authority_scope')=='synthetic_fixture_only':
        if device.type!='cpu' or torch.cuda.is_initialized() or data['nodes']>64:
            raise ValueError('Synthetic-only replay proof must remain tiny CPU data')
        return {'fixture_only':True,'device_binding':{'hostname':'fabricated_CPU_fixture','gpu_uuid':None},
                'versions':{'torch':torch.__version__},'CPU_provider_fence_AST':policy['CPU_provider_fence_AST']}
    if policy.get('schema')!='LinkPrototype-shared-source-replay-policy-v1' or policy.get('authority_scope')!='prospective_scientific_TRAIN_integers':
        raise ValueError('Explicit prospective source replay policy required')
    actual=physical_binding(torch,device)
    admission_path=confined(root,policy['physical_admission_directory'])/(actual['hostname']+'__'+actual['gpu_uuid']+'.json')
    admission=json.loads(admission_path.read_text())
    if (admission.get('schema')!='shared-source-replay-physical-admission-v1'
            or admission.get('admitted') is not True or not admission.get('root_authorization_reference')
            or admission.get('qualification_evidence_is_not_scientific_donor') is not True
            or admission.get('device_binding')!=actual
            or admission.get('integration_source_receipts')!=policy['integration_source_receipts']
            or not admission.get('source_epoch1_qualification_receipt')):
        raise ValueError('Separate exact source/runtime/device admission is required')
    runtime=copy.deepcopy(admission['runtime']);runtime['device_binding']=actual
    if (Path(sys.executable).resolve()!=Path(runtime['interpreter']).resolve()
            or file_pin(sys.executable)!=runtime['interpreter_pin']
            or {n:importlib.metadata.version(n) for n in runtime['versions']}!=runtime['versions']
            or not runtime['binary_file_pins']):
        raise ValueError('Physical replay runtime interpreter/versions are not admitted')
    for row in runtime['binary_file_pins']:
        if file_pin(row['path'])!={k:row[k] for k in ('bytes','sha256')}:
            raise ValueError('Physical replay numerical binary authority changed')
    profile={'deterministic':torch.are_deterministic_algorithms_enabled(),
             'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
             'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32}
    if runtime['profile']!=profile:raise ValueError('Physical replay numerical flags changed')
    for row in policy['integration_source_receipts'].values():bound(root,row)
    return runtime


class SourceReplayLease:
    def __init__(self, adapter):
        self.a=adapter;self.root=Path(adapter.root).resolve();self.policy=adapter.config['shared_source_replay']
        runtime=admitted_runtime(adapter.t,adapter.device,self.root,self.policy,adapter.data)
        source={**adapter.config['prototype_sources'],**self.policy.get('integration_source_receipts',{})}
        geometry=copy.deepcopy(adapter.data['identity']['TRAIN'])
        key=stream_key(input_receipts=adapter.config['link_inputs'],source_receipts=source,
                       geometry=geometry,master_seed=2026100401,runtime=runtime)
        key['authority_scope']=self.policy['authority_scope']
        # No qualified fixture directory or scientific model checkpoint is a donor.
        self.store=SharedReplayStore(confined(self.root,self.policy['store_root']),key)
        self.consumer='fit_'+digest({'study_id':adapter.provenance['data'],
                                    'run_id':adapter.run['run_id'],'run_config':adapter.config})
        self.leases={};self.costs=self.store.store/'consumer_costs'/self.consumer
        self.costs.mkdir(parents=True,exist_ok=True)

    def epoch(self, epoch):
        a=self.a;before=a.rt['state'].rng_digest(a.rt['state'].rng_state())
        tick=time.monotonic();attempt=self.costs/('epoch_'+str(epoch)+'_'+uuid.uuid4().hex+'.json')
        record={'schema':'source-shared-replay-consumer-cost-v1','consumer':self.consumer,'epoch':epoch,
                'key_hash':self.store.key_hash,'status':'RUNNING','source_draw_attempts':0,'source_rosters_written':0,
                'source_integer_bytes_written':0,'source_verifier_calls':0,'scientific_optimizer_updates':0}
        atomic_json(attempt,record);readers={}
        def producer(stage):
            record['source_draw_attempts']+=1
            permutation,negative=a.provider.draw_epoch(a.rt,a.data,2026100401,epoch)
            a.provider.write_epoch(a.rt,a.data,permutation,negative,2026100401,epoch,stage)
            record['source_rosters_written']+=1
            record['source_integer_bytes_written']+=sum((stage/n).stat().st_size for n in ('PERMUTATION.npy','NEGATIVE_PAIRS.npy'))
        def verifier(stage):
            state_before=a.rt['state'].rng_digest(a.rt['state'].rng_state())
            reader=a.provider.EpochReplay(a.rt,a.data,stage,file_pin(stage/'EPOCH.json')['sha256'])
            r=reader.receipt
            if r['master_seed']!=2026100401 or r['epoch']!=epoch or r['full_batches']!=a.config['train_minibatches_per_epoch']:
                raise ValueError('Complete native epoch/master/tail authority changed')
            if a.rt['state'].rng_digest(a.rt['state'].rng_state())!=state_before:
                raise ValueError('Shared source reader changed learner RNG')
            record['source_verifier_calls']+=1;readers.clear();readers[str(stage)]=reader
        try:
            directory,manifest=self.store.ensure_epoch(epoch,producer,verifier)
            lease=self.store.bind_consumer(self.consumer,epoch,directory,manifest)
            state={'epoch':epoch,'key_hash':self.store.key_hash,'manifest':manifest,'lease':file_pin(lease),
                   'source_epoch_receipt':file_pin(directory/'EPOCH.json')}
            if str(epoch) in self.leases and self.leases[str(epoch)]!=state:
                raise ValueError('A paid retry cannot replace its immutable epoch lease')
            self.leases[str(epoch)]=state
            if a.rt['state'].rng_digest(a.rt['state'].rng_state())!=before:
                raise ValueError('Source sharing altered the caller learning stream')
            record.update(status='PASS',generated_by_this_consumer=record['source_draw_attempts']==1,
                          reused_published_source_epoch=record['source_draw_attempts']==0,
                          manifest=manifest,lease=file_pin(lease),RNG_before=before,RNG_after=before)
            return readers[str(directory)]
        except BaseException as error:
            record.update(status='FAILED',error_type=type(error).__name__,error=str(error),failed_attempt_preserved=True)
            raise
        finally:
            record['wall_seconds']=time.monotonic()-tick
            record['RNG_restored']=a.rt['state'].rng_digest(a.rt['state'].rng_state())==before
            atomic_json(attempt,record)

    def capture(self):
        return {'schema':'paid-source-replay-lease-checkpoint-v1','key_hash':self.store.key_hash,
                'consumer':self.consumer,'epochs':copy.deepcopy(self.leases),'release_implemented':False}

    def restore(self, value):
        if (value.get('schema')!='paid-source-replay-lease-checkpoint-v1' or value.get('key_hash')!=self.store.key_hash
                or value.get('consumer')!=self.consumer or value.get('release_implemented') is not False):
            raise ValueError('Checkpoint/retry source/runtime/device/consumer authority changed')
        for text,state in value['epochs'].items():
            epoch=int(text);directory=self.store.store/('epoch_'+text)
            lease=self.store.store/'consumer_leases'/(self.consumer+'_epoch_'+text+'.json')
            publication=self.store.store/('PUBLISHED_'+text+'.json')
            if (not 1<=epoch<=100 or state['epoch']!=epoch or state['key_hash']!=self.store.key_hash
                    or file_pin(directory/'SHARED_MANIFEST.json')!=state['manifest']
                    or file_pin(directory/'EPOCH.json')!=state['source_epoch_receipt']
                    or file_pin(lease)!=state['lease']
                    or json.loads(publication.read_text())!={'key_hash':self.store.key_hash,'epoch':epoch,'manifest':state['manifest']}):
                raise ValueError('Paid source epoch/lease publication was lost or replaced')
        # Actual integer arrays are fully checked by source EpochReplay before
        # every subsequent use. This pointer restore never redraws old epochs.
        self.leases=copy.deepcopy(value['epochs'])
