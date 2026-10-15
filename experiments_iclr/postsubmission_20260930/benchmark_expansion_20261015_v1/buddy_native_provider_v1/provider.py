"""Compact authority around genuine native BUDDY inputs; no oracle tensor load."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

HERE=Path(__file__).resolve().parent
EXPANSION=HERE.parent
PHASE=EXPANSION.parent
SCHEMA='BUDDY-production-native-provider-state-v1'


def receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def load(path):
    path=Path(path).resolve();name='_BUDDY_native_provider_'+hashlib.sha256(str(path).encode()+path.read_bytes()).hexdigest()
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def tensor(value):
    array=value.detach().cpu().contiguous().numpy()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
            'sha256':hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def rng(torch):
    import numpy as np
    ns=np.random.get_state()
    return digest({'python':random.getstate(),'numpy':[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],'torch_cpu':tensor(torch.get_rng_state())})


def bindings(root,expected):
    root=Path(root).resolve()
    if not HERE.is_relative_to(root):raise ValueError('Bind actual source repository')
    pin=receipt(HERE/'MANIFEST.json')
    if pin['sha256']!=expected:raise ValueError('Pin exact new provider manifest')
    pins=[pin]
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path=(HERE/row['path']).resolve()
        if not path.is_relative_to(HERE):raise ValueError('Provider source escapes package')
        actual=receipt(path)
        if any(actual[key]!=row[key] for key in ('bytes','sha256')):raise ValueError('Frozen provider source changed')
        pins.append(actual)
    for row in json.loads((HERE/'SOURCES.json').read_text())['source_files']:
        path=(PHASE/row['path_relative_to_phase']).resolve()
        if not path.is_relative_to(root):raise ValueError('Original native source escapes repository')
        actual=receipt(path)
        if any(actual[key]!=row[key] for key in ('bytes','sha256')):raise ValueError('Original native source changed')
        pins.append(actual)
    return pins


class NativeProviderSession:
    def __init__(self,base,job,pins):
        self.base,self.native,self.job=base,base.native,copy.deepcopy(job)
        n=self.native;self.t=n.t
        if base.family!='BUDDY' or n.members!=4 or n.procedure!='base' or n.updates!=0 or base.candidate_id is not None:
            raise ValueError('Fresh native BUDDY boundary base required; method integration is separate')
        if n.args.year!=2007 or n.args.sign_k!=0 or n.args.batch_size!=1024 or int(n.seed)!=job['seed']:
            raise ValueError('Original native settings and actual fit seed required')
        static_pin=receipt(HERE/'STATIC_PROVIDER.json')
        if static_pin['sha256']!=job['method_settings']['static_provider_sha256']:
            raise ValueError('Pin the compact static provider authority')
        static=json.loads((HERE/'STATIC_PROVIDER.json').read_text())
        if base.native_binding['data_files']!=static['provider_data_files'] or n.recipe['native_fixed_config']!=static['native_fixed_config']:
            raise ValueError('Native GRAPH/TRAIN/VALID provider or fixed settings changed')
        from importlib.metadata import version
        if version('datasketch')!=static['datasketch_version']:
            raise ValueError('Native hashing dependency version changed')
        observed={name:tensor(value) for name,value in [('features',n.features),('degrees',n.degrees),('cards',n.cards),('source_positive_queries',n.positive)]}
        observed['hash_tables']={str(hop):{name:tensor(value) for name,value in table.items()} for hop,table in n.hash_tables.items()}
        if observed!=static['static_tensor_bindings']:
            raise ValueError('Actual seed-specific native static inputs differ from compact authority')
        self.metadata_record={'schema':'BUDDY-production-native-provider-v1','provider_mode':'production_native',
            'source_receipts':pins,'static_provider':static_pin,'native_binding':copy.deepcopy(base.native_binding),
            'fit_seed':int(n.seed),'split':int(job['split']),'body':int(job['body']),
            'initial_source_parameters':{name:tensor(p) for name,p in n.source_parameter_references.items()},
            'initial_wrapper_parameters':{name:tensor(p) for name,p in n.core.named_parameters()},
            'actual_sampled_negative_queries':tensor(n.train_negative),'actual_static_inputs':observed,
            'post_constructor_rng_sha256':rng(self.t),'native_epoch_layout':copy.deepcopy(base.layout),
            'oracle_tensor_files_loaded':0,'oracle_tensor_data_injected':False,
            'source_sampler_hydration_returns':'exact original returned objects; compact metadata only',
            'qualification':'input parity pending; method integration/full epoch/selected serving/CUDA remain separate',
            'TEST_accessed':False}
        self.ledger={'sampler_calls':0,'TRAIN_hydration_calls':0,'VALID_hydration_calls':0,
            'query_chain_sha256':digest([]),'TRAIN_observations':[],'VALID_observations':[]}
        self._role=None;self._query_traversal=False;self._pending=None;self._closed=False
        self._original_next,self._original_hydrate=n._next_batch,n._hydrate
        def next_batch(owner):
            if self._role not in ('TRAIN','QUERY_TRAIN'):raise RuntimeError('Owned native TRAIN or explicit query traversal required')
            before=rng(self.t);ids=self._original_next();after=rng(self.t)
            row={'epoch':int(n.epoch),'batch_cursor':int(n.batch_cursor),'source_sampler_ids':tensor(ids),
                'source_links':tensor(n.links[ids]),'source_labels':tensor(n.labels[ids]),'records':int(ids.numel()),
                'native_rng_before':before,'native_rng_after':after}
            self.ledger['sampler_calls']+=1
            self.ledger['query_chain_sha256']=digest([self.ledger['query_chain_sha256'],row])
            inspect=n.batch_cursor in (1,2,base.layout['optimizer_maps_per_epoch'])
            self._pending={'record':row,'ids':ids.detach().cpu().clone(),'inspect':inspect}
            if rng(self.t)!=after:raise RuntimeError('Provider metadata consumed source RNG')
            return ids
        def hydrate(owner,links):
            if self._role not in ('TRAIN','QUERY_TRAIN','VALID','QUERY_VALID'):raise RuntimeError('Owned native input role required')
            before=rng(self.t);values=self._original_hydrate(links);after=rng(self.t)
            if self._role in ('TRAIN','QUERY_TRAIN'):
                if self._pending is None or not self.t.equal(links.detach().cpu(),n.links[self._pending['ids']].detach().cpu()):
                    raise ValueError('Hydration does not match actual source sampler IDs')
                self.ledger['TRAIN_hydration_calls']+=1
                if self._pending['inspect']:
                    row=copy.deepcopy(self._pending['record']);row.update(inputs=[tensor(x) for x in values],hydration_rng_before=before,hydration_rng_after=after)
                    self.ledger['TRAIN_observations'].append(row)
            else:
                roles=[name for name in ('edge','edge_neg') if links.shape==n.valid[name].shape and self.t.equal(links.detach().cpu(),n.valid[name].detach().cpu())]
                if len(roles)!=1:raise ValueError('Complete original native VALID role required')
                self.ledger['VALID_hydration_calls']+=1
                self.ledger['VALID_observations'].append({'role':roles[0],'source_links':tensor(links),'inputs':[tensor(x) for x in values],
                    'hydration_rng_before':before,'hydration_rng_after':after})
            if rng(self.t)!=after:raise RuntimeError('Provider metadata consumed source RNG')
            return values
        n._next_batch,n._hydrate=types.MethodType(next_batch,n),types.MethodType(hydrate,n)
        def guard(module,inputs):
            if self._role in ('QUERY_TRAIN','QUERY_VALID') or self._query_traversal:
                raise RuntimeError('Input query traversal may not execute a model forward or claim training')
        self._forward_guard=n.model.register_forward_pre_hook(guard)
        self.assert_identity()

    def __getattr__(self,name):return getattr(self.base,name)

    def assert_identity(self):self.base.assert_identity()

    @contextmanager
    def operation(self,role):
        if self._closed or self._role is not None:raise RuntimeError('One open nonnested provider operation required')
        if role.startswith('QUERY_'):self._query_traversal=True
        elif self._query_traversal:raise RuntimeError('Query-only traversal cannot become a training fit')
        self._role=role
        try:yield self.native
        finally:self._role=None;self._pending=None;self.assert_identity()

    def train_update(self):
        with self.operation('TRAIN'):return self.base.train_update()

    def train_epoch(self):
        with self.operation('TRAIN'):return self.base.train_epoch()

    def validate(self):
        with self.operation('VALID'):return self.base.validate()

    def metadata(self):
        return {**self.base.metadata(),'native_BUDDY_provider':copy.deepcopy(self.metadata_record)}

    def state(self):
        if self._role is not None or self._query_traversal:raise RuntimeError('Only an idle genuine fit may be checkpointed')
        return {'schema':SCHEMA,'provider':copy.deepcopy(self.metadata_record),'ledger':copy.deepcopy(self.ledger),'native_integration':self.base.state()}

    def load_state(self,value):
        if self._role is not None or self._query_traversal or value.get('schema')!=SCHEMA or value.get('provider')!=self.metadata_record:
            raise ValueError('Exact idle seed/source/provider identity required for restore')
        self.base.load_state(value['native_integration']);self.ledger=copy.deepcopy(value['ledger']);self._pending=None
        self.assert_identity()

    def close(self):
        if self._role is not None:raise RuntimeError('Cannot close an active provider operation')
        if self._closed:return
        self.native._next_batch,self.native._hydrate=self._original_next,self._original_hydrate
        self._forward_guard.remove();self._closed=True;self.base.close()
