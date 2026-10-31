"""Optional route activation recomputation for the pinned PolyFormer bank API.

No source model/loss/optimizer/controller is rewritten. Install only around a
caller-owned forward/loss/backward block. Evaluation and no-grad calls use the
original route directly. Checkpoint inputs retain the existing token_0 object,
not a cloned token bank. Per-route checkpoints are non-reentrant and replay a
complete source call, including its ordinary hooks and validation.
"""
from contextlib import contextmanager,nullcontext
import time

_MISSING=object()

def _same_callable(left,right):
    return left is right or (getattr(left,'__self__',None) is getattr(right,'__self__',_MISSING)
      and getattr(left,'__func__',None) is getattr(right,'__func__',_MISSING))

def _clone_stream(torch,value):
    if not isinstance(value,dict) or set(value)!={'cpu','cuda'}:
        raise ValueError('Exact pinned private CPU/CUDA stream state required')
    if not torch.is_tensor(value['cpu']) or value['cpu'].dtype!=torch.uint8 or value['cpu'].device.type!='cpu':
        raise ValueError('Private CPU RNG state must be a CPU byte tensor')
    if value['cuda'] is not None and (not torch.is_tensor(value['cuda']) or value['cuda'].dtype!=torch.uint8):
        raise ValueError('Private CUDA RNG state must be a byte tensor or None')
    return {key:None if field is None else field.clone() for key,field in value.items()}

class RouteRecompute:
    """Reversible instance-only forward_route wrapper with explicit work counts.

The source forward must be pure in model/optimizer/factual tensors and may
advance only the selected private dropout stream. This is the pinned bank
contract; arbitrary running-stat/stateful forwards require another utility.
    """
    def __init__(self,torch,bank,*,enabled=True):
        if type(enabled) is not bool:raise ValueError('Explicit optional recomputation flag required')
        if (not isinstance(bank,torch.nn.Module) or not hasattr(bank,'token_0') or not callable(getattr(bank,'forward_route',None))
                or not isinstance(getattr(bank,'streams',None),list) or not bank.streams or not isinstance(getattr(bank,'members',None),int)):
            raise ValueError('Pinned bank streams/forward_route/token_0/members interface required')
        if (bank.token_0.dtype!=torch.float32 or bank.token_0.ndim!=2 or bank.token_0.requires_grad
                or bank.token_0.device.type not in ('cpu','cuda')):
            raise ValueError('Existing frozen float32 CPU/CUDA token_0 required')
        for state in bank.streams:_clone_stream(torch,state)
        if getattr(bank.forward_route,'_route_recompute_wrapper',False):raise ValueError('No nested wrapper on one bank')
        self.t,self.bank,self.enabled=torch,bank,enabled
        self.original=bank.forward_route;self.prior=bank.__dict__.get('forward_route',_MISSING)
        self.records=[];self.closed=False;self.installed=False
        self.counts={'logical_checkpoint_route_calls':0,'completed_logical_route_forwards':0,
          'recomputed_route_forward_attempts':0,'completed_recomputed_route_forwards':0,
          'bypassed_route_calls':0,'failed_route_invocations':0}
        self.host_seconds={'logical_route_calls':0.,'recomputed_route_calls':0.,'bypassed_route_calls':0.}
        def wrapped(route,*,slot=None):return self._forward(route,slot=slot)
        wrapped._route_recompute_wrapper=True
        self.wrapped=wrapped

    def __enter__(self):
        if self.closed or self.installed:raise ValueError('One wrapper installation per owned handle')
        if not _same_callable(self.bank.forward_route,self.original):
            raise ValueError('Route owner changed before wrapper installation')
        self.bank.forward_route=self.wrapped;self.installed=True;return self

    def close(self):
        if self.closed:return
        if self.installed:
            if self.bank.__dict__.get('forward_route') is not self.wrapped:raise ValueError('Cannot restore an unowned changed route wrapper')
            if self.prior is _MISSING:delattr(self.bank,'forward_route')
            else:self.bank.forward_route=self.prior
        self.installed=False;self.closed=True

    def __exit__(self,kind,error,traceback):self.close();return False

    def _forward(self,route,*,slot):
        t,bank=self.t,self.bank
        if type(route) is not int or not 0<=route<bank.members:raise ValueError('Exact source integer route required')
        active=route if slot is None else slot
        if type(active) is not int or not 0<=active<len(bank.streams):raise ValueError('Exact private stream slot required')
        if not self.enabled or not bank.training or not t.is_grad_enabled():
            self.counts['bypassed_route_calls']+=1;tick=time.perf_counter()
            try:return self.original(route,slot=slot)
            finally:self.host_seconds['bypassed_route_calls']+=time.perf_counter()-tick
        # Snapshot just this call's pre-draw state, not its post-call state.
        # Later views may logically advance the same slot before backward.
        pre=_clone_stream(t,bank.streams[active]);roster_size=len(bank.streams)
        original_modes=tuple((module,module.training) for module in bank.modules())
        token=bank.token_0;version=token._version
        record={'logical_call':len(self.records),'route':route,'slot':active,'logical_forward_completed':False,
          'recomputed_forward_attempts':0,'recomputed_forward_completed':0,
          'RNG_snapshot_bytes':sum(v.numel()*v.element_size() for v in pre.values() if v is not None),
          'token_input_cloned':False,'full_source_route_replay':True}
        self.records.append(record);self.counts['logical_checkpoint_route_calls']+=1
        phase={'recompute':False}
        @contextmanager
        def replay():
            if len(bank.streams)!=roster_size:raise ValueError('Private stream roster changed before backward')
            if bank.token_0 is not token or token._version!=version:raise ValueError('Frozen source token_0 changed before replay')
            # Keep the latest logical slot state. In paired losses this can
            # already be two or more source forwards beyond the stored pre.
            logical_state=_clone_stream(t,bank.streams[active])
            logical_object=bank.streams[active]
            caller_modes=tuple((module,module.training) for module in bank.modules())
            bank.streams[active]=_clone_stream(t,pre)
            for module,training in original_modes:module.training=training
            phase['recompute']=True
            try:yield
            finally:
                # Restore the already-advanced logical state even if the
                # source replay or autograd fails. Preserve its dict owner.
                logical_object.clear();logical_object.update(logical_state)
                bank.streams[active]=logical_object
                for module,training in caller_modes:module.training=training
                phase['recompute']=False
        def invoke(existing_token):
            if existing_token is not token or bank.token_0 is not token or token._version!=version:
                raise ValueError('Checkpoint must reuse the same factual source token_0')
            repeated=phase['recompute'];key='recomputed_route_calls' if repeated else 'logical_route_calls'
            if repeated:
                record['recomputed_forward_attempts']+=1;self.counts['recomputed_route_forward_attempts']+=1
            tick=time.perf_counter()
            try:
                result=self.original(route,slot=slot)
                if repeated:
                    record['recomputed_forward_completed']+=1;self.counts['completed_recomputed_route_forwards']+=1
                else:
                    record['logical_forward_completed']=True;self.counts['completed_logical_route_forwards']+=1
                return result
            except BaseException:
                self.counts['failed_route_invocations']+=1;raise
            finally:self.host_seconds[key]+=time.perf_counter()-tick
        # Reentrant checkpointing would require a differentiable input or
        # dummy interpolation. Non-reentrant records the actual source graph
        # even though every mono token is deliberately frozen.
        from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
        with set_checkpoint_early_stop(False):
            return checkpoint(invoke,token,use_reentrant=False,preserve_rng_state=True,
              context_fn=lambda:(nullcontext(),replay()),determinism_check='default')

    def report(self):
        logical=self.counts['completed_logical_route_forwards'];repeated=self.counts['completed_recomputed_route_forwards']
        return {'schema':'optional-source-route-recompute-work-v1','enabled':self.enabled,'counts':dict(self.counts),
          'logical_source_route_forwards':logical,'additional_full_source_route_forwards':repeated,
          'total_completed_training_route_forwards':logical+repeated,'recompute_overhead_forward_ratio':None if logical==0 else repeated/logical,
          'host_route_call_seconds':dict(self.host_seconds),'host_time_scope':'unsynchronized CPU-call wall time; not CUDA execution or study ETA',
          'records':[dict(row) for row in self.records],'route_calls_change_source_loss':False,'optimizer_actions':0,
          'token_bank_copies_created':0,'checkpoint_kind':'non-reentrant, full source forward_route replay, preserved global and private RNG',
          'methodological_novelty_claim':False,'actual_input_memory_qualified':False}

def route_recompute(torch,bank,*,enabled=True):
    """Use ``with route_recompute(torch, bank) as handle: ... loss.backward()``."""
    return RouteRecompute(torch,bank,enabled=enabled)
