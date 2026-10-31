"""One exact own-loss update and native serving; no fit or truth release here."""
import time
from runtime import require

def timed(rt,call):
    rt.t.cuda.synchronize(rt.device);start,end=rt.t.cuda.Event(enable_timing=True),rt.t.cuda.Event(enable_timing=True)
    tick=time.monotonic();start.record();value=call();end.record();rt.t.cuda.synchronize(rt.device)
    return value,dict(wall_seconds=time.monotonic()-tick,CUDA_seconds=start.elapsed_time(end)/1000)

def update(rt,model,optimizer,streams,roles,*,gradient_receipt=False):
    model.train();optimizer.zero_grad(set_to_none=True);outside=rt.fp(rt.rng());constants=model.bank_constants();values=[]
    for view in range(model.views):
        logits=model.view_forward(view,constants,lambda v:rt.scope(streams,v))
        require(logits.shape==(19717,3) and rt.t.isfinite(logits).all().item(),'Finite full graph logits')
        ce=rt.F.cross_entropy(logits[roles['train_ids']],roles['train_y']);require(rt.t.isfinite(ce).item(),'Finite complete own TRAIN CE')
        values.append(float(ce.item()))
        # Only this incoming bank's shared Ak/Av/Bq tape survives early views.
        # No common product is detached or carried across an optimizer update.
        (ce/model.views).backward(retain_graph=model.memory is not None and view+1<model.views)
        del logits,ce;rt.guard()
    live={name:p for name,p in model.named_parameters() if p.requires_grad}
    require(all(p.grad is not None and rt.t.isfinite(p.grad).all().item() for p in live.values()),'Finite every complete live CE gradient')
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'Frozen native source has no gradients')
    gradients={name:dict(nonzero=bool((p.grad!=0).any().item()),max_absolute=float(p.grad.abs().max().item())) for name,p in live.items()} if gradient_receipt else None
    optimizer.step();del constants
    require(all(rt.t.isfinite(p).all().item() for p in model.parameters()),'Finite parameter state after one Adam')
    require(all(rt.t.isfinite(v).all().item() for state in optimizer.state.values() for v in state.values() if rt.t.is_tensor(v)),'Finite private Adam moments')
    require(rt.fp(rt.rng())==outside,'Private dropout update changed outside global RNG')
    rt.guard();return dict(mean_own_CE=sum(values)/len(values),view_CE=values,backwards=model.views,Adam_steps=1,gradients=gradients,common_projected_banks_per_update=0 if model.memory is None else 1)

def serve(rt,model):
    modes=rt.modes(model);before=rt.fp(rt.rng())
    try:
        model.eval()
        with rt.t.no_grad():
            logits=model();require(logits.shape==(model.views,19717,3) and rt.t.isfinite(logits).all().item(),'Complete finite native served views')
            probability=logits.softmax(-1).mean(0)
    finally:
        for name,module in model.named_modules():module.training=modes[name]
    require(rt.fp(rt.rng())==before,'Eval changed global RNG');rt.guard();return logits,probability

def decision(rt,logits,probability,ids):
    return logits[0,ids].max(1)[1] if len(logits)==1 else probability[ids].argmax(-1)

def own_steps(rt,optimizer,expected):
    require(all(p in optimizer.state and int(optimizer.state[p]['step'].item())==expected for group in optimizer.param_groups for p in group['params']),'Every own live Adam state has exact trajectory step')
