"""Native Bayesian Cayley d2 and L4 source ports with explicit boundary routes.

The canonical models are BayesBundleSheafDiffusion, never deterministic BSNN.
Native SciPy samples, returned KL, grouped Adam and source stopping are retained.
"""
import copy
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import random
import sys
from . import common_native_wrapper as common
from .nsd_sources import extract
from .classifier import ClassifierAdapter
try:
    from ..engine import AdapterStepError
except ImportError:
    from engine import AdapterStepError

CACHE = {}
VARIANTS = {'bsnn_cayley_d2': (2, False, .3, 0.),
            'bsnn_richer_cayley_d2_l4': (4, True, .2, .4)}
CONTRACT = {'schema': 'native-Bayesian-BSNN-Cayley-d2-new-data-v1',
    'class': 'models.bayes_disc_models.BayesBundleSheafDiffusion',
    'training': 'one native posterior draw per route; own TRAIN NLL plus sigmoid(((epoch-1)%40)/2-10)*native incidence mean Cayley KL; no layer multiplier',
    'optimizer': 'source complete grouped_parameters Adam .01; both groups weight_decay .0005; no clipping or scheduler',
    'selection': 'source four posterior draws per route; AUROC, then lower NLL, then earlier epoch; no checkpoint0; cap500/patience200',
    'native_M1': 'one native slow body, one own trajectory, source4MC-selected state',
    'genuine_I4': 'four independently initialized and own-selected native trajectories, source4MC per body; body0 is exact M1 fit reference',
    'original_GNNM': 'explicit Bayesian task adaptation: source BEBlock replaces native lin1/lin2 only; one shared full native Bayesian interior; four private complete Python/NumPy/SciPy/Torch streams; four NLL+KL backwards averaged then one grouped Adam',
    'serving': 'four posterior draws per route/body; arithmetic probability pool for canonical native and I4; literal raw-head mean across4routes x4posterior samples for adapted original GNNM',
    'MC16_control': 'same source4MC-selected M1 fit; first4draws retained in one fixed16draw sequence;12extra source forwards, no reselection/retraining',
    'new_data': 'only fresh custodied TwitchDE MUSAE binary/raw features/10 role masks; development exposure, no original-paper dataset or historical states'}


def require(value, message):
    if not value:
        raise ValueError(message)


def native(torch, root, B):
    root = Path(root); D = root/B['householder_provider_relative_root']; key = str(root/B['author_directory'])
    for row in B['native_files']:
        common.bound(root, row)
    for row in B['provider']['files']:
        if Path(row['path']).suffix in ('.py', '.cpp', '.so'):
            common.bound(root, row)
    if key in CACHE:
        return CACHE[key]
    previous = {n:m for n,m in sys.modules.items() if n in ('models','lib') or n.startswith(('models.','lib.'))}
    old_paths = list(sys.path); old_ext = os.environ.get('TORCH_EXTENSIONS_DIR'); old_path = os.environ.get('PATH', '')
    for name in previous:
        sys.modules.pop(name)
    try:
        sys.path[:0] = [str(D/'toolchain_setuptools68'), str(D), key]
        os.environ.update(TORCH_EXTENSIONS_DIR=str(D/'extension_build'), MAX_JOBS='1', PATH=str(D/'ninja/data/bin')+':'+old_path)
        module = importlib.import_module('models.bayes_disc_models')
        house = importlib.import_module('torch_householder')
        require(Path(house.__file__).resolve().is_relative_to(D.resolve()), 'Exact owned Householder provider required')
        for n,m in sys.modules.items():
            if n in ('models','lib') or n.startswith(('models.','lib.')):
                require(Path(m.__file__).resolve().is_relative_to(Path(key).resolve()), 'Exact author models/lib origin required')
        CACHE[key] = module.BayesBundleSheafDiffusion
        return CACHE[key]
    finally:
        for name in list(sys.modules):
            if name in ('models','lib') or name.startswith(('models.','lib.')):
                sys.modules.pop(name)
        sys.modules.update(previous); sys.path[:] = old_paths; os.environ['PATH'] = old_path
        if old_ext is None:
            os.environ.pop('TORCH_EXTENSIONS_DIR', None)
        else:
            os.environ['TORCH_EXTENSIONS_DIR'] = old_ext


def helpers(torch, root, B):
    base = extract(root, B['baseline'], ('seed_all','capture_rng','restore_rng','metrics'), {'random':random})
    names = ('factory','numpy_pack','numpy_unpack','scipy_rng','capture','restore','seed_scipy','seed_streams','cpu_tree','exact','selector')
    source = extract(root, B['support'], names, {'require':require})
    return base, source


def optimizer(torch, model):
    sheaf, other = model.grouped_parameters()
    parameters = sheaf+other
    require(len({id(p) for p in parameters}) == len(parameters) == len(list(model.parameters()))
            and {id(p) for p in parameters} == {id(p) for p in model.parameters()}, 'Complete original source grouped ownership')
    return torch.optim.Adam([{'params':sheaf,'weight_decay':.0005}, {'params':other,'weight_decay':.0005}], lr=.01)


def boundary_bank(torch, root, B, body):
    BE = extract(root, B['boundary'], ('BEBlock',), {'torch':torch,'nn':torch.nn}).BEBlock
    nn = torch.nn
    class Boundary(nn.Module):
        def __init__(self, a, b, first):
            super().__init__(); self.block=BE(a,b,first,'cpu',4); self.member=0
        def forward(self, x):
            return self.block(x,self.member)
    prior = {n:p for n,p in body.named_parameters() if not n.startswith(('lin1.','lin2.'))}
    body.lin1=Boundary(body.input_dim,body.hidden_dim,True); body.lin2=Boundary(body.hidden_dim,2,False); body.to(body.device)
    require(all(dict(body.named_parameters())[n] is p for n,p in prior.items()), 'Every native Bayesian interior parameter retained')
    class Bank(nn.Module):
        def __init__(self):
            super().__init__(); self.body=body; self.members=4
        def grouped_parameters(self):
            return self.body.grouped_parameters()
        def route(self, member, x):
            old=body.lin1.member,body.lin2.member; body.lin1.member=body.lin2.member=member; raw=[]
            hook=body.lin2.register_forward_hook(lambda module,inputs,out:raw.append(out))
            try:
                logp,kl=body(x)
                require(len(raw)==1, 'One native raw head plus intact logp/KL tuple')
                return raw[0],logp,kl
            finally:
                hook.remove();body.lin1.member,body.lin2.member=old
    return Bank().to(body.device)


def construct(torch, root, B, graph, variant, arm, seed, device):
    import numpy as np
    require(variant in VARIANTS and arm in ('native_M1','original_GNNM'), 'Typed Bayesian variants and source controls only')
    h,s=helpers(torch,root,B); h.seed_all(np,torch,seed)
    protocol=json.loads(common.bound(root,B['protocols'][variant]).read_bytes()); cls=native(torch,root,B)
    expected=VARIANTS[variant]
    require((protocol['native_args']['layers'],protocol['native_args']['second_linear'],protocol['native_args']['dropout'],protocol['native_args']['input_dropout'])==expected, 'Exact fixed source tuple')
    args={**protocol['native_args'],'input_dim':graph['x'].shape[1],'output_dim':2,'graph_size':len(graph['x']),'device':str(device)}
    edges=graph['edge_index'].to(device)
    factory=s.factory(torch,cls,graph['edge_index'],edges,args)
    body=factory(); s.seed_scipy(np,body,seed)
    model=body if arm=='native_M1' else boundary_bank(torch,root,B,body)
    model.members=1 if arm=='native_M1' else 4
    opt=optimizer(torch,model)
    return model,opt,h,s,args


def plain_capture(torch, model):
    """All native nonregistered tensors and plain source flags, with object aliases."""
    tensors={}; identities={}; attributes={}; flags={}
    def pack(value):
        if torch.is_tensor(value):
            key=identities.setdefault(id(value),str(len(identities)))
            if key not in tensors:
                tensors[key]={'value':value.detach().cpu().clone(),'requires_grad':value.requires_grad}
            return {'tensor_alias':key}
        if isinstance(value,(tuple,list)) and all(torch.is_tensor(v) for v in value):
            return {'tuple' if isinstance(value,tuple) else 'list':[pack(v) for v in value]}
        return None
    for name,module in model.named_modules():
        attributes[name]={}; flags[name]={}
        for attr,value in vars(module).items():
            if attr.startswith('_') or attr=='training':
                continue
            encoded=pack(value)
            if encoded is not None:
                attributes[name][attr]=encoded
            elif attr=='L' and value is None:
                attributes[name][attr]={'source_none':True}
            elif type(value) in (str,int,float,bool):
                flags[name][attr]=value
            elif isinstance(value,torch.device):
                flags[name][attr]={'torch_device':str(value)}
    return {'tensors':tensors,'attributes':attributes,'flags':flags}


def plain_restore(torch, model, state, device):
    modules=dict(model.named_modules()); copies={k:v['value'].to(device).clone().requires_grad_(v['requires_grad']) for k,v in state['tensors'].items()}
    require(set(modules)==set(state['attributes'])==set(state['flags']), 'Complete native module state identity')
    def unpack(value):
        if 'tensor_alias' in value:return copies[value['tensor_alias']]
        if 'source_none' in value:return None
        if 'tuple' in value:return tuple(unpack(v) for v in value['tuple'])
        if 'list' in value:return [unpack(v) for v in value['list']]
        raise ValueError('Unknown source tensor state')
    for name,attrs in state['attributes'].items():
        for attr,value in attrs.items():setattr(modules[name],attr,unpack(value))
    for name,flags in state['flags'].items():
        for attr,value in flags.items():
            expected=torch.device(value['torch_device']) if isinstance(value,dict) else value
            require(getattr(modules[name],attr)==expected, 'Native plain source flag changed: '+name+'.'+attr)
    body=model.body if hasattr(model,'body') else model
    require(body.edge_index is body.laplacian_builder.edge_index, 'Exact native shared graph object alias')


class BayesianAdapter(ClassifierAdapter):
    def observe_validation(self, step, metrics):
        require(step==self._last_source_selection+1, 'Every completed source epoch must be selected once')
        key=(metrics['auroc'],-metrics['nll'],-step); improve=self._source_key is None or key>self._source_key
        if improve:self._source_key=key;self._best_step=step;self._source_stale=0
        else:self._source_stale+=1
        self._last_source_selection=step
        return {'save_selected':improve,'stop_requested':self._source_stale>=200,'best_saved_metric':self._source_key[0],
                'best_saved_step':self._best_step,'stale_steps':self._source_stale,'checkpoint_eligible':True}
    def validate(self, step):
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        with self.t.no_grad():
            z=self.serving_draws(step,4)
            p=z.exp().mean(0) if self.fusion=='mean_probabilities' else z.reshape(self.members,4,*z.shape[1:]).mean(1).mean(0).softmax(-1)
            v=self.h.metrics(self.t,roc_auc_score,p[self.select_ids].log(),self.select_y)
        return {**v,'objective':v['nll'],'ce':v['nll'],'regularization':0.,'selection_metric':v['auroc'],
                'source_epoch':step,'fusion':self.fusion,'operation_counts':{'model_forwards':4*self.members,
                'native_posterior_layer_samples':4*self.members*self.native_args['layers'],'optimizer_updates':0}}


def attach(torch, model, opt, h, s, args, graph, train, select, arm, seed, provenance):
    import numpy as np
    body=model.body if arm=='original_GNNM' else model
    x=graph['x'].to(next(model.parameters()).device); ids,y=train['ids'].to(x.device),train['y'].to(x.device)
    members=model.members; adapter=None; clock=0; states=[]
    initial=s.capture(np,torch,h,body)
    try:
        for member in range(members):
            if member:
                s.seed_streams(np,torch,h,body,seed+1000003*member)
            states.append(copy.deepcopy(s.capture(np,torch,h,body)))
    finally:s.restore(np,torch,h,body,initial)
    @contextmanager
    def route_scope(member, training=False, eval_epoch=None):
        before=s.capture(np,torch,h,body)
        try:
            s.restore(np,torch,h,body,states[member])
            if eval_epoch is not None:
                s.seed_streams(np,torch,h,body,seed+1000003*member+2000003+eval_epoch)
            yield
        finally:
            if training:states[member]=copy.deepcopy(s.capture(np,torch,h,body))
            s.restore(np,torch,h,body,before)
            require(s.exact(torch,s.capture(np,torch,h,body),before), 'Complete Python/NumPy/SciPy/Torch outer streams restored')
    def call(member):
        if arm=='original_GNNM':return model.route(member,x)
        logp,kl=model(x);return None,logp,kl
    def draws(epoch, number=4):
        require(number in (4,16) and (number==4 or members==1), 'Declared native posterior serving count only')
        output=[]
        for member in range(members):
            with route_scope(member,eval_epoch=epoch):
                for draw in range(number):
                    raw,logp,kl=call(member)
                    require(torch.isfinite(logp).all() and torch.isfinite(kl).all(), 'Finite intact native posterior/KL draw')
                    output.append(logp if arm=='native_M1' else raw)
        return torch.stack(output)
    def train_step(a, step):
        nonlocal clock
        counts={'optimizer_updates':0,'optimizer_step_attempts':0,'model_forward_attempts':0,'model_forwards':0,
                'backward_attempts':0,'backward_calls':0,'native_KL_evaluations':0,'native_posterior_layer_samples':0,
                'supervised_examples':len(ids)*members,'route_predictions':members}
        try:
            require(step==clock+1, 'Complete source epoch order required')
            model.train();opt.zero_grad(set_to_none=True);nlls=[];kls=[];objectives=[]
            # Literal CPU scalar construction from source runner.fit.
            beta=torch.sigmoid(torch.tensor(((step-1)%40)/2-10))
            for member in range(members):
                with route_scope(member,training=True):
                    counts['model_forward_attempts']+=1;_,logp,kl=call(member);counts['model_forwards']+=1
                    counts['native_KL_evaluations']+=1;counts['native_posterior_layer_samples']+=args['layers']
                    nll=torch.nn.functional.nll_loss(logp[ids],y);loss=nll+beta*kl
                    require(torch.isfinite(loss) and torch.isfinite(kl), 'Finite native NLL plus annealed Cayley KL')
                    counts['backward_attempts']+=1;(loss/members).backward();counts['backward_calls']+=1
                    nlls.append(float(nll.detach()));kls.append(float(kl.detach()));objectives.append(float(loss.detach()))
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'Native finite source gradients')
            counts['optimizer_step_attempts']=1;opt.step();counts['optimizer_updates']=1;clock=step
            return {'objective':sum(objectives)/members,'ce':sum(nlls)/members,'regularization':float(beta)*sum(kls)/members,
                    'native_KL':sum(kls)/members,'native_beta':float(beta),'operation_counts':counts}
        except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
    def capture():
        return {'source_args':copy.deepcopy(args),'source_epoch':clock,'route_states':copy.deepcopy(states),
                'outer_source_streams':s.capture(np,torch,h,body),'source_native_plain':plain_capture(torch,model),
                'source_selector':{'key':adapter._source_key,'stale':adapter._source_stale,'last':adapter._last_source_selection}}
    def restore(value):
        nonlocal clock
        require(value['source_args']==args and len(value['route_states'])==members, 'Complete Bayesian source/route ancestry')
        states[:]=copy.deepcopy(value['route_states']);clock=value['source_epoch']
        adapter._source_key=value['source_selector']['key'];adapter._source_stale=value['source_selector']['stale'];adapter._last_source_selection=value['source_selector']['last']
        plain_restore(torch,model,value['source_native_plain'],x.device)
        s.restore(np,torch,h,body,value['outer_source_streams'])
    adapter=BayesianAdapter(torch,torch.nn.functional,model,opt,logits_fn=lambda m,t,step:draws(step,4),
        train_ids=ids,train_y=y,select_ids=select['ids'].to(x.device),select_y=select['y'].to(x.device),
        regularizer=lambda m:x.new_zeros(()),fusion='mean_probabilities' if arm=='native_M1' else 'mean_logits',
        selector={'metric':'auroc','direction':'max','ties':'first','patience_steps':200,'save_every':1},
        custom_train=train_step,external_capture=capture,external_restore=restore,provenance=provenance)
    adapter.h,adapter.s,adapter.seed,adapter.members,adapter.native_args=h,s,seed,members,args
    adapter._source_key=None;adapter._source_stale=0;adapter._last_source_selection=0
    adapter.serving_draws=draws;adapter.predict_rows=lambda rows,step:draws(step,4)[:,rows.to(x.device)]
    return adapter


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    c=run['config'];B=c['bayesian_bsnn_sources']
    require(study['dataset']=='twitch-de-musae-raw' and c['classes']==2, 'Only custodied new binary TwitchDE task')
    read=lambda row:torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
    graph,train,select=(read(c['node_inputs'][key]) for key in ('graph','train','select'))
    require(set(graph)=={'x','edge_index'} and not torch.isin(train['ids'],select['ids']).any(), 'GRAPH/TRAIN/SELECT source role identity')
    require(c['initialization_seed']==run['seed']+1009*run['body'], 'Declared genuine I4 body seed')
    model,opt,h,s,args=construct(torch,project_root,B,graph,c['bayesian_bsnn_variant'],c['bayesian_bsnn_arm'],c['initialization_seed'],device)
    return attach(torch,model,opt,h,s,args,graph,train,select,c['bayesian_bsnn_arm'],c['initialization_seed'],
                  {'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'native_args':args})
