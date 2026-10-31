"""Literal source TABM routes and masks with own/collective source selectors."""
import copy
from pathlib import Path
from types import SimpleNamespace
from adapters.classifier import ClassifierAdapter
from adapters.source_common import bound, module, roles, Streams, OptimizerBank
from engine import AdapterStepError
from tabm_stopping import validate_tabm_run

POLICIES={'O4_stem_IID':'iid','O4_stem_SIGN':'activation_sign_paired','O4_graph_IID':'iid',
          'O4_graph_SIGN':'activation_sign_paired','O4_graph_BALANCED':'balanced_08','MS4_IID':'iid','MS4_SIGN':'activation_sign_paired','coupled_I4':'activation_sign_paired'}
PINS={'native':'07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
      'author':'dfd19cf6da0460a4de49697e2c5afe185e85e8192db4bb51e548868b9a2d1401',
      'joint':'c7fa6c4d608e7f209fc6f4727a4409c04f71ba03d01c78fdda5bb5d15b11cc82',
      'samplers':'ad2e0e4e9832682257e6c8d0c731812ed6d66386a2b5b274edd2dae33bcf90f5',
      'independent':'466edb10e53ab9cf613565ba250ea500630fd4454549d96fd6bf25fb41d5a43d'}

class PhysicsAdapter(ClassifierAdapter):
    def raw_logits(self, selected=False):
        t=self.t;rows=[];self.model.eval()
        for i,body in enumerate(self.model.bodies):
            state=copy.deepcopy(body.state_dict()) if selected and self.coupled else None
            if state is not None:
                if self.own_selected[i] is None:raise ValueError('Missing own selected state')
                body.load_state_dict(self.own_selected[i]['model'])
            try:
                if self.arm=='source_meanSAGE':rows.append(body(*self.author_inputs))
                else:
                    for member in range(self.members_per_body):
                        rows.append(self.joint.native_route_forward(body,self.graph,self.x,member,placement=self.placement,torch=t))
            finally:
                if state is not None:body.load_state_dict(state)
        return t.stack(rows)
    def train_step(self, step):
        counts={'optimizer_updates':0,'model_forwards':0,'route_predictions':0,'backward_calls':0,'supervised_examples':0,'source_detached_prefix_forwards':0}
        regularization=0.
        try:
            if self.coupled:
                flags=list(self.active)
                result=self.independent.coupled_I4_round(list(self.model.bodies),self.own_optimizers,flags,self.graph,self.x,self.train_ids,self.train_y,
                    torch=self.t,functional=self.F,mask_sampler=self.sampler,native_route_forward=self.joint.native_route_forward,
                    body_rng_scope=lambda i:self.streams.scope(i))
                values=[loss for loss in result['losses'] if loss is not None]
                loss=sum(values)/len(values);copies=sum(flags)
                counts.update(optimizer_updates=copies,model_forwards=copies,route_predictions=copies,backward_calls=copies,
                              source_detached_prefix_forwards=result['prefix_forwards'],source_stopped_companion_prefix_forwards=result['inactive_companion_prefix_forwards'])
                self.own_updates=[old+int(flag) for old,flag in zip(self.own_updates,flags)]
            else:
                body=self.model.bodies[0];opt=self.own_optimizers[0]
                with self.streams.scope(0):
                    if self.arm=='original_gnnm':
                        body.train();opt.zero_grad();parts=[]
                        for member in range(4):
                            z=body(self.graph,self.x,tabm_seed=member);ce=self.F.cross_entropy(z[self.train_ids],self.train_y)/4;ce.backward();parts.append(float(ce.detach()))
                        opt.step();loss=sum(parts);copies=4
                    elif self.arm in POLICIES:
                        loss=self.joint.train_step_joint_routes(body,self.graph,self.x,self.train_ids,self.train_y,opt,
                            placement=self.placement,torch=self.t,functional=self.F,mask_sampler=self.sampler,multisample_single=self.arm.startswith('MS4'))
                        copies=4;counts['source_detached_prefix_forwards']=4
                    else:
                        body.train();opt.zero_grad()
                        z=body(*self.author_inputs) if self.arm=='source_meanSAGE' else self.joint.native_route_forward(body,self.graph,self.x,0,placement=self.placement,torch=self.t)
                        penalty=body.regularizer() if self.arm=='source_meanSAGE' else z.sum()*0
                        total=self.F.cross_entropy(z[self.train_ids],self.train_y)+penalty;total.backward();opt.step();loss=float(total.detach());regularization=float(penalty.detach());copies=1
                self.own_updates[0]+=1
                counts.update(optimizer_updates=1,model_forwards=copies,route_predictions=copies,backward_calls=copies)
            counts['supervised_examples']=copies*len(self.train_ids)
            for p in self.model.parameters():
                if p.grad is not None and not self.t.isfinite(p.grad).all():raise FloatingPointError('Nonfinite source gradient')
            return {'objective':loss,'ce':loss-regularization,'regularization':regularization,'operation_counts':counts}
        except BaseException as error:raise AdapterStepError(str(error),operation_counts=counts) from error
    def validate(self, step):
        self.model.eval()
        with self.t.no_grad():
            logits=self.raw_logits();own=self.t.stack([self.F.cross_entropy(z[self.select_ids],self.select_y) for z in logits])
            mean=logits.mean(0);correct=mean[self.select_ids].argmax(-1).eq(self.select_y).sum()
            penalty=self.model.bodies[0].regularizer() if self.arm=='source_meanSAGE' else own.sum()*0
            value={'objective':float(own.mean()+penalty),'ce':float(own.mean()),'regularization':float(penalty),
                   'accuracy':float(correct)/len(self.select_y),'selection_metric':float(own.mean()+penalty) if self.arm=='source_meanSAGE' else float(correct)/len(self.select_y),
                   'own_correct':[int(z[self.select_ids].argmax(-1).eq(self.select_y).sum()) for z in logits],
                   'operation_counts':{'optimizer_updates':0,'model_forwards':len(logits),'route_predictions':len(logits),'backward_calls':0,'supervised_examples':0}}
        return value
    def observe_validation(self, step, metrics):
        if not self.coupled:return super().observe_validation(step,metrics)
        flags=list(self.active)
        for i,flag in enumerate(flags):
            if not flag:continue
            score=metrics['own_correct'][i]/len(self.select_y)
            above=self.own_best[i] is None or score>self.own_best[i]
            self.own_stale[i]=0 if above else self.own_stale[i]+1
            if above and step%10==0:
                self.own_best[i]=score
                self.own_selected[i]=self.cpu({'model':self.model.bodies[i].state_dict(),'optimizer':self.own_optimizers[i].state_dict(),
                    'stream':self.streams.states[i],'sampler':self.generator.get_state(),'global_rng':self.streams.capture(),
                    'selected_step':step,'SELECT_correct':metrics['own_correct'][i],'completed_own_updates':self.own_updates[i]})
            if self.own_stale[i]>=300 or step==5000:
                if self.own_selected[i] is None:raise ValueError('Own native checkpoint never selected')
                self.active[i]=False
        closed=not any(self.active)
        return {'save_selected':closed,'stop_requested':closed,'best_saved_metric':metrics['accuracy'],'best_saved_step':step if closed else None,
                'stale_steps':max(self.own_stale),'checkpoint_eligible':closed,'own_stale':list(self.own_stale),
                'own_selected_steps':[None if row is None else row['selected_step'] for row in self.own_selected],'active':list(self.active)}
    def cpu(self, value):
        if self.t.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:self.cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(self.cpu(v) for v in value)
        return copy.deepcopy(value)
    def capture_state(self):
        value=super().capture_state();value['source_Physics']=self.cpu({'streams':self.streams.states,'global_rng':self.streams.capture(),
            'sampler':None if self.generator is None else self.generator.get_state(),'active':self.active,'own_best':self.own_best,
            'own_stale':self.own_stale,'own_updates':self.own_updates,'own_selected':self.own_selected})
        return value
    def restore_state(self, value):
        super().restore_state(value);state=value['source_Physics'];self.streams.states=copy.deepcopy(state['streams']);self.streams.restore(state['global_rng'])
        if self.generator is not None:self.generator.set_state(state['sampler'].cpu())
        for name in ('active','own_best','own_stale','own_updates','own_selected'):setattr(self,name,copy.deepcopy(state[name]))

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
    import numpy as np
    from torch.nn import functional as F
    c=copy.deepcopy(run['config'])
    if study['stage']!='engineering':validate_tabm_run(run)
    root=Path(project_root);packs=roles(torch,root,c);x=packs['graph']['x'].to(device);edge=packs['graph']['edge_index'].to(device)
    arm=c['physics_arm'];sources=c['physics_sources']
    if set(sources)!=set(PINS) or any(sources[key]['sha256']!=digest for key,digest in PINS.items()):raise ValueError('Literal source TABM/joint/sampler/author pins required')
    joint=module(root,sources['joint'],'physics_source_joint');samplers=module(root,sources['samplers'],'physics_source_samplers')
    independent=module(root,sources['independent'],'physics_source_independent');native=module(root,sources['native'],'physics_source_native')
    author=module(root,sources['author'],'physics_source_author');streams=Streams(torch,np,device);coupled=arm=='coupled_I4'
    members_per_body=4 if arm.startswith('O4') or arm=='original_gnnm' else 1;bodies=[];optimizers=[]
    for i in range(4 if coupled else 1):
        streams.seed(c['initialization_seed']+1000003*i)
        body=author.model_class(torch,F)().to(device) if arm=='source_meanSAGE' else native.TABMModel('SAGE',2,x.shape[1],128,c['classes'],1,8,'LayerNorm',.2,tabm_inits=members_per_body,device=device).to(device)
        opt=author.optimizer_class(torch)(body.parameters()) if arm=='source_meanSAGE' else torch.optim.AdamW(body.parameters(),lr=.001,weight_decay=0)
        bodies.append(body);optimizers.append(opt);streams.states.append(streams.capture())
    class Bank(torch.nn.Module):
        def __init__(self):super().__init__();self.bodies=torch.nn.ModuleList(bodies)
    bank=Bank();generator=torch.Generator(device=device).manual_seed(run['seed']+211000007) if arm in POLICIES else None
    adapter=PhysicsAdapter(torch,F,bank,OptimizerBank(optimizers) if coupled else optimizers[0],logits_fn=lambda *_:None,
        train_ids=packs['train']['ids'].to(device),train_y=packs['train']['y'].to(device),select_ids=packs['select']['ids'].to(device),select_y=packs['select']['y'].to(device),
        regularizer=lambda _:torch.zeros((),device=device),fusion='mean_logits',selector={'metric':'objective' if arm=='source_meanSAGE' else 'accuracy',
            'direction':'min' if arm=='source_meanSAGE' else 'max','ties':'last' if arm=='source_meanSAGE' else 'first','patience_steps':50 if arm=='source_meanSAGE' else 300,'save_every':1 if arm=='source_meanSAGE' else 10},
        provenance={'run_id':run['run_id'],'config':c,'source':study['source'],'actual_model_seed':c['initialization_seed'],'actual_parent':None,'actual_H0_hash':None})
    adapter.arm,adapter.coupled,adapter.streams=arm,coupled,streams;adapter.graph=SimpleNamespace(edge_index=edge);adapter.x=x
    adapter.joint,adapter.independent,adapter.own_optimizers=joint,independent,optimizers;adapter.generator=generator
    adapter.sampler=samplers.make_mask_sampler(POLICIES[arm],torch=torch,generator=generator) if generator is not None else None
    adapter.members_per_body=members_per_body;adapter.placement=c['physics_placement'];adapter.own_updates=[0]*len(bodies)
    adapter.active=[True]*len(bodies);adapter.own_best=[None]*len(bodies);adapter.own_stale=[0]*len(bodies);adapter.own_selected=[None]*len(bodies)
    if arm=='source_meanSAGE':
        sparse=x.to_sparse_coo().coalesce();adj=torch.sparse_coo_tensor(edge.flip(0),torch.ones(edge.shape[1],device=device),(len(x),len(x))).coalesce()
        adapter.author_inputs=(sparse,adj,torch.bincount(adj.indices()[0],minlength=len(x)).float()[:,None])
    adapter.logits_fn=lambda _model,_training,_step:adapter.raw_logits(selected=adapter.coupled)
    return adapter
