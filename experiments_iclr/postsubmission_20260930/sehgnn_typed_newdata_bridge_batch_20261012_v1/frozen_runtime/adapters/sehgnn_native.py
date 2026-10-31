"""Native Bernoulli own learner and actual original boundary R/S/B control."""
import ast
import copy
import hashlib
import math
import types

try:from ..source_utils import bound
except ImportError:from source_utils import bound


def original_boundary(rt,sources,root,prototype):
    t=rt['torch'];tree=ast.parse(bound(root,sources['modules']['original_rsb']).read_bytes())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BEBlock')
    ns={'torch':t,'nn':t.nn,'__name__':'exact_original_GNNM_RSB'};exec(compile(ast.Module(body=[cls],type_ignores=[]),'<exact_original_RSB>', 'exec'),ns);Block=ns['BEBlock']
    # Native input maps have heterogeneous dimensions; each retained feature or
    # label map receives the exact input-projector block at its own boundary.
    prototype.boundary_feature=t.nn.ModuleDict({k:Block(prototype.data_size[k],512,True,'cpu',4) for k in prototype.feat_keys})
    prototype.boundary_labels=t.nn.ModuleDict({k:Block(5,512,True,'cpu',4) for k in prototype.label_feat_keys})
    prototype.embeding=t.nn.ParameterDict();prototype.labels_embeding=t.nn.ParameterDict()
    class Head(Block):
        def __init__(self):super().__init__(512,5,False,'cpu',4);self.member=0
        def forward(self,x):return Block.forward(self,x,self.member)
    prototype.task_mlp[-1][0]=Head();prototype.boundary_member=0
    native_tree=ast.parse(bound(root,sources['modules']['model']).read_bytes());native_cls=next(n for n in native_tree.body if isinstance(n,ast.ClassDef) and n.name=='SeHGNN')
    forward=next(copy.deepcopy(n) for n in native_cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');changes=[]
    class Replace(ast.NodeTransformer):
        def visit_BinOp(self,node):
            self.generic_visit(node)
            if isinstance(node.op,ast.MatMult) and isinstance(node.right,ast.Subscript) and isinstance(node.right.value,ast.Attribute) and node.right.value.attr in ('embeding','labels_embeding'):
                kind='boundary_feature' if node.right.value.attr=='embeding' else 'boundary_labels';changes.append(kind)
                return ast.Call(func=ast.Subscript(value=ast.Attribute(value=ast.Name('self',ast.Load()),attr=kind,ctx=ast.Load()),slice=node.right.slice,ctx=ast.Load()),
                    args=[node.left,ast.Attribute(value=ast.Name('self',ast.Load()),attr='boundary_member',ctx=ast.Load())],keywords=[])
            return node
    forward=Replace().visit(forward)
    if changes!=['boundary_feature','boundary_feature','boundary_labels']:raise ValueError('Only native typed input/label projector expressions may change')
    namespace=dict(rt['model_module'].__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),'<native_SeHGNN_original_boundary>', 'exec'),namespace)
    prototype.forward=types.MethodType(namespace['forward'],prototype)
    shared={id(p):p for p in prototype.parameters()};members=[]
    for member in range(4):
        model=copy.deepcopy(prototype,shared);model.boundary_member=member;model.task_mlp[-1][0].member=member;members.append(model)
    return t.nn.ModuleList(members),{'source_RSB_AST_sha256':hashlib.sha256(ast.dump(cls).encode()).hexdigest(),
        'native_forward_bound_AST_sha256':hashlib.sha256(ast.dump(forward).encode()).hexdigest(),'placement':'all25 feature and12 TRAIN-label input maps plus final task projector; typed extension of actual original blocks',
        'initializer':'actual source Xavier W/input Rademacher R/output R1/allS1/B0; native shared body and private BN buffers retained',
        'aggregation':'mean logits then Bernoulli sigmoid','parameter_row_storage':'actual originalM4 R/S/B tensors; one shared slow body and one optimizer'}


class BasicSession:
    def __init__(self,rt,engine,ctx,costs,mods,sources,root,arm,seed,lr):
        self.rt,self.engine,self.ctx,self.costs,self.arm=rt,engine,ctx,costs,arm;t=rt['torch'];rt['native'].set_random_seed(seed)
        prototype=engine.make_model(rt,ctx,costs);self.source_binding={}
        if arm=='original_gnnm':self.model,self.source_binding=original_boundary(rt,sources,root,prototype)
        elif arm=='factorized_native_single':
            bank=mods['factor'].install_member_bank(rt['model_module'],prototype,mods['factor'].AdapterConfig(enabled=True,members=1));self.model=bank.members
        elif arm=='native_single':self.model=t.nn.ModuleList([prototype])
        else:raise ValueError('Explicit native/original control required')
        self.model=self.model.to(rt['device']);self.models=tuple(self.model);self.optimizer=t.optim.Adam(self.model.parameters(),lr=lr,weight_decay=0)
        self.scalar=t.cuda.amp.GradScaler(enabled=rt['device'].type=='cuda');self.member_rng=[];caller=engine.capture_rng(rt['numpy'],t)
        try:
            for m in range(len(self.models)):
                rt['native'].set_random_seed(1000*seed+m+1);self.member_rng.append(engine.capture_rng(rt['numpy'],t))
        finally:engine.restore_rng(rt['numpy'],t,caller)
        self.counters={'own_epochs':0,'own_member_forwards':0,'own_backwards':0,'actual_Adam_steps':0}

    def train_epoch(self):
        t,engine=self.rt['torch'],self.engine;losses=[]
        for number,batch in enumerate(self.ctx.train_loader):
            if number!=0 or len(batch)!=self.ctx.train_count:raise ValueError('One complete native TRAIN minibatch required')
            self.optimizer.zero_grad(set_to_none=True);features={k:v[batch].to(self.rt['device']) for k,v in self.ctx.feats.items()};labels={k:v[batch].to(self.rt['device']) for k,v in self.ctx.label_feats.items()};truth=self.ctx.targets_cuda[batch]
            for m,model in enumerate(self.models):
                caller=engine.capture_rng(self.rt['numpy'],t)
                try:
                    engine.restore_rng(self.rt['numpy'],t,self.member_rng[m]);model.train()
                    with t.cuda.amp.autocast(enabled=self.rt['device'].type=='cuda'):
                        z=model(batch,features,labels,None);loss=t.nn.functional.binary_cross_entropy_with_logits(z,truth)
                    if not t.isfinite(z).all().item() or not math.isfinite(float(loss.detach())):raise ValueError('Finite complete native BCE required')
                    self.scalar.scale(loss/len(self.models)).backward();losses.append(float(loss.detach()));self.member_rng[m]=engine.capture_rng(self.rt['numpy'],t)
                    self.counters['own_member_forwards']+=1;self.counters['own_backwards']+=1
                finally:engine.restore_rng(self.rt['numpy'],t,caller)
            self.scalar.unscale_(self.optimizer)
            if any(p.grad is not None and not t.isfinite(p.grad).all().item() for p in self.model.parameters()):raise ValueError('Finite unscaled native gradients required')
            original=self.optimizer.step
            def observed(*args,**kwargs):
                value=original(*args,**kwargs);self.counters['actual_Adam_steps']+=1;return value
            self.optimizer.step=observed
            try:self.scalar.step(self.optimizer);self.scalar.update()
            finally:self.optimizer.step=original
        if len(losses)!=len(self.models):raise ValueError('Complete source own cycle required')
        self.counters['own_epochs']+=1
        return {'mean_native_own_BCE':sum(losses)/len(losses)}
