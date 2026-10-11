"""User-machine runtime adapter; no route, credentials or private states."""
import contextlib
import math
import random
import resource
import sys
import time
from common import BLOCKS,checked,load,read,require,runtime_versions

class Runtime:
    def __init__(self,prepared,device,wall_seconds):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np,self.t,self.F=np,torch,F;self.started=time.monotonic();self.wall_seconds=wall_seconds
        self.versions_metadata=runtime_versions();require(torch.get_default_dtype()==torch.float32,'Source float32 runtime')
        self.device=torch.device(device);require(device in ('cpu','cuda:0'),'Explicit CPU or one ordinary CUDA device')
        if self.device.type=='cuda':
            require(torch.cuda.is_available(),'Requested ordinary CUDA device unavailable');torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        self.custody=read(prepared);require(self.custody['complete'] and self.custody['blocks']==list(BLOCKS) and self.custody['role_rng']=='RandomState(block+5000011+223000019)','Exact fixed public role preparation')
        self.core,self.author,self.helper=load('core'),load('author_model'),load('state_helpers')
        self.native_core,self.scan_backend,self.tour_helper=load('native_core'),load('associative_scan'),load('tours')
        with np.load(checked(self.custody['files']['graph.npz']),allow_pickle=False) as values:
            require(set(values.files)=={'x','edge_index'},'Graph has no targets');x,e=values['x'].copy(),values['edge_index'].copy()
        require(x.shape==(34493,8415) and x.dtype==np.float32 and np.isfinite(x).all() and np.logical_or(x==0,x==1).all(),'Full raw binary X')
        require(e.shape==(2,530417) and e.dtype==np.int64,'Fixed full native support')
        self.features=torch.from_numpy(x).to_sparse_coo().coalesce().to(self.device)
        self.adjacency=torch.sparse_coo_tensor(torch.from_numpy(e).flip(0),torch.ones(530417),(34493,34493)).coalesce().to(self.device)
        require(self.adjacency._nnz()==530417,'Native support unchanged')
        self.degrees=torch.bincount(self.adjacency.indices()[0],minlength=34493).to(torch.float32).unsqueeze(1)
        require(self.degrees.min().item()>=1,'Positive native degrees')
        self.tours={};self.tour_bindings={};before=self.fp(self.rng())
        for block in BLOCKS:
            cpu=self.tour_helper.build_tours(torch,torch.from_numpy(e),block,expected_components=self.custody['components'])
            self.tour_bindings[str(block)]=[dict(seed=t['seed'],view=t['view'],components=t['components'],length=t['length'],payload_sha256={name:self.fp(t[name]) for name in ('tokens','reverse_index','starts','occurrences','offsets','generator_before_roots','generator_after_neighbors')}) for t in cpu]
            self.tours[block]=tuple(self.tour_helper.to_device(t,self.device) for t in cpu)
        require(self.fp(self.rng())==before,'Private fixed tour draws changed global RNG')
        self.fixed=(self.features,self.adjacency,self.degrees)+tuple(tensor for block in BLOCKS for tour in self.tours[block] for tensor in tour.values())
        self.fixed_versions=tuple(tensor._version for tensor in self.fixed)
        self.graph=dict(nodes=34493,features=8415,classes=5,edges=530417,feature_sha256=self.custody['feature_sha256'],graph_sha256=self.custody['graph_sha256'],components=self.custody['components'],tour_bindings=self.tour_bindings)
        del x,e;self.guard()
    def synchronize(self):
        if self.device.type=='cuda':self.t.cuda.synchronize(self.device)
    def guard(self):
        self.synchronize();require(time.monotonic()-self.started<self.wall_seconds,'Finite user wall limit exceeded; failure retained')
        require(self.fixed_versions==tuple(tensor._version for tensor in self.fixed),'Fixed public input/tour tensor mutated')
    def rng(self):
        value=self.np.random.get_state()
        return dict(python=random.getstate(),numpy=(value[0],value[1].tolist(),value[2],value[3],value[4]),cpu=self.t.get_rng_state(),cuda=self.t.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None)
    def restore_rng(self,value):
        random.setstate(value['python']);state=value['numpy'];self.np.random.set_state((state[0],self.np.asarray(state[1],dtype=self.np.uint32),state[2],state[3],state[4]))
        self.t.set_rng_state(value['cpu'].cpu())
        if self.device.type=='cuda':self.t.cuda.set_rng_state(value['cuda'].cpu(),self.device)
    def costs(self):
        return dict(seconds=time.monotonic()-self.started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),peak_CUDA_allocated_bytes=self.t.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,peak_CUDA_reserved_bytes=self.t.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None)

    def tensor(self,a):return self.t.from_numpy(a).to(self.device)

    def fp(self,v):return self.helper.fingerprint(self.t,v)

    def roles(self,block,select):
        def labels(name,count,perclass):
            with self.np.load(checked(self.custody['files'][f'block_{block}/{name}.npz']),allow_pickle=False) as a:
                require(set(a.files)=={'ids','y'},'Role-only labels');ids,y=a['ids'].copy(),a['y'].copy()
            require(ids.shape==y.shape==(count,) and ids.dtype==y.dtype==self.np.int64 and len(self.np.unique(ids))==count and ids.min()>=0 and ids.max()<34493 and y.min()>=0 and y.max()<5 and (self.np.bincount(y,minlength=5)==perclass).all(),'Exact stratified role')
            return ids,y
        tid,ty=labels('train',100,20);v=dict(train_ids=self.tensor(tid),train_y=self.tensor(ty))
        sid=self.np.load(checked(self.custody['files'][f'block_{block}/select_ids.npy']),allow_pickle=False)
        require(sid.dtype==self.np.int64 and sid.shape==(150,) and len(self.np.unique(sid))==150 and sid.min()>=0 and sid.max()<34493 and not self.np.intersect1d(tid,sid).size,'Label-free SELECT IDs')
        v['select_ids']=self.tensor(sid)
        if select:
            sid2,sy=labels('select',150,30);require(self.np.array_equal(sid,sid2),'SELECT ID authority');v['select_y']=self.tensor(sy)
        if not select:return v
        for role,n in (('cal',150),('dev',34093)):
            ids=self.np.load(checked(self.custody['files'][f'block_{block}/{role}_ids.npy']),allow_pickle=False)
            require(ids.dtype==self.np.int64 and ids.shape==(n,),'Label-free export roles');v[role+'_ids_numpy']=ids
        all_ids=self.np.concatenate([tid,sid,v['cal_ids_numpy'],v['dev_ids_numpy']])
        require(self.np.array_equal(self.np.sort(all_ids),self.np.arange(34493,dtype=self.np.int64)),'Whole disjoint role partition')
        return v

    def streams(self,block,members,start_index=0):
        result=[]
        for m in range(start_index,start_index+members):
            seed=block+1000003*m;g=self.t.Generator(device='cpu').manual_seed(seed)
            # Match CPU source-constructor draw progression without another model.
            for shape in ((8415,32),(8415,32),(32,5),(32,5)):
                std=math.sqrt(2.0/float(sum(shape)));bound=math.sqrt(3.0)*std
                self.t.empty(shape).uniform_(-bound,bound,generator=g)
            ng=self.np.random.RandomState(seed).get_state()
            result.append(dict(python=random.Random(seed).getstate(),numpy=(ng[0],ng[1].tolist(),ng[2],ng[3],ng[4]),cpu=g.get_state(),cuda=self.t.Generator(device=self.device).manual_seed(seed).get_state() if self.device.type=='cuda' else None))
        return result

    @contextlib.contextmanager
    def scope(self,streams,index):
        outside=self.rng();self.restore_rng(streams[index])
        try:yield
        finally:streams[index]=self.rng();self.restore_rng(outside)

    def construct(self,arm,block):
        core=self.core.build_arm(self.t,self.F,arm,block,native_core=self.native_core,scan_backend=self.scan_backend)
        for model in core['models']:model.to(self.device);model.tour_block=block
        audit=self.core.inspect_parameter_ownership(core)
        optimizers=self.core.make_optimizers(core,self.author.optimizer_class(self.t))
        for m,o in zip(core['models'],optimizers):
            params=[p for g in o.param_groups for p in g['params']]
            require(len({id(p) for p in params})==len(params) and {id(p) for p in params}=={id(p) for p in m.parameters() if p.requires_grad},'Complete disjoint optimizer ownership')
        return core,optimizers,audit

    def forward(self,model,streams=None):
        return model(self.features,self.adjacency,self.degrees,self.tours[model.tour_block],None if streams is None else lambda m:self.scope(streams,m))

    def save(self,path,model,optimizer,streams,meta):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_training=model.training,global_rng=self.rng(),route_rng=streams,meta=meta)
        state['fingerprint']=self.fp(state);self.t.save(state,path);return state['fingerprint']

    def restore(self,path,model,optimizer,streams,meta):
        state=self.t.load(path,map_location='cpu',weights_only=True);digest=state.pop('fingerprint')
        require(state['meta']==meta and self.fp(state)==digest,'Serialized selected-state custody')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);model.train(state['model_training'])
        streams[:]=state['route_rng'];self.restore_rng(state['global_rng'])
        actual=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_training=model.training,global_rng=self.rng(),route_rng=streams,meta=meta)
        require(self.fp(actual)==digest,'Exact model/Adam/mode/global/route RNG restore');return state
