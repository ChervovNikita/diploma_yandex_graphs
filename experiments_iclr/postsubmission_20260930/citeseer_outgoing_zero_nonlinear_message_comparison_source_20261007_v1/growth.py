"""Finite incoming, zero outgoing native growth. Numerical imports are deferred."""
import hashlib
import importlib.util
from math import comb
from pathlib import Path
import sys
from types import MethodType
import time

PHASE=Path(__file__).resolve().parents[1]
ORIGINAL=PHASE/'citeseer_nonlinear_preaggregation_growth_pilot_source_20261007_v2/growth.py'
ORIGINAL_SHA='681b7f6e7d29eaaf0e9c29d761d46558f5404e5186904376a3c20a99b5ec1d36'
CONDITIONS=('outgoing_graph','outgoing_unfiltered_partition','outgoing_feature_svd','outgoing_random',
            'outgoing_capable_single8','outgoing_independent4','incoming_graph_reference')
OPTIONAL_CONDITION='outgoing_graph_identity'
WIDTH=256

def original():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('Original growth source changed')
    name='outgoing_original_growth_v2'
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=ORIGINAL.resolve():raise ValueError('Original growth module shadowed')
        return module
    spec=importlib.util.spec_from_file_location(name,ORIGINAL);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def calibrate(torch,warm,features,support,queries):
    """Original calibration arithmetic; finite G required, norm only recorded."""
    base=original();base.require_site(torch,warm.encoder)
    modes=[(m,m.training) for m in warm.modules()];warm.eval()
    try:
        h=warm.encoder.xemb(features);prepared=warm.encoder.adjdrop(support)
        y=warm.encoder.convs[0](h,prepared);latent=base.finish(torch,warm.encoder,y)
        positive=warm.predictor(latent,support,queries[0]);negative=warm.predictor(latent,support,queries[1])
        objective=(-torch.nn.functional.logsigmoid(positive).mean()-torch.nn.functional.logsigmoid(-negative).mean())
        if not bool(torch.isfinite(objective)):raise FloatingPointError('Nonfinite TRAIN calibration loss')
        g=torch.autograd.grad(objective,y)[0]
        if not bool(torch.isfinite(h).all() and torch.isfinite(g).all()):raise FloatingPointError('Nonfinite calibration H/G')
        return h.detach(),g.detach(),{'objective':float(objective.detach()),'H_shape':list(h.shape),'G_shape':list(g.shape),
            'G_norm':float(g.detach().double().norm()),'G_nonzero_observed':bool((g!=0).any()),'dropout_off':True,
            'upstream_site':'PureConv output before native tail/JK','G_norm_is_not_fit_gate':True}
    finally:
        for module,mode in modes:module.training=mode

def rank(torch,singular):
    cutoff=max(1e-12,float(singular[0])*1e-6) if singular.numel() else 1e-12
    return int((singular>cutoff).sum()),cutoff

def bases(torch,encoder,h,g,support,factor_seed):
    """One TRAIN bank per seed, fixed RMS1 columns and realized diagnostics."""
    conv=original().require_site(torch,encoder)
    if not support.is_symmetric() or support.storage.value() is not None:raise ValueError('Symmetric unweighted TRAIN support required')
    row,col,_=support.coo()
    if bool((row==col).any()):raise ValueError('Native input support excludes selfloops')
    records=[];timing={};p_actions=[];right_graph=[]
    def propagate(value):
        p_actions.append(value.shape[1]);return conv(value,support)
    def left(k,n,label):
        if not bool(torch.isfinite(k).all()):raise FloatingPointError('Nonfinite initializer matrix')
        u,s,vh=torch.linalg.svd(k,full_matrices=False);usable,cutoff=rank(torch,s)
        if not bool(torch.isfinite(u).all() and torch.isfinite(s).all() and torch.isfinite(vh).all()):raise FloatingPointError('Nonfinite SVD result')
        if label.startswith('graph_band'):right_graph.append(vh[:2].to(device=h.device,dtype=h.dtype))
        records.append({'label':label,'rank_requested':n,'usable_rank':usable,'cutoff':cutoff,'singular_values':s.tolist()})
        return u[:,:n]
    def normalize(directions):
        pre=hd@directions
        rms=pre.square().mean(0).sqrt()
        if not bool(torch.isfinite(rms).all()) or bool((rms==0).any()):raise ValueError('Undefined fixed RMS1 directions; no rescale/redraw')
        v=(directions/rms).to(device=h.device,dtype=h.dtype)
        if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite finite-incoming bank')
        return v
    def partition(v):return v.reshape(WIDTH,4,2).permute(1,0,2).contiguous()
    with torch.no_grad():
        hd=h.detach().cpu().double();torch.cuda.synchronize();started=time.monotonic()
        packed=torch.cat([g]*4,dim=1)
        for step in range(3):
            pp=propagate(packed)
            packed=torch.cat([a+b if step<3-q else a-b for q,(a,b) in enumerate(zip(packed.split(WIDTH,1),pp.split(WIDTH,1)))],1)
        filtered=torch.cat([a*(comb(3,q)/8.) for q,a in enumerate(packed.split(WIDTH,1))],1)
        pfg=propagate(filtered).detach().cpu().double()
        graph=torch.stack([normalize(left(hd.t()@part,2,'graph_band'+str(q))) for q,part in enumerate(pfg.split(WIDTH,1))])
        torch.cuda.synchronize();timing['graph_filter_left_SVD_RMS_seconds']=time.monotonic()-started
        started=time.monotonic();pg=propagate(g).detach().cpu().double()
        unfiltered=partition(normalize(left(hd.t()@pg,8,'unfiltered_top8_left')))
        torch.cuda.synchronize();timing['unfiltered_left_SVD_RMS_seconds']=time.monotonic()-started
        started=time.monotonic();_,s,vh=torch.linalg.svd(hd,full_matrices=False);usable,cutoff=rank(torch,s)
        records.append({'label':'raw_H_feature_top8_right','rank_requested':8,'usable_rank':usable,'cutoff':cutoff,'singular_values':s.tolist()})
        feature=partition(normalize(vh[:8].t()));timing['feature_SVD_RMS_seconds']=time.monotonic()-started
        started=time.monotonic();generator=torch.Generator(device='cpu');generator.manual_seed(factor_seed+900001)
        random=partition(normalize(torch.randn((WIDTH,8),generator=generator,dtype=torch.float64)))
        timing['seeded_Gaussian_RMS_seconds']=time.monotonic()-started
        single=torch.cat([graph[m] for m in range(4)],1).unsqueeze(0)
        bank={'graph':graph,'graph_right':torch.stack(right_graph),'unfiltered':unfiltered,'feature':feature,'random':random,'single':single}
        ph=propagate(h).detach().cpu().double();uph,sph,_=torch.linalg.svd(ph,full_matrices=False);rph,_=rank(torch,sph);qph=uph[:,:rph]
        diagnostics={};started=time.monotonic()
        for key in ('graph','unfiltered','feature','random'):
            values=bank[key];v=torch.cat([values[m] for m in range(4)],1)
            pre=h@v;nonlinear=torch.tanh(pre);phi=propagate(nonlinear).detach().cpu().double()
            residual=phi-qph@(qph.t()@phi);_,sr,_=torch.linalg.svd(residual,full_matrices=False);rr,_=rank(torch,sr)
            node_q=[];member=[]
            for m,part in enumerate(phi.split(2,1)):
                u,s,_=torch.linalg.svd(part,full_matrices=False);r,cut=rank(torch,s);node_q.append(u[:,:r])
                member.append({'member':m,'usable_rank':r,'cutoff':cut,'singular_values':s.tolist()})
            distances=[]
            for a in range(4):
                for b in range(a):
                    qa,qb=node_q[a],node_q[b];total=qa.shape[1]+qb.shape[1]
                    sq=max(0.,float(total-2*(qa.t()@qb).square().sum()))
                    distances.append({'members':[a,b],'normalized_node_projector_distance':(sq/total)**.5 if total else 0.})
            diagnostics[key]={'member_features':member,'node_projector_distances':distances,
                'preactivation_RMS_per_column':pre.square().mean(0).sqrt().cpu().tolist(),
                'saturated_abs_tanh_ge_0_99_fraction':float((nonlinear.abs()>=.99).float().mean()),
                'tanh_derivative_RMS':float((1-nonlinear.square()).square().mean().sqrt()),
                'PH_usable_rank':rph,'nonlinear_residual_usable_rank':rr,
                'nonlinear_residual_frobenius':float(residual.norm()),
                'relative_nonlinear_residual_energy':float(residual.square().sum()/phi.square().sum()) if bool(phi.norm()>0) else 0.,
                'nonlinear_residual_TRAIN_G_correlation_frobenius':float((residual.t()@g.detach().cpu().double()).norm()),
                'diagnostics_are_interpretation_not_numeric_tolerance_fit_gates':True}
        torch.cuda.synchronize();timing['realized_dictionary_diagnostics_seconds']=time.monotonic()-started
    return bank,{'records':records,'diagnostics':diagnostics,'timing':timing,'native_P_column_widths':p_actions,
        'normalization':'full3327-public-node RMS(Hv)=1 per column; no centering/scale/rank search',
        'random_seed':factor_seed+900001,'random_generator':'isolated CPU float64 Gaussian',
        'graph_K':'H^T P^T Fq(P)G; left singular vectors; degree3 Bernstein',
        'single_exact_graph_dictionary_concatenation':True,'no_dense_N_by_N_projector':True,
        'rank_projector_saturation_residual_reported_no_rescue':True}

def make(torch,native,heads,models,initializer,warm,condition,seed,factor_seed,device,bank):
    if condition not in CONDITIONS+(OPTIONAL_CONDITION,):raise ValueError('Unknown fixed outgoing-zero condition')
    if condition=='incoming_graph_reference':
        values=bank['graph_right'].to(device=device,dtype=torch.float32)
        if tuple(values.shape)!=(4,2,256) or not bool(torch.isfinite(values).all()):raise ValueError('Matched right-graph reference bank shape/finiteness')
        model,partition=original().make(torch,native,heads,models,initializer,warm,'graph_growth',seed,factor_seed,device,{'graph':values})
        partition['growth'].update(condition=condition,initialization_end='incoming_zero',
            scope='Fresh paired new-study reference; original paper source/results are unchanged',
            incoming_and_outgoing_use_same_Kq_decomposition=True)
        return model,partition
    single=condition=='outgoing_capable_single8';independent=condition=='outgoing_independent4'
    key={'outgoing_unfiltered_partition':'unfiltered','outgoing_feature_svd':'feature','outgoing_random':'random'}.get(condition,'single' if single else 'graph')
    incoming=bank[key].to(device=device,dtype=torch.float32)
    expected=(1,256,8) if single else (4,256,2)
    if tuple(incoming.shape)!=expected or not bool(torch.isfinite(incoming).all()):raise ValueError('Incoming dictionary shape/finiteness')
    if not torch.equal(bank['single'].cpu(),torch.cat([bank['graph'][m] for m in range(4)],1).unsqueeze(0).cpu()):raise ValueError('Single8 must use literal same graph dictionary')
    old_condition='capable_single_rank8' if single else 'independent_graph_growth4' if independent else 'graph_growth'
    stub={'single' if single else 'graph':torch.zeros((expected[0],expected[2],256),device=device)}
    model,partition=original().make(torch,native,heads,models,initializer,warm,old_condition,seed,factor_seed,device,stub)
    routes=list(model.routes) if independent else [model]
    with torch.no_grad():
        for m,route in enumerate(routes):
            value=incoming[m:m+1] if independent else incoming
            if tuple(route.growth.V.shape)!=tuple(value.shape):raise ValueError('Finite V model shape mismatch')
            route.growth.V.copy_(value);route.growth.B.zero_()
            if bool((route.growth.B!=0).any()):raise ValueError('B must initialize exactlyzero')
    if condition==OPTIONAL_CONDITION:
        def forward(self,h,support,conv,members):
            packed=h@torch.cat([self.V[m] for m in members],1)
            groups=conv(packed,support).split(self.B.shape[1],1)
            return [group@self.B[m] for group,m in zip(groups,members)]
        model.growth.forward=MethodType(forward,model.growth)
    identities=[{id(p) for p in route.parameters()} for route in routes]
    if any(identities[a]&identities[b] for a in range(len(routes)) for b in range(a)):raise ValueError('Independent models share parameters')
    partition['growth'].update(condition=condition,V_initial='finite RMS1 incoming dictionary',B_initial=0.,
        B='zero outgoing matrix, trainable; no initial outgoing projector claim',
        incoming_bank=key,initialization_end='outgoing_zero',all_V_B_learn=True,
        message_activation='identity' if condition==OPTIONAL_CONDITION else 'tanh',no_new_biases=True)
    return model,partition

class PaidLiveness:
    """Observe only the actual first ordinary three committed backwards."""
    def __init__(self,torch,model):
        self.values={};self.handles=[]
        self.initial_end='incoming_zero' if not bool((model.growth.V!=0).any()) else 'outgoing_zero'
        for name,p in model.named_parameters():
            if name not in ('growth.V','growth.B'):continue
            self.values[name]=[]
            def observe(gradient,name=name):
                norms=[float(v.detach().double().norm()) for v in gradient]
                self.values[name].append({'pass_index':len(self.values[name]),'member_norms':norms,
                    'member_nonzero':[n>0 for n in norms],'all_exact_zero':not bool((gradient!=0).any()),
                    'finite':bool(torch.isfinite(gradient).all())})
            self.handles.append(p.register_hook(observe))
    def finish(self):
        for handle in self.handles:handle.remove()
        return {'paid_actual_backward_records':self.values,'extra_forward_backward_updates':0,
            'initialization_end':self.initial_end,
            'zero_nonzero_results_are_reported_not_tolerance_gates':True,
            'interpretation':'Outgoing-zero initially gives V-zero/B-live gradients; incoming-zero reverses them. Later wake is observed after the live factor updates. No nonzero norm superiority or optimizer guarantee.'}
