"""Frozen per-member temperature then probability mean; whole-panel DEV reader."""
import itertools
import json
import math
from pathlib import Path
import resource
import statistics
import time
from runtime import ARMS,BLOCKS,checked,receipt,require,write

METRICS=('accuracy_pct','calibrated_accuracy_pct','mean_member_accuracy_pct','worst_member_accuracy_pct','raw_NLL','raw_Brier','calibrated_NLL','calibrated_Brier','mean_member_NLL','worst_member_NLL','mean_member_Brier','worst_member_Brier')

def owner_closed(item):
    value=json.loads(checked(item).read_text())
    require(value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and value['direct_child_wait'] and value['child_pid_absent'] and value['owned_cuda_pid_absent'],'Successful prior direct finite owner exit/reap required')
    return value

def close_panel(config):
    owner_closed(config['acquisition_owner_end']);panel=json.loads(checked(config['panel']).read_text())
    index={(r['block'],r['arm']):r for r in panel['records']};expected={(b,a) for b in BLOCKS for a in ARMS}
    require(panel['complete'] and panel['terminal'] and len(panel['records'])==30 and set(index)==expected and panel['role_custody']==config['custody'],'Whole exact30bank panel before CAL/DEV')
    require((panel['banks'],panel['optimizer_trajectories'],panel['physical_backbones'],panel['exports'],panel['selected_optimizer_states'],panel['selected_physical_bodies'])==(30,48,48,30,48,48),'All optimizer/physical/exports closed')
    require((panel['prior_banks'],panel['new_banks'],panel['prior_optimizer_trajectories'],panel['new_optimizer_trajectories'],panel['prior_physical_bodies'],panel['new_physical_bodies'],panel['logical_member_view_outputs'])==(12,18,21,27,21,27,102),'Distinct21prior/27new work provenance required')
    require(panel['serving']=='mean_class_probabilities' and panel['CAL_DEV_truth_read'] is False,'Correct acquisition serving/role custody')
    require(sum(len(r['fits']) for r in index.values())==48 and sum(r['physical_backbones'] for r in index.values())==48,'Distinct trajectory versus physical-body accounting')
    from reuse import references
    prior=references(config);original={(r['block'],r['arm']):r for r in prior['records']}
    for (b,a),r in index.items():
        require(r['status']=='exported_complete' and r['probability_pool_authority'] and r['actual_served_members']==(1 if a in ('P1','F1') else 4),'Complete native membership')
        expected_fits=4 if a in ('I4','WI4') else 1
        require(len(r['fits'])==expected_fits and r['physical_backbones']==(4 if a in ('I4','WI4') else 1),'Frozen independent/joint ownership')
        if (b,a) in original:require(r==original[b,a],'Prior records must retain original state/export/source/owner provenance')
        else:require(r['origin']=='new_tour_scanner_fit','New scanner fit provenance')
        for f in r['fits']:
            require(f['status']=='selected_complete' and f['selected_restore_verified'] and 1<=f['selected_step']<=f['completed_updates']<=100000,'Every selected state restored')
            checked(f['checkpoint'])
        require(set(r['exports'])=={'cal','dev'},'Complete two label-free exports')
        for item in r['exports'].values():checked(item)
    return panel,index

def guard(config,start):
    require(time.monotonic()-start<config['limits']['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=config['limits']['rss_bytes'],'Finite reader CPU wall/RSS cap')

def bank_arrays(np,r,role):
    with np.load(checked(r['exports'][role]),allow_pickle=False) as a:
        require(set(a.files)=={'ids','member_log_probs','native_member_argmax','native_pool_argmax','native_pool_probabilities'},'Probability-mean export schema only');v={k:a[k].copy() for k in a.files}
    n=150 if role=='cal' else 34093;m=r['actual_served_members']
    require(v['ids'].dtype==np.int64 and v['ids'].shape==(n,) and len(np.unique(v['ids']))==n and v['member_log_probs'].dtype==np.float32 and v['member_log_probs'].shape==(m,n,5),'Full ordered role/members')
    require(v['native_member_argmax'].dtype==v['native_pool_argmax'].dtype==np.int64 and v['native_member_argmax'].shape==(m,n) and v['native_pool_argmax'].shape==(n,),'Native decision shape/type')
    p=v['native_pool_probabilities'];require(p.dtype==np.float32 and p.shape==(n,5) and np.isfinite(p).all() and (p>0).all() and np.max(np.abs(p.astype(np.float64).sum(-1)-1))<=1e-6,'Native positive FP32 pool distribution')
    require(np.isfinite(v['member_log_probs']).all() and np.array_equal(v['native_member_argmax'],v['member_log_probs'].argmax(-1)) and np.array_equal(v['native_pool_argmax'],p.argmax(-1)),'Stored native decision authority')
    # Float64 reconstitution is sensitivity only; primary counts retain native p32.
    reconstructed=np.exp(logprob(np,v['member_log_probs'].astype(np.float64))).mean(0)
    require(np.max(np.abs(reconstructed-p))<=1e-6,'Archive member/pool numeric custody')
    v['raw_float64_decision_sensitivity_count']=int((reconstructed.argmax(-1)!=v['native_pool_argmax']).sum())
    return v

def truth(np,custodian,b,role):
    prefix='calibration_only' if role=='cal' else 'scorer_only';item=custodian['files'][f'{prefix}/block_{b}_{role}.npz']
    with np.load(checked(item),allow_pickle=False) as a:
        require(set(a.files)=={'ids','y'},'Role-specific truth schema');ids,y=a['ids'].copy(),a['y'].copy()
    require(ids.dtype==y.dtype==np.int64 and ids.shape==y.shape==((150,) if role=='cal' else (34093,)) and len(np.unique(ids))==len(ids) and (y>=0).all() and (y<5).all(),'Complete fixed truth role');return ids,y

def logprob(np,z):
    shifted=z-z.max(-1,keepdims=True);return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))

def calibrated(np,lp,T):return np.exp(logprob(np,lp.astype(np.float64)/T)).mean(0)

def calibrate(config,custody,output):
    start=time.monotonic();panel,index=close_panel(config);custodian=json.loads(checked(config['custody']).read_text())
    import numpy as np
    import torch
    require(config['expected_runtime_versions']=={'numpy':np.__version__,'torch':torch.__version__},'Bound CAL runtime versions');torch.set_num_threads(2)
    rows=[dict(block=b,arm=a,status='unstarted_retained',updates=0) for b in BLOCKS for a in ARMS]
    from reuse import references
    prior=references(config)
    result=dict(complete=False,records=rows,requested_endpoints=30,requested_new_endpoints=18,requested_scalar_updates=9000,inherited_endpoints=12,inherited_scalar_updates=6000,DEV_truth_read=False,panel=config['panel'],role_custody=config['custody'],custody=custody,retry=False,prior_receipts=prior['receipts'])
    try:
        for row in rows:
            if (row['block'],row['arm']) in prior['endpoints']:
                row.update(prior['endpoints'][row['block'],row['arm']]);continue
            guard(config,start);row['status']='running';b,a=row['block'],row['arm'];ids,y=truth(np,custodian,b,'cal');raw=bank_arrays(np,index[b,a],'cal')
            row['origin']='new_tour_scanner_CAL'
            require(np.array_equal(ids,raw['ids']),'Exact CAL identity')
            lp=torch.from_numpy(raw['member_log_probs']).to(torch.float64);targets=torch.from_numpy(y);idx=torch.arange(len(y));M=len(lp)
            logT=torch.nn.Parameter(torch.zeros((),dtype=torch.float64));optimizer=torch.optim.Adam([logT],lr=.05,betas=(.9,.999),eps=1e-8,weight_decay=0)
            def objective():
                pooled=torch.logsumexp(torch.nn.functional.log_softmax(lp/logT.exp(),dim=-1),dim=0)-math.log(M)
                return -pooled[idx,targets].mean()
            with torch.no_grad():best=float(objective());best_logT=0.;best_step=0
            require(math.isfinite(best),'Finite initial CAL NLL')
            for step in range(1,501):
                guard(config,start);optimizer.zero_grad();loss=objective();require(torch.isfinite(loss).item(),'Finite fixed CAL loss');loss.backward();require(torch.isfinite(logT.grad).item(),'Finite scalar CAL gradient');optimizer.step()
                with torch.no_grad():
                    logT.clamp_(-5.,5.);value=float(objective())
                    require(math.isfinite(value),'Finite post-update CAL NLL')
                    if value<=best:best,best_logT,best_step=value,float(logT),step
                row['updates']=step
            T=math.exp(best_logT);require(math.isfinite(T) and T>0,'Finite selected fixed temperature')
            row.update(status='fixed_endpoint_complete',endpoint='latest_tied_minimum_step0_to500',selected_step=best_step,logT=best_logT,temperature=T,CAL_minimum_NLL=best,objective_forward_evaluations=1001,operator='mean_m softmax(member_log_probs_m/T)',cal_archive=index[b,a]['exports']['cal'],cal_truth=custodian['files'][f'calibration_only/block_{b}_cal.npz'])
            write(output/'CAL_PROGRESS.json',dict(closed_endpoints=sum(r['status']=='fixed_endpoint_complete' for r in rows),actual_scalar_updates=sum(r['updates'] for r in rows if r.get('origin')=='new_tour_scanner_CAL'),inherited_scalar_updates=6000))
        require(all(r['status']=='fixed_endpoint_complete' and r['updates']==500 for r in rows),'All30 CAL endpoints closed')
        require(sum(r.get('origin')=='prior_peer_reference' for r in rows)==12 and sum(r.get('origin')=='new_tour_scanner_CAL' for r in rows)==18,'Original12/new18 CAL endpoints')
        result.update(complete=True,endpoints=30,actual_scalar_updates=9000,total_bound_scalar_updates=15000,seconds=time.monotonic()-start,fitted_CAL_quality_interpretation=False)
    except BaseException as e:
        for row in rows:
            if row['status']=='running':row['status']='failed_retained'
        result.update(failure=dict(type=type(e).__name__,message=str(e)),actual_scalar_updates=sum(r['updates'] for r in rows if r.get('origin')=='new_tour_scanner_CAL'));raise
    finally:write(output/'TEMPERATURES.json',result)
    return result

def score_bank(np,raw,y,T):
    lp=logprob(np,raw['member_log_probs'].astype(np.float64));p=raw['native_pool_probabilities'].astype(np.float64);cp=calibrated(np,raw['member_log_probs'],T)
    P=raw['native_pool_argmax']==y;member=raw['native_member_argmax']==y[None];V=member.any(0);L=V & ~P;G=~V & P;CP=cp.argmax(-1)==y
    Q=(raw['native_member_argmax']==raw['native_member_argmax'][0:1]).all(0) & ~member[0]
    onehot=np.eye(5,dtype=np.float64)[y];mnll=-lp[np.arange(len(lp))[:,None],np.arange(len(y))[None],y[None]]
    rnll=-np.log(p[np.arange(len(y)),y]);cnll=-np.log(cp[np.arange(len(y)),y]);mbrier=((np.exp(lp)-onehot[None])**2).sum(-1)
    rbrier=((p-onehot)**2).sum(-1);cbrier=((cp-onehot)**2).sum(-1)
    def subset(mask):
        n=int(mask.sum());require(n>0,'Full class population');mc=member[:,mask].sum(1);acc=100*mc/n
        c=dict(nodes=n,served_correct=int(P[mask].sum()),calibrated_served_correct=int(CP[mask].sum()),member_correct=mc.tolist(),coverage=int(V[mask].sum()),pooling_losses=int(L[mask].sum()),aggregation_only_correct=int(G[mask].sum()),common_wrong_rivals=int(Q[mask].sum()))
        require(c['served_correct']==c['coverage']-c['pooling_losses']+c['aggregation_only_correct'],'Exact coverage/loss identity')
        q=dict(accuracy_pct=100*c['served_correct']/n,calibrated_accuracy_pct=100*c['calibrated_served_correct']/n,member_accuracy_pct=acc.tolist(),mean_member_accuracy_pct=float(acc.mean()),worst_member_accuracy_pct=float(acc.min()),raw_NLL=float(rnll[mask].mean()),calibrated_NLL=float(cnll[mask].mean()),raw_Brier=float(rbrier[mask].mean()),calibrated_Brier=float(cbrier[mask].mean()),member_NLL=mnll[:,mask].mean(1).tolist(),mean_member_NLL=float(mnll[:,mask].mean()),worst_member_NLL=float(mnll[:,mask].mean(1).max()),member_Brier=mbrier[:,mask].mean(1).tolist(),mean_member_Brier=float(mbrier[:,mask].mean()),worst_member_Brier=float(mbrier[:,mask].mean(1).max()))
        q.update(raw_ECE=None,calibrated_ECE=None,ECE_unavailable_reason='No prospectively frozen bin rule in the sealed scientific specification')
        require(all(math.isfinite(q[k]) for k in METRICS),'Finite complete quality metrics');return dict(counts=c,quality=q)
    return dict(**subset(np.ones(len(y),dtype=bool)),classes=[dict(class_id=c,**subset(y==c)) for c in range(5)],P=P,V=V,Q=Q,member=member,raw_float64_decision_sensitivity_count=raw['raw_float64_decision_sensitivity_count'])

def descriptive(v,signs=False):
    require(len(v)==3 and all(math.isfinite(x) for x in v),'Complete three-block vector');mean=statistics.mean(v);sd=statistics.stdev(v);half=4.302652729911275*sd/math.sqrt(3)
    r=dict(blocks=list(BLOCKS),values=v,mean=mean,sample_SD=sd,descriptive_95_CI=[mean-half,mean+half],scope='Three custom paired protocols on one encountered graph; descriptive only')
    if signs:r['eight_sign_flip_assignments']=[dict(signs=list(s),mean=statistics.mean(x*z for x,z in zip(v,s))) for s in itertools.product((-1,1),repeat=3)]
    return r

def delta(a,b,k):
    n=a['counts']['nodes'];require(n==b['counts']['nodes'],'Paired population')
    if k in ('accuracy_pct','calibrated_accuracy_pct'):
        field='served_correct' if k=='accuracy_pct' else 'calibrated_served_correct';return 100*(a['counts'][field]-b['counts'][field])/n
    if k=='mean_member_accuracy_pct':
        ac,bc=a['counts']['member_correct'],b['counts']['member_correct'];return 100*(sum(ac)*len(bc)-sum(bc)*len(ac))/(len(ac)*len(bc)*n)
    return a['quality'][k]-b['quality'][k]

def pair(a,b,mask):
    repair=(~b['P'] & a['P']) & mask;harm=(b['P'] & ~a['P']) & mask
    c=dict(repairs=int(repair.sum()),harms=int(harm.sum()),coverage_added=int((a['V'] & ~b['V'] & mask).sum()),coverage_removed=int((b['V'] & ~a['V'] & mask).sum()),introduced_common_wrong_rivals=int((a['Q'] & ~b['Q'] & mask).sum()),reference_common_rivals_covered=int((b['Q'] & a['V'] & mask).sum()),reference_common_rivals_served_repaired=int((b['Q'] & a['P'] & mask).sum()),repair_causes=dict(new_member_alternative=int((repair & a['V'] & ~b['V']).sum()),both_covered=int((repair & a['V'] & b['V']).sum()),aggregation_only=int((repair & ~a['V']).sum())),harm_causes=dict(available_alternative_lost=int((harm & a['V']).sum()),removed_alternative=int((harm & ~a['V'] & b['V']).sum()),lost_aggregation_only=int((harm & ~a['V'] & ~b['V']).sum())))
    d=dict(served_correct=int((a['P'] & mask).sum()-(b['P'] & mask).sum()),coverage=int((a['V'] & mask).sum()-(b['V'] & mask).sum()),pooling_losses=int((a['V'] & ~a['P'] & mask).sum()-(b['V'] & ~b['P'] & mask).sum()),aggregation_only_correct=int((~a['V'] & a['P'] & mask).sum()-(~b['V'] & b['P'] & mask).sum()))
    require(d['served_correct']==c['repairs']-c['harms']==d['coverage']-d['pooling_losses']+d['aggregation_only_correct'],'Exact paired decomposition')
    require(sum(c['repair_causes'].values())==c['repairs'] and sum(c['harm_causes'].values())==c['harms'],'Gross partition completeness');c['delta_counts']=d;return c

def gates(values):
    accuracy=[];confidence=[]
    def test(rows,name,reference,metric,mean_min=None,each_min=None,each_max=None,strict_positive=False):
        v=[delta(values[b,'SW4'],values[b,reference],metric) for b in BLOCKS]
        passed=(mean_min is None or statistics.mean(v)>=mean_min) and (each_min is None or all(x>=each_min for x in v)) and (each_max is None or all(x<=each_max for x in v)) and (not strict_positive or all(x>0 for x in v))
        row=dict(name=name,candidate='SW4',reference=reference,metric=metric,minimum_mean=mean_min,minimum_each=each_min,maximum_each=each_max,all_three_strict_positive=strict_positive,passed=passed,contrast=descriptive(v,True));rows.append(row)
    for a in ('P1','F1','I4','S4','PV1','WI4','V1'):
        test(accuracy,'worthwhile_raw_served_gain',a,'accuracy_pct',mean_min=.20,strict_positive=True)
    test(accuracy,'equipped_member_retention','WI4','mean_member_accuracy_pct',mean_min=-.25,each_min=-.50)
    for a in ('C4','R4'):
        test(accuracy,'private_tour' if a=='C4' else 'retained_path_state',a,'accuracy_pct',mean_min=.10,strict_positive=True)
    refs=('I4','WI4','PV1','V1')
    for a in refs:test(confidence,'calibrated_NLL_block_protection',a,'calibrated_NLL',each_max=.005)
    means={a:statistics.mean(values[b,a]['quality']['calibrated_NLL'] for b in BLOCKS) for a in refs}
    candidate=statistics.mean(values[b,'SW4']['quality']['calibrated_NLL'] for b in BLOCKS)
    confidence.append(dict(name='calibrated_NLL_best_reference_mean_protection',candidate='SW4',metric='calibrated_NLL',candidate_mean=candidate,reference_means=means,best_reference_mean=min(means.values()),passed=candidate<=min(means.values())))
    ap=all(r['passed'] for r in accuracy);cp=all(r['passed'] for r in confidence)
    return dict(accuracy_member_attribution_screen_pass=ap,calibrated_confidence_protection_pass=cp,accuracy_screens=accuracy,confidence_screens=confidence,combined_accuracy_and_confidence_pass=ap and cp,confidence_failure_kept_separate=True,R4_attribution_limit='Reset changes context,amplitude and conditioning while retaining original q=(1-a)u; no isolated long-range causal attribution',stronger_independent16view_reference_remains_missing=True,inference='Encountered one-graph development screen; no confirmation or primitive novelty')

def evaluate(config,custody,output):
    start=time.monotonic();panel,index=close_panel(config);owner_closed(config['calibration_owner_end']);temperatures=json.loads(checked(config['temperatures']).read_text())
    endpoints={(r['block'],r['arm']):r for r in temperatures['records']}
    from reuse import references
    prior=references(config)
    require(all(endpoints.get(k)==ep for k,ep in prior['endpoints'].items()),'Inherited12 CAL endpoints preserve every original receipt and scalar')
    require(temperatures['complete'] and temperatures['endpoints']==30 and temperatures['actual_scalar_updates']==9000 and temperatures['inherited_scalar_updates']==6000 and temperatures['total_bound_scalar_updates']==15000 and len(temperatures['records'])==30 and set(endpoints)==set(index) and temperatures['panel']==config['panel'] and temperatures['role_custody']==config['custody'] and temperatures['DEV_truth_read'] is False,'All30 exact CAL endpoints before whole DEV release')
    require(all(r['status']=='fixed_endpoint_complete' and r['updates']==500 and r['endpoint']=='latest_tied_minimum_step0_to500' and 0<=r['selected_step']<=500 and -5<=r['logT']<=5 and math.isfinite(r['temperature']) and r['temperature']==math.exp(r['logT']) and r['cal_archive']==index[k]['exports']['cal'] for k,r in endpoints.items()),'No calibration substitution')
    custodian=json.loads(checked(config['custody']).read_text())
    import numpy as np
    require(np.__version__==config['expected_runtime_versions']['numpy'],'Bound score runtime version')
    release=dict(one_whole_release=True,blocks=list(BLOCKS),after_optimizer_trajectories=48,after_physical_bodies=48,prior_trajectories=21,new_trajectories=27,after_exports=30,after_CAL_endpoints=30,temperatures=config['temperatures'],custody=custody,DEV_truth={str(b):custodian['files'][f'scorer_only/block_{b}_dev.npz'] for b in BLOCKS})
    write(output/'DEV_RELEASE.json',release);result=dict(complete=False,records=[],release=release,panel=config['panel'],temperatures=config['temperatures'],retry=False,survivor_subset=False)
    try:
        roles={b:truth(np,custodian,b,'dev') for b in BLOCKS};values={}
        for b in BLOCKS:
            ids,y=roles[b]
            for arm in ARMS:
                guard(config,start);raw=bank_arrays(np,index[b,arm],'dev');require(np.array_equal(ids,raw['ids']),'Exact complete DEV identity')
                value=score_bank(np,raw,y,endpoints[b,arm]['temperature']);values[b,arm]=value
                result['records'].append(dict(block=b,arm=arm,counts=value['counts'],quality=value['quality'],classes=value['classes'],raw_float64_decision_sensitivity_count=value['raw_float64_decision_sensitivity_count'],temperature=endpoints[b,arm]['temperature']))
                write(output/'DEV_PROGRESS.json',dict(scored_banks=len(result['records']),expected_banks=30))
        comparisons=[]
        for a in ARMS:
            for ref in ARMS:
                if a==ref:continue
                guard(config,start)
                comparisons.append(dict(candidate=a,reference=ref,metric_deltas={k:descriptive([delta(values[b,a],values[b,ref],k) for b in BLOCKS],True) for k in METRICS},per_block=[dict(block=b,**pair(values[b,a],values[b,ref],np.ones(34093,dtype=bool)),classes=[dict(class_id=c,**pair(values[b,a],values[b,ref],roles[b][1]==c)) for c in range(5)]) for b in BLOCKS]))
        regions=[]
        for b in BLOCKS:
            base=values[b,'S4'];candidate=values[b,'SW4'];masks={'S4_common_wrong_rival':base['Q'],'S4_common_wrong_rival_outside_I4_coverage':base['Q'] & ~values[b,'I4']['V'],'S4_common_wrong_rival_outside_WI4_coverage':base['Q'] & ~values[b,'WI4']['V']}
            for name,mask in masks.items():
                regions.append(dict(block=b,region=name,nodes=int(mask.sum()),new_correct_member_supply=int((~base['V'] & candidate['V'] & mask).sum()),actual_served_repairs=int((~base['P'] & candidate['P'] & mask).sum()),diagnostics=pair(candidate,base,mask),outside_diagnostics=pair(candidate,base,~mask)))
        paired_members=[dict(block=b,candidate='SW4',reference=a,member_accuracy_deltas_pct=[x-y for x,y in zip(values[b,'SW4']['quality']['member_accuracy_pct'],values[b,a]['quality']['member_accuracy_pct'])],member_NLL_deltas=[x-y for x,y in zip(values[b,'SW4']['quality']['member_NLL'],values[b,a]['quality']['member_NLL'])],independent_selected_horizons=[f['selected_step'] for f in index[b,a]['fits']]) for b in BLOCKS for a in ('I4','WI4')]
        serving_costs=[dict(block=r['block'],arm=r['arm'],origin=r['origin'],selected_forward_and_restore_seconds=[f['costs']['restoration_and_selected_forward_seconds'] for f in r['fits']],fit_wall_seconds=[f['costs']['fit_wall_seconds'] for f in r['fits']],operation_counts=[f['operation_counts'] for f in r['fits']],interpretation='Recorded full selected forward plus strict restore; no separate benchmark or speed claim') for r in panel['records']]
        require(len(result['records'])==30,'No survivor subset');result.update(complete=True,comparisons=comparisons,paired_member_diagnostics=paired_members,all_reference_paid_and_serving_costs=serving_costs,encountered_common_rival_regions=regions,gates=gates(values),arm_summaries=[dict(arm=a,metrics={k:descriptive([values[b,a]['quality'][k] for b in BLOCKS]) for k in METRICS}) for a in ARMS],paid_acquisition_counts=panel['counts'],paid_acquisition_costs=panel['costs'],paid_acquisition_owner=config['acquisition_owner_end'],paid_calibration_owner=config['calibration_owner_end'],new_calibration_scalar_updates=9000,inherited_calibration_scalar_updates=6000,total_bound_calibration_scalar_updates=15000,prior_trajectories=21,new_trajectories=27,prior_receipts=panel['prior_receipts'],reader_costs=dict(seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),scope='Complete fixed development study; no automatic next fit/confirmation/paper claim')
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e));raise
    finally:write(output/'DEVELOPMENT_REPORT.json',result)
    return result

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    if config['stage']=='calibrate':return calibrate(config,custody,output)
    if config['stage']=='score':return evaluate(config,custody,output)
    raise ValueError('Fixed reader stage required')
