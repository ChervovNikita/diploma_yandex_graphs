"""Fixed post-mean temperature endpoints, then one whole DEV truth release."""
import itertools
import json
import math
from pathlib import Path
import resource
import statistics
import time
from runtime import BLOCKS,HERE,checked,receipt,require,write

CANDIDATE='O4_graph_input_sign_pair'
GRAPH_IID='O4_graph_input_IID'
ORIGINAL='fresh_native_original_sequential_O4'
CAPABLE=('factor_M1_stem_IID','factor_M1_graph_input_IID','genuine_factor_I4_stem_IID','genuine_factor_I4_graph_input_IID','factor_M1_MS4_graph_input_IID','factor_M1_MS4_graph_input_sign_pair','fresh_source_meanSAGE_M1','unshared_coupled_I4')
PROPER_REFS=(ORIGINAL,'genuine_factor_I4_graph_input_IID','fresh_source_meanSAGE_M1','unshared_coupled_I4')
METRICS=('accuracy_pct','mean_member_accuracy_pct','worst_member_accuracy_pct','raw_NLL','raw_Brier','calibrated_NLL','calibrated_Brier','mean_member_NLL','worst_member_NLL','mean_member_Brier','worst_member_Brier')

def arms():
    proposal=json.loads((HERE/'STUDY_PROPOSAL.json').read_text())
    return tuple(a['id'] for a in proposal['roster']+proposal['references'])

def owner_closed(item):
    value=json.loads(checked(item).read_text())
    require(value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and value['direct_child_wait'] and value['child_pid_absent'] and value['owned_cuda_pid_absent'],'Prior complete direct owner exit/reap required')
    return value

def close_panel(config):
    panel=json.loads(checked(config['panel']).read_text());owner_closed(config['acquisition_owner_end'])
    expected={(b,a) for b in BLOCKS for a in arms()}
    index={(r['block'],r['arm']):r for r in panel['records']}
    require(panel['complete'] and panel['banks']==42 and panel['selected_states']==69 and panel['exports']==42 and panel['role_custody']==config['custody'] and len(panel['records'])==42 and set(index)==expected,'Whole fixed panel before CAL or DEV')
    require(all(r['status']=='exported_complete' for r in index.values()) and sum(len(r['fits']) for r in index.values())==69 and all(f['status']=='selected_complete' and f['selected_restore_verified'] for r in index.values() for f in r['fits']),'All selected bodies and banks closed')
    for record in index.values():
        for role in ('cal','dev'):checked(record['exports'][role])
    return panel,index

def guard(config,started):
    require(time.monotonic()-started<config['limits']['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=config['limits']['rss_bytes'],'Finite CPU reader wall/RSS cap exceeded')

def bank_arrays(np,record,role):
    with np.load(checked(record['exports'][role]),allow_pickle=False) as a:
        require(set(a.files)=={'ids','raw_logits','raw_logit_mean','member_decision','raw_logit_decision'},'Post-mean native raw-logit schema')
        value={k:a[k].copy() for k in a.files}
    n=150 if role=='cal' else 34093;m=record['actual_served_members']
    require(value['ids'].shape==(n,) and value['ids'].dtype==np.int64 and value['raw_logits'].shape==(m,n,5) and value['raw_logits'].dtype==np.float32 and value['raw_logit_mean'].shape==(n,5) and value['raw_logit_mean'].dtype==np.float32,'Actual served members and complete role shape')
    require(np.isfinite(value['raw_logits']).all() and np.isfinite(value['raw_logit_mean']).all() and np.array_equal(value['member_decision'],value['raw_logits'].argmax(-1)) and np.array_equal(value['raw_logit_decision'],value['raw_logit_mean'].argmax(-1)),'Finite native decision authorities')
    # The stored mean was computed by the native torch operator on CUDA. It is
    # the calibration/serving authority; no probability-pool substitution.
    return value

def truth(np,custodian,block,role):
    prefix='calibration_only' if role=='cal' else 'scorer_only'
    item=custodian['files'][f'{prefix}/block_{block}_{role}.npz']
    with np.load(checked(item),allow_pickle=False) as a:
        require(set(a.files)=={'ids','y'},'Role-specific truth schema');ids,y=a['ids'].copy(),a['y'].copy()
    require(ids.shape==y.shape==((150,) if role=='cal' else (34093,)) and ids.dtype==y.dtype==np.int64 and len(np.unique(ids))==len(ids) and (y>=0).all() and (y<5).all(),'Complete fixed role truth')
    return ids,y

def calibrate(config,custody,output):
    started=time.monotonic();panel,index=close_panel(config)
    custodian=json.loads(checked(config['custody']).read_text())
    import numpy as np
    import torch
    torch.set_num_threads(2)
    rows=[dict(block=b,arm=a,status='unstarted_retained',updates=0) for b in BLOCKS for a in arms()]
    result=dict(complete=False,records=rows,requested_endpoints=42,requested_scalar_updates=21000,retry=False,DEV_truth_read=False,panel=config['panel'],role_custody=config['custody'],custody=custody)
    try:
        for row in rows:
            guard(config,started);row['status']='running';tick=time.monotonic();b,a=row['block'],row['arm']
            ids,y=truth(np,custodian,b,'cal');raw=bank_arrays(np,index[b,a],'cal')
            require(np.array_equal(ids,raw['ids']),'Exact CAL row identity')
            logits=torch.from_numpy(raw['raw_logit_mean']).to(torch.float64);targets=torch.from_numpy(y)
            logT=torch.nn.Parameter(torch.zeros((),dtype=torch.float64,device='cpu'))
            optimizer=torch.optim.Adam([logT],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
            for update in range(1,501):
                guard(config,started);optimizer.zero_grad();objective=torch.nn.functional.cross_entropy(logits/logT.exp(),targets)
                require(torch.isfinite(objective).item(),'Finite fixed CAL objective')
                objective.backward();require(torch.isfinite(logT.grad).item(),'Finite scalar gradient');optimizer.step();row['updates']=update
            temperature=float(logT.detach().exp());require(math.isfinite(temperature) and temperature>0,'Finite positive fixed final scalar')
            row.update(status='fixed_endpoint_complete',endpoint='fixed_final_update_500',logT=float(logT.detach()),temperature=temperature,cal_archive=index[b,a]['exports']['cal'],cal_truth=custodian['files'][f'calibration_only/block_{b}_cal.npz'],operator='softmax(native_raw_logit_mean / T)',optimizer='CPUfloat64 Adam lr.01 betas(.9,.999) eps1e-8 wd0',seconds=time.monotonic()-tick)
            write(output/'CAL_PROGRESS.json',dict(closed_endpoints=sum(r['status']=='fixed_endpoint_complete' for r in rows),actual_scalar_updates=sum(r['updates'] for r in rows)))
            del logits,targets,logT,optimizer,objective,raw,ids,y
        require(len(rows)==42 and all(r['updates']==500 and r['status']=='fixed_endpoint_complete' for r in rows),'All 42 fixed endpoints before DEV')
        result.update(complete=True,endpoints=42,actual_scalar_updates=21000,seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,no_fitted_CAL_quality_interpretation=True)
    except BaseException as e:
        for row in rows:
            if row['status']=='running':row['status']='failed_retained'
        result.update(failure=dict(type=type(e).__name__,message=str(e)),actual_scalar_updates=sum(r['updates'] for r in rows),seconds=time.monotonic()-started);write(output/'TEMPERATURES.json',result);raise
    write(output/'TEMPERATURES.json',result);return result

def descriptive(values,signs=False):
    require(len(values)==3 and all(math.isfinite(x) for x in values),'Complete three-block descriptive vector')
    mean=statistics.mean(values);sd=statistics.stdev(values);half=4.302652729911275*sd/math.sqrt(3)
    result=dict(blocks=list(BLOCKS),values=values,mean=mean,sample_SD=sd,descriptive_95_CI=[mean-half,mean+half],df=2,scope='Three paired mask/optimizer blocks on one encountered graph; descriptive only')
    if signs:
        assignments=[dict(signs=list(s),mean=statistics.mean(x*z for x,z in zip(values,s))) for s in itertools.product((-1,1),repeat=3)]
        result.update(sign_flip_assignments=assignments,absolute_tail_fraction=sum(abs(r['mean'])>=abs(mean)-1e-12 for r in assignments)/8,sign_flip_scope='Eight symmetry assignments; sensitivity only, no confirmatory inference')
    return result

def logprob(np,z):
    shifted=z-z.max(-1,keepdims=True);return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))

def score_bank(np,raw,y,temperature):
    z=raw['raw_logits'].astype(np.float64);mean=raw['raw_logit_mean'].astype(np.float64)
    members=logprob(np,z);served=logprob(np,mean);calibrated=logprob(np,mean/temperature)
    P=raw['raw_logit_decision']==y;member=raw['member_decision']==y[None];V=member.any(0)
    loss=V & ~P;aggregation=~V & P
    target_raw=z[np.arange(len(z))[:,None],np.arange(len(y))[None],y[None]]
    S=((z>target_raw[:,:,None]).all(0) & (np.arange(5)[None]!=y[:,None])).any(-1)
    nll=-members[np.arange(len(z))[:,None],np.arange(len(y))[None],y[None]]
    served_nll=-served[np.arange(len(y)),y];cal_nll=-calibrated[np.arange(len(y)),y]
    onehot=np.eye(5,dtype=np.float64)[y]
    member_brier=((np.exp(members)-onehot[None])**2).sum(-1)
    raw_brier=((np.exp(served)-onehot)**2).sum(-1);cal_brier=((np.exp(calibrated)-onehot)**2).sum(-1)
    def subset(mask):
        count=int(mask.sum());require(count>0,'Every class has development rows')
        counts=dict(nodes=count,served_correct=int(P[mask].sum()),member_correct=member[:,mask].sum(1).tolist(),coverage=int(V[mask].sum()),pooling_losses=int(loss[mask].sum()),aggregation_only_correct=int(aggregation[mask].sum()),strict_wrong_rivals=int(S[mask].sum()),strict_wrong_rivals_corrected=int((S & P & mask).sum()))
        require(counts['served_correct']==counts['coverage']-counts['pooling_losses']+counts['aggregation_only_correct'],'Whole population count identity')
        acc=member[:,mask].mean(1)*100;mnll=nll[:,mask].mean(1);mbrier=member_brier[:,mask].mean(1)
        quality=dict(accuracy_pct=100*counts['served_correct']/count,member_accuracy_pct=acc.tolist(),mean_member_accuracy_pct=float(acc.mean()),worst_member_accuracy_pct=float(acc.min()),raw_NLL=float(served_nll[mask].mean()),raw_Brier=float(raw_brier[mask].mean()),calibrated_NLL=float(cal_nll[mask].mean()),calibrated_Brier=float(cal_brier[mask].mean()),member_NLL=mnll.tolist(),mean_member_NLL=float(mnll.mean()),worst_member_NLL=float(mnll.max()),member_Brier=mbrier.tolist(),mean_member_Brier=float(mbrier.mean()),worst_member_Brier=float(mbrier.max()))
        require(all(math.isfinite(quality[k]) for k in METRICS),'Finite complete development diagnostics')
        return dict(counts=counts,quality=quality)
    return dict(**subset(np.ones(len(y),dtype=bool)),classes=[dict(class_id=c,**subset(y==c)) for c in range(5)],P=P,V=V,S=S)

def error_pair(a,b,mask):
    repair=(~b['P'] & a['P']) & mask;harm=(b['P'] & ~a['P']) & mask;av,bv=a['V'],b['V'];ap,bp=a['P'],b['P']
    value=dict(repairs=int(repair.sum()),harms=int(harm.sum()),coverage_added=int((av & ~bv & mask).sum()),coverage_removed=int((bv & ~av & mask).sum()),repair_causes=dict(new_member_alternative=int((repair & av & ~bv).sum()),both_banks_have_member_alternative=int((repair & av & bv).sum()),aggregation_only=int((repair & ~av).sum())),harm_causes=dict(available_alternative_lost=int((harm & av).sum()),removed_member_alternative=int((harm & ~av & bv).sum()),lost_aggregation_only=int((harm & ~av & ~bv).sum())),strict_reference_rivals_repaired=int((b['S'] & ap & mask).sum()),new_strict_rivals_created=int((~b['S'] & a['S'] & mask).sum()))
    d=dict(served_correct=int((ap & mask).sum()-(bp & mask).sum()),coverage=int((av & mask).sum()-(bv & mask).sum()),pooling_losses=int((av & ~ap & mask).sum()-(bv & ~bp & mask).sum()),aggregation_only_correct=int((~av & ap & mask).sum()-(~bv & bp & mask).sum()))
    require(d['served_correct']==value['repairs']-value['harms']==d['coverage']-d['pooling_losses']+d['aggregation_only_correct'],'Exact paired net/count identity')
    require(sum(value['repair_causes'].values())==value['repairs'] and sum(value['harm_causes'].values())==value['harms'],'Gross repair/harm partitions')
    value['count_decomposition_delta']=d;return value

def metric_delta(a,b,metric):
    # Accuracy contrasts use integer counts before scaling. This preserves exact
    # zero/sign comparisons under the frozen whole-population count identities.
    n=a['counts']['nodes'];require(n==b['counts']['nodes'],'Paired population sizes')
    if metric=='accuracy_pct':return 100*(a['counts']['served_correct']-b['counts']['served_correct'])/n
    if metric=='mean_member_accuracy_pct':
        ac,bc=a['counts']['member_correct'],b['counts']['member_correct']
        return 100*(sum(ac)*len(bc)-sum(bc)*len(ac))/(len(ac)*len(bc)*n)
    return a['quality'][metric]-b['quality'][metric]

def comparisons(np,values,labels):
    rows=[]
    for reference in arms():
        if reference==CANDIDATE:continue
        deltas={k:descriptive([metric_delta(values[b,CANDIDATE],values[b,reference],k) for b in BLOCKS],signs=True) for k in METRICS}
        classes=[dict(class_id=c,quality_deltas={k:descriptive([metric_delta(values[b,CANDIDATE]['classes'][c],values[b,reference]['classes'][c],k) for b in BLOCKS]) for k in METRICS}) for c in range(5)]
        paired=[dict(block=b,**error_pair(values[b,CANDIDATE],values[b,reference],np.ones(len(labels[b]),dtype=bool)),classes=[dict(class_id=c,**error_pair(values[b,CANDIDATE],values[b,reference],labels[b]==c)) for c in range(5)]) for b in BLOCKS]
        rows.append(dict(candidate=CANDIDATE,reference=reference,quality_deltas=deltas,class_quality_deltas=classes,per_block_error_diagnostics=paired))
    return rows

def gates(values):
    def delta(ref,metric='accuracy_pct'):return [metric_delta(values[b,CANDIDATE],values[b,ref],metric) for b in BLOCKS]
    def gain(ref,metric='accuracy_pct',threshold=.25):
        vector=delta(ref,metric);return dict(reference=ref,metric=metric,required_mean_pp=threshold,all_three_strict_positive=True,passed=statistics.mean(vector)>=threshold and all(x>0 for x in vector),contrast=descriptive(vector,True))
    primary=[gain(GRAPH_IID),gain(ORIGINAL),gain(GRAPH_IID,'mean_member_accuracy_pct')]
    capable=[dict(reference=r,passed=sum(values[b,CANDIDATE]['counts']['served_correct']-values[b,r]['counts']['served_correct'] for b in BLOCKS)>=0,required_mean_pp=0,significance_gate=False,contrast=descriptive(delta(r),True)) for r in CAPABLE]
    rawloss=delta(GRAPH_IID,'raw_NLL');raw_protection=dict(reference=GRAPH_IID,maximum_mean_harm=.02,passed=statistics.mean(rawloss)<=.02,contrast=descriptive(rawloss,True))
    proper=[]
    for r in PROPER_REFS:
        vector=delta(r,'calibrated_NLL');proper.append(dict(reference=r,maximum_mean_harm=.02,maximum_each_block_harm=.05,passed=statistics.mean(vector)<=.02 and all(x<=.05 for x in vector),contrast=descriptive(vector,True)))
    stem=delta('O4_stem_sign_pair');uniform=delta('O4_graph_input_uniform_balance')
    interaction=[100*(values[b,CANDIDATE]['counts']['served_correct']-values[b,GRAPH_IID]['counts']['served_correct']-values[b,'O4_stem_sign_pair']['counts']['served_correct']+values[b,'O4_stem_IID']['counts']['served_correct'])/values[b,CANDIDATE]['counts']['nodes'] for b in BLOCKS]
    attribution=[dict(name=name,required_mean_pp=.10,all_three_nonnegative=True,passed=statistics.mean(v)>=.10 and all(x>=0 for x in v),contrast=descriptive(v,True)) for name,v in (('candidate_minus_stem_sign_pair',stem),('factorial_position_by_coupling_interaction',interaction),('candidate_minus_uniform_balance',uniform))]
    ms4=[metric_delta(values[b,'factor_M1_MS4_graph_input_sign_pair'],values[b,'factor_M1_MS4_graph_input_IID'],'accuracy_pct') for b in BLOCKS]
    sharing=[100*(values[b,CANDIDATE]['counts']['served_correct']-values[b,GRAPH_IID]['counts']['served_correct']-values[b,'factor_M1_MS4_graph_input_sign_pair']['counts']['served_correct']+values[b,'factor_M1_MS4_graph_input_IID']['counts']['served_correct'])/values[b,CANDIDATE]['counts']['nodes'] for b in BLOCKS]
    all_screens=primary+capable+[raw_protection]+proper+attribution
    return dict(development_screen_pass=all(r['passed'] for r in all_screens),primary=primary,capable_quality=capable,raw_NLL_protection=raw_protection,calibrated_NLL_protection=proper,attribution=attribution,O4_coupling_effect=descriptive(delta(GRAPH_IID),True),MS4_coupling_effect=descriptive(ms4,True),O4_minus_MS4_coupling_effect=descriptive(sharing,True),proper_loss_screen_scope='Practical screens; not statistical noninferiority',sharing_scope='A benefit reproduced by MS4 limits sharing-specific interpretation',interpretation='One encountered graph; development pilot, no confirmatory superiority or unused confirmation claim')

def evaluate(config,custody,output):
    started=time.monotonic();panel,index=close_panel(config);owner_closed(config['calibration_owner_end'])
    temperatures=json.loads(checked(config['temperatures']).read_text());expected={(b,a) for b in BLOCKS for a in arms()};endpoints={(r['block'],r['arm']):r for r in temperatures['records']}
    require(temperatures['complete'] and temperatures['endpoints']==42 and temperatures['actual_scalar_updates']==21000 and len(temperatures['records'])==42 and set(endpoints)==expected and temperatures['panel']==config['panel'] and temperatures['role_custody']==config['custody'] and temperatures['DEV_truth_read'] is False,'All 42 endpoints closed before one DEV release')
    require(all(r['status']=='fixed_endpoint_complete' and r['updates']==500 and r['endpoint']=='fixed_final_update_500' and math.isfinite(r['temperature']) and r['temperature']>0 and r['cal_archive']==index[k]['exports']['cal'] for k,r in endpoints.items()),'No endpoint substitution/retry/grid')
    custodian=json.loads(checked(config['custody']).read_text())
    import numpy as np
    release=dict(blocks=list(BLOCKS),expected_rows_per_block=34093,one_whole_release=True,after_selected_states=69,after_bank_exports=42,after_fixed_CAL_endpoints=42,temperatures=config['temperatures'],role_custody=config['custody'],DEV_truth_receipts={str(b):custodian['files'][f'scorer_only/block_{b}_dev.npz'] for b in BLOCKS},custody=custody)
    write(output/'DEV_RELEASE.json',release)
    result=dict(complete=False,retry=False,survivor_subset=False,release=release,records=[],panel=config['panel'],temperatures=config['temperatures'],custody=custody)
    try:
        # Release all three complete DEV roles together before any score.
        roles={b:truth(np,custodian,b,'dev') for b in BLOCKS};labels={b:v[1] for b,v in roles.items()};values={}
        for b in BLOCKS:
            for a in arms():
                guard(config,started);raw=bank_arrays(np,index[b,a],'dev');ids,y=roles[b]
                require(np.array_equal(ids,raw['ids']),'Exact complete DEV row identity')
                value=score_bank(np,raw,y,endpoints[b,a]['temperature']);values[b,a]=value
                result['records'].append(dict(block=b,arm=a,actual_served_members=index[b,a]['actual_served_members'],raw_archive=index[b,a]['exports']['dev'],temperature=endpoints[b,a]['temperature'],counts=value['counts'],quality=value['quality'],classes=value['classes']))
                del raw;write(output/'DEV_PROGRESS.json',dict(scored_banks=len(result['records']),expected_banks=42))
        require(len(result['records'])==42,'Whole panel scored; no survivor subset')
        result.update(complete=True,comparisons=comparisons(np,values,labels),gates=gates(values),arm_summaries=[dict(arm=a,metrics={k:descriptive([values[b,a]['quality'][k] for b in BLOCKS]) for k in METRICS}) for a in arms()],costs=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),operation_counts=panel['counts'],paid_acquisition_costs=panel['costs'],actual_scalar_updates=21000,scope='Complete numeric development screen; no selection from DEV, no automatic next study')
    except BaseException as e:
        result['failure']=dict(type=type(e).__name__,message=str(e));write(output/'DEVELOPMENT_REPORT.json',result);raise
    write(output/'DEVELOPMENT_REPORT.json',result);return result

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    if config['stage']=='calibrate':return calibrate(config,custody,output)
    if config['stage']=='score':return evaluate(config,custody,output)
    raise ValueError('Reader stage must be calibrate or score')
