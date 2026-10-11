"""Complete Physics CAL then one TEST readout; no model imports/forwards.

Stage entry requires complete60 prediction-bank/90-body custody. Failed endpoints
remain terminal records; incomplete paired rosters never become survivor means.
"""
import argparse
from datetime import datetime,timezone
import importlib
import math
from pathlib import Path
import resource
import time
import traceback
from contracts import ARMS,BLOCKS,MEMBERS,N,C,COUNTS,require,receipt,read,write,scoped,gate,close_predictions
from inference import adopted_report,descriptive

METRICS=('accuracy_pct','NLL','Brier','mean_member_accuracy_pct','worst_member_accuracy_pct',
         'mean_member_NLL','worst_member_NLL','mean_member_Brier','worst_member_Brier')

def now():
    return datetime.now(timezone.utc).isoformat()

def finite_stage(start,seconds):
    require(time.monotonic()-start<seconds,'Fixed CPU-stage cap reached; retain failure, no retry or TEST continuation')

def providers(cfg):
    for name,item in cfg['provider_modules'].items():
        require(Path(importlib.import_module(name).__file__).resolve()==receipt(item),'Pinned provider differs: '+name)
    np=importlib.import_module('numpy');torch=importlib.import_module('torch')
    torch.set_num_threads(2)
    return np,torch

def ids_only(np,item,count):
    ids=np.load(receipt(item),allow_pickle=False)
    require(isinstance(ids,np.ndarray) and ids.dtype==np.int64 and ids.shape==(count,)
            and len(np.unique(ids))==count and np.array_equal(ids,np.sort(ids))
            and ids.min()>=0 and ids.max()<N,'Fixed sorted label-free role IDs')
    return ids

def truth(np,cfg,block,role):
    row=cfg['roles'][str(block)];prefix='cal' if role=='CAL' else 'confirm'
    with np.load(receipt(row[prefix+'_truth']),allow_pickle=False) as archive:
        require(set(archive.files)=={'ids','y'},'Separate proper-role truth only')
        ids,y=archive['ids'].copy(),archive['y'].copy()
    expected=ids_only(np,row[prefix+'_ids'],COUNTS[role])
    require(np.array_equal(ids,expected) and y.dtype==np.int64 and y.shape==expected.shape
            and set(np.unique(y))==set(range(C)),'Fixed role membership/five-class truth')
    return ids,y

def load_bank(np,bank,role,expected_ids):
    item=bank['selected_'+role+'_archive']
    with np.load(receipt(item),allow_pickle=False) as a:
        require(set(a.files)=={'ids','member_log_probs','native_member_argmax','native_pool_argmax'},
                'Label-free archived native probability-mean authority')
        ids,logp,member,pool=(a[k].copy() for k in ('ids','member_log_probs','native_member_argmax','native_pool_argmax'))
    m=MEMBERS[bank['arm']];rows=COUNTS[role]
    require(ids.dtype==np.int64 and np.array_equal(ids,expected_ids)
            and logp.dtype==np.float32 and logp.shape==(m,rows,C) and np.isfinite(logp).all()
            and np.allclose(np.exp(logp.astype(np.float64)).sum(-1),1.,rtol=1e-5,atol=1e-7),
            'Finite native float32 member log probabilities and role identity')
    require(member.dtype==pool.dtype==np.int64 and member.shape==(m,rows) and pool.shape==(rows,)
            and np.isin(member,np.arange(C)).all() and np.isin(pool,np.arange(C)).all(),
            'Archived native member/pool integer decisions')
    if m==1:
        require(np.array_equal(member[0],pool),'A single member is its probability pool')
    return logp,member,pool

def log_predictions(torch,logp,log_T):
    temperature=log_T.exp()
    require(logp.device.type=='cpu' and logp.dtype==torch.float64
            and bool(torch.isfinite(temperature)) and float(temperature.detach())>0,'Positive CPUfloat64 temperature')
    # Re-normalize stored float32 log probabilities stably in float64. Integer
    # raw decisions remain the archived native authority, including ties.
    members=(logp/temperature).log_softmax(-1)
    pooled=torch.logsumexp(members,0)-math.log(len(logp))
    require(bool(torch.isfinite(members).all()) and bool(torch.isfinite(pooled).all()),'Finite fixed readout')
    return pooled,members

def calibrate(cfg,adm,custody,predictions,output):
    np,torch=providers(cfg)
    # No TEST truth load, prediction comparison or partial endpoint promotion.
    start=time.monotonic();index={(r['block'],r['arm']):r for r in predictions['records']}
    roles={b:truth(np,cfg,b,'CAL') for b in BLOCKS}
    rows=[]
    for block in BLOCKS:
        ids,target=roles[block];y=torch.from_numpy(target)
        for arm in ARMS:
            finite_stage(start,cfg['stage_seconds']['calibrate'])
            bank=index[block,arm];before=time.monotonic()
            record={'block':block,'arm':arm,'status':'failed_retained','requested_updates':500,'updates':0,
                    'role':'CAL','parameters':1,'TEST_truth_access':False,'retry':False}
            try:
                require(bank['status']=='selected_finite','Missing native prediction bank retained; no replacement')
                raw,_,_=load_bank(np,bank,'CAL',ids)
                logp=torch.from_numpy(raw).to(torch.float64)
                log_T=torch.nn.Parameter(torch.zeros((),dtype=torch.float64,device='cpu'))
                optimizer=torch.optim.Adam([log_T],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
                trace=[]
                for update in range(1,501):
                    finite_stage(start,cfg['stage_seconds']['calibrate'])
                    optimizer.zero_grad();pooled,_=log_predictions(torch,logp,log_T)
                    objective=-pooled[torch.arange(len(y)),y].mean()
                    require(bool(torch.isfinite(objective)),'Finite fixed CAL objective')
                    objective.backward();require(bool(torch.isfinite(log_T.grad)),'Finite scalar gradient')
                    optimizer.step();record['updates']=update
                    if update in (1,100,200,300,400,500):
                        trace.append({'update':update,'pre_update_CAL_NLL':float(objective.detach())})
                log_predictions(torch,logp,log_T)
                record.update(status='finite_fixed_endpoint',log_T=float(log_T.detach()),temperature=float(log_T.detach().exp()),
                              endpoint='fixed_final_update',trace=trace,selected_CAL_archive=bank['selected_CAL_archive'])
            except Exception as error:
                record.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            record['seconds']=time.monotonic()-before;rows.append(record)
    require(len(rows)==60 and {(r['block'],r['arm']) for r in rows}=={(b,a) for b in BLOCKS for a in ARMS},
            'All60 fixed endpoints terminal before TEST release')
    close_predictions(cfg['prediction_closure'],custody['protocol_sha256'])
    value={'terminal':True,'scalar_endpoints':60,'records':rows,'all_finite':all(r['status']=='finite_fixed_endpoint' for r in rows),
           'requested_scalar_updates':30000,'actual_scalar_updates':sum(r['updates'] for r in rows),
           'comparisons':[],'TEST_truth_access':False,'custody':custody,'prediction_closure':cfg['prediction_closure'],
           'CAL_truth_receipts':{str(b):cfg['roles'][str(b)]['cal_truth'] for b in BLOCKS},
           'seconds':time.monotonic()-start,'process_lifetime_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'selection':'Fixed final update; no grid/retry/fullSELECT_CAL refit','interpretation_eligible':False}
    write(output/'COMPLETE_CAL.json',value)
    return value

def score(torch,logp,y,member_authority,pool_authority,log_T,calibrated):
    pooled,members=log_predictions(torch,logp.to(torch.float64),log_T if calibrated else torch.zeros((),dtype=torch.float64))
    decision=pooled.argmax(-1) if calibrated else pool_authority
    correct=decision.eq(y);member_correct=member_authority.eq(y[None])
    covered=member_correct.any(0);lost=covered & ~correct;aggregation=~covered & correct
    target_logp=members.gather(2,y[None,:,None].expand(len(members),-1,1)).squeeze(-1)
    false_class=torch.nn.functional.one_hot(y,C).eq(0)
    strict_rival=((members>target_logp[:,:,None]).all(0) & false_class).any(-1)
    member_nll=-target_logp;pool_nll=-pooled[torch.arange(len(y)),y]
    target=torch.nn.functional.one_hot(y,C).to(torch.float64)
    pool_brier=((pooled.exp()-target)**2).sum(-1)
    member_brier=((members.exp()-target[None])**2).sum(-1)
    def subset(mask):
        count=int(mask.sum());p=correct[mask];v=covered[mask];l=lost[mask];g=aggregation[mask]
        member=member_correct[:,mask];nll=member_nll[:,mask];brier=member_brier[:,mask]
        counts={'nodes':count,'pooled_correct':int(p.sum()),'member_correct':member.sum(1).tolist(),
                'coverage':int(v.sum()),'pooling_losses':int(l.sum()),'aggregation_only_correct':int(g.sum()),
                'strict_wrong_rivals':int(strict_rival[mask].sum()),
                'strict_wrong_rivals_corrected':int((strict_rival[mask] & p).sum())}
        require(counts['pooled_correct']==counts['coverage']-counts['pooling_losses']+counts['aggregation_only_correct'],
                'Exact coverage/loss/aggregation decomposition')
        quality={'accuracy_pct':100*counts['pooled_correct']/count,'NLL':float(pool_nll[mask].mean()),'Brier':float(pool_brier[mask].mean()),
                 'member_accuracy_pct':(member.double().mean(1)*100).tolist(),'mean_member_accuracy_pct':float(member.double().mean()*100),
                 'worst_member_accuracy_pct':float(member.double().mean(1).min()*100),'member_NLL':nll.mean(1).tolist(),
                 'mean_member_NLL':float(nll.mean()),'worst_member_NLL':float(nll.mean(1).max()),'member_Brier':brier.mean(1).tolist(),
                 'mean_member_Brier':float(brier.mean()),'worst_member_Brier':float(brier.mean(1).max())}
        require(all(math.isfinite(quality[k]) for k in METRICS),'Finite complete metrics')
        return {'counts':counts,'quality':quality}
    result=subset(torch.ones(len(y),dtype=torch.bool))
    result['classes']=[dict(subset(y==k),class_id=k) for k in range(C)]
    result['float64_member_argmax_disagreements_with_native_archive']=int((members.argmax(-1)!=member_authority).sum())
    result.update(P=correct,V=covered,S=strict_rival)
    return result

def error_pair(a,b,mask):
    repair=(~b['P'] & a['P']) & mask;harm=(b['P'] & ~a['P']) & mask
    av,bv=a['V'],b['V'];ap,bp=a['P'],b['P']
    counts={'repairs':int(repair.sum()),'harms':int(harm.sum()),
            'coverage_added':int((av & ~bv & mask).sum()),'coverage_removed':int((bv & ~av & mask).sum()),
            'repair_causes':{'new_member_alternative':int((repair & av & ~bv).sum()),
                             'both_banks_have_member_alternative':int((repair & av & bv).sum()),
                             'aggregation_only':int((repair & ~av).sum())},
            'harm_causes':{'available_alternative_lost':int((harm & av).sum()),
                           'removed_member_alternative':int((harm & ~av & bv).sum()),
                           'lost_aggregation_only':int((harm & ~av & ~bv).sum())},
            'count_decomposition_delta':{'pooled_correct':int((ap & mask).sum()-(bp & mask).sum()),
                                         'coverage':int((av & mask).sum()-(bv & mask).sum()),
                                         'pooling_losses':int((av & ~ap & mask).sum()-(bv & ~bp & mask).sum()),
                                         'aggregation_only_correct':int((~av & ap & mask).sum()-(~bv & bp & mask).sum())},
            'fixed_reference_strict_rivals_repaired':int((b['S'] & ap & mask).sum()),
            'new_strict_rivals_created':int((~b['S'] & a['S'] & mask).sum())}
    d=counts['count_decomposition_delta']
    require(d['pooled_correct']==counts['repairs']-counts['harms']==d['coverage']-d['pooling_losses']+d['aggregation_only_correct'],
            'Exact paired net accounting')
    require(sum(counts['repair_causes'].values())==counts['repairs'] and sum(counts['harm_causes'].values())==counts['harms'],
            'Gross repair/harm partition')
    return counts

def pair_report(torch,values,candidate,reference,labels,t_distribution):
    pairs=[(values[b,candidate],values[b,reference]) for b in BLOCKS]
    delta={k:descriptive([None if a is None or b is None else a['quality'][k]-b['quality'][k] for a,b in pairs],t_distribution) for k in METRICS}
    class_delta=[{'class_id':k,'quality_deltas':{metric:descriptive([None if a is None or b is None else a['classes'][k]['quality'][metric]-b['classes'][k]['quality'][metric] for a,b in pairs],t_distribution) for metric in METRICS}} for k in range(C)]
    rows=[]
    for block,(a,b) in zip(BLOCKS,pairs):
        if a is None or b is None:
            rows.append({'block':block,'incomplete':True,'no_complete_case_subset':True})
        else:
            y=labels[block]
            rows.append({'block':block,**error_pair(a,b,torch.ones(len(y),dtype=torch.bool)),
                         'classes':[{'class_id':k,**error_pair(a,b,y==k)} for k in range(C)]})
    return {'candidate':candidate,'reference':reference,'complete_ten_blocks':all(a is not None and b is not None for a,b in pairs),
            'quality_deltas':delta,'class_quality_deltas':class_delta,'per_block_error_diagnostics':rows,
            'scope':'Paired selected procedures; overlapping graph readouts are descriptive, not causal node trials',
            'no_complete_case_subset':True}

def test(cfg,adm,custody,predictions,output):
    calibration=read(receipt(adm['calibration_terminal']))
    require(adm.get('calibration_owner_closed_and_clean') is True and calibration.get('terminal') is True
            and calibration.get('scalar_endpoints')==60 and len(calibration.get('records',[]))==60
            and calibration.get('TEST_truth_access') is False and calibration.get('prediction_closure')==cfg['prediction_closure']
            and calibration['custody']['config_sha256']==custody['config_sha256']
            and calibration['custody']['source_seal_sha256']==custody['source_seal_sha256'],
            'All60 fixed CAL endpoints/root owner close before one complete TEST release')
    endpoints={(r['block'],r['arm']):r for r in calibration['records']}
    require(set(endpoints)=={(b,a) for b in BLOCKS for a in ARMS} and all(r['status'] in ('finite_fixed_endpoint','failed_retained') for r in endpoints.values()),
            'Exact terminal scalar roster without topup')
    require(all(r['updates']==500 and r['endpoint']=='fixed_final_update' for r in endpoints.values() if r['status']=='finite_fixed_endpoint'),
            'Every finite endpoint is the fixed500th update')
    np,torch=providers(cfg);t_distribution=importlib.import_module('scipy.stats').t
    start=time.monotonic();release=now()
    # All ten TEST role truths are released together, only after the gates above.
    roles={b:truth(np,cfg,b,'CONFIRM') for b in BLOCKS};labels={b:torch.from_numpy(v[1]) for b,v in roles.items()}
    index={(r['block'],r['arm']):r for r in predictions['records']}
    scores={'raw':{},'calibrated':{}};rows=[]
    for block in BLOCKS:
        ids,_=roles[block];y=labels[block]
        for arm in ARMS:
            finite_stage(start,cfg['stage_seconds']['test'])
            bank=index[block,arm];endpoint=endpoints[block,arm];before=time.monotonic()
            row={'block':block,'arm':arm,'selected_predictions':bank.get('selected_CONFIRM_archive'),
                 'fits':bank['fits'],'calibration_endpoint':endpoint,'same_selected_native_state':True}
            try:
                require(bank['status']=='selected_finite','Native failure retained')
                logp,member,pool=load_bank(np,bank,'CONFIRM',ids)
                logp=torch.from_numpy(logp);member=torch.from_numpy(member);pool=torch.from_numpy(pool)
            except Exception as error:
                for kind in ('raw','calibrated'):
                    scores[kind][block,arm]=None;row[kind]={'failed_retained':True,'error_type':type(error).__name__,'error':str(error),'retry':False}
                row['traceback']=traceback.format_exc()
            else:
                for kind in ('raw','calibrated'):
                    try:
                        require(kind=='raw' or endpoint['status']=='finite_fixed_endpoint','Fixed CAL endpoint failure retained')
                        log_T=torch.zeros((),dtype=torch.float64) if kind=='raw' else torch.tensor(endpoint['log_T'],dtype=torch.float64)
                        value=score(torch,logp,y,member,pool,log_T,kind=='calibrated')
                        row[kind]={k:value[k] for k in ('quality','counts','classes','float64_member_argmax_disagreements_with_native_archive')}
                    except Exception as error:
                        value=None;row[kind]={'failed_retained':True,'error_type':type(error).__name__,'error':str(error),'retry':False,
                                              'traceback':traceback.format_exc()}
                    scores[kind][block,arm]=value
            row['reader_seconds']=time.monotonic()-before;rows.append(row)
    aggregate={kind:{arm:{'quality':{metric:descriptive([None if values[b,arm] is None else values[b,arm]['quality'][metric] for b in BLOCKS],t_distribution) for metric in METRICS},
        'classes':[{'class_id':k,'quality':{metric:descriptive([None if values[b,arm] is None else values[b,arm]['classes'][k]['quality'][metric] for b in BLOCKS],t_distribution) for metric in METRICS}} for k in range(C)]} for arm in ARMS} for kind,values in scores.items()}
    contrasts={kind:{a+'_MINUS_'+b:pair_report(torch,values,a,b,labels,t_distribution) for a in ARMS for b in ARMS if a!=b} for kind,values in scores.items()}
    inference=adopted_report(scores['raw'],t_distribution,COUNTS['CONFIRM'])
    confidence={}
    for ref in ('BASE','equipped_I4'):
        differences=[None if scores['calibrated'][b,'E4'] is None or scores['calibrated'][b,ref] is None else
                     scores['calibrated'][b,'E4']['quality']['NLL']-scores['calibrated'][b,ref]['quality']['NLL'] for b in BLOCKS]
        description=descriptive(differences,t_distribution);complete=description['complete_ten_blocks']
        confidence[ref]={'NLL_harm':description,'practical_diagnostic_pass':complete and description.get('mean') is not None
                          and description['mean']<=.02 and max(differences)<=.05,
                         'each_block_allowance':.05,'mean_allowance':.02,'not_statistical_noninferiority':True,
                         'separate_from_primary_raw_accuracy':True}
    close_predictions(cfg['prediction_closure'],custody['protocol_sha256']);receipt(adm['calibration_terminal'])
    for block in BLOCKS:receipt(cfg['roles'][str(block)]['confirm_truth'])
    finite_stage(start,cfg['stage_seconds']['test'])
    value={'terminal':True,'one_complete_TEST_truth_release':True,'TEST_truth_access':True,'truth_release_UTC':release,
           'blocks':list(BLOCKS),'arms':list(ARMS),'rows':rows,'arm_aggregates':aggregate,'all_ordered_pair_contrasts':contrasts,
           'prospectively_fixed_raw_inference':inference,'separate_calibrated_NLL_risk_screen':confidence,
           'calibration_terminal':adm['calibration_terminal'],'custody':custody,'prediction_closure':cfg['prediction_closure'],
           'TEST_truth_receipts':{str(b):cfg['roles'][str(b)]['confirm_truth'] for b in BLOCKS},
           'native_preparation_acquisition_export_cost_records':predictions['cost_records'],
           'native_source_bindings':predictions['source_bindings'],
           'all_raw_finite':all(scores['raw'][b,a] is not None for b in BLOCKS for a in ARMS),
           'all_calibrated_finite':all(scores['calibrated'][b,a] is not None for b in BLOCKS for a in ARMS),
           'seconds':time.monotonic()-start,'process_lifetime_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'no_new_model_fits_forwards_or_state_deserializations':True,'missing_pairs_inconclusive_no_survivor_subset':True,
           'cost_scope':'Native acquisition/export/checkpoint/restore scopes retained in fit/closure records; CPU serving/calibration separated; Penn overlap prevents isolated speed claims',
           'no_novelty_or_manuscript_verdict':True}
    write(output/'COMPLETE_TEST.json',value)
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('calibrate','test'),required=True);args=parser.parse_args()
    cfg,adm,custody,predictions=gate(args.config,args.admission,args.stage)
    output=scoped(args.output);require(not output.exists(),'Fresh stage directory only');output.mkdir()
    value=calibrate(cfg,adm,custody,predictions,output) if args.stage=='calibrate' else test(cfg,adm,custody,predictions,output)
    write(output/'STAGE_END.json',{'terminal':True,'stage':args.stage,'complete_roster':True,
                                  'all_finite':value['all_finite'] if args.stage=='calibrate' else value['all_raw_finite'] and value['all_calibrated_finite'],
                                  'TEST_truth_access':args.stage=='test'})
    print('{"terminal":true,"complete_stage":true,"arrays_exported":false}')

if __name__=='__main__':main()
