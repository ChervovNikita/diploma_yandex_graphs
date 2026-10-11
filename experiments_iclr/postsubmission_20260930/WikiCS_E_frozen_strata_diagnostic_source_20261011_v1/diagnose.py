"""Frozen CPU-only descriptive pass over complete encountered WikiCS E banks.

Archived error flags own all decisions. Raw logits are used only for the added
same-false-class strict-rival rank predicate. No replay, calibration or model.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS=(7301,7403,7507)
ARMS=('BASE','E4','N4','equipped_I4')
REFS=('BASE','N4','equipped_I4')
N,ROWS,C,M=11701,5274,10,4

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):result.update(part)
    return result.hexdigest()

def receipt(value):
    path=Path(value['path']).resolve()
    require(path.is_relative_to(R) and path.stat().st_size==value['bytes'] and sha(path)==value['sha256'],
            'Pinned original input changed: '+str(path))
    return path

def interrupted(signum,frame):raise TimeoutError('Fixed CPU diagnostic wall cap; retain partial output without interpretation')

def bank(np,item):
    with np.load(receipt(item),allow_pickle=False) as a:
        require(set(a.files)=={'ids','y','raw_logits','probability_mean','member_errors','pooled_errors'},'Original exact6-field archive')
        ids,y,prob,errors,pool,logits=(a[k].copy() for k in ('ids','y','probability_mean','member_errors','pooled_errors','raw_logits'))
    require(ids.dtype==y.dtype==np.int64 and ids.shape==y.shape==(ROWS,) and np.array_equal(ids,np.sort(ids))
            and len(np.unique(ids))==ROWS and ids.min()>=0 and ids.max()<N and np.isin(y,np.arange(C)).all(),'Original VALID IDs/classes')
    require(errors.dtype==pool.dtype==np.bool_ and errors.shape==(M,ROWS) and pool.shape==(ROWS,), 'Original boolean decision authority')
    require(prob.dtype==logits.dtype==np.float32 and prob.shape==(ROWS,C) and logits.shape==(M,ROWS,C)
            and np.isfinite(prob).all() and np.isfinite(logits).all(),'Original finite float32 scores/probabilities')
    correct_members=~errors;correct=~pool;k=correct_members.sum(0);coverage=k>0
    # Same one false class must strictly outrank truth in every member. Members
    # may have different wrong top classes; no argmax or decision is replaced.
    truth=logits[:,np.arange(ROWS),y]
    better=(logits>truth[:,:,None]).all(0);better[np.arange(ROWS),y]=False
    strict=better.any(-1)
    del logits,truth,better
    return {'ids':ids,'y':y,'prob':prob,'P':correct,'V':coverage,'L':coverage & ~correct,
            'G':~coverage & correct,'member':correct_members,'k':k,'S':strict}

def cell(np,a,b,mask):
    n=int(mask.sum());ap,bp=a['P'],b['P'];av,bv=a['V'],b['V']
    repair=~bp & ap & mask;harm=bp & ~ap & mask
    result={'n_readouts':n,'candidate_correct':int((ap & mask).sum()),'reference_correct':int((bp & mask).sum()),
            'reference_wrong':int((~bp & mask).sum()),'repairs':int(repair.sum()),'harms':int(harm.sum()),
            'candidate_member_correct_total':int(a['member'][:,mask].sum()),'reference_member_correct_total':int(b['member'][:,mask].sum()),
            'candidate_coverage':int((av & mask).sum()),'reference_coverage':int((bv & mask).sum()),
            'candidate_pool_losses':int((a['L'] & mask).sum()),'reference_pool_losses':int((b['L'] & mask).sum()),
            'candidate_pool_only_correct':int((a['G'] & mask).sum()),'reference_pool_only_correct':int((b['G'] & mask).sum()),
            'coverage_added':int((av & ~bv & mask).sum()),'coverage_removed':int((bv & ~av & mask).sum()),
            'repairs_new_member_alternative':int((repair & av & ~bv).sum()),
            'repairs_both_have_member_alternative':int((repair & av & bv).sum()),'repairs_pool_only':int((repair & ~av).sum()),
            'harms_available_alternative_lost':int((harm & av).sum()),'harms_removed_member_alternative':int((harm & ~av & bv).sum()),
            'harms_lost_pool_only':int((harm & ~av & ~bv).sum()),
            'candidate_strict_rival':int((a['S'] & mask).sum()),'reference_strict_rival':int((b['S'] & mask).sum()),
            'reference_strict_rival_repaired':int((b['S'] & repair).sum()),
            'new_strict_rival_created':int((~b['S'] & a['S'] & mask).sum()),
            'strict_rival_removed':int((b['S'] & ~a['S'] & mask).sum()),
            'candidate_strict_rival_served_correct':int((a['S'] & ap & mask).sum()),
            'reference_strict_rival_served_correct':int((b['S'] & bp & mask).sum())}
    for route in range(M):
        result['candidate_member_route'+str(route)+'_correct']=int((a['member'][route] & mask).sum())
        result['reference_member_route'+str(route)+'_correct']=int((b['member'][route] & mask).sum())
    for k in range(5):
        result['candidate_k'+str(k)]=int(((a['k']==k) & mask).sum())
        result['reference_k'+str(k)]=int(((b['k']==k) & mask).sum())
    for side in ('candidate','reference'):
        require(result[side+'_correct']==result[side+'_coverage']-result[side+'_pool_losses']+result[side+'_pool_only_correct']
                and sum(result[side+'_k'+str(k)] for k in range(5))==n,'Per-bank stored-flag identity and complete member histogram')
    result['net_correct']=result['candidate_correct']-result['reference_correct']
    result['coverage_delta']=result['candidate_coverage']-result['reference_coverage']
    result['pool_loss_delta']=result['candidate_pool_losses']-result['reference_pool_losses']
    result['pool_only_delta']=result['candidate_pool_only_correct']-result['reference_pool_only_correct']
    require(result['net_correct']==result['repairs']-result['harms']
            ==result['coverage_delta']-result['pool_loss_delta']+result['pool_only_delta'],'Exact stored-flag net accounting')
    require(result['repairs']==sum(result[k] for k in ('repairs_new_member_alternative','repairs_both_have_member_alternative','repairs_pool_only'))
            and result['harms']==sum(result[k] for k in ('harms_available_alternative_lost','harms_removed_member_alternative','harms_lost_pool_only')),
            'Gross stored-flag repair/harm partition')
    return result

def rates(result,unique_nodes):
    n=result['n_readouts'];result['unique_nodes']=unique_nodes
    result['small_readout_cell']=n<20;result['small_unique_node_cell']=unique_nodes<20
    result['accuracy_delta_pp']=100*result['net_correct']/n if n else None
    result['candidate_accuracy_pct']=100*result['candidate_correct']/n if n else None
    result['reference_accuracy_pct']=100*result['reference_correct']/n if n else None
    result['candidate_mean_member_accuracy_pct']=100*result['candidate_member_correct_total']/(M*n) if n else None
    result['reference_mean_member_accuracy_pct']=100*result['reference_member_correct_total']/(M*n) if n else None
    result['member_accuracy_delta_pp']=(100*(result['candidate_member_correct_total']-result['reference_member_correct_total'])/(M*n)) if n else None
    result['pooling_gain_delta_pp']=result['accuracy_delta_pp']-result['member_accuracy_delta_pp'] if n else None
    result['repair_fraction_of_reference_wrong']=result['repairs']/result['reference_wrong'] if result['reference_wrong'] else None
    result['harm_fraction_of_reference_correct']=result['harms']/result['reference_correct'] if result['reference_correct'] else None
    for side in ('candidate','reference'):
        coverage=result[side+'_coverage']
        result[side+'_pool_loss_fraction_of_coverage']=result[side+'_pool_losses']/coverage if coverage else None
        for route in range(M):
            result[side+'_member_route'+str(route)+'_accuracy_pct']=100*result[side+'_member_route'+str(route)+'_correct']/n if n else None
    for total,names in (('repairs',('repairs_new_member_alternative','repairs_both_have_member_alternative','repairs_pool_only')),
                        ('harms',('harms_available_alternative_lost','harms_removed_member_alternative','harms_lost_pool_only'))):
        for name in names:result[name+'_fraction']=result[name]/result[total] if result[total] else None
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--admission',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    require(socket.gethostname()=='anogena-2-0' and Path.cwd()==R,'Authorized scientific host/repository before project reads')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID],
            'Sole authorized allocation; diagnostic creates no CUDA context')
    here=Path(__file__).resolve().parent;plan_path=here/'PLAN.json';plan=json.loads(plan_path.read_text())
    admission=json.loads(args.admission.read_text())
    require(admission.get('admitted') is True and admission.get('action')=='closed_WikiCS_E_frozen_strata_diagnostic'
            and admission.get('source_sha256')==sha(__file__) and admission.get('plan_sha256')==sha(plan_path)
            and admission.get('TEST_access') is False and admission.get('current_Physics_Penn_access') is False,
            'Root exact source/frozen-plan CPU admission required')
    require(type(admission.get('wall_seconds')) is int and 0<admission['wall_seconds']<=900,'Finite CPU diagnostic cap')
    signal.signal(signal.SIGALRM,interrupted);signal.alarm(admission['wall_seconds']);start=time.monotonic()
    complete=json.loads(receipt(plan['complete_analysis']).read_text())
    require(complete['complete'] and complete['banks']==15 and complete['optimizer_acquisition_bundles']==24
            and complete['native_body_fit_records']==24 and complete['TEST_access'] is False
            and complete['fixed_seed_roster']==list(SEEDS),'Original entire E family closed before arrays')
    for value in plan['all_archive_receipts']:receipt(value)
    import numpy as np  # Numerical data code starts only after closed-custody checks.
    with np.load(receipt(plan['safe_train']),allow_pickle=False) as graph:
        require(set(graph.files)=={'x','edge_index','ids','y'},'Exact safe TRAIN container; only edges decoded')
        edges=graph['edge_index'].copy()
    require(edges.dtype==np.int64 and edges.shape==(2,442907) and edges.min()>=0 and edges.max()<N,'Original full factual support')
    nonself=edges[0]!=edges[1]
    require(int((~nonself).sum())==N,'Original11701 self loops; excluded from degree')
    degree=np.bincount(edges[1,nonself],minlength=N)
    q=np.quantile(degree,[.25,.5,.75],method='higher')
    degree_band=np.searchsorted(q,degree,side='left');del edges
    index={(r['seed'],r['arm']):r for r in plan['all_archive_receipts']}
    values={(s,a):bank(np,index[s,a]) for s in SEEDS for a in ARMS}
    anchor=values[SEEDS[0],'BASE']
    require(all(np.array_equal(v['ids'],anchor['ids']) and np.array_equal(v['y'],anchor['y']) for v in values.values()),
            'Same original5274 IDs/classes across all arms/three dependent readouts')
    masks={}
    for seed in SEEDS:
        base=values[seed,'BASE'];confidence=base['prob'].max(1)
        top=np.partition(base['prob'],-2,axis=1)[:,-2:];margin=top.max(1)-top.min(1)
        conf_band=np.searchsorted(np.array([.5,.8],dtype=np.float32),confidence,side='right')
        margin_band=np.searchsorted(np.array([.1,.3],dtype=np.float32),margin,side='right')
        d=degree_band[base['ids']]
        for cls,cb,db in itertools.product(range(10),range(3),range(4)):
            masks[seed,'class_conf_degree',cls,cb,db]=(base['y']==cls) & (conf_band==cb) & (d==db)
        for mb in range(3):masks[seed,'margin',mb,-1,-1]=margin_band==mb
        masks[seed,'all',-1,-1,-1]=np.ones(ROWS,dtype=bool)
        require(sum(int(masks[seed,'class_conf_degree',cl,cb,db].sum()) for cl,cb,db in itertools.product(range(10),range(3),range(4)))==ROWS,
                'All bins including ties/empty bands partition the complete readout')
    keys=[('all',-1,-1,-1)]+[('class_conf_degree',cl,cb,db) for cl,cb,db in itertools.product(range(10),range(3),range(4))]+[('margin',mb,-1,-1) for mb in range(3)]
    rows=[];overall={}
    for ref in REFS:
        for key in keys:
            pieces=[];distinct=set()
            for seed in SEEDS:
                mask=masks[(seed,)+key];value=cell(np,values[seed,'E4'],values[seed,ref],mask)
                pieces.append(value);distinct.update(int(i) for i in anchor['ids'][mask])
                rows.append({'reference':ref,'view':'seed_'+str(seed),'stratum':key[0],'axis1':key[1],'axis2':key[2],'axis3':key[3],
                             **rates(dict(value),int(mask.sum()))})
            pooled={name:sum(piece[name] for piece in pieces) for name in pieces[0]}
            rows.append({'reference':ref,'view':'sum_three_dependent_readouts','stratum':key[0],'axis1':key[1],'axis2':key[2],'axis3':key[3],
                         **rates(pooled,len(distinct))})
            if key[0]=='all':overall[ref]=pooled
    require(len(rows)==1488,'Complete frozen class/confidence/degree and marginal-margin table')
    output=args.output.resolve();require(output.is_relative_to(P) and not output.exists(),'Fresh diagnostic output in authorized phase')
    output.mkdir()
    with (output/'STRATA.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for value in plan['all_archive_receipts']:receipt(value)
    receipt(plan['safe_train'])
    summary={'complete':True,'plan_sha256':sha(plan_path),'source_sha256':sha(__file__),'rows':len(rows),'overall':overall,
             'degree_quartiles_higher_all11701':q.tolist(),'graph_nonself_directed_entries':int(degree.sum()),
             'true_class_axis':'Stored factual class IDs0..9','confidence_axis':['p<.50','.50<=p<.80','p>=.80'],
             'degree_axis':['d<=q25','q25<d<=q50','q50<d<=q75','d>q75'],'margin_axis':['m<.10','.10<=m<.30','m>=.30'],
             'strict_rival':'One same false class raw score strictly exceeds truth in every member; rank diagnostic only, not argmax/coverage authority or classifier-bottleneck cause',
             'small_cell_rule':'n_readouts<20 or distinct_nodes<20; zero cells retained; no subgroup tests',
             'count_scope':'Three selected optimizer readouts reuse the same5274 nodes; sums are15822 dependent readouts',
             'raw_error_authority_unchanged':True,'numerical_replay_gate':False,'fits_forwards_calibration':0,
             'TEST_access':False,'current_Physics_Penn_access':False,'causal_or_unused_confirmation_claim':False,
             'seconds':time.monotonic()-start,'process_lifetime_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'overlap_scope':'CPU diagnostic during unchanged live fits; no isolated performance claim'}
    (output/'COMPLETE_DIAGNOSTIC.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    signal.alarm(0)
    print(json.dumps({'complete':True,'rows':len(rows),'TEST_access':False,'model_operations':0}))

if __name__=='__main__':main()
