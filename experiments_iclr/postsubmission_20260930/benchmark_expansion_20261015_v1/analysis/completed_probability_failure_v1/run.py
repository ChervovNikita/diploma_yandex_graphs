"""One CPU saved-prediction diagnostic of all completed33 MolBACE banks.

No model/checkpoint load, forward, update, calibrator, selection or new readout.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
SEEDS=[101,202,303]
Q=[0.,.01,.05,.25,.5,.75,.95,.99,1.]


def require(ok,message):
    if not ok:raise ValueError(message)


def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def quantiles(values):
    values=sorted(values)
    if not values:return None
    result={}
    for q in Q:
        position=(len(values)-1)*q;left=math.floor(position);right=math.ceil(position)
        result[str(q)]=values[left]+(values[right]-values[left])*(position-left)
    return result


def scalar_predictions(logits,labels):
    require(len(logits)==len(labels)>0 and all(math.isfinite(z) for z in logits) and set(labels)=={0,1},'Aligned finite binary VALID required')
    margin=[(2*y-1)*z for z,y in zip(logits,labels)]
    ptrue=[1/(1+math.exp(-m)) if m>=0 else math.exp(m)/(1+math.exp(m)) for m in margin]
    losses=[max(0.,-m)+math.log1p(math.exp(-abs(m))) for m in margin]
    brier=[(1-p)**2 for p in ptrue];correct=[(z>=0)==bool(y) for z,y in zip(logits,labels)]
    def subset(positions):
        n=len(positions)
        return {'graphs':n,'mean_BCE':math.fsum(losses[i] for i in positions)/n if n else None,
            'mean_Brier':math.fsum(brier[i] for i in positions)/n if n else None,
            'total_BCE':math.fsum(losses[i] for i in positions),
            'mean_probability_true_label':math.fsum(ptrue[i] for i in positions)/n if n else None,
            'probability_true_label_quantiles':quantiles([ptrue[i] for i in positions])}
    def group(positions):
        return {**subset(positions),'logit_quantiles':quantiles([logits[i] for i in positions]),
            'true_class_margin_quantiles':quantiles([margin[i] for i in positions]),
            'correct':subset([i for i in positions if correct[i]]),'wrong':subset([i for i in positions if not correct[i]])}
    result={'all':group(list(range(len(labels)))),'by_class':{str(c):group([i for i,y in enumerate(labels) if y==c]) for c in (0,1)},
        'threshold':'positive iff native raw logit >=0; zero is positive','quantile_rule':'linear interpolation at (n-1)q; no clipping',
        'BCE_rule':'float64 softplus of negative true-class margin; uniform per graph; no refit'}
    return result,losses


def paired(candidate,reference,ids,labels):
    require(len(candidate)==len(reference)==len(ids)==len(labels)>0,'Aligned paired graph losses required')
    delta=[a-b for a,b in zip(candidate,reference)];total=math.fsum(delta)
    positive=math.fsum(max(v,0.) for v in delta);negative=math.fsum(min(v,0.) for v in delta)
    ordered=sorted((v for v in delta if v>0),reverse=True)
    concentration={}
    for k in (1,5):
        top=math.fsum(ordered[:k])
        concentration['top'+str(k)]={'largest_positive_graph_contributions':k,'actual_positive_graphs_used':min(k,len(ordered)),
            'positive_BCE_sum':top,'fraction_of_total_excess_BCE':top/total if total>0 else None,
            'fraction_of_all_positive_BCE_mass':top/positive if positive else None,
            'remaining_signed_BCE_sum_including_all_negative_offsets':total-top}
    require(math.isclose(total,positive+negative,rel_tol=1e-12,abs_tol=1e-12),'Signed loss accounting differs')
    return {'graphs':len(delta),'mean_signed_BCE_difference':total/len(delta),'total_signed_BCE_difference':total,
        'positive_BCE_mass':positive,'negative_BCE_offsets':negative,'net_excess_BCE':total>0,
        'higher_loss_graphs':sum(v>0 for v in delta),'lower_loss_graphs':sum(v<0 for v in delta),'equal_loss_graphs':sum(v==0 for v in delta),
        'signed_difference_quantiles':quantiles(delta),'concentration':concentration,
        'by_class':{str(c):{'graphs':sum(y==c for y in labels),
            'mean_signed_BCE_difference':math.fsum(v for v,y in zip(delta,labels) if y==c)/sum(y==c for y in labels),
            'total_signed_BCE_difference':math.fsum(v for v,y in zip(delta,labels) if y==c)} for c in (0,1)},
        'concentration_rule':'Top1/top5 mean1/5 graphs; denominator is net signed excess including negative offsets. Fraction can exceed1; null when net excess<=0. No selective positive-only total.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest-sha256',required=True);args=parser.parse_args()
    require(socket.gethostname()=='anogena-2-0' and subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],
        'Dedicated allocation guard required before project metadata reads')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU saved-prediction diagnostic requires CUDA hidden')
    started=time.perf_counter();plan=json.loads((HERE/'PLAN.json').read_text());root=Path(plan['project_root'])
    require(HERE==root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/analysis/completed_probability_failure_v1' and
        Path(sys.prefix).resolve()==(root/'.venv').resolve(),'Exact reviewed source and original interpreter required')
    require(pin(HERE/'MANIFEST.json')['sha256']==args.manifest_sha256,'Root-reviewed manifest differs')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path=(HERE/row['path']).resolve();require(path.is_relative_to(HERE) and pin(path)['sha256']==row['sha256'],'Own source changed')
    allocation_output=E/'analysis_outputs/completed_probability_failure_v1'
    require(not any((HERE/name).exists() for name in ('RESULT.json','REPORT.md','GATE.json')) and not allocation_output.exists(),
        'Fresh scalar and allocation-only output required; root executes once, no overwrite/retry')
    loader_path=Path(plan['verified_loader']['path']);require(pin(loader_path)==plan['verified_loader'],'Retained saved-bank verification helper differs')
    definition=importlib.util.spec_from_file_location('_completed_saved_bank_custody',loader_path)
    helper=importlib.util.module_from_spec(definition);definition.loader.exec_module(helper);verified=helper.verified
    report_path=verified(plan['completed33_authority']);original=json.loads(report_path.read_text());records=[]
    try:
        require(original['schema']=='residual-only-selected-VALID-33-bank-addition-v1' and original['status']in ('complete','complete_with_warnings') and
            original['TEST_accessed']is False and len(original['rows'])==33 and len(original['training_fit_outcomes'])==39,'Exact completed33-bank/39-fit authority required')
        require({(r['seed'],r['bank']) for r in original['rows']}=={(seed,bank) for seed in SEEDS for bank in plan['banks']},'All11 methods x3 seeds required')
        fits={}
        for row in original['training_fit_outcomes']:
            value=json.loads(verified(row['receipt']).read_text())
            require(value['status']=='complete' and value['job_id']==row['job_id'] and value['TEST_accessed']is False and not value.get('failure') and not value.get('close_error'),
                'Completed own fit metadata differs')
            verified(value['selected_VALID_export']);fits[row['job_id']]=value
        collections={}
        for row in original['route_collection_outcomes']:
            value=json.loads(verified(row['receipt']).read_text());fit=fits[row['job_id']]
            require(value['status']in ('complete','complete_with_sanity_warning') and value['role']=='VALID' and value['TEST_accessed']is False and
                value['job_id']==row['job_id'] and value['job_hash']==fit['job_hash'] and value['selected_epoch']==fit['selected_epoch'] and
                value['selected_checkpoint']==fit['selected_checkpoint'] and value['saved_VALID_export']==fit['selected_VALID_export'] and
                value['actual_work']['TRAIN_callbacks']==value['actual_work']['optimizer_steps']==0 and
                value['evaluated_VALID_graphs']==151 and value['members']==4 and not value.get('close_error'),'Complete paid selected route collection differs')
            verified(value['raw_tensor_receipt']);collections[row['job_id']]=value
            records.append({'job_id':row['job_id'],'collection':row['receipt'],'raw':value['raw_tensor_receipt'],'status':value['status'],
                'sanity':value['sanity_against_saved_export']})
        require(len(fits)==39 and len(collections)==27,'All completed exports/collections required before tensor loading')
        train_summary=json.loads(verified(plan['TRAIN_support_authority']).read_text())['TRAIN_support']
        require((train_summary['positive'],train_summary['negative'])==(480,730),'Actual TRAIN support authority differs')
        import torch
        torch.set_num_threads(2);exports={};reference_ids=reference_labels=None
        for ident,fit in fits.items():
            saved=torch.load(verified(fit['selected_VALID_export']),map_location='cpu',weights_only=False)
            require(saved['schema']=='own-selected-VALID-logits-v1' and saved['role']=='VALID' and saved['TEST_accessed']is False and
                saved['job_id']==ident and saved['job_hash']==fit['job_hash'] and saved['selected_epoch']==fit['selected_epoch'] and
                saved['selected_checkpoint']==fit['selected_checkpoint'] and saved['validation']==fit['selected_VALID'],'Saved own-selected authority differs')
            ids,y,z=saved['ids'],saved['labels'],saved['raw_logits']
            require(tuple(z.shape)==tuple(y.shape)==(151,1) and ids.shape==(151,) and torch.unique(ids).numel()==151 and
                z.dtype==torch.float32 and torch.isfinite(z).all().item() and torch.isfinite(y).all().item() and
                int((y==1).sum())==130 and int((y==0).sum())==21,'Complete finite original VALID151 required')
            if reference_ids is None:reference_ids,reference_labels=ids,y
            require(torch.equal(ids,reference_ids) and torch.equal(y,reference_labels),'Exact original VALID order/labels differ')
            exports[ident]=saved
        raw_routes={}
        for ident,collection in collections.items():
            value=torch.load(verified(collection['raw_tensor_receipt']),map_location='cpu',weights_only=False);saved=exports[ident]
            require(value['role']=='VALID' and value['TEST_accessed']is False and value['job_id']==ident and value['job_hash']==saved['job_hash'] and
                value['selected_epoch']==saved['selected_epoch'] and value['selected_checkpoint']==saved['selected_checkpoint'] and
                torch.equal(value['ids'],reference_ids) and torch.equal(value['labels'],reference_labels) and
                tuple(value['member_raw_logits'].shape)==(4,151,1) and torch.isfinite(value['member_raw_logits']).all().item(),'Saved selected member logits differ')
            raw_routes[ident]=value['member_raw_logits']  # no preheads or checkpoint payload retained
        ids=reference_ids.tolist();labels=[int(v) for v in reference_labels[:,0].tolist()];rows=[];losses={}
        for original_row in original['rows']:
            seed,name,owners=original_row['seed'],original_row['bank'],original_row['source_job_ids']
            if name=='I4':
                require(len(owners)==4 and [exports[i]['body'] for i in owners]==[0,1,2,3] and
                    len({exports[i]['seed'] for i in owners})==4,'Four genuine independent own-selected bodies required')
                members=torch.stack([exports[i]['raw_logits'] for i in owners]);pool=members.mean(0)
            else:
                require(len(owners)==1,'Single fitted bank authority required');pool=exports[owners[0]]['raw_logits']
                members=pool.unsqueeze(0) if name=='M1' else raw_routes[owners[0]]
            pooled,per_graph=scalar_predictions(pool[:,0].double().tolist(),labels)
            require(abs(pooled['all']['mean_BCE']-original_row['uniform_graph_BCE'])<1e-12 and
                abs(pooled['all']['mean_Brier']-original_row['pooled_metrics']['Brier'])<1e-12,'Descriptive loss must match original uniform-graph score')
            member=[]
            for index,z in enumerate(members):
                item,_=scalar_predictions(z[:,0].double().tolist(),labels)
                require(abs(item['all']['mean_BCE']-original_row['members'][index]['BCE'])<1e-12,'Native member loss differs from completed authority')
                member.append(item)
            rows.append({'seed':seed,'bank':name,'owners':owners,'selected_epochs':original_row['selected_epochs'],
                'original_pool_AUC':original_row['pool_AUC'],'original_uniform_graph_BCE':original_row['uniform_graph_BCE'],
                'pool':pooled,'native_members':member})
            losses[(seed,name)]=per_graph
        allocation_output.mkdir(parents=True);per_example_path=allocation_output/'PER_EXAMPLE.jsonl';example_rows=0
        with per_example_path.open('x') as example_file:
            for row in rows:
                row['paired_pooled_loss']={}
                for reference in ('I4','residual_only_private_BN'):
                    candidate,baseline=losses[(row['seed'],row['bank'])],losses[(row['seed'],reference)]
                    row['paired_pooled_loss'][reference]=paired(candidate,baseline,ids,labels)
                    for ident,y,a,b in zip(ids,labels,candidate,baseline):
                        example_file.write(json.dumps({'seed':row['seed'],'bank':row['bank'],'reference':reference,
                            'graph_id':ident,'label':y,'candidate_BCE':a,'reference_BCE':b,'signed_BCE_difference':a-b},
                            separators=(',',':'),allow_nan=False)+'\n');example_rows+=1
        require(example_rows==9966,'Exact33 banks x2 references x151 graph contributions required')
        summaries=[]
        for name in plan['banks']:
            selected=sorted((r for r in rows if r['bank']==name),key=lambda r:r['seed']);require([r['seed'] for r in selected]==SEEDS,'All paired seeds required')
            fields={'pool_BCE':[r['pool']['all']['mean_BCE'] for r in selected],'pool_Brier':[r['pool']['all']['mean_Brier'] for r in selected]}
            for cls in ('0','1'):
                for key in ('mean_BCE','mean_Brier'):fields['class'+cls+'_'+key]=[r['pool']['by_class'][cls][key] for r in selected]
            summaries.append({'bank':name,'seeds':SEEDS,'summary':{k:{'values':v,'mean':statistics.mean(v),'sample_SD':statistics.stdev(v)} for k,v in fields.items()}})
        result={'schema':'completed-probability-failure-descriptive-v1','status':'complete','rows':rows,'three_seed_summaries':summaries,
            'source':pin(__file__),'manifest':pin(HERE/'MANIFEST.json'),'PLAN':pin(HERE/'PLAN.json'),
            'completed33_authority':plan['completed33_authority'],'verified_loader':plan['verified_loader'],
            'allocation_only_per_example':{'receipt':pin(per_example_path),'rows':example_rows,
                'custody':'Allocation-only scalar graph contributions; do not synchronize locally or commit. Main RESULT contains summaries and this receipt only.'},
            'TRAIN_support_authority':plan['TRAIN_support_authority'],'source_fit_receipts':[r['receipt'] for r in original['training_fit_outcomes']],
            'saved_collection_receipts':records,'observed_prevalence':{'TRAIN':{'positive':480,'negative':730,'positive_fraction':480/1210},
                'VALID':{'positive':130,'negative':21,'positive_fraction':130/151},
                'limit':'Observed scaffold-role label prevalence differs. No pure label-shift assumption, causal explanation of paired excess, prior correction or VALID refit follows.'},
            'interpretation':'Descriptive decomposition of fixed saved predictions on exposed selected VALID. AUC can recover while probability losses worsen; inspect class/threshold/margin and paired signed-loss contributions before attributing blanket overconfidence. This analysis does not identify a causal mechanism.',
            'readout':'Original saved selected pools; I4 exact original FP32 mean of four own exports. No recalibration/readout change or replacement of original scores.',
            'new_models':0,'native_forwards':0,'optimizer_steps':0,'checkpoint_payloads_loaded':0,'TEST_accessed':False,
            'wall_seconds':time.perf_counter()-started,'own_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        lines=['# Completed probability-failure diagnostics','','All11 methods x3 seeds; original serving and report remain unchanged.','',
            'TRAIN positives:480/1210 (39.67%); VALID positives:130/151 (86.09%). This observed difference does not establish pure label shift or explain paired excess BCE.','',
            '| Seed | Bank | Pool BCE | Class0 BCE | Class1 BCE | Wrong graphs | Mean BCE on wrong | Mean true-label probability on wrong |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            p=row['pool'];w=p['all']['wrong'];fmt=lambda v:'unsupported' if v is None else f'{v:.5f}'
            lines.append(f"| {row['seed']} | {row['bank']} | {p['all']['mean_BCE']:.5f} | {p['by_class']['0']['mean_BCE']:.5f} | {p['by_class']['1']['mean_BCE']:.5f} | {w['graphs']} | {fmt(w['mean_BCE'])} | {fmt(w['mean_probability_true_label'])} |")
        lines += ['','RESULT.json retains pooled/native-member raw-logit/true-margin quantiles, per-class BCE/Brier, correct/wrong splits and wrong-prediction true-label probabilities, signed paired-loss summaries and top1/top5 graph concentration including all negative offsets. All graph-level contributions remain in the separately receipted allocation-only PER_EXAMPLE.jsonl; do not sync or commit it. Fractions of net excess can exceed1; no net excess is reported as unsupported concentration.','',
            'No fitted calibrator, prior adjustment, bootstrap, significance test, ECE or new scoring rule is used. No model/checkpoint, current partial fit or TEST payload is read.','']
        (HERE/'REPORT.md').write_text('\n'.join(lines));(HERE/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({'status':'complete','banks':33,'methods':11,'new_models':0,'native_forwards':0,'optimizer_steps':0,'wall_seconds':result['wall_seconds']}));return 0
    except BaseException as error:
        (HERE/'GATE.json').write_text(json.dumps({'status':'failed_no_partial_diagnostics','error':{'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(limit=8)},'source':pin(__file__),'completed33_authority':plan['completed33_authority'],
            'new_models':0,'native_forwards':0,'optimizer_steps':0,'TEST_accessed':False,'wall_seconds':time.perf_counter()-started},indent=2,sort_keys=True)+'\n')
        raise


if __name__=='__main__':raise SystemExit(main())
