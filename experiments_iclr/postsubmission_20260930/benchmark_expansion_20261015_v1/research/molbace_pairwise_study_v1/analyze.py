"""Complete selected-VALID comparison; CPU saved predictions, no models."""
import json
from pathlib import Path
import resource
import statistics
import time
import traceback
import study as s


def run(root,spec,jobs,fits,admission,start,end_pin,out_pin):
    h=s.helpers();started=time.perf_counter();collection_records=[]
    study=h.safe(root,spec['study_output']);collected=h.safe(root,admission['collection_output']);output=h.safe(root,admission['analysis_output'])
    h.require(not output.exists() and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in (study,collected)),
        'Fresh analysis output separate from fit and collection trees required')
    output.mkdir(parents=True)
    try:
        shared=[r for r in jobs if r['kind']!='native_common'];end_path=collected/'END.json';end=json.loads(end_path.read_text())
        s.collection_gate(end,shared)
        h.require(end['admission']==start['admission'] and end['manifest']==start['manifest'] and
            end['terminal_END']==end_pin and end['terminal_OUTCOMES']==out_pin,'Collection owner authority differs')
        collections={};all_fits={r['id']:f for r,f in zip(jobs,fits)}
        for row,outcome in zip(shared,end['outcomes']):
            path=h.verified(root,outcome['RESULT']);h.require(path==collected/row['token']/'RESULT.json','Collector path differs')
            value=json.loads(path.read_text());s.collection(value,all_fits[row['id']],h.pin(study/'fits'/row['token']/'RESULT.json'))
            h.require(value['analysis_source']['sha256']==h.pin(s.E/'analysis/selected_molecular.py')['sha256'],'Unchanged collector source required')
            h.verified(root,value['raw_tensor_receipt']);collections[row['id']]=value
            collection_records.append({'id':row['id'],'origin':'new','RESULT':outcome['RESULT'],**{k:value.get(k) for k in
                ('status','actual_work','wall_seconds','own_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','failure','close_error','sanity_against_saved_export')}})
        paid=json.loads((s.HERE/'BASELINE.json').read_text());authority_path=h.verified(root,paid['completed33']);authority=json.loads(authority_path.read_text())
        h.require(authority['schema']=='residual-only-selected-VALID-33-bank-addition-v1' and authority['status']in ('complete','complete_with_warnings') and
            authority['TEST_accessed']is False and len(authority['rows'])==33 and len(authority['training_fit_outcomes'])==39,
            'Completed paid33-bank/39-fit authority required')
        old_cost={r['job_id']:r for r in authority['training_fit_outcomes']};reused=[]
        for row in paid['fits']:
            path=h.verified(root,row['RESULT']);value=json.loads(path.read_text())
            h.require(value['job_id']==row['id'] and value['status']=='complete' and value['TEST_accessed']is False and
                value['selected_VALID_export']==row['export'] and value['selected_checkpoint']==row['checkpoint'] and
                old_cost[row['id']]['receipt']==row['RESULT'] and not value.get('failure') and not value.get('close_error'),
                'Exact paid fit/export/checkpoint authority differs')
            h.verified(root,row['export']);all_fits[row['id']]=value
            reused.append(old_cost[row['id']])
            if 'collection'in row:
                cp=h.verified(root,row['collection']);cv=json.loads(cp.read_text());s.collection(cv,value,row['RESULT'])
                h.verified(root,cv['raw_tensor_receipt']);collections[row['id']]=cv
                collection_records.append({'id':row['id'],'origin':'reused','RESULT':row['collection'],**{k:cv.get(k) for k in
                    ('status','actual_work','wall_seconds','own_peak_rss_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','failure','close_error','sanity_against_saved_export')}})
        h.require(len(reused)==18 and len(collections)==15,'All18 paid fits and6 paid shared collections required')
        # All21 terminal and all9 new +6 reused collector metadata gates have
        # passed before any tensor import, metric or score-bearing family output.
        import torch
        from ogb.graphproppred import Evaluator
        torch.set_num_threads(2)
        metric=s.load(s.E/'analysis/selected_molecular.py','_pair_unchanged_binary_metrics')
        diagnostic=s.diagnostics();aggregate=s.load(s.E/'runtime/aggregate_selected.py','_pair_original_I4_aggregator')
        evaluator=Evaluator(name='ogbg-molbace');exports={};reference_ids=reference_labels=None
        for ident,fit in all_fits.items():
            value=torch.load(h.verified(root,fit['selected_VALID_export']),map_location='cpu',weights_only=False);s.accept_export(torch,value,fit)
            if reference_ids is None:reference_ids,reference_labels=value['ids'],value['labels']
            h.require(torch.equal(value['ids'],reference_ids) and torch.equal(value['labels'],reference_labels),'Exact original VALID order/labels differ')
            checked=metric.binary_metrics(torch,evaluator,value['raw_logits'],value['labels'])
            h.require(abs(checked['rocauc']-fit['selected_VALID']['metrics']['rocauc'])<1e-12,'Saved selected AUC differs from own fit authority')
            exports[ident]=value
        labels=reference_labels[:,0].tolist();banks=[];predictions={};warnings=[];aggregations=[]
        for ident,value in collections.items():
            if value['status']!='complete':warnings.append({'id':ident,'sanity':value['sanity_against_saved_export']})

        def bank(seed,name,owners,pool,members,native_CE,origin):
            member_metrics=[metric.binary_metrics(torch,evaluator,z,reference_labels) for z in members]
            pooled=metric.binary_metrics(torch,evaluator,pool,reference_labels)
            arrays=[z[:,0].double().tolist() for z in members];served=pool[:,0].double().tolist()
            pair,margin,manual=diagnostic.pair_diagnostics(arrays,served,labels)
            h.require(pair['pairs']==2730 and abs(pooled['rocauc']-statistics.mean(map(diagnostic.credit,margin)))<1e-12 and
                all(abs(a-v['rocauc'])<1e-12 for a,v in zip(manual,member_metrics)),'Official AUC and exact half-tie pair credit differ')
            row={'seed':seed,'bank':name,'origin':origin,'source_job_ids':owners,'selected_epochs':[all_fits[i]['selected_epoch'] for i in owners],
                'members':member_metrics,'pool_metrics':pooled,'pool_AUC':pooled['rocauc'],'uniform_graph_BCE':pooled['BCE'],
                'saved_native_VALID_BCE':native_CE,'mean_member_AUC':statistics.mean(v['rocauc'] for v in member_metrics),
                'worst_member_AUC':min(v['rocauc'] for v in member_metrics),'mean_member_BCE':statistics.mean(v['BCE'] for v in member_metrics),
                'worst_member_BCE':max(v['BCE'] for v in member_metrics),**pair}
            row['pool_minus_mean_member_AUC_pp']=100*(row['pool_AUC']-row['mean_member_AUC'])
            row['absolute_common_reversal_rate']=row['all_members_reversed']/2730
            banks.append(row);predictions[(seed,name)]={'members':list(zip(*(diagnostic.margins(z,labels) for z in arrays))), 'pool':margin}
            return row

        def native_bank(seed,name,owners,origin):
            before=time.perf_counter();folder=output/(name+'_seed'+str(seed))
            value=aggregate.aggregate([all_fits[i]['selected_VALID_export']['path'] for i in owners],root,folder)
            raw=torch.load(h.verified(root,value['bank_receipt']),map_location='cpu',weights_only=False)
            h.require(raw['M1_exact_member0_job_hash']==all_fits[owners[0]]['job_hash'] and raw['extra_training_fits']==0 and
                raw['pooled_epoch_reselection']is False and torch.equal(raw['ids'],reference_ids) and torch.equal(raw['labels'],reference_labels),
                'Genuine four-body own-selected I4 authority differs')
            aggregations.append({'bank':name,'seed':seed,'origin':origin,'RESULT':h.pin(folder/'RESULT.json'),'bank_receipt':value['bank_receipt'],
                'wall_seconds':time.perf_counter()-before,'new_training_fits':0,'new_native_forwards':0})
            return bank(seed,name,owners,raw['raw_logits'],raw['member_raw_logits'],raw['loss'],origin)

        def shared_bank(seed,name,ident,origin):
            saved=exports[ident];cv=collections[ident]
            raw=torch.load(h.verified(root,cv['raw_tensor_receipt']),map_location='cpu',weights_only=False)
            h.require(raw['schema']=='own-selected-molecular-route-VALID-tensors-v1' and raw['role']=='VALID' and raw['TEST_accessed']is False and
                raw['job_id']==ident and raw['job_hash']==all_fits[ident]['job_hash'] and raw['selected_epoch']==saved['selected_epoch'] and
                raw['selected_checkpoint']==saved['selected_checkpoint'] and torch.equal(raw['ids'],reference_ids) and torch.equal(raw['labels'],reference_labels) and
                tuple(raw['member_raw_logits'].shape)==(4,151,1) and torch.isfinite(raw['member_raw_logits']).all().item(),
                'Complete own-selected route export differs')
            h.require(torch.equal(raw['member_raw_logits'].mean(0),raw['pooled_raw_logits']),'Collector reconstructed pool differs from its route mean')
            row=bank(seed,name,[ident],saved['raw_logits'],raw['member_raw_logits'],saved['validation']['loss'],origin)
            row['reconstruction']={'sanity':cv['sanity_against_saved_export'],
                'maximum_absolute_difference':float((raw['pooled_raw_logits'].double()-saved['raw_logits'].double()).abs().max())}
            return row

        for seed in s.SEEDS:
            for row in [r for r in paid['banks'] if r['seed']==seed]:
                owners=row['owners'];name=row['bank']
                if name=='BCE_I4':current=native_bank(seed,name,owners,'reused')
                elif name=='BCE_M1':
                    z=exports[owners[0]];current=bank(seed,name,owners,z['raw_logits'],z['raw_logits'].unsqueeze(0),z['validation']['loss'],'reused')
                else:current=shared_bank(seed,name,owners[0],'reused')
                old=next(v for v in authority['rows'] if v['seed']==seed and v['bank']==row['prior_bank'])
                h.require(abs(current['pool_AUC']-old['pool_AUC'])<1e-12 and abs(current['uniform_graph_BCE']-old['uniform_graph_BCE'])<1e-12,
                    'Reused bank prediction metrics differ from completed33-bank authority')
            native=sorted((r for r in jobs if r['cohort_seed']==seed and r['kind']=='native_common'),key=lambda r:r['job']['body'])
            owners=[r['id'] for r in native];z=exports[owners[0]]
            bank(seed,'pair_M1',[owners[0]],z['raw_logits'],z['raw_logits'].unsqueeze(0),z['validation']['loss'],'new')
            native_bank(seed,'pair_I4',owners,'new')
            for row in [r for r in shared if r['cohort_seed']==seed]:shared_bank(seed,row['kind'],row['id'],'new')
        h.require(len(banks)==27 and len({(r['seed'],r['bank']) for r in banks})==27,'All27 banks required')
        comparisons=[]
        pairs=[(arm,base) for arm in ('common','structured','permuted') for base in ('BCE_M1','BCE_I4','original_G0','residual_F','pair_M1','pair_I4')]
        pairs += [('structured','common'),('structured','permuted'),('permuted','common'),('pair_M1','BCE_M1'),('pair_I4','BCE_I4')]
        fields=['pool_AUC','uniform_graph_BCE','saved_native_VALID_BCE','mean_member_AUC','worst_member_AUC','mean_member_BCE','worst_member_BCE',
            'pool_minus_mean_member_AUC_pp','absolute_common_reversal_rate','all_members_reversed','pool_reversed_despite_some_member_correct']
        for seed in s.SEEDS:
            lookup={r['bank']:r for r in banks if r['seed']==seed}
            for candidate,reference in pairs:
                c,r=lookup[candidate],lookup[reference];old,new=predictions[(seed,reference)],predictions[(seed,candidate)]
                changes={('delta_'+k+('_pp' if k.endswith('AUC') else '')):(100 if k.endswith('AUC') else 1)*(c[k]-r[k]) for k in fields}
                value={'seed':seed,'comparison':candidate+'_minus_'+reference,'candidate':candidate,'reference':reference,**changes,
                    **diagnostic.compare_pairs(old['pool'],new['pool']),**s.alternatives(old['members'],new['members'],old['pool'],new['pool'])}
                h.require(abs(value['delta_pool_AUC_pp']-value['delta_AUC_pp'])<1e-10 and
                    abs(value['delta_AUC_pp']-100*(value['tie_aware_repair_credit']-value['tie_aware_harm_credit'])/2730)<1e-10,
                    'Served AUC and tie-aware repairs minus harms disagree')
                comparisons.append(value)
        cost_fields=('seed','body','selected_epoch','completed_epochs','stop_reason','convergence_established','invocation_wall_seconds',
            'optimizer_calls','optimizer_calls_this_invocation','optimizer_calls_are_trajectory_cumulative','resources','failure','close_error',
            'selected_checkpoint','selected_VALID_export','source_metadata')
        new_cost=[{'job_id':r['id'],'origin':'new','receipt':h.pin(study/'fits'/r['token']/'RESULT.json'),'status':f['status'],
            **{k:f.get(k) for k in cost_fields}} for r,f in zip(jobs,fits)]
        h.require(len(reused+new_cost)==len({r['job_id'] for r in reused+new_cost})==39,'Exactly39 relevant unique fits; M1/body0 counted once')
        numeric=[k for k,v in comparisons[0].items() if type(v)in (float,int) and k!='seed']
        result={'schema':'molbace-pairwise-complete27-bank-selected-VALID-v1','status':'complete_with_warnings' if warnings else 'complete',
            'rows':banks,'summaries':diagnostic.summarize(banks,'bank',fields),'contrasts':comparisons,
            'contrast_summaries':diagnostic.summarize(comparisons,'comparison',numeric),'frozen_decision':s.decision(banks),
            'training_outcomes':reused+new_cost,'collection_outcomes':collection_records,'I4_aggregations':aggregations,'warnings':warnings,
            'paid33_bank_authority':paid['completed33'],'historical_original_collection':authority['historical_full_original_collection'],
            'terminal_END':end_pin,'terminal_OUTCOMES':out_pin,'collection_END':h.pin(end_path),**start,
            'new_fits':21,'reused_unique_fits':18,'M1_extra_fits':0,'new_collectors':9,'reused_shared_collectors':6,
            'new_analysis_models':0,'new_analysis_native_forwards':0,'new_analysis_optimizer_steps':0,'TEST_accessed':False,'full_campaign_GO':False,
            'serving_authority':'Own-selected saved pools; exact original four-own-export I4 means. Reconstructed shared means are sanity only.',
            'BCE_rule':'Uniform graph FP64 BCE for every bank plus original saved native VALID/FP32 aggregate BCE separately; no clipping/calibration.',
            'cost_rule':'21 new +18 relevant paid fits once;9 new +6 reused shared collectors, original historical collection receipt,6 CPU I4 aggregations and reader cost separate. No summing concurrent walls/peaks.',
            'scope':'Three paired optimizer seeds, one exposed scaffold VALID151 (130 positive/21 negative;2730 dependent pairs). No TEST/generalization, independent-pair replication or novelty claim.',
            'wall_seconds':time.perf_counter()-started,'own_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        lines=['# Pairwise competence: complete selected VALID comparison','',result['scope'],'',
            '| Seed | Bank | Pool AUC % | FP64 BCE | Mean / worst member AUC % | Pool minus mean AUC points |',
            '| --- | --- | ---: | ---: | --- | ---: |']
        for r in banks:lines.append(f"| {r['seed']} | {r['bank']} | {100*r['pool_AUC']:.4f} | {r['uniform_graph_BCE']:.5f} | {100*r['mean_member_AUC']:.4f} / {100*r['worst_member_AUC']:.4f} | {r['pool_minus_mean_member_AUC_pp']:.4f} |")
        lines += ['','All signed paired AUC/BCE/member/repair-harm/tie/alternative contrasts and mean/sample-SD summaries, frozen screens, checkpoint epochs and individual costs are retained in RESULT.json. An AUC gain with worse BCE remains a ranking/probability trade-off. No baseline was retrained or recollected.','']
        (output/'REPORT.md').write_text('\n'.join(lines));h.write(output/'RESULT.json',result)
        return 0
    except BaseException as error:
        h.write(output/'GATE.json',{'status':'blocked_no_partial_summary','TEST_accessed':False,'error':{'type':type(error).__name__,
            'message':str(error),'traceback':traceback.format_exc(limit=8)},'collection_records_read':collection_records,
            'wall_seconds':time.perf_counter()-started})
        raise
