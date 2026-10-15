"""All18 selected VALID banks; CPU saved tensors only, no model or TEST."""
import json
from pathlib import Path
import resource
import statistics
import time
import traceback

import study as s


def run(root,spec,rows,cohorts,fits,admission,start,end_pin,outcomes_pin):
    h = s.helpers(); started = time.perf_counter()
    collected,output = h.safe(root,admission['collection_output']),h.safe(root,admission['analysis_output'])
    fitted = h.safe(root,spec['study_output'])
    h.require(not output.exists() and output != collected and not output.is_relative_to(collected) and
        not collected.is_relative_to(output) and not output.is_relative_to(fitted) and not fitted.is_relative_to(output),
        'Fresh analysis output disjoint from collection and fit trees required')
    output.mkdir(parents=True); records = []
    try:
        end_path = collected/'END.json';end = json.loads(end_path.read_text())
        shared = [r for r in rows if r['kind'] != 'native'];s.collection_gate(end,shared)
        h.require(end['admission'] == start['admission'] and end['manifest'] == start['manifest'] and
            end['terminal_END'] == end_pin and end['terminal_OUTCOMES'] == outcomes_pin,'Collection owner authority differs')
        collections = {}
        for row,outcome in zip(shared,end['outcomes']):
            p = h.verified(root,outcome['RESULT']);h.require(p == collected/row['token']/'RESULT.json','Collection path differs')
            v = json.loads(p.read_text()); fit = fits[row['index']]
            h.require(v['status'] in ('complete','complete_with_sanity_warning') and v['job_id'] == row['id'] and
                v['job_hash'] == row['job_hash'] and v['role'] == 'VALID' and v['TEST_accessed'] is False and
                v['selected_epoch'] == fit['selected_epoch'] and v['selected_checkpoint'] == fit['selected_checkpoint'] and
                v['fit_RESULT'] == h.pin(h.safe(root,spec['study_output'])/'fits'/row['token']/'RESULT.json') and
                v['terminal_END'] == end_pin and v['terminal_OUTCOMES'] == outcomes_pin and v['admission'] == start['admission'] and
                v['saved_VALID_export'] == fit['selected_VALID_export'] and v['members'] == 4 and v['VALID_nodes'] == 3943 and
                v['VALID_class_counts'] == [72,1919,1952] and v['actual_work']['TRAIN_callbacks'] ==
                v['actual_work']['optimizer_steps'] == 0 and v['actual_work']['native_forwards_completed'] == 4 and
                not v.get('close_error') and v['source']['sha256'] == h.pin(Path(__file__).with_name('collect.py'))['sha256'],
                'Exact complete own-selected collection and zero TRAIN required')
            h.verified(root,v['raw_tensor_receipt']);collections[row['id']] = v
            records.append({'id':row['id'],'RESULT':outcome['RESULT'],**{k:v.get(k) for k in
                ('status','actual_work','wall_seconds','resources','failure','close_error','sanity')}})
        # Every terminal and12collection metadata gate precedes tensor imports/quality.
        import torch
        torch.set_num_threads(2)
        aggregator = s.load(s.E/'runtime/aggregate_selected.py','_pubmed_original_saved_I4_aggregator')
        exports = {}
        for row,fit in zip(rows,fits):
            saved = torch.load(h.verified(root,fit['selected_VALID_export']),map_location='cpu',weights_only=False)
            s.accept_export(torch,saved,fit,row['job'],spec['provider']);exports[row['id']] = saved
        banks,prediction,aggregation_receipts,warnings = [],{},{},[]

        def add(seed,bank,pool,members,labels,native_CE,authorities,epochs):
            metric = s.metrics(pool,labels); member = [s.metrics(z,labels) for z in members]
            record = {'seed':seed,'bank':bank,'pool':metric,'members':member,'source_JOB_ids':authorities,'selected_epochs':epochs,
                'pool_accuracy':metric['accuracy'],'pool_NLL':metric['NLL'],'macro_accuracy':metric['macro_accuracy'],
                'macro_NLL':metric['macro_NLL'],'class0_accuracy':metric['per_class'][0]['accuracy'],
                'class0_NLL':metric['per_class'][0]['NLL'],'saved_native_CE':native_CE,
                'mean_member_accuracy':statistics.mean(r['accuracy'] for r in member),
                'worst_member_accuracy':min(r['accuracy'] for r in member),
                'mean_member_NLL':statistics.mean(r['NLL'] for r in member),
                'worst_member_NLL':max(r['NLL'] for r in member),**s.errors(members,pool,labels)}
            record['pool_minus_mean_member_accuracy_pp'] = 100*(record['pool_accuracy']-record['mean_member_accuracy'])
            banks.append(record);prediction[(seed,bank)] = {'pool':pool,'members':members}

        for seed,cohort in zip(s.SEEDS,cohorts):
            native_rows = [r for r in rows if r['cohort_seed'] == seed and r['kind'] == 'native']
            native_rows.sort(key=lambda r:r['job']['body']); body0 = native_rows[0]
            first = exports[body0['id']]; labels = first['labels'].tolist()
            add(seed,'M1',first['raw_logits'].tolist(),[first['raw_logits'].tolist()],labels,
                first['validation']['loss'],[body0['id']],[first['selected_epoch']])
            paths = [fits[r['index']]['selected_VALID_export']['path'] for r in native_rows]
            before = time.perf_counter(); aggregate = aggregator.aggregate(paths,root,output/('I4_seed'+str(seed)))
            bank = torch.load(h.verified(root,aggregate['bank_receipt']),map_location='cpu',weights_only=False)
            h.require(torch.equal(bank['ids'],first['ids']) and torch.equal(bank['labels'],first['labels']) and
                bank['M1_exact_member0_job_hash'] == body0['job_hash'] and bank['extra_training_fits'] == 0 and
                bank['pooled_epoch_reselection'] is False,'Genuine own-selected I4/body0 reuse differs')
            aggregation_receipts[str(seed)] = {'RESULT':h.pin(output/('I4_seed'+str(seed))/'RESULT.json'),
                'bank':aggregate['bank_receipt'],'wall_seconds':time.perf_counter()-before,'extra_training_fits':0,'new_native_forwards':0}
            add(seed,'I4',bank['raw_logits'].tolist(),bank['member_raw_logits'].tolist(),labels,bank['loss'],
                [r['id'] for r in native_rows],[exports[r['id']]['selected_epoch'] for r in native_rows])
            for row in [r for r in shared if r['cohort_seed'] == seed]:
                saved = exports[row['id']];v = collections[row['id']]
                raw = torch.load(h.verified(root,v['raw_tensor_receipt']),map_location='cpu',weights_only=False)
                h.require(raw['schema'] == 'pubmed-cheb-selected-VALID-member-logits-v1' and raw['role'] == 'VALID' and
                    raw['TEST_accessed'] is False and raw['job_id'] == row['id'] and raw['job_hash'] == row['job_hash'] and
                    raw['selected_epoch'] == saved['selected_epoch'] and raw['selected_checkpoint'] == saved['selected_checkpoint'] and
                    raw['saved_VALID_export'] == fits[row['index']]['selected_VALID_export'] and
                    torch.equal(raw['ids'],saved['ids']) and torch.equal(raw['labels'],saved['labels']) and
                    tuple(raw['member_raw_logits'].shape) == (4,3943,3) and torch.isfinite(raw['member_raw_logits']).all().item(),
                    'Aligned complete selected route tensors required')
                add(seed,row['kind'],saved['raw_logits'].tolist(),raw['member_raw_logits'].tolist(),labels,
                    saved['validation']['loss'],[row['id']],[saved['selected_epoch']])
                if v['status'] != 'complete':warnings.append({'seed':seed,'bank':row['kind'],'sanity':v['sanity']})
            for row in [r for r in rows if r['cohort_seed'] == seed]:
                saved = exports[row['id']]
                counted = s.metrics(saved['raw_logits'].tolist(),saved['labels'].tolist())
                h.require(abs(counted['accuracy']-saved['validation']['metrics']['accuracy']) < 1e-7,
                    'Saved native FP32 accuracy differs from exact recounted decisions')
        h.require(len(banks) == 18 and len({(r['seed'],r['bank']) for r in banks}) == 18,'All18 banks required')
        contrasts = []
        pairs = [('G0',r) for r in ('M1','I4')]
        pairs += [(c,r) for c in ('uniform','structured','permuted') for r in ('M1','I4','G0')]
        pairs += [('structured','uniform'),('structured','permuted'),('permuted','uniform')]
        for seed in s.SEEDS:
            labels = exports[next(r['id'] for r in rows if r['cohort_seed'] == seed)]['labels'].tolist()
            lookup = {r['bank']:r for r in banks if r['seed'] == seed}
            for candidate,reference in pairs:
                c,r = lookup[candidate],lookup[reference]
                value = {'seed':seed,'comparison':candidate+'_minus_'+reference,'candidate':candidate,'reference':reference,
                    'delta_accuracy_pp':100*(c['pool_accuracy']-r['pool_accuracy']),'delta_NLL':c['pool_NLL']-r['pool_NLL'],
                    'delta_saved_native_CE':c['saved_native_CE']-r['saved_native_CE'],
                    'delta_macro_accuracy_pp':100*(c['macro_accuracy']-r['macro_accuracy']),
                    'delta_macro_NLL':c['macro_NLL']-r['macro_NLL'],
                    'delta_class0_accuracy_pp':100*(c['class0_accuracy']-r['class0_accuracy']),
                    'delta_class0_NLL':c['class0_NLL']-r['class0_NLL'],
                    'delta_mean_member_accuracy_pp':100*(c['mean_member_accuracy']-r['mean_member_accuracy']),
                    'delta_worst_member_accuracy_pp':100*(c['worst_member_accuracy']-r['worst_member_accuracy']),
                    'delta_pool_minus_mean_member_accuracy_pp':c['pool_minus_mean_member_accuracy_pp']-r['pool_minus_mean_member_accuracy_pp'],
                    'delta_no_correct_member':c['no_correct_member']-r['no_correct_member'],
                    'delta_strict_common_false_rival':c['strict_common_false_rival']-r['strict_common_false_rival'],
                    **s.repairs(prediction[(seed,reference)],prediction[(seed,candidate)],labels)}
                h.require(abs(value['delta_accuracy_pp']-100*(value['repairs']-value['harms'])/3943) < 1e-10,'Repairs/harms and accuracy disagree')
                value['per_class_signed'] = [{'class':i,'support':c['pool']['per_class'][i]['support'],
                    'delta_accuracy_pp':100*(c['pool']['per_class'][i]['accuracy']-r['pool']['per_class'][i]['accuracy']),
                    'delta_NLL':c['pool']['per_class'][i]['NLL']-r['pool']['per_class'][i]['NLL']} for i in range(3)]
                contrasts.append(value)
        numeric = ['delta_accuracy_pp','delta_NLL','delta_saved_native_CE','delta_macro_accuracy_pp','delta_macro_NLL',
            'delta_class0_accuracy_pp','delta_class0_NLL','delta_mean_member_accuracy_pp','delta_worst_member_accuracy_pp',
            'delta_pool_minus_mean_member_accuracy_pp','delta_no_correct_member','delta_strict_common_false_rival','repairs','harms',
            'new_correct_member_alternatives','lost_correct_member_alternatives','new_alternative_served','new_alternative_still_lost']
        numeric += ['reference_common_rival_acquires_correct_member','reference_common_rival_served_repairs',
            'new_strict_common_false_rivals','cleared_strict_common_false_rivals']
        class_summaries,class_contrast_summaries = [],[]
        for cls,support in enumerate(spec['provider']['VALID_class_counts']):
            class_rows = [dict(r['pool']['per_class'][cls],seed=r['seed'],bank=r['bank']) for r in banks]
            class_summaries += [dict(v,**{'class':cls,'support':support}) for v in
                s.summarize(class_rows,'bank',['accuracy','NLL'])]
            class_contrasts = [dict(r['per_class_signed'][cls],seed=r['seed'],comparison=r['comparison'],
                repairs=r['per_class'][cls]['repairs'],harms=r['per_class'][cls]['harms']) for r in contrasts]
            class_contrast_summaries += [dict(v,**{'class':cls,'support':support}) for v in
                s.summarize(class_contrasts,'comparison',['delta_accuracy_pp','delta_NLL','repairs','harms'])]
        training = [{'id':r['id'],'cohort_seed':r['cohort_seed'],'kind':r['kind'],'RESULT':h.pin(
            h.safe(root,spec['study_output'])/'fits'/r['token']/'RESULT.json'),**{k:f.get(k) for k in
            ('seed','body','selected_epoch','completed_epochs','stop_reason','convergence_established','invocation_wall_seconds',
            'optimizer_calls','optimizer_calls_this_invocation','resources','selected_checkpoint','selected_VALID_export','failure','close_error')}}
            for r,f in zip(rows,fits)]
        result = {'schema':'pubmed-cheb-selected-VALID-18-bank-analysis-v1','status':'complete_with_warnings' if warnings else 'complete',
            'rows':banks,'summaries':s.summarize(banks,'bank',['pool_accuracy','pool_NLL','macro_accuracy','macro_NLL','class0_accuracy',
                'class0_NLL','mean_member_accuracy','worst_member_accuracy','mean_member_NLL','worst_member_NLL',
                'pool_minus_mean_member_accuracy_pp','no_correct_member','pool_lost_correct_alternative','strict_common_false_rival']),
            'contrasts':contrasts,'contrast_summaries':s.summarize(contrasts,'comparison',numeric),
            'per_class_summaries':class_summaries,'per_class_contrast_summaries':class_contrast_summaries,
            'training_outcomes':training,'collection_outcomes':records,'I4_aggregation':aggregation_receipts,'warnings':warnings,
            'terminal_END':end_pin,'terminal_OUTCOMES':outcomes_pin,'collection_END':h.pin(end_path),**start,
            'VALID_class_counts':[72,1919,1952],'minority_limit':'class0 only72/3943=1.83%; per-class and macro diagnostics do not change native selection.',
            'serving_authority':'Each saved selected mean raw-logit export; route reconstruction is a disclosed sanity check only.',
            'proper_loss':'Uniform-per-node float64 stable CE/NLL for all banks, plus signed original saved FP32 native/aggregate CE; no clipping or calibration.',
            'scope':'Three paired optimizer seeds on one encountered transductive graph/source split. Nodes/routes/classes are not independent graph replications. No TEST/generalization/novelty claim.',
            'M1_extra_training_fits':0,'unique_training_fits':24,'banks':18,'new_analysis_models':0,'new_analysis_native_forwards':0,
            'new_analysis_optimizer_steps':0,'TEST_accessed':False,'full_campaign_GO':False,
            'cost_rule':'Retain24 unique fits once,12 model-restoring collection costs,3 saved I4 aggregations and reader cost separately; do not add concurrent peaks/walls.',
            'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        h.write(output/'RESULT.json',result)
        lines = ['# PubMed-Cheb complete selected VALID comparison','','VALID class counts: **72 / 1919 / 1952**. Class0 is only1.83%; native selection remains whole-VALID accuracy.',
            '',result['scope'],'','| Seed | Bank | Pool accuracy % | NLL | Macro accuracy % | Class0 accuracy % | Mean / worst member accuracy % |',
            '| --- | --- | ---: | ---: | ---: | ---: | --- |']
        for r in banks:lines.append(f"| {r['seed']} | {r['bank']} | {100*r['pool_accuracy']:.4f} | {r['pool_NLL']:.5f} | {100*r['macro_accuracy']:.4f} | {100*r['class0_accuracy']:.4f} | {100*r['mean_member_accuracy']:.4f} / {100*r['worst_member_accuracy']:.4f} |")
        lines += ['','Signed per-seed and three-seed mean/sample-SD contrasts, class-specific accuracy/NLL/repairs/harms, common-rival witnesses, ties, own-selected epochs and all costs are retained in RESULT.json. No partial family, new pool, calibration or selector change is used.','']
        (output/'REPORT.md').write_text('\n'.join(lines));return 0
    except BaseException as error:
        h.write(output/'GATE.json',{'status':'blocked_no_partial_summary','error':{'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(limit=8)},'collection_records_read':records,'TEST_accessed':False,
            'wall_seconds':time.perf_counter()-started})
        raise
