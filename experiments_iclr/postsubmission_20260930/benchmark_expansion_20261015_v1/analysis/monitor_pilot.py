"""Finite selected-state analysis of the frozen pilot; never trains or retries."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from selected_molecular import require_allocation, safe_path, binary_metrics, error_summary, pin


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def comparison(root,plan,output):
    import torch
    from ogb.graphproppred import Evaluator
    torch.set_num_threads(2)
    by_id={row['id']:row for row in plan['jobs']};cohorts=[]
    for cohort in plan['cohorts']:
        ids=cohort['native_fit_ids']+cohort['method_fit_ids']
        folders=[safe_path(by_id[key]['output'],root) for key in ids]
        if not all((folder/'RESULT.json').is_file() for folder in folders):continue
        results=[json.loads((folder/'RESULT.json').read_text()) for folder in folders]
        if not all(r['status']=='complete' for r in results):continue
        exports=[torch.load(folder/'selected_VALID.pt',map_location='cpu',weights_only=False) for folder in folders]
        graph_ids,truth=exports[0]['ids'],exports[0]['labels']
        if not all(torch.equal(e['ids'],graph_ids) and torch.allclose(e['labels'],truth,rtol=0,atol=0,equal_nan=True) for e in exports):
            raise ValueError('Cannot compare unmatched official VALID identities/labels')
        evaluator=Evaluator(name=results[0]['dataset'])
        native=torch.stack([e['raw_logits'] for e in exports[:4]])
        single=binary_metrics(torch,evaluator,native[0],truth)
        ordinary=binary_metrics(torch,evaluator,native.mean(0),truth)
        base=binary_metrics(torch,evaluator,exports[4]['raw_logits'],truth)
        arms=[]
        for result,export in zip(results[4:],exports[4:]):
            metric=binary_metrics(torch,evaluator,export['raw_logits'],truth)
            arms.append({'method':result['procedure'],'metrics':metric,
                         'ROC_AUC_delta_pp_vs_base':100*(metric['rocauc']-base['rocauc']),
                         'ROC_AUC_delta_pp_vs_I4':100*(metric['rocauc']-ordinary['rocauc']),
                         'BCE_delta_vs_base':metric['BCE']-base['BCE'],
                         'BCE_delta_vs_I4':metric['BCE']-ordinary['BCE'],
                         'selected_epoch':result['selected_epoch'],'trained_epochs':result['completed_epochs'],
                         'stop_reason':result['stop_reason']})
        cohorts.append({'cohort':cohort['id'],'M1':single,'I4':ordinary,
                        'I4_native_error_overlap':error_summary(torch,native,native.mean(0),truth),
                        'methods':arms,'authorities':[pin(folder/'RESULT.json') for folder in folders],
                        'VALID_graphs':len(graph_ids)})
    grouped={}
    for cohort in cohorts:
        for arm in cohort['methods']:grouped.setdefault(arm['method'],[]).append(arm)
    aggregate=[]
    for method,rows in grouped.items():
        a=[r['ROC_AUC_delta_pp_vs_base'] for r in rows];b=[r['ROC_AUC_delta_pp_vs_I4'] for r in rows]
        aggregate.append({'method':method,'complete_paired_trials':len(rows),
                          'all_ROC_AUC_deltas_pp_vs_base':a,'all_ROC_AUC_deltas_pp_vs_I4':b,
                          'mean_delta_pp_vs_base':statistics.mean(a),'mean_delta_pp_vs_I4':statistics.mean(b),
                          'sample_SD_delta_pp_vs_base':statistics.stdev(a) if len(a)>1 else None,
                          'sample_SD_delta_pp_vs_I4':statistics.stdev(b) if len(b)>1 else None})
    report={'schema':'bounded-molecular-pilot-comparison-v1','scope':'selected VALID exploratory; no heldout or significance claim',
            'planned_paired_trials':3,'complete_paired_trials':len(cohorts),'cohorts':cohorts,'aggregate':aggregate,
            'TEST_accessed':False,'optimizer_steps':0,'partial_results_not_used_for_promotion':len(cohorts)<3}
    write(output/'COMPARISON.json',report)
    lines=['# Frozen molecular pilot comparison','','Selected VALID results. This validation set also selected checkpoints. TEST remains closed.',
           '',f'Complete paired trials: {len(cohorts)}/3. Incomplete trials are not silently omitted from a final claim.','',
           '| Method | Paired trials | Mean delta vs base, ROC-AUC points | Mean delta vs I4, ROC-AUC points |',
           '| --- | ---: | ---: | ---: |']
    for row in aggregate:lines.append(f"| {row['method']} | {row['complete_paired_trials']} | {row['mean_delta_pp_vs_base']:+.4f} | {row['mean_delta_pp_vs_I4']:+.4f} |")
    lines+=['','All per-trial differences, stopping reasons and source receipts are in COMPARISON.json. Three seeds on one scaffold split do not establish significance or transfer.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--plan',required=True)
    parser.add_argument('--maximum-wait-seconds',type=int,default=129600)
    args=parser.parse_args();require_allocation()
    root=Path(args.project_root).resolve();path=safe_path(args.plan,root)
    plan=json.loads(path.read_text())
    if plan['fit_count']!=24 or plan['full_campaign_GO'] is not False:raise ValueError('Only current bounded24-fit pilot')
    output=path.parent/'selected_analysis_v1';output.mkdir(exist_ok=False)
    write(output/'START.json',{'source':pin(Path(__file__)),'plan':pin(path),'maximum_wait_seconds':args.maximum_wait_seconds,'fits':24,'TEST_accessed':False,'training_steps':0})
    start=time.monotonic();seen={};terminal=None
    while time.monotonic()-start<args.maximum_wait_seconds:
        for index,binding in enumerate(plan['jobs']):
            if binding['id'] in seen:continue
            fit=safe_path(binding['output'],root);result=fit/'RESULT.json'
            if not result.exists():continue
            value=json.loads(result.read_text())
            if value['status']!='complete':
                seen[binding['id']]={'status':'fit_failed','fit_result':pin(result)};continue
            destination=output/('fit_'+str(index).zfill(3))
            command=[sys.executable,'-B',str(Path(__file__).with_name('selected_molecular.py')),'--project-root',str(root),'--fit',str(fit),'--output',str(destination)]
            with (output/('fit_'+str(index).zfill(3)+'.log')).open('x') as log:
                child=subprocess.run(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            seen[binding['id']]={'exit_code':child.returncode,'analysis_output':str(destination),'automatic_retries':0}
        report=comparison(root,plan,output)
        write(output/'PROGRESS.json',{'analyzed_or_failed_fits':len(seen),'records':seen,'complete_paired_trials':report['complete_paired_trials'],'elapsed_seconds':time.monotonic()-start})
        if (path.parent/'QUEUE_RESULT.json').exists():
            terminal=json.loads((path.parent/'QUEUE_RESULT.json').read_text())
            break
        if len(seen)==24:break
        time.sleep(30)
    write(output/'END.json',{'status':'complete' if len(seen)==24 else 'queue_terminal_partial' if terminal else 'wait_cap_partial',
                            'analyzed_or_failed_fits':len(seen),'records':seen,'queue_terminal':terminal,
                            'elapsed_seconds':time.monotonic()-start,'TEST_accessed':False,'training_steps':0})

if __name__=='__main__':main()
