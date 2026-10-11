"""Fixed-selected-state CAL/CONFIRM reader. No native imports or deserialization."""
import argparse
import math
from pathlib import Path
import statistics
import time
import traceback
from inference import adopted_report
from contracts import (ARMS, SEEDS, ALTERNATES, require, gate, read, write, scoped, receipt, describe, close_native, closed_owner)


def fusions():
    return [(arm, 'logit_mean' if arm == ARMS[1] else 'probability_mean') for arm in ARMS] + [(arm, 'logit_mean') for arm in ALTERNATES]


def load_logits(np, row, cfg, with_authority=False):
    with np.load(receipt(row['selected_logits']), allow_pickle=False) as a:
        require(set(a.files) == {'node_ids', 'raw_logits', 'raw_probability_pool', 'raw_logit_pool',
                'raw_probability_decision', 'raw_logit_decision', 'member_decision'} and np.array_equal(a['node_ids'], np.arange(41554, dtype=np.int64)), 'Complete selected identity')
        raw = a['raw_logits'].copy()
        authority = {k:a[k].copy() for k in ('raw_probability_decision','raw_logit_decision','member_decision')}
        for name in ('raw_probability_pool','raw_logit_pool'):
            require(a[name].dtype == np.float32 and a[name].shape == (41554,2) and np.isfinite(a[name]).all(), 'Native float32 saved serving pools')
    shape = (4, 41554) if row['arm'] == ARMS[1] else (1 if row['arm'] in (ARMS[3], ARMS[4], ARMS[6]) else 4, 41554, 2)
    require(raw.dtype == np.float32 and raw.shape == shape and np.isfinite(raw).all(), 'Finite native selected logits schema')
    require(all(v.dtype == np.int64 and np.isin(v,(0,1)).all() for v in authority.values())
        and authority['raw_probability_decision'].shape == authority['raw_logit_decision'].shape == (41554,)
        and authority['member_decision'].shape == (len(raw),41554), 'Archived native-device binary decision authority')
    return (raw,authority) if with_authority else raw


def log_predictions(torch, raw, representation, fusion, log_T):
    temperature = log_T.exp()
    require(bool(torch.isfinite(temperature)) and float(temperature.detach()) > 0, 'Finite positive bank-global T')
    scaled = raw / temperature
    if representation == 'BCE1':
        if fusion == 'logit_mean':
            z = scaled.mean(0)
            pooled = torch.stack((torch.nn.functional.logsigmoid(-z), torch.nn.functional.logsigmoid(z)), -1)
        else:
            member = torch.stack((torch.nn.functional.logsigmoid(-scaled), torch.nn.functional.logsigmoid(scaled)), -1)
            pooled = torch.logsumexp(member, 0)-math.log(len(raw))
        members = torch.stack((torch.nn.functional.logsigmoid(-scaled), torch.nn.functional.logsigmoid(scaled)), -1)
    else:
        members = scaled.log_softmax(-1)
        pooled = (torch.logsumexp(members, 0)-math.log(len(raw))) if fusion == 'probability_mean' else scaled.mean(0).log_softmax(-1)
    require(bool(torch.isfinite(pooled).all()) and bool(torch.isfinite(members).all()), 'Finite selected-state readout')
    return pooled, members


def role(np, item, cfg, key):
    with np.load(receipt(item), allow_pickle=False) as a:
        require(set(a.files) == {'ids', 'y'}, 'Separate proper-role truth file only')
        ids, y = a['ids'].copy(), a['y'].copy()
    require(ids.dtype == y.dtype == np.int64 and ids.shape == y.shape == (cfg['role_counts'][key],)
            and len(np.unique(ids)) == len(ids) and np.array_equal(ids, np.sort(ids))
            and ids.min() >= 0 and ids.max() < 41554 and set(np.unique(y)) == {0, 1}, 'Proper-role identity and binary targets')
    with np.load(receipt(cfg['inputs'][key+'_ids']), allow_pickle=False) as a:
        require(set(a.files) == {'ids'} and np.array_equal(a['ids'], ids), 'Truth membership equals admitted label-free role export')
    return ids, y


def calibrate(cfg, admission, custody, output):
    # Custody and every native terminal record precede CAL target access.
    closed_owner(admission['native_owner_receipt'])
    native = close_native(admission['native_terminal_receipt'])
    require(native['custody']['config_sha256'] == custody['config_sha256'] and native['placement_native_initial_bytes_equal'], 'Same frozen procedure and placement custody')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import importlib
    for name,item in cfg['provider_modules'].items():
        require(Path(importlib.import_module(name).__file__).resolve()==receipt(item), 'Actual reader provider module differs: '+name)
    ids, y_array = role(np, cfg['inputs']['cal_truth'], cfg, 'cal')
    y = torch.from_numpy(y_array)
    rows, start = [], time.perf_counter()
    index = {(r['seed'], r['arm']): r for r in native['records']}
    for seed in SEEDS:
        for arm, fusion in fusions():
            record = {'seed': seed, 'arm': arm, 'fusion': fusion, 'status': 'failed_retained', 'updates': 0,
                'parameters': 1, 'role': 'CAL', 'CONFIRM_truth_access': False}
            before = time.perf_counter()
            try:
                bank = index[seed, arm]
                require(bank['status'] == 'selected_finite', 'Native failure retained; no scalar attempt or seed rescue')
                raw = torch.from_numpy(load_logits(np, bank, cfg)[:, ids]).to(torch.float64)
                log_T = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
                optimizer = torch.optim.Adam([log_T], lr=.01, betas=(.9,.999), eps=1e-8, weight_decay=0)
                trace = []
                for update in range(1, 501):
                    optimizer.zero_grad()
                    pooled, _ = log_predictions(torch, raw, bank['representation'], fusion, log_T)
                    objective = -pooled[torch.arange(len(y)), y].mean()
                    require(bool(torch.isfinite(objective)), 'Finite fixed CAL objective')
                    objective.backward(); require(bool(torch.isfinite(log_T.grad)), 'Finite scalar gradient')
                    optimizer.step(); record['updates'] = update
                    if update in (1,100,200,300,400,500): trace.append({'update': update, 'pre_update_CAL_NLL': float(objective.detach())})
                pooled, _ = log_predictions(torch, raw, bank['representation'], fusion, log_T)
                record.update(status='finite_fixed_endpoint', log_T=float(log_T.detach()), temperature=float(log_T.detach().exp()),
                    endpoint='fixed_final_update', trace=trace, selected_logits=bank['selected_logits'])
            except Exception as error:
                record.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), retry=False)
            record['seconds'] = time.perf_counter()-before; rows.append(record)
    require(len(rows) == 90 and len({(r['seed'],r['arm'],r['fusion']) for r in rows}) == 90, 'All90 primary/alternate scalar endpoints terminal')
    value = {'terminal': True, 'scalar_endpoints': 90, 'records': rows, 'all_finite': all(r['status']=='finite_fixed_endpoint' for r in rows),
        'native_terminal_receipt': admission['native_terminal_receipt'], 'custody': custody,
        'CAL_truth_receipt': cfg['inputs']['cal_truth'], 'CONFIRM_truth_access': False, 'comparisons': [],
        'requested_scalar_updates': 45000, 'actual_scalar_updates': sum(r['updates'] for r in rows),
        'seconds': time.perf_counter()-start, 'interpretation_eligible': False}
    write(output/'COMPLETE_CAL.json', value)
    return value


def score(torch, raw, y, representation, fusion, log_T, calibrated, authority):
    # Raw decision authority remains native float32; NLL/Brier use stable float64.
    zero = torch.zeros((), dtype=torch.float64)
    pooled, members = log_predictions(torch, raw.to(torch.float64), representation, fusion, log_T if calibrated else zero)
    if calibrated:
        decision = pooled.argmax(-1)
    else:
        # Original cached logit decision is strict native float32 mean(z)>0.
        # All raw integer decisions were cached on the acquisition device.
        decision = authority['raw_logit_decision' if fusion=='logit_mean' else 'raw_probability_decision']
    correct = decision.eq(y)
    member_correct = authority['member_decision'].eq(y[None])
    coverage = member_correct.any(0); lost = coverage & ~correct; aggregation = ~coverage & correct
    rival = ((raw * (2*y[None]-1) < 0).all(0) if representation == 'BCE1' else ((raw.gather(2,(1-y)[None,:,None].expand(len(raw),-1,1))
             > raw.gather(2,y[None,:,None].expand(len(raw),-1,1))).all(0).squeeze(-1)))
    member_nll = -members.gather(2,y[None,:,None].expand(len(raw),-1,1)).squeeze(-1)
    pooled_nll = -pooled[torch.arange(len(y)), y]
    truth = torch.nn.functional.one_hot(y,2).to(torch.float64)
    pooled_brier = ((pooled.exp()-truth)**2).sum(-1)
    member_brier = ((members.exp()-truth[None])**2).sum(-1)
    counts = {'nodes': len(y), 'pooled_correct': int(correct.sum()), 'member_correct': member_correct.sum(1).tolist(),
        'coverage': int(coverage.sum()), 'pooling_losses': int(lost.sum()), 'aggregation_only_correct': int(aggregation.sum()),
        'strict_wrong_rivals': int(rival.sum()), 'strict_wrong_rivals_corrected': int((rival & correct).sum())}
    require(counts['pooled_correct']==counts['coverage']-counts['pooling_losses']+counts['aggregation_only_correct'], 'Exact binary coverage decomposition')
    quality = {'accuracy_pct':100*counts['pooled_correct']/len(y),'NLL':float(pooled_nll.mean()),'Brier':float(pooled_brier.mean()),
        'member_accuracy_pct':(member_correct.double().mean(1)*100).tolist(), 'mean_member_accuracy_pct':float(member_correct.double().mean()*100),
        'worst_member_accuracy_pct':float(member_correct.double().mean(1).min()*100), 'member_NLL':member_nll.mean(1).tolist(),
        'member_Brier':member_brier.mean(1).tolist(),
        'float64_member_argmax_disagreements_with_native_archived_decisions':int((members.argmax(-1)!=authority['member_decision']).sum()),
        'mean_member_NLL':float(member_nll.mean()),'worst_member_NLL':float(member_nll.mean(1).max()),
        'mean_member_Brier':float(member_brier.mean()),'worst_member_Brier':float(member_brier.mean(1).max())}
    classes = [{'class': k, 'nodes': int((y==k).sum()), 'pooled_correct': int((correct & (y==k)).sum()),
        'coverage':int((coverage & (y==k)).sum()),'pooling_losses':int((lost & (y==k)).sum()),
        'aggregation_only_correct':int((aggregation & (y==k)).sum()),
        'member_accuracy_pct':(member_correct[:,y==k].double().mean(1)*100).tolist(),
        'member_NLL':member_nll[:,y==k].mean(1).tolist(),'member_Brier':member_brier[:,y==k].mean(1).tolist(),'NLL':float(pooled_nll[y==k].mean()),'Brier':float(pooled_brier[y==k].mean())} for k in range(2)]
    return {'quality':quality,'counts':counts,'classes':classes,'P':correct,'V':coverage,'S':rival}


def paired_diagnostics(torch, values, candidate, reference, labels):
    rows=[]; deltas=[]
    for seed in SEEDS:
        a,b=values[seed,candidate],values[seed,reference]
        if a is None or b is None: return {'failed_retained':True,'complete_seed_roster_required':True}
        repair,harm=~b['P'] & a['P'], b['P'] & ~a['P']
        delta=100*(a['counts']['pooled_correct']-b['counts']['pooled_correct'])/len(labels);deltas.append(delta)
        rows.append({'block':seed,'base_seed':10000*seed,'accuracy_pp':delta,'repairs':int(repair.sum()),'harms':int(harm.sum()),
            'coverage_added':int((a['V'] & ~b['V']).sum()),'coverage_removed':int((b['V'] & ~a['V']).sum()),
            'fixed_reference_strict_rivals_repaired':int((b['S'] & a['P']).sum()),
            'new_strict_rivals_created':int((~b['S'] & a['S']).sum()),
            'classes':[{'class':k,'repairs':int((repair & (labels==k)).sum()),'harms':int((harm & (labels==k)).sum())} for k in range(2)]})
    quality_deltas = {metric: [values[s,candidate]['quality'][metric]-values[s,reference]['quality'][metric] for s in SEEDS]
        for metric in ('NLL','Brier','mean_member_accuracy_pct','worst_member_accuracy_pct','mean_member_NLL','worst_member_NLL')}
    return {'quality_deltas':{k:{'seed_deltas':v,'mean':statistics.mean(v),'SD':statistics.stdev(v)} for k,v in quality_deltas.items()},
        'paired_mean_accuracy_pp':statistics.mean(deltas),'paired_seed_deltas_pp':deltas,
        'paired_seed_SD_pp':statistics.stdev(deltas),'min_pp':min(deltas),'max_pp':max(deltas),'per_seed':rows,
        'uncertainty_scope':'Ten optimizer repeats of the same graph/split, not ten independent tasks.',
        'inference_or_progression_pass':'See separate root_adopted_paired_inference; these are descriptive paired diagnostics'}


def confirm(cfg, admission, custody, output):
    closed_owner(admission['native_owner_receipt'])
    native = close_native(admission['native_terminal_receipt'])
    closed_owner(admission['calibration_owner_receipt'])
    calibration = read(receipt(admission['calibration_terminal_receipt']))
    require(calibration['terminal'] is True and calibration['scalar_endpoints']==90 and len(calibration['records'])==90
        and calibration['native_terminal_receipt']==admission['native_terminal_receipt']
        and calibration['custody']['config_sha256']==custody['config_sha256'], 'All native and90 CAL endpoints close before CONFIRM truth')
    require(admission.get('one_complete_CONFIRM_release') is True and admission.get('no_prior_CONFIRM_score_in_scope') is True,
            'Root must authorize one complete heldout scoring release')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    import importlib
    for name,item in cfg['provider_modules'].items():
        require(Path(importlib.import_module(name).__file__).resolve()==receipt(item), 'Actual reader provider module differs: '+name)
    ids,y_array=role(np,cfg['inputs']['confirm_truth'],cfg,'confirm');y=torch.from_numpy(y_array)
    endpoints={(r['seed'],r['arm'],r['fusion']):r for r in calibration['records']}
    require(set(endpoints)=={(s,a,f) for s in SEEDS for a,f in fusions()}, 'Exact90 terminal endpoint roster')
    rows, scores, banks = [], {'raw':{},'calibrated':{}}, {(r['seed'],r['arm']):r for r in native['records']}
    for seed in SEEDS:
        for arm,fusion in fusions():
            name=arm+'__'+fusion; row={'seed':seed,'arm':arm,'fusion':fusion,'same_selected_native_state':True,
                'acquisition_costs':[r.get('costs') for r in banks[seed,arm]['fits']],
                'calibration_endpoint':endpoints[seed,arm,fusion]}
            bank,endpoint=banks[seed,arm],endpoints[seed,arm,fusion]
            if bank['status'] != 'selected_finite':
                row['raw']=row['calibrated']={'failed_retained':True}
                scores['raw'][seed,name]=scores['calibrated'][seed,name]=None
            else:
                raw_array, cached = load_logits(np,bank,cfg,True)
                raw=torch.from_numpy(raw_array[:,ids])
                authority={k:torch.from_numpy(v[:,ids] if v.ndim==2 else v[ids]) for k,v in cached.items()}
                for kind in ('raw','calibrated'):
                    if kind=='calibrated' and endpoint['status']!='finite_fixed_endpoint':
                        value=None;row[kind]={'failed_retained':True,'endpoint_failure_preserved':True}
                    else:
                        value=score(torch,raw,y,bank['representation'],fusion,torch.tensor(endpoint.get('log_T',0.),dtype=torch.float64),kind=='calibrated',authority)
                        row[kind]={k:value[k] for k in ('quality','counts','classes')}
                    scores[kind][seed,name]=value
            rows.append(row)
    candidate=ARMS[0]+'__probability_mean'
    references=[a+'__'+('logit_mean' if a==ARMS[1] else 'probability_mean') for a in ARMS[1:]]
    from scipy.stats import t as t_distribution
    inference = adopted_report(scores['raw'], t_distribution, ARMS, SEEDS, len(y))
    contrasts={kind:{ref:paired_diagnostics(torch,values,candidate,ref,y) for ref in references} for kind,values in scores.items()}
    fixed_state={arm: {kind:paired_diagnostics(torch,values,arm+'__probability_mean',arm+'__logit_mean',y)
        for kind,values in scores.items()} for arm in ALTERNATES}
    confidence={}
    for ref in (references[0], references[2], references[3], references[4]):
        values=[(scores['calibrated'][s,candidate],scores['calibrated'][s,ref]) for s in SEEDS]
        if any(a is None or b is None for a,b in values):confidence[ref]={'failed_retained':True,'pass':False}
        else:
            delta=[a['quality']['NLL']-b['quality']['NLL'] for a,b in values]
            confidence[ref]={'NLL_harm_per_seed':delta,'mean':statistics.mean(delta),'max':max(delta),
                'pass':statistics.mean(delta)<=.02 and max(delta)<=.05,'secondary_does_not_veto_raw_accuracy':True}
    # Rehash original native/checkpoint bytes and stage/input pins before output.
    close_native(admission['native_terminal_receipt']); receipt(admission['calibration_terminal_receipt'])
    for item in cfg['inputs'].values(): receipt(item)
    value={'terminal':True,'complete_CONFIRM_release':True,'rows':rows,'contrasts':contrasts,'fixed_state_pooling':fixed_state,
        'separate_secondary_confidence':confidence,'native_terminal_receipt':admission['native_terminal_receipt'],
        'calibration_terminal_receipt':admission['calibration_terminal_receipt'],'custody':custody,
        'root_adopted_paired_inference':inference,
        'statistical_operator_source_binding_required':True,'manuscript_acceptance_verdict':None,
        'no_same_state_placement_cause_inferred':True,'original_whole_recipe_comparison_separate':True,
        'CONFIRM_truth_access':True,'TEST_in_model_or_CAL_STAGE':False,'no_new_native_fits_or_forwards':True,
        'no_novelty_claim':True,'scope':'One Penn94 graph/fixed split with ten paired seed blocks; failures/costs retained. Alternate pooling uses identical selected logits.'}
    write(output/'COMPLETE_CONFIRM.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('calibrate','confirm'),required=True);args=parser.parse_args()
    cfg,admission,custody=gate(args.config,args.admission,args.stage)
    output=scoped(args.output);require(not output.exists(),'Fresh one-stage output only');output.mkdir()
    value=calibrate(cfg,admission,custody,output) if args.stage=='calibrate' else confirm(cfg,admission,custody,output)
    write(output/'STAGE_END.json',{'terminal':True,'success':True,'stage':args.stage,'CONFIRM_truth_access':args.stage=='confirm'})
    print('{"terminal":true,"complete_stage":true,"arrays_exported":false}')


if __name__=='__main__':main()
