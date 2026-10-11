"""Whole fresh PubMed LM4 CAL/DEV reader; no training modules or TEST access."""
import json
import math
from pathlib import Path
import resource
import statistics
import time
from runtime import (ACCOUNTING,SEEDS,ENDPOINTS,RAW_REFERENCES,STAGE2,HERE,
    checked,require,receipt,write,owner_closed,source_check,providers,guard)
from arithmetic import calibrated,delta,descriptive,logprob,pair,truth

METRICS=('accuracy_pct','calibrated_accuracy_pct','mean_member_accuracy_pct',
    'worst_member_accuracy_pct','raw_NLL','raw_Brier','calibrated_NLL','calibrated_Brier',
    'mean_member_NLL','worst_member_NLL','mean_member_Brier','worst_member_Brier')
REPORT_METRICS=METRICS+('macro_accuracy_pct','calibrated_macro_accuracy_pct',
    'mean_member_macro_accuracy_pct','worst_member_macro_accuracy_pct')
VIEW_FIELDS={'ids','member_log_probs','native_member_argmax','native_pool_argmax',
    'native_pool_probabilities','view_body_ids','native_body_probabilities','native_body_argmax'}


def layout(endpoint):
    if endpoint.startswith('F1_b') or endpoint in ('B1','S64','FS64'):
        return 1,1,[0]
    if endpoint in ('I4[S64]','I4[FS64]','GA4'):
        return 4,4,list(range(4))
    require(endpoint in ('B4','T16','LM4'),'Fixed declared endpoint')
    return 4,1,[0]*4


def endpoint_temperature(ep, archive, cal_truth):
    require(ep['status'] == 'fixed_endpoint_complete' and type(ep['updates']) is int and ep['updates'] == 500
        and ep['endpoint'] == 'latest_tied_minimum_step0_to500'
        and type(ep['selected_step']) is int and 0 <= ep['selected_step'] <= 500 and -5 <= ep['logT'] <= 5
        and math.isfinite(ep['temperature']) and ep['temperature'] == math.exp(ep['logT'])
        and math.isfinite(ep['CAL_minimum_NLL']) and ep['objective_forward_evaluations'] == 1001
        and ep['operator'] == 'mean_m softmax(member_log_probs_m/T)'
        and ep['cal_archive'] == archive and ep['cal_truth'] == cal_truth,
        'Exact per-view-before-mean scalar CAL endpoint required')


def validate_arrays(np, v, V, n):
    require(v['ids'].dtype == np.int64 and v['ids'].shape == (n,)
        and len(np.unique(v['ids'])) == n and v['ids'].min() >= 0 and v['ids'].max() < 19717
        and v['member_log_probs'].dtype == np.float32
        and v['member_log_probs'].shape == (V, n, 3), 'Complete role/view log-probabilities')
    require(v['native_member_argmax'].dtype == v['native_pool_argmax'].dtype == np.int64
        and v['native_member_argmax'].shape == (V, n) and v['native_pool_argmax'].shape == (n,),
        'Native decision shape/type')
    p = v['native_pool_probabilities']
    require(p.dtype == np.float32 and p.shape == (n, 3) and np.isfinite(p).all()
        and (p > 0).all() and np.max(np.abs(p.astype(np.float64).sum(-1) - 1)) <= 1e-6,
        'Native positive FP32 pool distribution')
    require(np.isfinite(v['member_log_probs']).all()
        and (v['native_member_argmax'] >= 0).all() and (v['native_member_argmax'] < 3).all()
        and np.array_equal(v['native_pool_argmax'], v['native_member_argmax'][0] if V == 1 else p.argmax(-1)),
        'Author-native raw-logit single decision; native probability-mean multiview decision')
    vp = np.exp(logprob(np, v['member_log_probs'].astype(np.float64)))
    reconstructed = vp.mean(0)
    require(np.max(np.abs(reconstructed - p)) <= 1e-6, 'Fixed numerical export custody tolerance')
    v['raw_float64_decision_sensitivity_count'] = int((reconstructed.argmax(-1) != v['native_pool_argmax']).sum())
    v['view_logprob_decision_sensitivity_counts'] = [int((v['member_log_probs'][m].argmax(-1)
        != v['native_member_argmax'][m]).sum()) for m in range(V)]
    return vp


def bank_arrays(np,r,role):
    with np.load(checked(r['exports'][role]),allow_pickle=False) as archive:
        require(set(archive.files)==VIEW_FIELDS,'Exact new LM4 eight-field export')
        raw={k:archive[k].copy() for k in archive.files}
    V,B,mapping=layout(r['endpoint_id']);n=1313 if role=='cal' else 1316
    vp=validate_arrays(np,raw,V,n)
    require(raw['view_body_ids'].dtype==np.int64 and raw['view_body_ids'].shape==(V,)
        and np.array_equal(raw['view_body_ids'],np.asarray(mapping,dtype=np.int64)), 'Exact view/body ownership')
    bp,ba=raw['native_body_probabilities'],raw['native_body_argmax']
    require(bp.dtype==np.float32 and bp.shape==(B,n,3) and np.isfinite(bp).all()
        and (bp>0).all() and np.max(np.abs(bp.astype(np.float64).sum(-1)-1))<=1e-6
        and ba.dtype==np.int64 and ba.shape==(B,n) and (ba>=0).all() and (ba<3).all(),
        'Native selected body distribution/decision custody')
    for b in range(B):
        views=[v for v,body in enumerate(mapping) if body==b]
        expected=raw['native_member_argmax'][views[0]] if len(views)==1 else bp[b].argmax(-1)
        require(np.array_equal(ba[b],expected),'Exact author-native one-view body or native multiview body mean')
    reconstructed=np.stack([vp[raw['view_body_ids']==b].mean(0) for b in range(B)])
    require(np.max(np.abs(reconstructed-bp))<=1e-6,'Body-view numerical custody tolerance')
    raw['body_float64_decision_sensitivity_counts']=[int((reconstructed[b].argmax(-1)!=ba[b]).sum()) for b in range(B)]
    raw['body_probability_origin']='native_Torch_FP32_own_selected_body'
    return raw


def unit_summary(np, probabilities, argmax, calibrated_probabilities, y, mask):
    n = int(mask.sum())
    correct = argmax == y[None]
    counts = correct[:, mask].sum(1)
    nll = -np.log(probabilities[np.arange(len(probabilities))[:, None], np.arange(len(y))[None], y[None]])
    cnll = -np.log(calibrated_probabilities[np.arange(len(probabilities))[:, None], np.arange(len(y))[None], y[None]])
    onehot = np.eye(3, dtype=np.float64)[y]
    brier = ((probabilities - onehot[None]) ** 2).sum(-1)
    cbrier = ((calibrated_probabilities - onehot[None]) ** 2).sum(-1)
    return dict(units=len(probabilities), correct=counts.tolist(), accuracy_pct=(100 * counts / n).tolist(),
        mean_accuracy_pct=float(100 * counts.mean() / n), worst_accuracy_pct=float(100 * counts.min() / n),
        raw_NLL=nll[:, mask].mean(1).tolist(), raw_Brier=brier[:, mask].mean(1).tolist(),
        calibrated_accuracy_pct=(100 * (calibrated_probabilities.argmax(-1)[:, mask] == y[None, mask]).sum(1) / n).tolist(),
        calibrated_NLL=cnll[:, mask].mean(1).tolist(), calibrated_Brier=cbrier[:, mask].mean(1).tolist())


def score_bank(np, r, raw, y, T):
    vp = np.exp(logprob(np, raw['member_log_probs'].astype(np.float64)))
    cvp = np.exp(logprob(np, raw['member_log_probs'].astype(np.float64) / T))
    bp = raw['native_body_probabilities'].astype(np.float64)
    cbp = np.stack([cvp[raw['view_body_ids'] == b].mean(0) for b in range(len(bp))])
    p, cp = raw['native_pool_probabilities'].astype(np.float64), calibrated(np, raw['member_log_probs'], T)
    # LM4 uses four private routes; I4 uses four independently selected bodies.
    body_primary = r['endpoint_id'] in ('I4[S64]', 'I4[FS64]', 'GA4')
    decisions = raw['native_body_argmax'] if body_primary else raw['native_member_argmax']
    primary_p, primary_cp = (bp, cbp) if body_primary else (vp, cvp)
    unit = ('independently_selected_body' if body_primary else
        ('private_route' if r['endpoint_id'] in ('LM4',) else
         ('single' if len(vp)==1 else 'tied_view')))
    P, CP = raw['native_pool_argmax'] == y, cp.argmax(-1) == y
    member = decisions == y[None]
    V = member.any(0)
    same_wrong_argmax = (decisions == decisions[0:1]).all(0) & ~member[0]
    true_score = primary_p[np.arange(len(primary_p))[:,None], np.arange(len(y))[None], y[None]]
    strict_rivals = (primary_p > true_score[:,:,None]).all(0)
    strict_rivals[np.arange(len(y)), y] = False
    Q = strict_rivals.any(-1)
    onehot = np.eye(3, dtype=np.float64)[y]
    rnll, cnll = -np.log(p[np.arange(len(y)), y]), -np.log(cp[np.arange(len(y)), y])
    rbrier, cbrier = ((p - onehot) ** 2).sum(-1), ((cp - onehot) ** 2).sum(-1)
    def subset(mask):
        n = int(mask.sum())
        require(n > 0, 'Full class population')
        units = unit_summary(np, primary_p, decisions, primary_cp, y, mask)
        c = dict(nodes=n, served_correct=int(P[mask].sum()), calibrated_served_correct=int(CP[mask].sum()),
            member_correct=units['correct'], coverage=int(V[mask].sum()),
            pooling_losses=int((V & ~P & mask).sum()), aggregation_only_correct=int((~V & P & mask).sum()),
            strict_common_false_rivals=int(Q[mask].sum()), common_wrong_argmax=int(same_wrong_argmax[mask].sum()),
            all_members_correct=int(member[:,mask].all(0).sum()), no_correct_member_count=int((~V & mask).sum()))
        require(c['served_correct'] == c['coverage'] - c['pooling_losses'] + c['aggregation_only_correct'], 'Exact primary-unit coverage identity')
        q = dict(accuracy_pct=100 * c['served_correct'] / n, calibrated_accuracy_pct=100 * c['calibrated_served_correct'] / n,
            member_accuracy_pct=units['accuracy_pct'], mean_member_accuracy_pct=units['mean_accuracy_pct'],
            worst_member_accuracy_pct=units['worst_accuracy_pct'], raw_NLL=float(rnll[mask].mean()),
            calibrated_NLL=float(cnll[mask].mean()), raw_Brier=float(rbrier[mask].mean()), calibrated_Brier=float(cbrier[mask].mean()),
            member_NLL=units['raw_NLL'], mean_member_NLL=float(np.mean(units['raw_NLL'])), worst_member_NLL=max(units['raw_NLL']),
            member_Brier=units['raw_Brier'], mean_member_Brier=float(np.mean(units['raw_Brier'])), worst_member_Brier=max(units['raw_Brier']),
            raw_ECE=None, calibrated_ECE=None, ECE_unavailable_reason='No prospectively frozen bin rule')
        require(all(math.isfinite(q[k]) for k in METRICS), 'Finite whole-bank metrics')
        return dict(counts=c, quality=q, primary_metric_unit=unit,
            views=unit_summary(np, vp, raw['native_member_argmax'], cvp, y, mask),
            bodies=unit_summary(np, bp, raw['native_body_argmax'], cbp, y, mask))
    result = dict(**subset(np.ones(len(y), dtype=bool)),
        classes=[dict(class_id=c, **subset(y == c)) for c in range(3)], P=P, V=V, Q=Q, member=member,
        raw_float64_decision_sensitivity_count=raw['raw_float64_decision_sensitivity_count'],
        view_logprob_decision_sensitivity_counts=raw['view_logprob_decision_sensitivity_counts'],
        body_float64_decision_sensitivity_counts=raw['body_float64_decision_sensitivity_counts'],
        body_probability_origin=raw['body_probability_origin'])

    result['quality'].update(macro_accuracy_pct=statistics.mean(c['quality']['accuracy_pct'] for c in result['classes']),
        calibrated_macro_accuracy_pct=statistics.mean(c['quality']['calibrated_accuracy_pct'] for c in result['classes']))
    macro=[statistics.mean(c['quality']['member_accuracy_pct'][m] for c in result['classes']) for m in range(len(decisions))]
    result['quality'].update(mean_member_macro_accuracy_pct=statistics.mean(macro),worst_member_macro_accuracy_pct=min(macro))
    result['member_macro_accuracy_pct']=macro
    require(all(math.isfinite(result['quality'][k]) for k in REPORT_METRICS),'Finite complete bank/macro metrics')
    return result


def gates(values):
    accuracy, confidence = [], []
    def test(rows, name, reference, metric, mean_min=None, each_min=None, each_max=None, strict_positive=False):
        v = [delta(values[b, 'LM4'], values[b, reference], metric) for b in SEEDS]
        passed = ((mean_min is None or statistics.mean(v) >= mean_min)
            and (each_min is None or all(z >= each_min for z in v))
            and (each_max is None or all(z <= each_max for z in v))
            and (not strict_positive or all(z > 0 for z in v)))
        rows.append(dict(name=name, candidate='LM4', reference=reference, metric=metric,
            minimum_mean=mean_min, minimum_each=each_min, maximum_each=each_max,
            all_three_strict_positive=strict_positive, passed=passed, contrast=descriptive(v, True)))
    for ref in RAW_REFERENCES:
        test(accuracy, 'worthwhile_raw_served_gain', ref, 'accuracy_pct', mean_min=.20, strict_positive=True)
    for ref in ('I4[S64]', 'I4[FS64]'):
        require(all(values[b, 'LM4']['views']['units'] == 4
            and values[b, 'LM4']['primary_metric_unit'] == 'private_route' and values[b, ref]['bodies']['units'] == 4
            and values[b, ref]['primary_metric_unit'] == 'independently_selected_body' for b in SEEDS),
            'Competence compares FOUR LM4 routes to FOUR independently selected equipped bodies')
        test(accuracy, 'route_vs_equipped_independent_body_retention', ref,
            'mean_member_accuracy_pct', mean_min=-.25, each_min=-.50)
    for ref in RAW_REFERENCES:
        test(confidence, 'calibrated_NLL_block_protection', ref, 'calibrated_NLL', each_max=.005)
    means = {ref: statistics.mean(values[b, ref]['quality']['calibrated_NLL'] for b in SEEDS) for ref in RAW_REFERENCES}
    cm = statistics.mean(values[b, 'LM4']['quality']['calibrated_NLL'] for b in SEEDS)
    confidence.append(dict(name='calibrated_NLL_best_reference_mean_protection', candidate='LM4',
        metric='calibrated_NLL', candidate_mean=cm, reference_means=means,
        best_reference_mean=min(means.values()), passed=cm <= min(means.values())))
    ap, cp = all(r['passed'] for r in accuracy), all(r['passed'] for r in confidence)
    return dict(raw_and_member_screen_pass=ap, calibrated_confidence_protection_pass=cp,
        combined_accuracy_and_confidence_pass=ap and cp, accuracy_screens=accuracy,
        confidence_screens=confidence, confidence_failure_kept_separate=True,
        confidence_evaluation_population='DEVELOPMENT', CAL_supplies_temperature_only=True,
        topology_private_order_causal_attribution_tested=False,
        continuation='Raw/member failure closes exact LM4 pilot. Passing requires separately frozen matched private-address/topology controls before causal attribution/unused confirmation/promotion.',
        inference='Three optimizer seeds on encountered PubMed and one fixed DEV role; no confirmation or primitive novelty')


def close_panel(config):
    source=source_check(config['acquisition_source_manifest'])
    end,action=owner_closed(config['acquisition_owner_end'],'acquire',config['acquisition_config'],
        config['acquisition_owner_start'],config['acquisition_source_manifest'])
    require(Path(config['panel']['path'])==Path(action['output'])/'PANEL.json'
        and action['seeds']==list(SEEDS) and action['arms']==list(STAGE2), 'Exact complete acquisition producer output/roster')
    prepare_end,prepare=owner_closed(config['prepare_owner_end'],'prepare',config['prepare_config'],
        config['prepare_owner_start'],config['acquisition_source_manifest'])
    require(Path(config['prepared_custody']['path'])==Path(prepare['output'])/'CUSTODY.json'
        and action['prepared_custody']==config['prepared_custody']
        and action['prepare_owner_end']==config['prepare_owner_end'], 'Exact prepared roles/canonical geometry producer')
    prepared=json.loads(checked(config['prepared_custody']).read_text())
    require(prepared['complete'] is True and prepared['retry'] is False and prepared['TEST_read'] is False
        and prepared['source_manifest']==config['acquisition_source_manifest']
        and prepared['original_VALID_exact_disjoint_union'] is True
        and prepared['role_counts']==dict(select=1313,cal=1313,dev=1316)
        and set(prepared['role_files'])=={'select','select_ids','cal','cal_ids','dev','dev_ids'}, 'Original fresh PolyFormer role custody')
    # Byte custody only; target arrays stay unopened until all fit/owner closure.
    for item in prepared['role_files'].values():checked(item)
    checked(prepared['geometry_capsule']);checked(prepared['role_rng_and_permutations'])
    qualified_end=owner_closed(action['qualification_owner_end'],'qualify')
    qualification=json.loads(checked(action['qualification']).read_text())
    require(qualified_end['source_manifest']==config['acquisition_source_manifest']
        and qualified_end['produced_artifact']==action['qualification']
        and qualification['complete'] is True and qualification['status']=='pass'
        and qualification['source_manifest']==config['acquisition_source_manifest']
        and qualification['prepared_custody']==config['prepared_custody'],
        'Passed complete source/resource qualification bound to exact acquisition')
    panel=json.loads(checked(config['panel']).read_text())
    index={(r['seed'],r['endpoint_id']):r for r in panel['records']}
    require(panel['complete'] is True and panel['terminal'] is True and panel['retry'] is False
        and panel['CAL_DEV_truth_read'] is False and panel['TEST_read'] is False
        and panel['serving']=='mean_class_probabilities' and panel['stage1_closed_before_stage2'] is True
        and len(panel['records'])==len(index)==39 and set(index)=={(s,e) for s in SEEDS for e in ENDPOINTS}
        and panel['config']==config['acquisition_config'] and panel['source_manifest']==config['acquisition_source_manifest']
        and panel['root_decision']==action['root_decision'] and panel['prepared_custody']==config['prepared_custody']
        and panel['prepare_owner_end']==config['prepare_owner_end']
        and panel['owner_start']==config['acquisition_owner_start']
        and panel['expected_owner_end_path']==config['acquisition_owner_end']['path'], 'Exact whole39-row producer linkage before CAL/DEV')
    require(all(panel['counts'].get(k)==v for k,v in ACCOUNTING.items()), '36 fresh groups/54 fresh trajectories/39 endpoints; zero inherited')
    groups={(g['seed'],g['endpoint_id']):g for g in panel['groups']}
    trajectories={t['trajectory_id']:t for t in panel['trajectories']}
    require(len(panel['groups'])==len(groups)==36 and set(groups)=={(s,e) for s in SEEDS for e in ENDPOINTS if e!='GA4'}
        and len(panel['trajectories'])==len(trajectories)==54
        and all(g['status']=='selected_complete' for g in groups.values())
        and panel['records']==panel['endpoints'], 'All36 explicit groups and54 trajectory rows; no aliases or survivor subset')
    checkpoints=[]
    for (s,e),row in index.items():
        V,B,mapping=layout(e)
        stage='stage1' if e.startswith('F1_b') else ('derived' if e=='GA4' else 'stage2')
        authority='native FP32 mean of four own-selected source softmax' if e=='GA4' else 'native FP32 mean of per-view softmax'
        require(row['stage']==stage and row['status']=='selected_complete'
            and row['actual_served_views']==V and row['actual_native_bodies']==B and row['view_body_ids']==mapping
            and row['probability_pool_authority']==authority and isinstance(row['construction'],dict)
            and isinstance(row['costs'],dict) and isinstance(row['provenance'],dict), 'Fresh native row/body/view/cost custody')
        nf=0 if e=='GA4' else (4 if e in ('I4[S64]','I4[FS64]') else 1)
        require(len(row['fits'])==nf, 'Exact independent optimizer trajectory count')
        if e!='GA4':require(row['fits']==groups[s,e]['fits'], 'Selected endpoint retains every complete own group trajectory')
        for fit in row['fits']:
            lower,maximum=(1,2000) if stage=='stage1' else (0,500)
            require(fit['status']=='selected_complete' and fit['selected_restore_verified'] is True
                and type(fit['selected_step']) is type(fit['completed_updates']) is int
                and lower<=fit['selected_step']<=fit['completed_updates']<=maximum and fit['completed_updates']>=1
                and isinstance(fit['costs'],dict) and isinstance(fit['operation_counts'],dict), 'Every exact selected fresh body restored')
            require(fit==trajectories[fit['trajectory_id']], 'Exact explicit fresh trajectory metadata')
            checked(fit['checkpoint']);checkpoints.append(fit['checkpoint']['path'])
        require(set(row['exports'])=={'cal','dev'}, 'Both complete label-free role exports')
        for item in row['exports'].values():checked(item)
    require(len(checkpoints)==len(set(checkpoints))==54
        and sum(len(r['fits']) for r in index.values() if r['stage']=='stage1')==12
        and sum(len(r['fits']) for r in index.values() if r['stage']=='stage2')==42, '54 globally distinct selected fresh trajectories')
    for s in SEEDS:
        ancestors=[dict(body=b,endpoint_id=f'F1_b{b}',checkpoint=index[s,f'F1_b{b}']['fits'][0]['checkpoint']) for b in range(4)]
        for e in STAGE2+('GA4',):
            expected=ancestors if e in ('I4[S64]','I4[FS64]','GA4') else ancestors[:1]
            actual=index[s,e]['ancestors']
            require(len(actual)==len(expected) and all(all(a[k]==v for k,v in b.items()) for a,b in zip(actual,expected)),
                'Candidate uses source body0; independent references own four ancestors')
            if e!='GA4':require(all(a['cache']==index[s,a['endpoint_id']]['cache'] for a in actual), 'Exact unchanged source cache ancestry')
    return panel,index,prepared


def calibrate(config,custody,output):
    start=time.monotonic();panel,index,prepared=close_panel(config)
    actual_providers=providers(config)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    rows=[dict(seed=s,endpoint_id=e,arm=index[s,e]['arm'],status='unstarted_retained',updates=0,inherited=False)
        for s in SEEDS for e in ENDPOINTS]
    result=dict(complete=False,records=rows,requested_endpoints=39,requested_scalar_updates=19500,
        inherited_endpoints=0,inherited_scalar_updates=0,DEV_truth_read=False,TEST_read=False,
        config=config['_config_receipt'],panel=config['panel'],prepared_custody=config['prepared_custody'],custody=custody,retry=False,
        providers=actual_providers,failed_or_unstarted_rows_retained=True)
    try:
        ids,y=truth(np,prepared,'cal')
        for row in rows:
            guard(config,start);s,e=row['seed'],row['endpoint_id'];row['status']='running'
            raw=bank_arrays(np,index[s,e],'cal');require(np.array_equal(ids,raw['ids']), 'Complete identical CAL role/order')
            lp=torch.from_numpy(raw['member_log_probs']).to(torch.float64)
            targets,idx,M=torch.from_numpy(y),torch.arange(len(y)),len(lp)
            logT=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
            optimizer=torch.optim.Adam([logT],lr=.05,betas=(.9,.999),eps=1e-8,weight_decay=0)
            def objective():
                pooled=torch.logsumexp(torch.nn.functional.log_softmax(lp/logT.exp(),dim=-1),dim=0)-math.log(M)
                return -pooled[idx,targets].mean()
            with torch.no_grad():best,best_logT,best_step=float(objective()),0.,0
            require(math.isfinite(best),'Finite initial CAL objective')
            for step in range(1,501):
                guard(config,start);optimizer.zero_grad();loss=objective()
                require(torch.isfinite(loss).item(),'Finite fixed CAL loss');loss.backward()
                require(torch.isfinite(logT.grad).item(),'Finite scalar CAL gradient');optimizer.step()
                with torch.no_grad():
                    logT.clamp_(-5.,5.);value=float(objective());require(math.isfinite(value),'Finite post-update CAL NLL')
                    if value<=best:best,best_logT,best_step=value,float(logT),step
                row['updates']=step
            row.update(status='fixed_endpoint_complete',endpoint='latest_tied_minimum_step0_to500',
                selected_step=best_step,logT=best_logT,temperature=math.exp(best_logT),CAL_minimum_NLL=best,
                objective_forward_evaluations=1001,operator='mean_m softmax(member_log_probs_m/T)',
                temperature_axis='one scalar inside each actual view/body before original mean',
                cal_archive=index[s,e]['exports']['cal'],cal_truth=prepared['role_files']['cal'])
            write(output/'CAL_PROGRESS.json',dict(closed_endpoints=sum(r['status']=='fixed_endpoint_complete' for r in rows),
                actual_scalar_updates=sum(r['updates'] for r in rows),expected_endpoints=39,expected_scalar_updates=19500))
        require(all(r['status']=='fixed_endpoint_complete' and r['updates']==500 and r['inherited'] is False for r in rows)
            and sum(r['updates'] for r in rows)==19500,'All39 fresh temperatures; no inherited fit or survivor subset')
        result.update(complete=True,endpoints=39,actual_scalar_updates=19500,total_bound_scalar_updates=19500,
            seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            fitted_CAL_quality_interpretation=False,CAL_supplies_temperature_only=True)
    except BaseException as error:
        for row in rows:
            if row['status']=='running':row['status']='failed_retained'
        result.update(failure=dict(type=type(error).__name__,message=str(error)),actual_scalar_updates=sum(r['updates'] for r in rows))
        raise
    finally:write(output/'TEMPERATURES.json',result)
    return result


def evaluate(config,custody,output):
    start=time.monotonic();panel,index,prepared=close_panel(config)
    cal_end,cal_config=owner_closed(config['calibration_owner_end'],'calibrate',config['calibration_config'],
        config['calibration_owner_start'],config['source_manifest'])
    require(Path(config['temperatures']['path'])==Path(cal_config['output'])/'TEMPERATURES.json'
        and cal_config['panel']==config['panel'] and cal_config['prepared_custody']==config['prepared_custody']
        and cal_config['acquisition_config']==config['acquisition_config']
        and cal_config['acquisition_owner_end']==config['acquisition_owner_end'], 'Exact complete CAL producer/inputs before DEV')
    temperatures=json.loads(checked(config['temperatures']).read_text())
    endpoints={(r['seed'],r['endpoint_id']):r for r in temperatures['records']}
    require(temperatures['complete'] is True and temperatures['retry'] is False and temperatures['endpoints']==39
        and temperatures['actual_scalar_updates']==temperatures['total_bound_scalar_updates']==19500
        and temperatures['inherited_endpoints']==temperatures['inherited_scalar_updates']==0
        and len(temperatures['records'])==len(endpoints)==39 and set(endpoints)==set(index)
        and temperatures['panel']==config['panel'] and temperatures['prepared_custody']==config['prepared_custody']
        and temperatures['DEV_truth_read'] is False and temperatures['TEST_read'] is False
        and temperatures['custody']['stage']=='calibrate'
        and temperatures['custody']['source_manifest']==config['source_manifest']
        and temperatures['config']==config['calibration_config']
        and temperatures['custody']['root_decision']==cal_config['root_decision'], 'All39 exact fresh endpoints and producer owner closed before DEV')
    for key,ep in endpoints.items():
        require(ep['inherited'] is False,'No inherited temperature or refit');endpoint_temperature(ep,index[key]['exports']['cal'],prepared['role_files']['cal'])
    actual_providers=providers(config)
    import numpy as np
    release=dict(one_whole_release=True,seeds=list(SEEDS),after_fresh_fit_banks=36,after_fresh_optimizer_trajectories=54,
        after_exports=39,after_CAL_endpoints=39,after_scalar_CAL_updates=19500,inherited_fits_or_CAL=0,
        panel=config['panel'],temperatures=config['temperatures'],calibration_config=config['calibration_config'],
        calibration_owner_start=config['calibration_owner_start'],calibration_owner_end=config['calibration_owner_end'],
        custody=custody,DEV_truth=prepared['role_files']['dev'],TEST_closed=True)
    write(output/'DEV_RELEASE.json',release)
    result=dict(complete=False,records=[],release=release,panel=config['panel'],temperatures=config['temperatures'],
        retry=False,survivor_subset=False,providers=actual_providers)
    try:
        ids,y=truth(np,prepared,'dev');values={}
        for s in SEEDS:
            for e in ENDPOINTS:
                guard(config,start);raw=bank_arrays(np,index[s,e],'dev')
                require(np.array_equal(ids,raw['ids']),'Complete original DEV role/order; same population across seeds')
                value=score_bank(np,index[s,e],raw,y,endpoints[s,e]['temperature']);values[s,e]=value
                result['records'].append(dict(seed=s,endpoint_id=e,arm=index[s,e]['arm'],
                    **{k:v for k,v in value.items() if k not in ('P','V','Q','member')},
                    temperature=endpoints[s,e]['temperature'],stage=index[s,e]['stage'],origin=index[s,e]['origin']))
                write(output/'DEV_PROGRESS.json',dict(scored_rows=len(result['records']),expected_rows=39))
        comparisons=[]
        for a in ENDPOINTS:
            for ref in ENDPOINTS:
                if a==ref:continue
                guard(config,start)
                comparisons.append(dict(candidate=a,reference=ref,
                    metric_deltas={k:descriptive([delta(values[s,a],values[s,ref],k) for s in SEEDS],True) for k in REPORT_METRICS},
                    per_seed=[dict(seed=s,**pair(values[s,a],values[s,ref],np.ones(1316,dtype=bool)),
                        classes=[dict(class_id=c,**pair(values[s,a],values[s,ref],y==c)) for c in range(3)]) for s in SEEDS]))
        regions=[]
        for s in SEEDS:
            for ref in RAW_REFERENCES:
                candidate,reference=values[s,'LM4'],values[s,ref]
                mask=reference['Q'];regions.append(dict(seed=s,reference=ref,region='strict_common_false_rival',nodes=int(mask.sum()),
                    new_correct_route_supply=int((candidate['V'] & mask).sum()),
                    actual_served_repairs=int((candidate['P'] & ~reference['P'] & mask).sum()),
                    diagnostics=pair(candidate,reference,mask),outside_diagnostics=pair(candidate,reference,~mask),
                    interpretation='Reference cohort descriptive only; whole-population primary; no diagnosed cause'))
        paired_members=[dict(seed=s,candidate='LM4',reference=ref,candidate_unit='four private routes',
            reference_unit='four independently selected equipped bodies',candidate_route_accuracy_pct=values[s,'LM4']['views']['accuracy_pct'],
            reference_body_accuracy_pct=values[s,ref]['bodies']['accuracy_pct'],
            independent_selected_horizons=[f['selected_step'] for f in index[s,ref]['fits']]) for s in SEEDS for ref in ('I4[S64]','I4[FS64]')]
        require(len(result['records'])==39 and len(comparisons)==156,'Complete39-row/156ordered-contrast report; no survivor subset')
        result.update(complete=True,comparisons=comparisons,paired_member_diagnostics=paired_members,
            encountered_strict_common_rival_regions=regions,gates=gates(values),
            general_source_memory_augmentation_contrasts=[r for r in comparisons
                if r['candidate'] in ('S64','FS64') and r['reference'] in ('F1_b0','B1','B4')],
            arm_summaries=[dict(endpoint_id=e,metrics={k:descriptive([values[s,e]['quality'][k] for s in SEEDS]) for k in REPORT_METRICS}) for e in ENDPOINTS],
            all_reference_paid_and_serving_records=panel['records'],paid_acquisition_accounting=panel['counts'],
            complete_acquisition_metadata={k:v for k,v in panel.items() if k!='records'},
            paid_prepare_metadata=prepared,paid_prepare_owner=config['prepare_owner_end'],paid_acquisition_owner=config['acquisition_owner_end'],
            paid_calibration_owner=config['calibration_owner_end'],calibration_scalar_updates=19500,inherited_scalar_updates=0,
            reader_costs=dict(seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            numeric_reconstitution='Sensitivity only; native single-logit and pool/body decisions remain primary',
            confidence_population='DEVELOPMENT; CAL supplies T only',
            strict_rival_definition='Some common wrong class strictly above truth in every primary route/body; distinct from same wrong argmax',
            scope='Three optimizer seeds on one encountered PubMed graph/fixed role; no independent-node/seed generalization interval,confirmation,causal,novelty or automatic next fit claim')
    except BaseException as error:
        result['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:write(output/'DEVELOPMENT_REPORT.json',result)
    return result


def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    if config['stage']=='calibrate':return calibrate(config,custody,output)
    if config['stage']=='score':return evaluate(config,custody,output)
    raise ValueError('Exact complete reader stage required')
