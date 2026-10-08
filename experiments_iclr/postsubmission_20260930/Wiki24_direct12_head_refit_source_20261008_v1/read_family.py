"""Comparative opening only after all twelve original endpoints/failures seal."""
import argparse
import os
from common import *
from heads import logits
from cohorts import masks, transitions


def metrics(torch, np, L, P, pool, y):
    require(np.isfinite(L).all() and np.isfinite(P).all() and np.isfinite(pool).all(), 'Finite complete predictions')
    yt = torch.from_numpy(y)
    logp = torch.from_numpy(L).log_softmax(-1).gather(-1,yt[None,:,None].expand(len(L),-1,1)).squeeze(-1)
    member_accuracy = (P.argmax(-1) == y[None]).mean(1)
    return dict(served_accuracy=float((pool.argmax(-1)==y).mean()),
        mean_member_accuracy=float(member_accuracy.mean()), worst_member_accuracy=float(member_accuracy.min()),
        members=[dict(member_index0=m, accuracy=float(member_accuracy[m]), NLL=float(-logp[m].mean())) for m in range(len(L))],
        pooled_NLL=float(-(torch.logsumexp(logp,dim=0)-__import__('math').log(len(L))).mean()),
        global_any_member_correct=float((P.argmax(-1)==y[None]).any(0).mean()))


def fitted(torch, np, a, endpoint):
    H = torch.from_numpy(a['H_dev'])
    with torch.no_grad():
        L = logits(torch,H,torch.from_numpy(endpoint['A']),torch.from_numpy(endpoint['bias']))
        P = L.softmax(-1); pool = P.mean(0)
    return L.numpy(), P.numpy(), pool.numpy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release',type=Path,required=True); parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    cfg,output=release(args.release,args.release_sha256,'read')
    require(cfg.get('whole12_comparative_opening_authorized') is True, 'Separate whole12 comparative opening authority')
    sealed=read(bound(cfg['fit_seal']))
    require(sealed.get('closed') is True and sealed.get('whole12_accounted') is True
            and sealed['source_manifest_sha256']==cfg['source_manifest_sha256'], 'Sealed all12 source identity')
    closure=read(bound(sealed['closure']))
    terminal=read(bound(cfg['fit_terminal_evidence']))
    require(terminal.get('owner_and_children_terminal') is True and terminal.get('fit_seal')==cfg['fit_seal'],
            'Separate actual fit-family owner exit/reap evidence')
    require(closure.get('closed') is True and closure.get('whole12_accounted') is True
            and closure.get('all_owned_workers_reaped') is True
            and closure.get('development_scores_opened') is False and closure['roster']==roster(), 'All original12 endpoints and worker terminals first')
    require([{k:r[k] for k in ('condition','seed','bundle','arm','members')} for r in closure['rows']]==roster(), 'No surviving-only roster')
    # Hash every supplied endpoint/result and prepared asset BEFORE any comparative arrays.
    for row in closure['rows']:
        for key in ('worker_job','worker_log','failure'):
            if key in row: bound(row[key])
        if row['status'] in ('converged','finite_nonconverged'):
            bound(row['result']); bound(row['endpoint']); bound(row['asset']['prepared'])
        require(row['status'] in ('converged','finite_nonconverged','failed','timeout','source_unavailable','not_attempted'), 'Closed terminal status')
    pins=read(ROOT/'SOURCE_BINDINGS.json')
    decision=read(bound(pins['frozen_decision'])); bound(pins['cohort_clarification'])
    require(decision['whole12_closure_before_comparative_opening'] is True, 'Frozen prospective screen')
    import numpy as np
    import torch
    backend(torch,np)
    metadata=read(bound(pins['metadata_export']))
    old={(r['arm'],r['seed']):r for r in metadata['cells']}
    stored_quality=read(bound(pins['original_native_quality_readout']))
    quality={(r['arm'],r['seed']):r for r in stored_quality['cells']}
    native={}; native_rows=[]; prepared={}; refits=[]; arrays={}
    for row in closure['rows']:
        key=(row['arm'],row['seed'])
        asset=row.get('asset',{})
        if key not in native and 'prepared' in asset:
            source=asset['source']; a=load_arrays(np,source['features']); prepared[key]=load_arrays(np,asset['prepared'])
            require(a['y_dev'].shape==(5274,) and a['dev_ids'].shape==(5274,), 'Complete development only')
            L,P,pool=a['native_dev_logits'],a['native_dev_probability'],a['native_dev_pool']
            native[key]=(a,masks(np,L,P,pool,a['y_dev']))
            native_rows.append(dict(arm=key[0],seed=key[1],
                authoritative_historical_selected_VALID=old[key]['selected_VALID'],
                authoritative_historical_member_VALID=old[key]['member_VALID'],
                authoritative_closed_native_pooled_NLL=quality[key]['full_population']['served_nll'],
                native_reference_NLL_custody=pins['original_native_quality_readout'],
                native_FP32_checkpoint_prediction_event=metrics(torch,np,L,P,pool,a['y_dev']),
                historical_scores_replaced=False))
        record={k:row[k] for k in ('condition','seed','bundle','status')}
        if row['status'] not in ('converged','finite_nonconverged'):
            record['unavailable_reason']=row['unavailable_reason']; refits.append(record); continue
        a=prepared[key]; e=load_arrays(np,row['endpoint']); receipt=read(bound(row['result']))
        require(e['A'].shape==(row['members'],10,512) and e['A'].dtype==np.float64
                and np.isfinite(e['A']).all() and np.isfinite(e['bias']).all(), 'Original finite FP64 endpoint')
        L,P,pool=fitted(torch,np,a,e)
        flag=masks(np,L,P,pool,a['y_dev'])
        record.update(metrics=metrics(torch,np,L,P,pool,a['y_dev']), optimization=receipt)
        arrays[(row['condition'],row['seed'])]=(L,P,pool,flag,record)
        refits.append(record)
    pairs=[]
    for seed in SEEDS:
        baseline=native.get(('be_unit_contrastive',seed))
        pair=dict(seed=seed,available=False)
        if baseline is None:
            pair['unavailable_reason']='Native unit+contrast source unavailable'; pairs.append(pair); continue
        a,b=baseline
        source=next(r['asset']['source'] for r in closure['rows'] if r['condition']=='BE_factor_refit' and r['seed']==seed)
        fixed=load_arrays(np,source['native_cohorts'])
        require(np.array_equal(fixed['dev_ids'],a['dev_ids']) and np.array_equal(fixed['y_dev'],a['y_dev']), 'Frozen native cohort roles')
        require(all(np.array_equal(fixed[k],b[k]) for k in b), 'Original prefit native cohort membership')
        for condition in CONDITIONS:
            candidate=arrays.get((condition,seed))
            if candidate is None: continue
            L,P,pool,flag,record=candidate
            prepared_key=(ARM[condition],seed)
            require(np.array_equal(prepared[prepared_key]['dev_ids'],a['dev_ids'])
                    and np.array_equal(prepared[prepared_key]['y_dev'],a['y_dev']), 'Identical complete comparison population')
            eligible=b['all_members_wrong'] & b['strict_common_wrong_rival']
            cohort_names=('full_population','pooled_correct','pooled_wrong','any_member_correct','all_members_wrong','strict_common_wrong_rival')
            record['native_unit_contrast_transitions']={name:transitions(np,b,flag,a['native_dev_logits'],L,a['y_dev'],b[name]) for name in cohort_names}
            record['native_all_wrong_common_rival_transitions']=transitions(np,b,flag,a['native_dev_logits'],L,a['y_dev'],eligible)
            record['fitted_all_wrong']=int(flag['all_members_wrong'].sum())
            record['fitted_common_rival']=int(flag['strict_common_wrong_rival'].sum())
            record['fitted_all_wrong_common_rival']=int((flag['all_members_wrong'] & flag['strict_common_wrong_rival']).sum())
        f=arrays.get(('BE_factor_refit',seed)); u=arrays.get(('BE_full_refit',seed))
        if f is not None and u is not None:
            fm,um=f[4]['metrics'],u[4]['metrics']
            pair.update(available=True,factor_metrics=fm,full_metrics=um,
                deltas_full_minus_factor={k:um[k]-fm[k] for k in ('served_accuracy','pooled_NLL','mean_member_accuracy','worst_member_accuracy','global_any_member_correct')},
                excess_newly_covered_native_all_wrong_nodes=int((u[3]['any_member_correct'] & b['all_members_wrong']).sum())-int((f[3]['any_member_correct'] & b['all_members_wrong']).sum()),
                excess_actual_pooled_repairs_on_native_all_wrong_common_rival=u[4]['native_all_wrong_common_rival_transitions']['pooled_repairs']-f[4]['native_all_wrong_common_rival_transitions']['pooled_repairs'])
        pairs.append(pair)
    evaluable=len(arrays)==12 and all(r['status'] in ('converged','finite_nonconverged') for r in closure['rows'])
    affirmative=evaluable and all(r['status']=='converged' for r in closure['rows'])
    screen=dict(evaluable=evaluable,affirmative_complete_screen_allowed=affirmative,
                full_specific_opportunity=None,coverage_opportunity=None,
                ordinary_quality_gap_closed=None,frozen_weakness_remedied=None)
    if affirmative:
        mean=lambda key:sum(p['deltas_full_minus_factor'][key] for p in pairs)/3
        screen['full_specific_opportunity']=all(p['deltas_full_minus_factor']['served_accuracy']>0 for p in pairs) and mean('pooled_NLL')<=0 and mean('mean_member_accuracy')>=0 and mean('worst_member_accuracy')>=0
        screen['coverage_opportunity']=all(p['excess_newly_covered_native_all_wrong_nodes']>0 for p in pairs) and mean('global_any_member_correct')>0 and sum(p['excess_actual_pooled_repairs_on_native_all_wrong_common_rival'] for p in pairs)>0
        fullmean=lambda key:sum(arrays[('BE_full_refit',s)][4]['metrics'][key] for s in SEEDS)/3
        ordinarymean=lambda key:sum(arrays[('ordinary_full_refit',s)][4]['metrics'][key] for s in SEEDS)/3
        nativeordinary=[r for r in native_rows if r['arm']=='independent4']
        original_accuracy=sum(r['authoritative_historical_selected_VALID'] for r in nativeordinary)/3
        original_NLL=sum(r['authoritative_closed_native_pooled_NLL'] for r in nativeordinary)/3
        screen['ordinary_quality_gap_closed']=fullmean('served_accuracy')>=max(original_accuracy,ordinarymean('served_accuracy')) and fullmean('pooled_NLL')<=min(original_NLL,ordinarymean('pooled_NLL'))
        screen['frozen_weakness_remedied']=screen['full_specific_opportunity'] and screen['coverage_opportunity'] and screen['ordinary_quality_gap_closed']
    output.mkdir(mode=0o700)
    report=dict(schema='Wiki24-direct12-whole-family-readout-v1',whole12_accounted=True,native=native_rows,refits=refits,paired_BE=pairs,
        frozen_decision=pins['frozen_decision'],screen=screen,automatic_next_fit=False,TEST_access=False,
        limitations=['Frozen C10 head utility only; no whole-network capacity conclusion',
            'Factor stationarity does not certify global optimality; raw/effective gradient norms both reported',
            'Same development role selected native states and supplies this descriptive diagnostic',
            'Ordinary private biases and original backbone objective/selection differ from the BE pair',
            'Three seed blocks; nodes/members are not independent fitted-model replicates',
            'Partial opportunity can motivate a separately reviewed proposal; it does not launch or establish a remedy',
            'Original native checkpoint scores remain authoritative; FP64 fitted scores are a separate diagnostic'])
    write(output/'REPORT.json',report)
    write(output/'SEAL.json',dict(report=binding(output/'REPORT.json'),fit_seal=cfg['fit_seal'],source_manifest_sha256=cfg['source_manifest_sha256']))


if __name__=='__main__': main()
