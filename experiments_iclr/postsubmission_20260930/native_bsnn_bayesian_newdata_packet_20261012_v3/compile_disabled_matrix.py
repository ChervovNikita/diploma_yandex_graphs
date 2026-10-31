"""Compile two source Bayesian variants; metadata only, no run authority."""
import copy
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
sys.path.insert(0,str(HERE/'runtime'))
from spec import canonical,fingerprint,validate
from compile_master_plan import receipt as master_receipt,typed_counts,bank_roster

def receipt(root,path):
    return master_receipt(root,Path(path).resolve().relative_to(Path(root).resolve()))
from bsnn_stopping import BUDGET,SELECTION,VARIANTS,validate_bsnn_run
from adapters.bsnn_bayesian import CONTRACT


def main():
    old=PHASE/'retained_final_five_batch_20261012_v1/bsnn_ready_v1'
    oldplan=json.loads((old/'matrix/PLAN.json').read_text())
    oldfile=PHASE/oldplan['studies'][0]['path']
    oldstudy=json.loads(gzip.decompress(oldfile.read_bytes()))
    B=json.loads((HERE/'BSNN_BAYESIAN_SOURCE_BINDINGS.json').read_text())
    out=HERE/'matrix';out.mkdir();(out/'studies').mkdir()
    studies=[];specs=[];groups=[];bank_details=[]
    root_files=[receipt(PHASE,p) for p in sorted((HERE/'runtime').rglob('*.py'))]
    root_files += [receipt(PHASE,HERE/name) for name in ('BSNN_BAYESIAN_SOURCE_BINDINGS.json','check_synthetic_CPU.py','compile_disabled_matrix.py')]
    for variant,(layers,second,dropout,input_dropout) in VARIANTS.items():
        study=copy.deepcopy(oldstudy);study['study_id']='native-Bayesian-'+variant+'-new-TwitchDE-v1'
        study['hypothesis']='Fixed native Bayesian Cayley '+variant+' against genuine own-selected I4, explicit shared Bayesian boundary-GNNM and same-paid-M1 MC16 serving control.'
        study['source']['files']+=root_files
        study['data']['evidence'].update(source_Bayesian_KL_SciPy_preserved=True,source_total500_complete_updates=False,
            source_successful_completed_update_bounds=[201,500],native_posterior_selection_draws=4,
            extra_MC16_control_reuses_exact_M1_selected_fit=True,original_paper_dataset_or_historical_state_used=False)
        study['comparisons']=[{'candidates':[variant],'references':[variant+'_genuine_I4',variant+'_original_boundary_GNNM',variant+'_source_M1_MC16'],
            'scope':'10 complete existing custodied TwitchDE binary roles x3 paired seeds; fixed source tuple/groupedAdam; source cap500/patience200 and4MC selector; no HPO or reselection',
            'interpretation':'development; posterior draws are not independent learned parameter members; shared boundary adaptation retains native KL/SciPy; MC16 isolates serving sample count from four routes/bodies'}]
        runs=study['runs'];models={}
        for r in runs:
            r['adapter']=str((HERE/'runtime/adapters').relative_to(PHASE))+'/'+('bsnn_bayesian.py:make_adapter' if r['kind']=='model_fit' else 'bsnn_bayesian_endpoints.py:'+r['adapter'].rsplit(':',1)[1])
            if r['kind']=='model_fit':
                c=r['config'];arm=c.pop('bsnn_arm');c.pop('bsnn_sources');c.pop('bsnn_source_contract')
                c.update(bayesian_bsnn_arm=arm,bayesian_bsnn_variant=variant,bayesian_bsnn_sources=B,
                    bayesian_bsnn_source_contract=CONTRACT,fusion='mean_probabilities' if arm=='native_M1' else 'mean_logits',backbone='original_BayesBundleSheafDiffusion_d2',
                    architecture='native_Bayesian_BSNN_typed',width=32,depth=layers,second_linear=second,
                    dropout=dropout,input_dropout=input_dropout,weight_decay=.0005,source_complete_body_constructors=1,
                    source_SELECT_draws=4*c['members'],source_KL_rule='native_incidence_mean_sigmoid40_no_layer_multiplier',
                    source_optimizer_rule='source_grouped_Adam_lr.01_both_decay.0005',historical_state_inherited=False)
                r['budget']=copy.deepcopy(BUDGET);r['selection']=copy.deepcopy(SELECTION)
                r['method']=variant if arm=='native_M1' else variant+'_original_boundary_GNNM'
                validate_bsnn_run(r);models[r['run_id']]=r
            elif r['kind']=='ensemble_assembly':
                c=r['config'];c.update(posterior_draws_per_route=4,posterior_count_control=False,
                    source_posterior_draws=16 if len(c['fit_parents'])==4 or models[c['fit_parents'][0]]['config']['members']==4 else 4,
                    source_selection_draws_per_route=4)
                r['method']=variant+'_genuine_I4' if len(c['fit_parents'])==4 else models[c['fit_parents'][0]]['method']
            else:
                r['method']=variant+r['method'].replace('bsnn_deterministic_d2','').replace('BSNN_native_genuine_I4','_genuine_I4').replace('BSNN_original_boundary_GNNM','_original_boundary_GNNM')
        # All source comparative fits and endpoint triplets keep their exact
        # ancestry. Add one serving-only MC16 triplet per paid native body0.
        originals=list(runs)
        for ex in [r for r in originals if r['kind']=='ensemble_assembly' and len(r['config']['fit_parents'])==1
                   and models[r['config']['fit_parents'][0]]['config']['bayesian_bsnn_arm']=='native_M1']:
            cal=next(r for r in originals if r['kind']=='calibration' and ex['run_id'] in r['parents'])
            ev=next(r for r in originals if r['kind']=='evaluation' and ex['run_id'] in r['parents'])
            mapping={r['run_id']:r['run_id']+'-MC16' for r in (ex,cal,ev)}
            for parent in (ex,cal,ev):
                r=copy.deepcopy(parent);r['run_id']=mapping[r['run_id']];r['method']=variant+'_source_M1_MC16'
                r['parents']=[mapping.get(x,x) for x in r['parents']]
                for key in ('export_parent','calibration_parent'):
                    if key in r['config']:r['config'][key]=mapping.get(r['config'][key],r['config'][key])
                if r['kind']=='ensemble_assembly':
                    r['config'].update(posterior_draws_per_route=16,posterior_count_control=True,source_posterior_draws=16,
                        extra_source_forward_draws=12,source4MC_selected_fit=ex['config']['fit_parents'][0],reselection=False,retraining=False)
                runs.append(r)
        fit=[r for r in runs if r['kind']=='model_fit'];banks=[r for r in runs if r['kind']=='ensemble_assembly']
        assert len(fit)==150 and len(banks)==120 and len(runs)==510
        validate(study);path=out/'studies'/(study['study_id']+'.json.gz')
        path.write_bytes(gzip.compress((canonical(study)+'\n').encode(),mtime=0))
        specs.append({'path':str(path.relative_to(PHASE)),'spec_hash':fingerprint(study)});studies.append(study)
        group=copy.deepcopy(oldplan['groups'][0]);group['id']=study['study_id'];group['run_keys']=[study['study_id']+'/'+r['run_id'] for r in runs]
        group['estimated_seconds']=len(fit)*500;group['assignment_status']='PROVISIONAL_UNMEASURED_DISABLED_source_family_resources_pending'
        groups.append(group);bank_details+=bank_roster(study)
    plan=copy.deepcopy(oldplan);plan.update(studies=specs,groups=groups,output_root=str(out.relative_to(PHASE))+'/execution',
        runtime_package=str((HERE/'runtime').relative_to(PHASE)),runtime_launcher=str((HERE/'runtime/launcher.py').relative_to(PHASE)),
        preparation_scope='Two exact native Bayesian Cayley tuples with true posterior/KL/SciPy/native stop, shared typed boundary adaptation, and same-paid-source MC16 serving control')
    for gate in plan['global_gates'].values():gate['path']=str(out.relative_to(PHASE))+'/gate_receipts/'+Path(gate['path']).name
    counts=typed_counts(studies)
    summary={'schema':'native-Bayesian-BSNN-disabled-two-variant-summary-v1','status':'DISABLED_CPU_PROOF_ACTUAL_INPUT_RESOURCES_AND_GO_PENDING',
        'plan_hash':fingerprint(plan),'counts':counts,'canonical_conditions':list(VARIANTS),'planned_model_fits':300,
        'genuine_I4_banks':60,'original_GNNM_new_data_fits':60,'native_M1_MC16_extra_training_fits':0,
        'native_M1_MC16_banks':60,'native_M1_MC16_extra_forward_draws':720,'all_banks':240,
        'source_optimizer_update_total_exact':False,'successful_complete_fit_optimizer_map_bounds':[60300,150000],
        'source_contract':CONTRACT,'source_4MC_SELECTED_without_reselection':True,
        'all_prior_deterministic_source_packets_preserved':True,'historical_states_reused':False,
        'actual_input_resources_qualified':False,'scientific_fits':0,'GO':False}
    for name,value in [('PLAN.json',plan),('SUMMARY.json',summary),('BANK_ROSTER.json',bank_details)]:
        (out/name).write_text(canonical(value)+'\n')
    entry={'id':'native_Bayesian_BSNN_two_source_variants_new_data','plan':receipt(PHASE,out/'PLAN.json'),
           'summary':receipt(PHASE,out/'SUMMARY.json'),'plan_hash':fingerprint(plan),
           'coverage':[receipt(PHASE,HERE/'BSNN_BAYESIAN_SOURCE_BINDINGS.json')]}
    (HERE/'MASTER_INPUT_ENTRY.json').write_text(canonical(entry)+'\n')
    print(json.dumps({'summary':summary,'entry':entry}))


if __name__=='__main__':main()
