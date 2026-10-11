"""Metadata-only join of closed core/context archives; no target/array reads.

Normalizes the existing SELECTED_COMPLETE status and author context arm name.
Root retains ownership of all model/export execution and whole-stage cleanup.
"""
import argparse
from pathlib import Path
from contracts import ARMS,BLOCKS,require,receipt,read,write,scoped,sha,route,close_predictions,validate_predictions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();route()
    here=scoped(Path(__file__).parent);seal=read(here/'SEAL.json')
    require(all(sha(here/name)==digest for name,digest in seal['files'].items()),'Readout/join source seal unchanged')
    cfg_path,adm_path=scoped(args.config),scoped(args.admission);cfg,adm=read(cfg_path),read(adm_path)
    protocol_sha=sha(here/'PROTOCOL.json')
    require(cfg.get('enabled') is True and cfg.get('protocol_sha256')==protocol_sha
            and adm.get('admitted') is True and adm.get('stage')=='join_predictions'
            and adm.get('config_sha256')==sha(cfg_path) and adm.get('source_seal_sha256')==sha(here/'SEAL.json')
            and adm.get('root_validated_all90_selected_state_custody') is True
            and adm.get('all_native_owners_closed_and_clean') is True
            and adm.get('CAL_truth_access') is False and adm.get('TEST_truth_access') is False,
            'Root exact joined metadata/custody admission required')
    for item in cfg['source_bindings'].values():receipt(item)
    core=read(receipt(cfg['core_predictions']));context=read(receipt(cfg['author_context']))
    require(core.get('terminal') is True and core.get('complete') is True
            and core.get('CAL_CONFIRM_truth_access') is False and len(core.get('records',[]))==50
            and context.get('terminal') is True and context.get('complete') is True
            and context.get('CAL_CONFIRM_truth_access') is False and len(context.get('records',[]))==10,
            'Both entire selected-prediction source cohorts closed; no partial/survivor join')
    rows=[]
    statuses={'SELECTED_COMPLETE':'selected_finite','FAILED_RETAINED':'failed_retained',
              'selected_finite':'selected_finite','failed_retained':'failed_retained'}
    for original in core['records']+context['records']:
        row=dict(original);row['source_arm']=original['arm'];row['source_status']=original['status']
        if row['arm']=='author_meanSAGE_M1':row['arm']=ARMS[-1]
        require(row['arm'] in ARMS and row['block'] in BLOCKS and row['status'] in statuses,'Fixed bank/status mapping')
        row['status']=statuses[row['status']]
        row['fits']=[dict(fit,source_status=fit['status'],status=statuses[fit['status']]) for fit in row['fits']]
        rows.append(row)
    value={'schema':'Physics_selected_predictions_complete_v1','terminal':True,'blocks':list(BLOCKS),'arms':list(ARMS),
           'banks':60,'body_fit_units':90,'protocol_sha256':protocol_sha,'CAL_truth_access':False,'TEST_truth_access':False,
           'all_native_owners_closed_and_clean':True,'records':rows,'source_bindings':cfg['source_bindings'],
           'cost_records':cfg['cost_records'], 'source_closures':{'core':cfg['core_predictions'],'author_context':cfg['author_context']},
           'root_admission':{'config_sha256':sha(cfg_path),'admission_sha256':sha(adm_path)},
           'status_and_arm_normalization_only_no_predictions_changed':True}
    validate_predictions(value,protocol_sha)
    path=scoped(args.output);write(path,value)
    close_predictions({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)},protocol_sha)
    print('{"terminal":true,"banks":60,"body_fit_units":90,"arrays_read":false,"truth_access":false}')

if __name__=='__main__':main()
