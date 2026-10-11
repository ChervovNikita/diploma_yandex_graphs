"""One root-admitted selected-state export child; no fit/update invocation."""
import argparse
from pathlib import Path
from common import H,CORE,BLOCKS,require,checked,read,item,sha,write,load,freeze_recheck,pipeline_gate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--admission',required=True,type=Path);parser.add_argument('--freeze',required=True,type=Path)
    parser.add_argument('--release',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();cfg,adm,cfg_receipt,adm_receipt=pipeline_gate(args.config,args.admission)
    release=read(args.release);freeze_receipt=item(args.freeze);freeze=read(checked(freeze_receipt))
    require(release.get('admitted') is True and release.get('stage')=='selected_core_export'
            and release.get('pipeline_config')==cfg_receipt and release.get('pipeline_admission')==adm_receipt
            and release.get('freeze')==freeze_receipt and release.get('pipeline_seal_sha256')==sha(Path(__file__).with_name('SEAL.json'))
            and release.get('CAL_truth_access') is False and release.get('TEST_truth_access') is False,
            'One source-bound export release only after exact90-state freeze')
    freeze_recheck(freeze)
    prepared=read(checked(cfg['prepared_receipt']))
    role_blocks={str(b):{k:prepared['files'][f'block_{b}/{k}.npy'] for k in ('cal_ids','confirm_ids')} for b in BLOCKS}
    for block in role_blocks.values():
        for receipt in block.values():checked(receipt)
    integration=load('Physics_pipeline_core_constructor',cfg['source_receipts']['core_constructor'])
    spec=read(checked(cfg['core_integration_spec']))
    require(spec['H']==str(H),'Same root-acquired full Physics cohort')
    family,port_receipt=integration.construct_core(spec,H/'selected_core_loader_v1')
    require(family.cfg['seeds']==list(BLOCKS) and family.cfg['root_frozen_protocol']==read(checked(cfg['protocol'])),
            'Constructor retains exact frozen fit config/protocol')
    exporter=load('Physics_pipeline_selected_core_exporter',cfg['source_receipts']['selected_export'])
    export_release={'status':'ROOT_SELECTED_EXPORT_ADMITTED','joined_selected_states90':'ROOT_VERIFIED_COMPLETE',
                    'limits':cfg['export_limits'],'freeze':freeze_receipt}
    def root_validate(actual,**kwargs):
        require(actual==family.cfg and kwargs.get('action')=='fixed_Physics_selected_core_export'
                and kwargs.get('source_receipt')==cfg['source_receipts']['selected_export']
                and kwargs.get('complete_fit_receipt')==freeze['core_complete']
                and kwargs.get('role_blocks')==role_blocks and kwargs.get('release')==export_release,
                'Exact root-approved export source/config/states/roles/limits')
        freeze_recheck(freeze)
        return {'status':'ROOT_SELECTED_EXPORT_ADMITTED','pipeline_config':cfg_receipt,'pipeline_admission':adm_receipt,
                'freeze':freeze_receipt,'port_receipt':port_receipt,'CAL_TEST_truth_access':False,'fit_updates':0}
    result=exporter.export_all(family,checked(freeze['core_complete']),role_blocks,args.output,root_validate,export_release)
    require(result['terminal'] and result['complete'] and result['banks']==50 and result['fits']==80
            and result['fit_updates']==0 and result['CAL_CONFIRM_truth_access'] is False,'Whole50 core prediction banks, no updates')
    freeze_recheck(freeze)
    write(Path(args.output)/'EXPORT_STAGE_END.json',{'terminal':True,'complete':True,'freeze':freeze_receipt,
          'prediction_closure':item(Path(args.output)/'COMPLETE_SELECTED_CORE_PREDICTIONS.json'),
          'fit_updates':0,'CAL_TEST_truth_access':False})
    print('{"terminal":true,"complete":true,"core_banks":50,"fit_updates":0,"truth_access":false}')

if __name__=='__main__':main()
