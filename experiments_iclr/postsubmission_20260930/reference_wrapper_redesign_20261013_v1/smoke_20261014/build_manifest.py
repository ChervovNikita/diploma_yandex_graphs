"""167 actual source-configuration checks; no scientific campaign GO."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REMOTE='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
PHASE=REMOTE+'/experiments_iclr/postsubmission_20260930'
DESIGN=PHASE+'/reference_wrapper_redesign_20261013_v1'


def main():
    roster=json.loads((HERE.parent/'ROSTER.json').read_text());cases=[]
    for original in roster['recipes']:
        recipe=dict(original);recipe['_redesign_root']=DESIGN;recipe['_seed']=0
        if recipe['task']=='node_classification':
            callback=DESIGN+'/smoke_20261014/node_port.py'
            root=PHASE+'/training_pipeline_v2_20261012/provider_cache'
        elif recipe['task']=='graph_classification':
            callback=DESIGN+'/smoke_20261014/ogb_port/session.py'
            folder='massive_molecular_packet_20261012_v1' if recipe['dataset']=='ogbg-molhiv' else 'massive_molecular_packet_20261012_v3'
            root=PHASE+'/'+folder+'/provider_cache'
        else:
            callback=DESIGN+'/smoke_20261014/link_port/callbacks.py'
            root=PHASE+'/training_pipeline_v2_20261012/qa/link_admitted_transfer_v1/cache/ogbl_collab'
            recipe['data_files']={'graph':PHASE+'/buddy_newdata_root_20261012_v1/provider_data/ogbl_collab/processed/geometric_data_processed.pt',
                                 'train':root+'/split/time/train.pt','valid':root+'/split/time/valid.pt'}
        for procedure in ['native_single']+[m['id'] for m in recipe['wrappers']]+['independent4']:
            cfg=dict(recipe)
            if procedure=='independent4':cfg['_native_callback']=callback
            cases.append({'id':recipe['id']+'/'+procedure,'recipe':cfg,'procedure':procedure,
                          'callback_module':DESIGN+'/smoke_20261014/independent_bank.py' if procedure=='independent4' else callback,
                          'data_root':root,'device':'cuda:0','rtol':1e-4,'atol':1e-6,
                          'expected_optimizer_owners':4 if procedure=='independent4' else 1})
    assert len(cases)==167
    packet={'schema':'reference-wrapper-real-data-smoke-v1','authorization_client_date':'2026-10-14',
            'human_authorization':'You have to do that','scientific_GO':False,
            'campaign_plan_hash':json.loads((HERE.parent/'SUMMARY.json').read_text())['plan_hash'],
            'two_initial_updates':True,'charged_resume_validation':{'third_uninterrupted':1,'third_resumed':1},
            'expected_case_count':len(cases),'TEST_scoring':False,'cases':cases}
    (HERE/'MANIFEST.json').write_text(json.dumps(packet,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'cases':len(cases),'node':56,'molecular':96,'link':15,'scientific_GO':False}))


if __name__=='__main__':main()
