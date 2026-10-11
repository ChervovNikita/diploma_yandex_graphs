"""Reuse only the complete original peer reference cohort and its receipts."""
import copy
import json
from runtime import ARMS,BLOCKS,PRIOR_ARMS,checked,require

PEER_ARMS=('P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4')
PEER_MANIFEST_SHA='f83cb34bae29bda7bd542075bb3009d973fc7ab3bf494bd2cdfcf0801e704cd4'
PEER_BINDING_SHA='704783e8f71c2031da7e98b10849610a4811488debc6154550f51f844c4ac4c5'
IDENTITIES=('source','data','roles','initialization','native_dropout_rng','objective','optimizer','selector','stop','selected_restore','native_probability_exports','calibration','finite_owner')

def references(config):
    from reader import owner_closed
    owner_closed(config['prior_acquisition_owner_end']);owner_closed(config['prior_calibration_owner_end'])
    checked(config['prior_source_manifest']);checked(config['prior_binding'])
    require(config['prior_source_manifest']['sha256']==PEER_MANIFEST_SHA and config['prior_binding']['sha256']==PEER_BINDING_SHA,'Original sealed peer source/binding required')
    audit=json.loads(checked(config['reuse_audit']).read_text())
    receipts={k:config[k] for k in ('prior_panel','prior_acquisition_owner_end','prior_temperatures','prior_calibration_owner_end','prior_source_manifest','prior_binding','custody','source_manifest')}
    require(audit.get('complete') is True and audit.get('references')==receipts and audit.get('identities')=={k:True for k in IDENTITIES} and audit.get('prior_trajectories')==21 and audit.get('new_trajectories')==27,'Root complete baseline identity audit required')
    panel=json.loads(checked(config['prior_panel']).read_text());index={(r['block'],r['arm']):r for r in panel['records']}
    require(panel['complete'] and panel['terminal'] and len(panel['records'])==30 and set(index)=={(b,a) for b in BLOCKS for a in PEER_ARMS} and panel['role_custody']==config['custody'],'Original entire30bank peer family closed with exact roles')
    require((panel['banks'],panel['optimizer_trajectories'],panel['physical_backbones'],panel['exports'],panel['selected_optimizer_states'],panel['selected_physical_bodies'])==(30,48,57,30,48,57) and panel['serving']=='mean_class_probabilities' and panel['CAL_DEV_truth_read'] is False,'Original peer whole-family state/export closure')
    temperatures=json.loads(checked(config['prior_temperatures']).read_text());endpoints={(r['block'],r['arm']):r for r in temperatures['records']}
    require(temperatures['complete'] and temperatures['endpoints']==30 and temperatures['actual_scalar_updates']==15000 and len(temperatures['records'])==30 and set(endpoints)==set(index) and temperatures['panel']==config['prior_panel'] and temperatures['role_custody']==config['custody'] and temperatures['DEV_truth_read'] is False,'Original complete30endpoint CAL closure')
    for key,r in index.items():
        a=key[1];fits=4 if a in ('I4','IA4') else 1;bodies=4 if a in ('I4','IA4','UX4') else 1
        require(r['status']=='exported_complete' and r['probability_pool_authority'] and len(r['fits'])==fits and r['physical_backbones']==bodies,'Original every-bank selected/export status')
        for f in r['fits']:
            require(f['status']=='selected_complete' and f['selected_restore_verified'] and 1<=f['selected_step']<=f['completed_updates']<=100000,'Original complete selected-state restore')
            checked(f['checkpoint'])
        require(set(r['exports'])=={'cal','dev'},'Original complete exports')
        for item in r['exports'].values():checked(item)
        ep=endpoints[key]
        require(ep['status']=='fixed_endpoint_complete' and ep['updates']==500 and ep['endpoint']=='latest_tied_minimum_step0_to500' and 0<=ep['selected_step']<=500 and ep['cal_archive']==r['exports']['cal'],'Original fixed CAL endpoint')
    records=[];prior_endpoints={}
    for b in BLOCKS:
        for a in PRIOR_ARMS:
            row=copy.deepcopy(index[b,a]);row.update(origin='prior_peer_reference',original_panel=config['prior_panel'],original_source_manifest=config['prior_source_manifest'],original_acquisition_owner=config['prior_acquisition_owner_end'],reuse_audit=config['reuse_audit'])
            records.append(row);ep=copy.deepcopy(endpoints[b,a]);ep.update(origin='prior_peer_reference',original_temperatures=config['prior_temperatures'],original_calibration_owner=config['prior_calibration_owner_end'],reuse_audit=config['reuse_audit']);prior_endpoints[b,a]=ep
    require(len(records)==12 and sum(len(r['fits']) for r in records)==21 and sum(r['physical_backbones'] for r in records)==21,'Exactly12prior banks/21prior own trajectories and bodies')
    return dict(panel=panel,records=records,endpoints=prior_endpoints,receipts=receipts,reuse_audit=config['reuse_audit'])

def operation_counts(records):
    keys={k for r in records for f in r['fits'] for k in f['operation_counts']}
    return {k:sum(f['operation_counts'].get(k,0) for r in records for f in r['fits']) for k in sorted(keys)}
