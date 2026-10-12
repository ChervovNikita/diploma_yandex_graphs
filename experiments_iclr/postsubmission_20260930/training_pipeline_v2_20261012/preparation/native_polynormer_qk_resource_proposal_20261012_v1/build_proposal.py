"""Local-only exact native Q/K resource proposal; no numerical or remote APIs."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

from qualify_native_qk_resources import canonical,fingerprint,validate_proposal,H_SHA,Z_SHA

HERE=Path(__file__).resolve().parent
V=HERE.parents[1]
P=V.parent
MIRROR=V/'preparation/native_polynormer_qk_cpu_handoff_20261012_v1/allocation_artifacts'
SOURCE_DIRECTORY='native_polynormer_qk_resource_source_20261012_v1'


def metadata_receipt(path,relative):
    raw=path.read_bytes();return {'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    if (HERE/'PROPOSAL.json').exists():raise ValueError('Fresh immutable proposal required')
    packet='massive_native_polynormer_qk_packet_20261012_v2'
    plan=json.loads((MIRROR/packet/'PLAN.json').read_text())
    if plan['scientific_launch_enabled'] is not False:raise ValueError('Disabled source packet only')
    manifest=json.loads((MIRROR/'master_controller_source_20261012_v6/MANIFEST.json').read_text())
    dependencies={row['path']:row for row in manifest['files']}
    cases=[]
    for entry in plan['studies']:
        path=MIRROR/entry['path'];study=json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(study)!=entry['spec_hash'] or study['dataset'] not in ('wikics','amazon-photo'):raise ValueError('Exact new-data source study required')
        custody_rows=[r for r in study['source']['files'] if Path(r['path']).name=='CUSTODY.json']
        if len(custody_rows)!=1:raise ValueError('One actual prepared custody metadata receipt required')
        custody_path=V/'preparation/current_node_custody_metadata_root_20261012_v1'/study['dataset']/'CUSTODY.json'
        if metadata_receipt(custody_path,custody_rows[0]['path'])!=custody_rows[0]:raise ValueError('Current custody mirror changed')
        custody=json.loads(custody_path.read_text());counts={str(r['split_id']):r['role_counts']['train'] for r in custody['splits']}
        if counts['0']!=max(counts.values()):raise ValueError('Source split0 must cover maximum TRAIN count; no implicit provider graft')
        for arm in ('full_qk_be_init','original_gnnm'):
            matches=[r for r in study['runs'] if r['kind']=='model_fit' and r['config']['polynormer_qk_arm']==arm
                     and r['config']['stage']=='search' and r['seed']==78101 and r['body']==0
                     and r['config']['learning_rate']==.001 and (r['config']['in_dropout'],r['config']['dropout'],r['config']['global_dropout'])==(.5,.5,.5)]
            if len(matches)!=1:raise ValueError('Literal source anchor required')
            run=matches[0];c=run['config']
            adapter_path=run['adapter'].partition(':')[0]
            adapter=next(row for row in study['source']['files'] if row['path']==adapter_path)
            for row in study['source']['files']:
                if Path(row['path']).suffix in ('.py','.sh') or row in c['polynormer_qk_sources'].values():dependencies[row['path']]=row
            for branch in ('local','global'):
                cases.append({'case_id':'native-qk-'+study['dataset']+'-'+arm+'-'+branch,'dataset':study['dataset'],'arm':arm,'branch':branch,
                              'study_receipt':metadata_receipt(path,entry['path']),'study_hash':entry['spec_hash'],
                              'source_run_id':run['run_id'],'source_run_hash':fingerprint(run),'source_adapter_receipt':adapter,
                              'source_run_config':copy.deepcopy(c),'GRAPH_TRAIN_bindings':{k:c['node_inputs'][k] for k in ('graph','train')},
                              'provider_custody':custody_rows[0],'representative_mask':0,'train_rows':counts['0'],'all_mask_TRAIN_counts':counts,
                              'model_graph_shape':custody['model_graph'],
                              'shape_config':{k:c[k] for k in ('width','depth','global_layers','heads','members','classes')},
                              'expected_actions_per_cycle':{'model_forwards':8,'backward_calls':8,'autograd_grad_VJP_calls':0,'optimizer_attempts':1},
                              'warm_callback_steps':[1,1,1,1],'timeout_seconds':600,
                              'branch_status':'explicit fresh untrained geometry; never learned/selected',
                              'nested_local_snapshot_status':'missing; report actual None and same-shape tensor-byte estimate only',
                              'scientific_fits_allowed':0,'selected_metrics_allowed':0,'real_endpoint_targets_allowed':False,'automatic_retry':False})
    h=metadata_receipt(V/'qa/practical_zero_update_resources.py','private_source_zero_update_helpers_20261012_v2/runtime/qa/practical_zero_update_resources.py')
    z=metadata_receipt(V/'qa/private_source_zero_update_resources.py','private_source_zero_update_helpers_20261012_v2/runtime/qa/private_source_zero_update_resources.py')
    if h['sha256']!=H_SHA or z['sha256']!=Z_SHA:raise ValueError('Reviewed inherited h/z helper changed')
    dependencies[h['path']]=h;dependencies[z['path']]=z
    bindings={'schema':'native-polynormer-qk-resource-source-bindings-v1','role_optimizer_cap_helper':h,'tensor_commit_tree_helper':z,
              'frozen_controller_manifest':metadata_receipt(MIRROR/'master_controller_source_20261012_v6/MANIFEST.json','master_controller_source_20261012_v6/MANIFEST.json'),
              'runtime_package':'master_controller_source_20261012_v6/runtime','runtime_dependency_receipts':[dependencies[k] for k in sorted(dependencies)]}
    proposal={'schema':'native-polynormer-qk-resource-proposal-v1','requested_cases':cases,'execution_authorized':False,'scientific_GO':False,
              'max_actual_optimizer_updates':0,'actual_CUDA_executions':0,'all_cases_reuse_reviewed_h_z_guards':True,
              'original_five_excluded_for_every_method':True,'scientific_source_packet':{'path':packet+'/PLAN.json','fingerprint':fingerprint(plan)},
              'total_requested_callbacks':32,'total_requested_native_forwards':256,'total_requested_backwards':256,'total_denied_Adam_attempts':32,
              'summed_case_deadlines_seconds':4800,'all_fit_ETA':None,'separate_root_source_review_and_explicit_case_admission_required':True,
              'limits':['No optimizer kernels, learned model, selected metrics or real endpoint targets.',
                        'Missing learned nested local model/Adam snapshot remains unqualified; no fake selected checkpoint.',
                        'Full Q/K maximizes Q/K weight storage within fixed bodies; does not establish peak activation/runtime dominance for gamma/delta operators.',
                        'Observed allocator reservations, concurrent pairs and other host runtimes remain separate admissions.',
                        'No lower-shape arms or duplicate cases; full fixed native H512/L7/global2/head1 only.']}
    validate_proposal(proposal)
    for name,value in (('PROPOSAL.json',proposal),('SOURCE_BINDINGS.json',bindings)):
        (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    template={'schema':'root-approved-zero-update-resource-admission-v1','root_source_review_confirmed':False,
              'helper_sha256':hashlib.sha256((HERE/'qualify_native_qk_resources.py').read_bytes()).hexdigest(),
              'scientific_GO':False,'max_actual_optimizer_updates':0,'approved_case_ids':[],
              'proposal_receipt':metadata_receipt(HERE/'PROPOSAL.json',SOURCE_DIRECTORY+'/PROPOSAL.json'),**{k:v for k,v in bindings.items() if k!='schema'}}
    (HERE/'ENGINEERING_ADMISSION_TEMPLATE.json').write_text(json.dumps(template,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source_only':True,'cases':len(cases),'callbacks_requested':32,'actual_CUDA_executions':0,'fits':0,'GO':False}))


if __name__=='__main__':main()
