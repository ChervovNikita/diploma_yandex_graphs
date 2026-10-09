"""Inactive original18 metadata/cost/hash receipt builder; never inspect scores/arrays."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
OWNER=(563273,6035067559)
WORKER=(563276,6035067565)
BOOT='24c315a7-3c08-471f-b550-b9a3e1faf75d'
VANILLA='bc6264234cc003c5e89cc46bedf045e0edfbf097ec35b79a8d194fd0c3d11a22'
SEEDS=[7409,8501,9607]


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def bound(row):
    path=Path(row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and sha(path)==row['sha256'],'Exact closed status/custody input')
    return path


def artifact(path,label):
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to(PHASE),'Artifacts stay in research custody')
    return dict(label=label,path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def closure(cfg):
    """Only custody/status metadata is read until actual original process closure."""
    require(socket.gethostname()=='anogena-2-0','Existing normal host only')
    custody=read(bound(cfg['original18_custody']))
    require(custody['release_owner']=='root' and custody['all18_complete_verified'] is True
            and custody['comparative_outcomes_opened'] is False,'Root status-only original18 closure, outcomes held until21')
    require((custody['parent_pid'],custody['parent_birth'])==OWNER and (custody['worker_pid'],custody['worker_birth'])==WORKER
            and custody['boot_id']==BOOT and custody['parent_group']==OWNER[0] and custody['worker_group']==WORKER[0],
            'Exact original science owner/worker/birth/boot/groups')
    require(all(custody[key] is True for key in ('actual_parent_absent','actual_parent_group_absent','actual_worker_absent','actual_worker_group_absent',
            'actual_worker_CUDA_absent','reaped')),'Actual original parent/worker/groups/CUDA absent and reaped')
    terminal=read(bound(custody['terminal']));launch=read(bound(custody['launch']))
    require(terminal['complete'] is True and terminal['exit_code']==0 and terminal['reaped'] is True
            and terminal['actual_worker_absent'] is True and terminal['actual_worker_CUDA_absent'] is True
            and terminal['child']['pid']==WORKER[0] and terminal['child']['start_ticks']==WORKER[1] and terminal['child']['boot_id']==BOOT
            and launch['parent']['pid']==OWNER[0] and launch['parent']['start_ticks']==OWNER[1] and launch['parent']['boot_id']==BOOT,
            'Exact successful launch/terminal metadata')
    require(not Path('/proc',str(OWNER[0])).exists() and not Path('/proc',str(WORKER[0])).exists(),'Fresh actual original PID absence before header reads')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:raw=(path/'stat').read_text()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        values=raw[raw.rfind(')')+2:].split()
        require(int(values[2]) not in (OWNER[0],WORKER[0]),'Fresh actual original process groups absent')
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    require(not any(len(parts)>1 and parts[1].strip() in (str(OWNER[0]),str(WORKER[0])) for parts in (row.split(',') for row in rows)),
            'Fresh actual original CUDA PID absence; other jobs untouched')
    return custody


def cost_name(name):
    return any(word in name for word in ('seconds','bytes','cost','forwards','constructors','backwards','Adam'))


def costs(record):
    """Explicit shallow whitelist; no metric/output/diagnostic dictionaries traversed."""
    result={}
    def scalar(prefix,values):
        for name,value in values.items():
            target=prefix+name
            if cost_name(target) and isinstance(value,(int,float,bool)) and value is not None:result[target]=value
    scalar('',record)
    if isinstance(record.get('counters'),dict):scalar('counters.',record['counters'])
    if record['kind']=='independent_member':
        native=record['native_result'];scalar('native_result.',native)
        scalar('native_result.forward_counts.',native['forward_counts'])
        scalar('native_result.parameters.',native['parameters'])
    return result


def key(row):return row['kind'],row['base_seed'],row.get('member',-1)


def build(cfg,output):
    require(cfg['enabled'] is True and cfg['release_owner']=='root' and cfg['action']=='original18_origins_then_centered3'
            and cfg['builder_source_sha256']==sha(__file__),'Exact explicit continuation approval')
    custody=closure(cfg)
    complete_path=bound(cfg['original_complete']);complete=read(complete_path)
    expected={('shared_fit',seed,-1) for seed in SEEDS}|{('independent_pool',seed,-1) for seed in SEEDS}|{('independent_member',seed,m) for seed in SEEDS for m in range(4)}
    require(complete['complete'] is True and len(complete['records'])==18 and {key(row) for row in complete['records']}==expected
            and all(row['status']=='complete' for row in complete['records']),'All original18 status headers complete; no survivor receipt')
    root=complete_path.parent
    require(root.name=='geometry_only_core_scientific18_execution_root_20261009_v1','Exact original18 output custody')
    rows=[]
    for kind,seed,member in sorted(expected):
        if kind=='shared_fit':folder=root/'shared'/('seed'+str(seed));path=folder/'RESULT.json'
        else:folder=root/'independent'/('base'+str(seed));path=folder/('POOL_RESULT.json' if kind=='independent_pool' else 'MEMBER'+str(member)+'.json')
        require(path.is_file(),'Every original component result header retained')
        record=read(path)
        require(key(record)==(kind,seed,member) and record['status']=='complete','Exact complete component header')
        identity=(record['native_result']['identity'] if kind=='independent_member' else record['identity'] if kind=='shared_fit'
                  else record['components'][0]['native_result']['identity'])
        require(identity['source_seal_sha256']==VANILLA and identity['configuration']['id']=='d4_f16_L4'
                and identity['execution_source_commit']=='e80b02c44a926bb29f348af62f4f47f35215bb4e'
                and identity['admission_receipt_sha256']=='67d89b4486d23306dd887213f0700b2225fdee4cd9a3cd40e8bb3b7c6661b538'
                and identity['role_archive_sha256']==cfg['roles_sha256'] and identity['role_metadata_sha256']==cfg['role_metadata_sha256'],
            'Same original source/admission/config/runtime/role origins')
        require(identity['runtime_versions']==cfg['expected_runtime_versions'] and identity['device']=='cuda:0'
                and identity['deterministic_algorithms'] is False and identity['allow_tf32'] is False,'Exact qualified original runtime policy')
        artifacts=[]
        selected=None
        if kind=='independent_member':
            require(record['seed']==seed+1000003*member and record['reused'] is False and record['complete_native_encoder_head_and_optimizer_owned'] is True
                    and record['no_learned_parameter_sharing'] is True,'Exact newly fitted complete independent body, no historical reuse')
            native_folder=folder/('native_single__d4_f16_L4__seed'+str(record['seed']))
            require(Path(record['checkpoint_path']).resolve()==(native_folder/'SELECTED_STATE.pt').resolve(),'Own exact selected state path')
            native=record['native_result'];selected=native['selected_epoch']
            require(native['status']=='complete' and 1<=selected<=native['epochs_completed']<=500,'Complete own selected-epoch header')
            own=read(native_folder/'RESULT.json')
            require(own['status']=='complete' and own['seed']==record['seed'] and own['selected_epoch']==selected
                    and own['identity']==identity and own['checkpoint_sha256']==record['checkpoint_sha256'],
                    'Same exact independently fitted native result headers')
            artifacts=[artifact(native_folder/'RESULT.json','native_result'),artifact(native_folder/'HISTORY.jsonl','history'),artifact(native_folder/'SELECTED_STATE.pt','checkpoint')]
        elif kind=='shared_fit':
            selected=record['selected_epoch'];require(1<=selected<=record['epochs_completed']<=500,'Complete shared selected-epoch header')
            artifacts=[artifact(folder/'HISTORY.jsonl','history'),artifact(folder/'SELECTED_STATE.pt','checkpoint'),artifact(folder/'SELECTED_SERVING_REFERENCE.pt','serving_reference')]
        else:
            require(record['trained_independent_bodies']==4 and record['native_serving_full_paths']==4 and len(record['member_bindings'])==4
                    and [row['member'] for row in record['member_bindings']]==list(range(4)),'Exactly four own independent selected-serving paths')
            artifacts=[artifact(folder/'SELECTED_SERVING_REFERENCE.pt','serving_reference')]
        by_label={item['label']:item for item in artifacts}
        if 'checkpoint' in by_label:require(by_label['checkpoint']['sha256']==record['checkpoint_sha256'],'Owned exact selected checkpoint hash')
        if 'serving_reference' in by_label:require(by_label['serving_reference']['sha256']==record['serving_reference_sha256'],'Actual exact fresh-serving reference hash')
        if kind=='independent_pool':
            for item in record['member_bindings']:
                source=read(folder/('MEMBER'+str(item['member'])+'.json'))
                require(item['seed']==seed+1000003*item['member'] and item['checkpoint_sha256']==source['checkpoint_sha256']
                        and item['own_selected_epoch']==source['native_result']['selected_epoch'],'Pool references these exact complete own selected states')
        row=dict(kind=kind,base_seed=seed,status='complete',result=artifact(path,'result'),artifacts=artifacts,
            cost_fields=costs(record),costs_charged_in_full=True,selected_epoch=selected,
            identity=identity)
        if kind=='independent_member':row.update(member=member,seed=record['seed'])
        rows.append(row)
    result=dict(schema='complete-original18-cost-and-paired-origin-receipt-v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_complete=artifact(complete_path,'original_complete'),records=rows,comparative_outcomes_unopened=True,all_original_costs_charged=True,
        builder_source_sha256=sha(__file__),
        original_custody_sha256=cfg['original18_custody']['sha256'],metadata_only=True,metric_fields_accessed=False,
        arrays_or_checkpoint_contents_loaded=False,model_forwards=0,new_independent_fits=0,reference_reuse='exact12 bodies once for centered3; source control costs retained',
        prior_original_terminal_cost=custody['terminal'])
    output=Path(output).resolve()
    require(output.is_relative_to(PHASE) and not output.is_relative_to(root) and not output.exists(),'Fresh origin receipt outside original results')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return dict(path=str(output),sha256=sha(output),bytes=output.stat().st_size,records=18,independent_bodies=12,scores_exposed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--release',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute:print(json.dumps(dict(inactive=True,numerical_or_score_or_array_access=False)));return
    require(args.release and args.output,'Explicit approved continuation metadata and fresh receipt path required')
    print(json.dumps(build(read(args.release),args.output)))


if __name__=='__main__':main()
