"""Portable bounded real-work body; original public core/driver stay unchanged."""
import argparse,ast,gc,hashlib,importlib.util,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','queue-source','train','valid','geometry','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args();started=time.monotonic()
    assert sha(args.source/'MANIFEST.json')=='190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
    for row in json.loads((args.source/'MANIFEST.json').read_text())['files']:
        path=args.source/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    assert sha(args.queue_source/'MANIFEST.json')=='fdc9dadadac324f0760983adc0bf16a17b88dea4ef5012ae0cbb01bc0ef2faa3'
    sys.path.insert(0,str(args.source))
    from portable import Session
    from data_interface import load_train_valid,validation_batches,_data
    ops=module(args.source/'train.py','_public_train_ops');queue=module(args.queue_source/'queue.py','_public_queue_metadata')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    device=torch.device(args.device);torch.cuda.set_device(device)
    train,valid,data=load_train_valid('molhiv',args.train,args.valid)
    data={key:data[key] for key in ('train_npz_sha256','valid_npz_sha256')}
    geometry=json.loads(args.geometry.read_text());selected=geometry['global_max_nodes']
    assert geometry['complete'] and geometry['batches_scanned']==77400 and not geometry['label_based_selection']
    assert (selected['seed'],selected['epoch'],selected['batch_index_zero_based'],selected['graphs'],selected['nodes'],selected['edges'])==(6101,68,166,128,3948,8404)
    order=torch.randperm(len(train['ids']),generator=torch.Generator().manual_seed(6101+19709+1000*68))
    positions=order[166*128:167*128];assert positions.tolist()==selected['positions']
    args.output.mkdir(exist_ok=False)
    tree=ast.parse((args.source/'train.py').read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=next(n for n in function.body if isinstance(n,ast.Try)).body
    epoch_loop=next(n for n in body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    selector=[]
    for node in epoch_loop.body:
        if not isinstance(node,ast.If):continue
        names={n.id for n in ast.walk(node.test) if isinstance(n,ast.Name)}
        if 'best' in names or 'best_local' in names or ast.unparse(node.test)=='session.model.independent':selector.append(node)
    assert len(selector)==3
    select_code=compile(ast.Module(body=selector,type_ignores=[]),str(args.source/'train.py'),'exec')
    bank=next(n for n in body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='ordinary_independent')
    bank_code=compile(ast.Module(body=[bank],type_ignores=[]),str(args.source/'train.py'),'exec')
    runtime=queue.runtime_identity();physical=queue.gpu_observation(args.device);cases=[]
    def case(arm):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);began=time.monotonic()
        session=Session('molhiv',arm,6101,args.device)
        assert all(parameter.dtype==torch.float32 for parameter in session.model.parameters() if parameter.is_floating_point())
        directory=args.output/arm;directory.mkdir()
        batch,labels=_data().molecular_batch(train,positions,device)
        assert batch['graph'].num_graphs==128 and batch['graph'].num_nodes==3948 and batch['graph'].edge_index.shape[1]==8404
        before={name:parameter.detach().cpu().clone() for name,parameter in session.model.named_parameters()}
        private={id(parameter) for layer in session.model.modules() if isinstance(layer,session.core['factors'].FactorLinear) for parameter in (layer.r,layer.s)}
        private.update(id(parameter) for name,parameter in session.model.named_parameters() if name.endswith('message_factors'))
        session.train_step(batch,labels)
        changed_private=changed_common=0;member_changes=[False]*len(session.model.models)
        for name,parameter in session.model.named_parameters():
            changed=not torch.equal(before[name],parameter.detach().cpu())
            if changed:
                if id(parameter) in private:changed_private+=1
                else:changed_common+=1
                member_changes[int(name.split('.')[1])]=True
        assert changed_common>0 and all(member_changes)
        if arm=='be_init_contrastive':assert changed_private>0
        del before,batch,labels
        metric,per=ops.evaluate(session,train,valid)  # Complete VALID; values never leave private artifacts.
        run={'resource_only':True,'data':data,'source_manifest_sha256':sha(args.source/'MANIFEST.json')}
        ns=dict(session=session,args=SimpleNamespace(task='molhiv',output=directory),torch=torch,
                ordinary_independent=arm=='independent4',metric=metric,per=per,epoch=1,best=-float('inf'),
                best_local=-float('inf'),own_best=[-float('inf')]*session.model.members,
                own_local=[-float('inf')]*session.model.members,run=run,config=session.config,
                joint_snapshot=ops.joint_snapshot,json_write=ops.json_write,evaluate=ops.evaluate,train=train,valid=valid)
        exec(select_code,ns)
        chosen={path.name:sha(path) for path in directory.glob('*.pt')}
        if arm=='independent4':assert 'selected.pt' not in chosen and len(chosen)==4
        else:assert 'selected.pt' in chosen
        ns['epoch']=2;exec(select_code,ns)  # Cached complete-VALID tie tests strict first maximum; no epoch2 fit claim.
        assert chosen=={path.name:sha(path) for path in directory.glob('*.pt')}
        exec(bank_code,ns)
        saved=torch.load(directory/'selected.pt',map_location=device,weights_only=False)
        session.model.load_state_dict(saved['model'])
        if arm=='independent4':assert saved['evaluation_only'] and 'optimizers' not in saved
        else:
            assert len(saved['optimizers'])==len(session.optimizers)
            for optimizer,state in zip(session.optimizers,saved['optimizers']):optimizer.load_state_dict(state)
        session.core['selection'].finite_state(session.model,session.optimizers)
        session.model.eval();served=0
        with torch.no_grad():
            for batch,labels in validation_batches('molhiv',train,valid,device):
                logits,_=session.forward(batch);session.core['selection'].finite_predictions(logits,session.serving(logits));served+=logits.shape[1]
        assert served==4113
        torch.cuda.synchronize(device)
        measured={'arm':arm,'seed':6101,'source_batch_epoch':68,'source_batch_index_zero_based':166,
            'fresh_model_updates':1,'TRAIN_graphs':128,'TRAIN_nodes':3948,'TRAIN_edges':8404,'members':session.model.members,
            'own_views':2,'VALID_graphs':4113,'VALID_batches':33,'reload_served_graphs':served,
            'changed_common_parameter_tensors':changed_common,'changed_private_parameter_tensors':changed_private,
            'all_own_bodies_changed':all(member_changes),'strict_cached_VALID_tie_keeps_first_checkpoint':True,
            'joint_or_own_selector_branches_from_unchanged_train_py':True,
            'complete_TRAIN_update_VALID_snapshot_reload_serving':True,'quality_values_closed':True,
            'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'seconds':time.monotonic()-began,'selected_file_sha256':sha(directory/'selected.pt'),
            'artifact_storage_bytes':sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
            'evaluation_only_bank':arm=='independent4'}
        return measured
    stage='representative_cases'
    try:
        for arm in ('single','be_init_contrastive','independent4'):
            cases.append(case(arm));gc.collect();torch.cuda.empty_cache()
        candidate={'schema':'portable-representative-work-candidate-v1','complete':True,'task':'molhiv',
            'source_manifest_sha256':sha(args.source/'MANIFEST.json'),'data':data,'runtime':runtime,
            'physical_gpu_uuid':physical['physical_uuid'],'cases':cases,
            'peak_GPU_bytes':max(max(row['peak_CUDA_allocated_bytes'],row['peak_CUDA_reserved_bytes']) for row in cases),
            'inclusive_worker_seconds':time.monotonic()-started,'peak_worker_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'geometry_sha256':sha(args.geometry),'max_node_batch_qualified':True,'global_max_edge_batch_qualified':False,
            'unqualified_max_edge_excess':26,'scientific_fit':False,'quality_values_closed':True,
            'resource_weights_used_as_fit_start':False,'TEST_scoring':False,'automatic_retry':False}
        write(args.output/'CANDIDATE.json',candidate)
        print(json.dumps({'complete':True,'cases':len(cases),'quality_values_closed':True}))
    except BaseException as error:
        write(args.output/'FAILURE.json',{'complete':False,'stage':stage,'error_type':type(error).__name__,
            'error':str(error),'completed_cases':cases,'seconds':time.monotonic()-started,'automatic_retry':False})
        raise


if __name__=='__main__':main()
