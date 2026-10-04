"""Fresh native GNNM TRAIN-only full-shape component; no scientific driver call."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent
PHASE=ROOT.parent
REPO=PHASE.parent.parent
UUID='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
PHYSICAL_UUIDS=[UUID,'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
MANIFEST='fbe1aec5eafb029fce9c9d42ebf9a5b94b98cfbc2ce9881502f0e3fc8e192ca6'
PROTOCOL='9da2f5ebed5660d09b77ef4a620caa50bda81e4dc8ee999444aefc8f87e58524'
COVERAGE_SHA='0e05618da1fe4916eebd77ccadd26b8aa476fb23bd872f20b6fa5391c48bf74c'


def verify(record):
    path=PHASE/record['path']
    assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
    assert path.stat().st_size==record['bytes']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
    return path


def tree_difference(torch,left,right):
    result=dict(exact=True,close=True,maximum_absolute_error=0.0,tensors=0)
    def visit(a,b):
        if isinstance(a,torch.Tensor):
            assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype
            result['tensors']+=1;result['exact'] &= torch.equal(a,b)
            if a.is_floating_point() and a.numel():
                delta=float((a.to(torch.float64)-b.to(torch.float64)).abs().max())
                result['maximum_absolute_error']=max(result['maximum_absolute_error'],delta)
                result['close'] &= torch.allclose(a,b,rtol=5e-5,atol=5e-6)
            else:result['close'] &= torch.equal(a,b)
        elif isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b)
            for key in a:visit(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert type(a) is type(b) and len(a)==len(b)
            for x,y in zip(a,b):visit(x,y)
        else:assert a==b
    visit(left,right);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('local','global'),required=True);args=parser.parse_args()
    assert Path.cwd()==REPO and os.uname().nodename=='peptide'
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve()
    assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==PHYSICAL_UUIDS
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    source=ROOT/'staged_phase/accuracy_first_native_gnnm_reference_source_preparation_20261004_v2'
    sys.path.insert(0,str(source))
    from runtime import load_runtime
    from bank import build_bank,set_stage,snapshot,restore,rng_state,cpu_tree
    from bank_driver import eval_logits
    from native_driver import native_train_step
    from views import TrainRole,construct_views,native_edges
    rt=load_runtime(execute=True,manifest_sha256=MANIFEST,protocol_sha256=PROTOCOL)
    torch,np=rt.torch,rt.numpy
    import torch_geometric
    torch.set_num_threads(1)
    assert torch.cuda.device_count()==1
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    runtime=dict(python_executable=sys.executable,python_version=sys.version,torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,numpy_version=np.__version__,torch_geometric_version=torch_geometric.__version__,
        physical_gpu_UUIDs=PHYSICAL_UUIDS,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    runtime['package_descriptors']=[dict(name=name,path=module.__file__,bytes=Path(module.__file__).stat().st_size,
        sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        for name,module in [('torch',torch),('numpy',np),('torch_geometric',torch_geometric)]]
    expected_runtime={'python_executable': '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12', 'python_version': '3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0]', 'torch_version': '2.7.1', 'torch_cuda_version': '12.6', 'numpy_version': '1.26.4', 'torch_geometric_version': '2.4.0', 'package_descriptors': [{'name': 'torch', 'path': '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/lib/python3.12/site-packages/torch/__init__.py', 'bytes': 100862, 'sha256': '9801e75447c7f585545f989f8a21940b60e0c4cc888effb2f08e739664b4b904'}, {'name': 'numpy', 'path': '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/lib/python3.12/site-packages/numpy/__init__.py', 'bytes': 17005, 'sha256': '22cd1535fa14d74ef6f457cca149ffdc80875f460be313b8f895273f78bc402e'}, {'name': 'torch_geometric', 'path': '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/lib/python3.12/site-packages/torch_geometric/__init__.py', 'bytes': 1043, 'sha256': '949147cecf1e6d7d1858f4f13a0008a0172084177932d6812163c664f70e59db'}]}
    assert all(runtime[k]==v for k,v in expected_runtime.items())
    bindings=json.loads((ROOT/'INPUT_BINDINGS.json').read_text())
    data=json.loads(verify(bindings['manifest']).read_text())
    assert data['public_graph']==bindings['public_graph'] and data['train_labels']==bindings['train_labels']
    with np.load(verify(data['public_graph']),allow_pickle=False) as archive:
        assert set(archive.files)=={'features','edge_index','train_mask','val_mask','test_mask'}
        x_cpu=archive['features'].copy();raw=archive['edge_index'].copy()
        ids=tuple(int(i) for i in np.flatnonzero(archive['train_mask'][:,0]))
        val_ids=tuple(int(i) for i in np.flatnonzero(archive['val_mask'][:,0]))
        test_ids=tuple(int(i) for i in np.flatnonzero(archive['test_mask'][:,0]))
    with np.load(verify(data['train_labels']['0']),allow_pickle=False) as archive:
        assert set(archive.files)=={'ids','labels'} and np.array_equal(archive['ids'],ids)
        labels=tuple(int(y) for y in archive['labels'])
    assert x_cpu.shape==(24492,300) and x_cpu.dtype==np.float32 and np.isfinite(x_cpu).all()
    assert len(ids)==12246 and len(val_ids)==len(test_ids)==6123
    role=TrainRole(24492,0,ids,labels,val_ids,test_ids)
    bundle=construct_views(role,native_edges(24492,((int(a),int(b)) for a,b in raw.T)),17)
    coverage_path=ROOT/'staged_phase/graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json'
    assert hashlib.sha256(coverage_path.read_bytes()).hexdigest()==COVERAGE_SHA
    saved=json.loads(coverage_path.read_text())
    assert saved==dict(bundle['coverage'],coverage_origin='official_TRAIN',scientific_freeze_eligible=bundle['coverage']['coverage_eligible'])
    assert saved['scientific_freeze_eligible']
    x=torch.from_numpy(x_cpu).to(device)
    native=torch.tensor(bundle['native'],dtype=torch.long,device=device).t().contiguous()
    torch.cuda.reset_peak_memory_stats()
    model,optimizer,_=build_bank(rt,'tied_persistent',17,device)
    set_stage(model,args.stage=='global')
    update=1 if args.stage=='local' else 201
    binding=dict(engineering_only=True,reference='ordinary_native_gnnm_full_TRAIN',stage=args.stage,
        manifest_sha256=MANIFEST,protocol_sha256=PROTOCOL,role=role.identity(),native_edges_sha256=bundle['coverage']['native_edges_sha256'])
    selection=dict(global_=args.stage=='global',actual_update=update);selection['global']=selection.pop('global_')
    def step(u):return native_train_step(rt,model,optimizer,x,native,ids,labels,u)
    torch.cuda.synchronize();started=time.perf_counter();_=step(update);torch.cuda.synchronize();step_seconds=time.perf_counter()-started
    started=time.perf_counter();expected=eval_logits(rt,model,x,native).detach().cpu();torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-started
    started=time.perf_counter();image=snapshot(rt,model,optimizer,device,selection,binding);snapshot_seconds=time.perf_counter()-started
    started=time.perf_counter();restore(rt,model,optimizer,device,image,binding);actual=eval_logits(rt,model,x,native).detach().cpu();torch.cuda.synchronize();restore_seconds=time.perf_counter()-started
    function_replay=tree_difference(torch,expected,actual)
    restore(rt,model,optimizer,device,image,binding);_=step(update+1)
    first=cpu_tree(rt,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),gradients={n:p.grad for n,p in model.named_parameters()},rng=rng_state(rt,device)))
    restore(rt,model,optimizer,device,image,binding);_=step(update+1)
    second=cpu_tree(rt,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),gradients={n:p.grad for n,p in model.named_parameters()},rng=rng_state(rt,device)))
    next_update_replay=tree_difference(torch,first,second);torch.cuda.synchronize()
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),family='native_reference',condition='ordinary_native_gnnm',stage=args.stage,
        graph_shape=list(x.shape),source_manifest_sha256=rt.manifest_sha256,protocol_sha256=rt.protocol_sha256,
        source_provenance=list(rt.source_provenance),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_bindings_sha256=hashlib.sha256((ROOT/'INPUT_BINDINGS.json').read_bytes()).hexdigest(),coverage_sha256=COVERAGE_SHA,
        selected_device='cuda:0',GPU_UUID=UUID,runtime=runtime,parameters=sum(p.numel() for p in model.parameters()),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),step_seconds=step_seconds,evaluation_seconds=evaluation_seconds,
        snapshot_seconds=snapshot_seconds,restore_and_evaluation_seconds=restore_seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training_updates_checked=3,training_forwards_and_backwards=12,native_evaluation_forwards=8,
        function_replay=function_replay,next_update_replay=next_update_replay,
        exact_function_requirement_met=function_replay['exact'],implementation_close=function_replay['close'] and next_update_replay['close'],
        independent_scientific_result=False,predictive_values_reported=False,VALIDATION_or_TEST_targets_read=False,
        scientific_training_updates=0,checkpoint_or_logits_saved=False,eligible_as_donor=False,
        component_stage_fresh_initialization=True,co_resident_with_DDI_queue=True,exact_whole_schedule_or_competence_qualification=False)
    with (ROOT/('native_reference_ordinary_native_gnnm_'+args.stage+'.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except BaseException:print(traceback.format_exc(),file=sys.stderr);raise
