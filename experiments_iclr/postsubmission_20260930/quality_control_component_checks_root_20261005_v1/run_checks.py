"""Stage reviewed source and run one bounded CPU correctness check per identity."""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
REMOTE_PHASE=REPO+'/experiments_iclr/postsubmission_20260930'
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
HELPER=PHASE/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'


def single_body():
    return '''import ast,importlib.util,sys
from pathlib import Path
phase=Path(PHASE_PATH)
prototype=phase/'graph_ncNC_member_completion_qualification_preparation_20261003_v2'
pilot=phase/'exact_cb_support_bucket_paired_predictive_preparation_20261004_v1'
control=phase/'joint_pattern_capable_single_control_source_20261005_v1'
sys.path.insert(0,str(pilot));sys.path.insert(0,str(prototype));sys.path.insert(0,str(control))
import native_reference,graph_ops,pilot_model,pattern_teacher,conditional_loss
native,utils=native_reference.native_modules()
p=phase/'graph_ncNC_collab_predictive_pilot_design_20261003_v1/design_spec.py'
spec=importlib.util.spec_from_file_location('bounded_control_design',p)
design=importlib.util.module_from_spec(spec);sys.modules[spec.name]=design;spec.loader.exec_module(design)
mods=dict(native=native,native_utils=utils,design=design)
# Load only the exact native-adjacency function, not the unrelated count model.
p=phase/'ncnc_cardinality_single_comparator_implementation_preparation_20261003_v1/cardinality_model.py'
source=p.read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_adjacency')
ns=dict(torch=torch)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
from component_check import check_components
result=check_components(mods,graph_ops,pilot_model.make_native,ns['native_adjacency'],pattern_teacher.ObservationTeacher,conditional_loss)
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=('native','single'));parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args();assert args.attempt>0
    scope=json.loads((HERE/'CHECK_SCOPE.json').read_text())
    wanted=[r for r in scope['sources'] if ('native_fixed_view' in r['path'])==(args.kind=='native')]
    payloads=[]
    for row in wanted:
        raw=(PHASE/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        encoded=base64.b64encode(raw).decode()
        payloads.append(dict(row,chunks=[encoded[i:i+36000] for i in range(0,len(encoded),36000)]))
    body='''import json,os,resource,time,traceback
started=time.perf_counter();cpu=time.process_time()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
assert not torch.cuda.is_available() and torch.cuda.device_count()==0
'''
    if args.kind=='native':
        body+='''import importlib.util,sys
from pathlib import Path
p=Path(PHASE_PATH)/'native_fixed_view_independent_control_source_20261005_v1/control.py'
spec=importlib.util.spec_from_file_location('bounded_native_control',p)
control=importlib.util.module_from_spec(spec);sys.modules[spec.name]=control;spec.loader.exec_module(control)
rt=control.load_runtime(numerical=True)
result=control.verify_cpu_components(rt)
'''
    else:
        records=json.loads((PHASE/'joint_pattern_capable_single_control_source_20261005_v1/SOURCE_BINDINGS.json').read_text())['records']
        names={'pilot_model.py','pattern_teacher.py','conditional_loss.py','prototype.py','graph_ops.py','native_reference.py','model.py','utils.py','design_spec.py','cardinality_model.py'}
        records=[r for r in records if Path(r['path']).name in names]
        body+='import hashlib\nfrom pathlib import Path\n'
        body+='for row in '+repr(records)+':\n p=Path(PHASE_PATH)/row["path"];raw=p.read_bytes();assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n'
        body+=single_body()
    body+='''print(json.dumps(dict(result=result,elapsed_wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CUDA_available=torch.cuda.is_available(),CUDA_device_count=torch.cuda.device_count(),torch_version=str(torch.__version__),optimizer_steps=0,predictive_fits=0,real_dataset_checkpoint_outcome_reads=False)))
'''
    body='PHASE_PATH='+repr(REMOTE_PHASE)+'\n'+body
    outer='''from pathlib import Path
import base64,hashlib,json,os,subprocess,time
repo=Path(REPO_PATH);phase=Path(PHASE_PATH);assert Path.cwd().resolve()==repo.resolve()
assert subprocess.run(['hostname'],capture_output=True,text=True,check=True).stdout.strip()=='peptide'
uuids=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert uuids==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
for row in PAYLOADS:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase.resolve()) and not p.is_symlink()
 raw=base64.b64decode(''.join(row['chunks']),validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'.gnnm_runtime/buddy_extra_v1/site'))
started=time.perf_counter()
try:
 r=subprocess.run([PYTHON_PATH,'-B','-c',NUMERICAL_CODE],cwd=repo,env=env,capture_output=True,text=True,timeout=60)
 value=dict(physical_exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_seconds=time.perf_counter()-started,observation_timeout=False)
 if r.returncode==0:value['numeric_result']=json.loads(r.stdout.splitlines()[-1])
except subprocess.TimeoutExpired as e:
 value=dict(physical_exit_code=None,exception='CPU_component_timeout',stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.perf_counter()-started,observation_timeout=True,automatic_retry=False)
print(json.dumps(value))
'''
    outer='REPO_PATH='+repr(REPO)+'\nPHASE_PATH='+repr(REMOTE_PHASE)+'\nPYTHON_PATH='+repr(PYTHON)+'\nPAYLOADS='+repr(payloads)+'\nNUMERICAL_CODE='+repr(body)+'\n'+outer
    assert len(outer.encode())<95000
    identity='quality_'+args.kind+'_CPU_components_once_20261005_v%d'%args.attempt
    (HERE/(identity+'_NUMERICAL_SOURCE.py.txt')).write_text(body)
    spec=importlib.util.spec_from_file_location('bounded_control_transport',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.HERE=HERE
    result=helper.run(identity,outer)
    with (HERE/(identity+'_RESULT.json')).open('x') as h:json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
