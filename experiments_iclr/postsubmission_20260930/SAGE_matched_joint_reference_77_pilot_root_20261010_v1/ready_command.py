import hashlib,json,os,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='peptide'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert uuids==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');P=R/'experiments_iclr/postsubmission_20260930';H=P/'SAGE_matched_joint_reference_77_pilot_root_20261010_v1'
assert Path(sys.executable).resolve()==Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python').resolve()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='ab8419011cebd09beedc35f3e756baab95995b7e'
import numpy as np,torch,torch_geometric
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
receipt=lambda q:dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q))
D=P/'label_correction_full_WikiCS_role_projection_allocation_20261008_v1'
roles={}
for name,expected in [('train',{'x','edge_index','ids','y'}),('valid',{'ids','y'})]:
 q=D/(name+'.npz')
 with np.load(q,allow_pickle=False) as a:
  assert set(a.files)==expected
  roles[name]=dict(file=receipt(q),arrays={k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype),C_bytes_sha256=hashlib.sha256(a[k].tobytes(order='C')).hexdigest()) for k in a.files})
assert roles['train']['arrays']['x']['shape']==[11701,300] and roles['train']['arrays']['ids']['shape']==[580] and roles['valid']['arrays']['ids']['shape']==[5274]
assert roles['valid']['arrays']['ids']['C_bytes_sha256']=='48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801'
assert roles['valid']['arrays']['y']['C_bytes_sha256']=='2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'
files={n:receipt(R/n) for n in ('models.py','run_base.py','run_common.py')}
for n in ('shared_fast_graph_model_interface_20261010_v1/common_routes.py','portable_internal_be_public_interface_20261007_v2/core/factors.py','SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py'):
 files[n]=receipt(P/n)
assert files['shared_fast_graph_model_interface_20261010_v1/common_routes.py']['sha256']=='84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773'
assert files['portable_internal_be_public_interface_20261007_v2/core/factors.py']['sha256']=='9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'
assert files['SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py']['sha256']=='5ea88c5eba68df512c20ea10ee778d25f46499705059a91cb80104c4fc8943c4'
from torch_geometric.nn import SAGEConv
import inspect
provider=Path(inspect.getfile(SAGEConv))
runtime=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,pyg=torch_geometric.__version__,sage_provider=receipt(provider),hostname=socket.gethostname())
fingerprint=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
value=dict(hostname=socket.gethostname(),gpu_uuids=uuids,gpu_resources=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines(),runtime=runtime,runtime_fingerprint_sha256=fingerprint,roles=roles,sources=files,TEST_access=False,no_quality_scored=True)
H.mkdir(exist_ok=True);q=H/'ACTUAL_READY_V1.json';assert not q.exists();q.write_text(json.dumps(value,indent=2)+'\n');print('F_READY_JSON='+json.dumps(value))
