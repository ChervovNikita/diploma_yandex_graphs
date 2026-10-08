from pathlib import Path
import socket,subprocess,hashlib,json,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_correction_full_WikiCS_role_projection_allocation_20261008_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not D.exists()
import torch,numpy as np
f=P/'wikics_official_acquisition_root_20261007_v1/available/wikics_split0.pt'
assert hashlib.sha256(f.read_bytes()).hexdigest()=='74868c33350221f3a38f26f156a72003f6ee943f6275fb87c205f6bc1cf775a5'
x=torch.load(f,map_location='cpu',weights_only=True)
assert set(x)=={'x','edge_index','train_ids','train_y','valid_ids','valid_y'}
assert x['x'].shape==(11701,300) and x['edge_index'].shape==(2,442907) and len(x['train_ids'])==580 and len(x['valid_ids'])==5274
D.mkdir()
np.savez(D/'train.npz',x=x['x'].numpy(),edge_index=x['edge_index'].numpy(),ids=x['train_ids'].numpy(),y=x['train_y'].numpy())
np.savez(D/'valid.npz',ids=x['valid_ids'].numpy(),y=x['valid_y'].numpy())
def bind(p):return {'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
meta={'schema':'internal-be-official-role-projection-v2','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'wikics','format':'NPZ_numeric_only','official_split_preserved':True,'split_index':0,'train_count':580,'valid_count':5274,'TEST_values_in_payload':False,'source_custody':{'safe_payload':bind(f),'official_manifest':bind(f.parents[1]/'AVAILABLE_MANIFEST.json'),'original_acquisition_route':'Authorized one-GPU allocation; existing official source','TEST_values_excluded_from_safe_payload':True},'payloads':{'train':bind(D/'train.npz'),'valid':bind(D/'valid.npz')},'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'GPU_initialized':torch.cuda.is_initialized()},'scientific_training':False,'comparative_scoring':False,'Mac_numeric_payload_copy':False}
assert not meta['runtime']['GPU_initialized']
(D/'PROJECTION_MANIFEST.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
