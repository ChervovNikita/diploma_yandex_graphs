from pathlib import Path
import sys,json,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
sys.path.insert(0,str(R/'.venv/lib/python3.11/site-packages'))
import torch
assert torch.__version__=='2.1.2+cu118' and not torch.cuda.is_initialized()
scope=json.loads((P/'amazon_G0_complete_comparison_activation_root_20261006_v2/COMPARISON_SCOPE.json').read_bytes())
for name,rows in scope['ordinary'].items():
 f=P/rows['endpoint']['path'];h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 assert f.stat().st_size==rows['endpoint']['bytes'] and h.hexdigest()==rows['endpoint']['sha256']
 image=torch.load(f,map_location='cpu',weights_only=True)
 summary={k:v for k,v in image.items() if isinstance(v,(str,int,float,bool,type(None)))}
 summary['top_level_keys']=list(image);summary['simple_lists']={k:v for k,v in image.items() if isinstance(v,(list,tuple)) and all(isinstance(a,(str,int,float,bool,type(None))) for a in v)}
 print(json.dumps({'endpoint':name,'scalar_metadata':summary,'recipe_equal':image.get('recipe')==json.loads((P/scope['candidate_complete']['path']).read_bytes())['recipe'],'Torch_CUDA_initialized':torch.cuda.is_initialized(),'predictions_or_labels_computed':False}),flush=True)
 del image
