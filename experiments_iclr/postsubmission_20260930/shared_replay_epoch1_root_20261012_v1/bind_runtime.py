
import hashlib,importlib.metadata,json,os,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930';out=Path(__file__).resolve().parent
import torch,numpy,scipy,torch_sparse,torch_scatter
assert Path(sys.executable).resolve()==Path(repo/'.venv/bin/python').resolve()
def pin(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
profile={'deterministic':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32}
binary=[]
for p in (Path(torch.__file__).parent/'_C.cpython-311-x86_64-linux-gnu.so',Path(numpy.__file__).parent/'core/_multiarray_umath.cpython-311-x86_64-linux-gnu.so'):
 assert p.is_file();binary.append(pin(p))
for module in (torch_sparse,torch_scatter):
 for p in sorted(Path(module.__file__).parent.glob('*cuda.so')):binary.append(pin(p))
assert len(binary)>=4
runtime={'interpreter':str(repo/'.venv/bin/python'),'interpreter_pin':{k:v for k,v in pin(sys.executable).items() if k!='path'},'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','torch-geometric','torch-sparse','torch-scatter']},'binary_file_pins':binary,'profile':profile}
host={'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),'disk':dict(zip(['total','used','free'],__import__('shutil').disk_usage(repo))), 'MemAvailable_kib':int([x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0])}
p=phase/'private_GRU_acquisition_root_20261011_v1/owner_v1';start=json.loads((p/'OWNER_START.json').read_text());observed={}
for k in ('owner','child'):
 ident=start[k];f=Path('/proc')/str(ident['pid'])/'stat'
 if f.exists():
  a=f.read_text().rsplit(')',1)[1].split();actual={'pid':ident['pid'],'start_ticks':int(a[19]),'pgid':int(a[2])};assert all(actual[z]==ident[z] for z in actual);observed[k]=actual
 else:assert (p/'OWNER_END.json').exists();observed[k]=None
result={'runtime':runtime,'host':host,'Physics':observed,'models':0,'optimizers':0,'scientific_fits':0,'GO':False}
(out/'RUNTIME_BINDING.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
