from pathlib import Path
import importlib,importlib.util,json,socket,subprocess,sys
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
out={'python':sys.executable,'modules':{},'models_or_data_loaded':False}
for name in ('torch','numpy','scipy','torch_geometric','torch_sparse','torch_scatter','torch_householder'):
 try:
  m=importlib.import_module(name);out['modules'][name]={'version':getattr(m,'__version__',None),'file':getattr(m,'__file__',None)}
 except Exception as e:out['modules'][name]={'error':type(e).__name__+': '+str(e)}
print(json.dumps(out))
