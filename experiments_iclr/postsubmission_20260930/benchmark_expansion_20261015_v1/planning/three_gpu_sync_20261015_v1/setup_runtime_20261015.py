import subprocess,json,os,time,sys
from pathlib import Path
R=Path(__file__).resolve().parent;u=R/'.tools/uv';u.chmod(0o755);env=dict(os.environ,UV_PYTHON_INSTALL_DIR=str(R/'.tools/python'),UV_CACHE_DIR=str(R/'.cache/uv'));start=time.time();state={'commands':[],'status':'running'}
commands=[[str(u),'venv','--python','3.11.14',str(R/'.venv')],[str(u),'pip','install','--python',str(R/'.venv/bin/python'),'torch==2.1.2+cu118','--index-url','https://download.pytorch.org/whl/cu118'],[str(u),'pip','install','--python',str(R/'.venv/bin/python'),'numpy==1.26.4','scipy==1.14.1','scikit-learn==1.5.2','torch-geometric==2.7.0','ogb==1.3.6','einops','pyyaml','tqdm'],[str(u),'pip','install','--python',str(R/'.venv/bin/python'),'torch-scatter==2.1.2+pt21cu118','torch-sparse==0.6.18+pt21cu118','--no-index','--find-links','https://data.pyg.org/whl/torch-2.1.0+cu118.html']]
try:
 for argv in commands:
  p=subprocess.run(argv,cwd=R,env=env,capture_output=True,text=True,timeout=2400);state['commands'].append({'argv':argv,'returncode':p.returncode,'stdout':p.stdout[-3000:],'stderr':p.stderr[-6000:]});(R/'RUNTIME_SETUP_20261015_PROGRESS.json').write_text(json.dumps(state,indent=2));assert p.returncode==0,p.stderr
 p=subprocess.run([str(R/'.venv/bin/python'),'-B','-c',"import torch,torch_geometric,torch_sparse,torch_scatter,json,sys;print(json.dumps({'python':sys.version,'torch':torch.__version__,'pyg':torch_geometric.__version__,'CUDA':torch.cuda.is_available(),'devices':torch.cuda.device_count()}))"],cwd=R,env=env,capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr;state['runtime']=json.loads(p.stdout);state['status']='complete'
except BaseException as e:state['status']='failed';state['error']=str(e)
state['elapsed_seconds']=time.time()-start;(R/'RUNTIME_SETUP_20261015_RESULT.json').write_text(json.dumps(state,indent=2)+'\n')
