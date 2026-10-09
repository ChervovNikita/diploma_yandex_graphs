from pathlib import Path
import importlib.metadata,json,socket,subprocess,sys
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
runtime=json.loads((P/'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json').read_text());sys.path[:0]=[str(P/'private_sheaf_dependency_overlay_20261009_v1'),*runtime['PYTHONPATH']]
import torch,numpy,scipy,scipy.stats,torch_householder
packages=('torch','numpy','scipy','scikit-learn','torch-geometric','torch-sparse','torch-scatter','torch-householder')
print(json.dumps({'python':sys.executable,'versions':{k:importlib.metadata.version(k) for k in packages},'providers':{name:module.__file__ for name,module in [('torch',torch),('numpy',numpy),('scipy',scipy),('torch_householder',torch_householder)]},'SciPy_SO_sampler_available':hasattr(scipy.stats,'special_ortho_group'),'model_or_role_data_reads':False,'numerical_or_sampler_execution':False}))
