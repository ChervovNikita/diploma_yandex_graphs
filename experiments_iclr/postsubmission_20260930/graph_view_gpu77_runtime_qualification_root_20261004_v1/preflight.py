from remote_transport import run,HERE,REPO,PHASE,PYTHON
from datetime import datetime,timezone
import json
paths=[
 'amazon_ratings_native_warm_execution_root_20261003_v3/data/DATA_MANIFEST.json',
 'amazon_ratings_native_warm_execution_root_20261003_v3/data/public_graph.npz',
 'amazon_ratings_native_warm_execution_root_20261003_v3/data/split0_train.npz',
 'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json',
 'accuracy_first_graph_view_source_preparation_20261004_v2/MANIFEST.json',
 'amazon_polynormer_paired_family_source_preparation_20261003_v6/MANIFEST.json']
code='from pathlib import Path\nimport json,os,hashlib,subprocess\n'
code+='repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(PHASE))+');python='+repr(PYTHON)+'\n'
code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\npaths='+repr(paths)+'\n'
code+='result={"hostname":os.uname().nodename,"repository":str(repo),"python":python,"paths":[],"nvidia_smi":subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.free","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout}\n'
code+='for rel in paths:\n f=phase/rel;assert f.resolve().is_relative_to(phase);row={"path":rel,"exists":f.is_file(),"symlink":f.is_symlink()}\n if f.is_file():row.update(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())\n result["paths"].append(row)\n'
code+='print(json.dumps(result))\n'
value=run('graph_view_gpu77_runtime_preflight_20261004_v1',code)
(HERE/'PREFLIGHT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
