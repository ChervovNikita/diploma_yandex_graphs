"""CPU-only label-free source support/CSR preparation; no model or targets."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from adapters.native_label_families import feature_graph,bound,receipt
from adapters.shared_message_port import prepare_operator


def prepare(torch,root,packet):
 root=Path(root).resolve();packet=Path(packet).resolve();bindings=json.loads((packet/'SOURCE_BINDINGS.json').read_text());output=packet/'derived_inputs'
 if output.exists():raise ValueError('Fresh source preparation; retain previous attempts')
 output.mkdir();records={'schema':'fitted-label-source-derived-graph-inputs-v1','sparse_supports':{},'mean_operators':{},'observations':[],'GRAPH_only':True,'target_tensors_opened':False,'CUDA':0,'model_constructors':0,'optimizer_objects':0,'scientific_fits':0,'GO':False}
 for dataset in ('wikics','amazon-photo'):
  custody=json.loads((packet/'metadata'/dataset/'CUSTODY.json').read_text());row=next(r for r in custody['files'] if r['path']=='graph.pt')
  graph_binding={'path':'training_pipeline_v2_20261012/prepared_data/'+dataset+'/graph.pt','bytes':row['bytes'],'sha256':row['sha256']}
  print(json.dumps({'dataset':dataset,'phase':'GRAPH_ONLY_SOURCE_PREPARATION_START'}),flush=True)
  graph=torch.load(bound(root,graph_binding),map_location='cpu',weights_only=True)
  if set(graph)!={'x','edge_index'}:raise ValueError('Target-free graph authority')
  x,edges=graph['x'],graph['edge_index'];helper,edits=feature_graph(root,{'family_sources':bindings['sparse_feature_kernel']},x.shape[1],len(x))
  tick=time.monotonic();support,meta=helper.build_support(torch,x,edges);elapsed=time.monotonic()-tick
  path=output/(dataset+'_sparse_support.pt');torch.save({'edge_index':support,'source_graph':graph_binding,'feature_graph_source':bindings['sparse_feature_kernel']['feature_graph'],'provider_sites':edits,'preparation':meta},path)
  records['sparse_supports'][dataset]=receipt(root,str(path.relative_to(root)))
  tick=time.monotonic();p,a=prepare_operator(torch,x,edges);mean_seconds=time.monotonic()-tick
  path=output/(dataset+'_mean_operator.pt');torch.save({'forward':p,'adjoint':a,'source_graph':graph_binding,'policy':'incoming_record_mean_float32;factualCOO;no new loops'},path)
  records['mean_operators'][dataset]=receipt(root,str(path.relative_to(root)))
  records['observations'].append({'dataset':dataset,'source_graph':graph_binding,'sparse_support':meta,'sparse_preparation_CPU_seconds':elapsed,'mean_forward_adjoint_CPU_seconds':mean_seconds,
    'preparation_once_per_dataset':True,'source_support_total_tensor_bytes':support.numel()*support.element_size(),'constructor_or_target_work':0})
  (packet/'PREPARED_DERIVED_GRAPH_INPUTS.json').write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps({'dataset':dataset,'phase':'GRAPH_ONLY_SOURCE_PREPARATION_COMPLETE','union_edges':meta['union_directed_pairs'],'CPU_seconds':elapsed}),flush=True)
 return records


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);a=p.parse_args()
 if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
  raise ValueError('Exact allocation identity before project operations')
 import torch
 torch.set_num_threads(2)
 if torch.cuda.is_initialized():raise ValueError('CPU-only source/data preparation')
 result=prepare(torch,Path(a.project_root),Path(a.packet));print(json.dumps({'derived_inputs':4,'CUDA':0,'models':0,'targets':0,'fits':0,'GO':False}))
