"""Freeze fixed source Exp(1)/class-mean-one tables from TRAIN only; no models."""
import argparse
import json
from pathlib import Path
import time
from adapters.private_bootstrap_source import bootstrap_module,source_paths,receipt,CONTRACT
from compile_native_matrix import bind_custody
from prepare_node_data import verify_allocation
from spec import canonical,confined

SEARCH_SEEDS=(77101,77103)
COMPARATIVE_SEEDS=(77201,77203,77207)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('project-root','custody','source-bindings','output'):p.add_argument('--'+k,required=True,nargs='+' if k=='custody' else None)
 a=p.parse_args();verify_allocation();root=Path(a.project_root).resolve();output=confined(root,a.output)
 if output.exists():raise ValueError('Fresh fixed TRAIN weight artifact directory required')
 output.mkdir();bindings=json.loads(confined(root,a.source_bindings).read_text());source_paths(root,bindings)
 import numpy as np
 import torch
 from adapters.context_exclusion import tensor_hash
 if torch.cuda.is_initialized():raise ValueError('TRAIN setup must remain CPU-only')
 tables=[];tick=time.monotonic()
 for value in a.custody:
  custody,data_dir,_=bind_custody(root,confined(root,value));dataset=custody['dataset_id']
  if dataset not in ('wikics','amazon-photo'):raise ValueError('Exact admitted bootstrap providers only')
  for split in custody['splits']:
   path=data_dir/split['node_inputs']['train'];train=torch.load(path,map_location='cpu',weights_only=True)
   if set(train)!={'ids','y'} or train['ids'].dtype!=torch.long or train['y'].dtype!=torch.long:raise ValueError('Exact categorical TRAIN only')
   helper=bootstrap_module(root,bindings,len(train['ids']),custody['model_graph']['classes'])
   for seed in (*SEARCH_SEEDS,*COMPARATIVE_SEEDS) if split['split_id']==0 else COMPARATIVE_SEEDS:
    start=time.monotonic();raw,w,record=helper.draw_weights(np,seed,train['ids'].numpy(),train['y'].numpy())
    weights=torch.from_numpy(w.copy());payload={'schema':'private-bootstrap-fixed-TRAIN-v1','seed':seed,'train':receipt(root,str(path.relative_to(root))),
      'bootstrap_source':bindings['bootstrap'],'ids':train['ids'].clone(),'y':train['y'].clone(),'raw_exponential':torch.from_numpy(raw.copy()),
      'weights':weights,'weights_tensor_sha256':tensor_hash(weights),'source_record':record}
    saved=output/dataset/('mask'+str(split['split_id'])+'_seed'+str(seed)+'.pt');saved.parent.mkdir(exist_ok=True);torch.save(payload,saved);saved.chmod(0o444)
    tables.append({'dataset':dataset,'split':split['split_id'],'seed':seed,'binding':receipt(root,str(saved.relative_to(root))),
      'train':payload['train'],'weights_tensor_sha256':payload['weights_tensor_sha256'],'source_record':record,'setup_wall_seconds':time.monotonic()-start,
      'shared_across_all_three_weighted_arms_and_GAT_SAGE':True,'bootstrap_source':bindings['bootstrap']})
 assert len(tables)==94
 value={'schema':'private-bootstrap-TRAIN-table-manifest-v1','source_contract':CONTRACT,'tables':tables,'setup_wall_seconds':time.monotonic()-tick,
  'source_dimension_changes':['TRAIN_NODES:prepared_TRAIN_count','CLASSES:prepared_classes'],'GRAPH_modified':False,'SELECT_CAL_evaluate_decoded':False,
  'weights_fitted':False,'scientific_fits':0,'CUDA_executions':0,'scores_emitted':0,'GO_created':False}
 path=output/'MANIFEST.json';path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
 print(canonical({'tables':len(tables),'manifest':receipt(root,str(path.relative_to(root))),'scientific_fits':0,'CUDA_executions':0}))


if __name__=='__main__':main()
