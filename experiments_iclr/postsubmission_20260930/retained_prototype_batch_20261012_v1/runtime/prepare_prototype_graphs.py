"""Bounded CPU GRAPH/TRAIN-only preprocessing. No model, optimizer or SELECT label."""
import argparse,ast,base64,gzip,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def receipt(path,phase):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True);parser.add_argument('--packet',required=True);args=parser.parse_args()
 assert socket.gethostname()=='anogena-2-0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 phase,packet=Path(args.phase),Path(args.packet);out=packet/'prepared_graph_metadata';assert not out.exists();out.mkdir()
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 sampler=load(packet/'sources/shared_ordered_path_operator_source_20261011_v1/ordered_path_sampler.py','prototype_ordered_sampler')
 source=packet/'sources/accuracy_first_graph_view_source_preparation_20261004_v2/views.py';original=source.read_text();rows=[];walk_rows=[];reads=[]
 seeds=(88101,88103,88201,88203,88207)
 for dataset in ('wikics','amazon-photo'):
  copied=packet/'metadata'/dataset/'CUSTODY.json';declared=json.loads((packet/'metadata'/dataset/'RECEIPT.json').read_text());custody_path=phase/declared['path']
  assert receipt(custody_path,phase)==declared and copied.read_bytes()==custody_path.read_bytes();custody=json.loads(copied.read_text());classes=custody['model_graph']['classes'];nodes=custody['model_graph']['nodes'];base=custody_path.parent;files={r['path']:r for r in custody['files']}
  def tensor(relative):
   assert relative=='graph.pt' or relative.endswith('/train.pt')
   path=base/relative;expected=files[relative];actual=receipt(path,phase);assert actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256']
   reads.append({'dataset':dataset,'authority':'GRAPH' if relative=='graph.pt' else 'TRAIN','receipt':actual});return torch.load(path,map_location='cpu',weights_only=True)
  graph=tensor('graph.pt');assert set(graph)=={'x','edge_index'};edges=tuple(map(tuple,graph['edge_index'].T.tolist()));assert len(edges)==custody['model_graph']['edges']
  # Source class-domain guard only: learned/view deletion laws unchanged.
  tree=ast.parse(original);edits=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and ast.unparse(node.func)=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==5:
    node.args[0].value=classes;edits.append('TrainRole.class_guard:5->provider_classes')
  assert edits==['TrainRole.class_guard:5->provider_classes']
  namespace={'__name__':'prototype_bound_views_'+dataset};sys.modules[namespace['__name__']]=type(sys)(namespace['__name__']);namespace=sys.modules[namespace['__name__']].__dict__
  exec(compile(ast.fix_missing_locations(tree),'<provider_class_bound_view_source>','exec'),namespace)
  walker=sampler.sample_walks(nodes,edges);walk_file=out/(dataset+'-ordered-walks.json.gz');walk_file.write_bytes(gzip.compress(json.dumps({'receipt':walker.receipt(),'indices':walker.indices,'swap_mask':walker.swap_mask},separators=(',',':')).encode(),mtime=0));walk_rows.append({'dataset':dataset,'graph':reads[-1]['receipt'],'walks':walker.receipt(),'artifact':receipt(walk_file,phase)})
  for split in custody['splits']:
   train=tensor(split['node_inputs']['train']);assert set(train)=={'ids','y'} and len(train['ids'])==split['role_counts']['train']
   ids=tuple(train['ids'].tolist());labels=tuple(train['y'].tolist());known=set(ids);complement=tuple(i for i in range(nodes) if i not in known)
   # The view algorithm needs only TRAIN labels and all other rows as context.
   # Keep scientific SELECT/CAL/evaluate roles in CUSTODY; never decode their targets.
   role=namespace['TrainRole'](nodes,split['split_id'],ids,labels,(),complement)
   for seed in seeds:
    tick=time.monotonic();bundle=namespace['construct_views'](role,edges,seed);namespace['verify_bundle'](bundle)
    suffix=dataset+'-sp'+str(split['split_id'])+'-s'+str(seed)
    deleted=out/(suffix+'-deleted-pairs.json.gz');deleted.write_bytes(gzip.compress(json.dumps(bundle['deleted_pairs'],sort_keys=True,separators=(',',':')).encode(),mtime=0))
    record={'dataset':dataset,'split':split['split_id'],'view_seed':seed,'coverage':bundle['coverage'],'deleted_pairs':receipt(deleted,phase),'CPU_preparation_seconds':time.monotonic()-tick,
     'provider_bindings':edits,'role_adapter':'TRAIN versus label-free all-other context partition for view construction; scientific four-role CUSTODY retained separately','heldout_targets_read':False,'scientific_fits':0}
    meta=out/(suffix+'-coverage.json');meta.write_text(json.dumps(record,indent=2)+'\n');rows.append(record)
   print(json.dumps({'prepared_dataset':dataset,'split':split['split_id'],'view_seed_cases':len(seeds),'all_coverage_eligible':all(r['coverage']['coverage_eligible'] for r in rows[-len(seeds):]),'scientific_fits':0}),flush=True)
 result={'schema':'retained-prototype-actual-GRAPH-TRAIN-preprocessing-v1','status':'prepared','view_cases':len(rows),'eligible_view_cases':sum(r['coverage']['coverage_eligible'] for r in rows),'walk_tables':walk_rows,'views':rows,'actual_authorized_payload_reads':reads,
  'classes_binding_only':True,'scientific_four_role_authority':'original exact prepared CUSTODY; no role relabeling for scientific fits','TRAIN_labels_used_for_view_deletion_only':True,
  'CPU_only':True,'heldout_targets_read':0,'benchmark_fits':0,'toy_optimizer_steps':0,'CUDA_compute':0,'GO':False}
 assert len(rows)==150 and len(walk_rows)==2 and not torch.cuda.is_initialized()
 (packet/'GRAPH_TRAIN_PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'view_cases':len(rows),'eligible_view_cases':result['eligible_view_cases'],'walk_tables':2,'benchmark_fits':0,'heldout_targets_read':0}),flush=True)
if __name__=='__main__':main()
