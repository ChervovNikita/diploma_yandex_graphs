"""Full released DBLP GRAPH and fresh development-only role custody; no model."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
from adapters.hgt_native import source,receipt,PINS

def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True);args=p.parse_args();root=Path(args.project_root).resolve();packet=root/args.packet
 metadata=json.loads((packet/'DBLP_INPUT_METADATA.json').read_text());rows=json.loads((packet/'EXACT_SOURCE_BINDINGS.json').read_text());rows={k:{x:v for x,v in row.items() if x in ('path','bytes','sha256')} for k,row in rows.items()}
 if any(rows[k]['sha256']!=v for k,v in PINS.items()):raise ValueError('Exact HGT/native/readers/mixed source pins')
 inputs=source(root,rows['inputs'],'inputs');implementation=source(root,rows['implementation'],'implementation')
 schema,attributes,edges=inputs.stream_schema(metadata['archive'],metadata['members'])
 if schema['member_sha256']!=metadata['member_sha256'] or schema['node_counts']!=metadata['expected_node_counts'] or schema['input_dims']!=metadata['expected_input_dims']:raise ValueError('Complete byte-bound release node/link geometry')
 if {str(r['raw_id']):[r['source'],r['target']] for r in schema['relations']}!=metadata['expected_relations'] or schema['support_edges']!=239566 or schema['label_members_opened']:raise ValueError('Native full directed raw relations; label members not read by GRAPH')
 development=inputs.read_development_labels(metadata['development_labels'],metadata['archive']['sha256'])
 if development['source_label_member_sha256']!=metadata['source_label_member_sha256'] or len(development['node_ids'])!=1217 or development['train_class_schema']!=[0,1,2,3]:raise ValueError('Released development-only compact target0 labels')
 import torch,numpy as np
 torch.set_num_threads(2);graph,features,schema=inputs.materialize(schema,attributes,edges,implementation,metadata['relation_row_order'],torch.device('cpu'))
 directory=packet/'data';directory.mkdir();graph_path=directory/'graph.pt'
 torch.save({'schema':schema,'features':features,'relations':[{'raw_id':r.raw_id,'source':r.source,'target':r.target,'src':r.src,'dst':r.dst} for r in graph.relations]},graph_path)
 labels=dict(zip(development['node_ids'],development['labels']));pool=np.array(development['node_ids'],dtype=np.int64);splits=[]
 for split in range(10):
  seed=99001+104729*split;rng=np.random.RandomState(seed);roles={r:[] for r in ('train','select','cal','evaluate')}
  for cls in range(4):
   ids=np.array([i for i in development['node_ids'] if labels[i]==cls],dtype=np.int64)
   if len(ids)<=80:raise ValueError('Every class must retain evaluation nodes after20/30/30')
   shuffled=rng.permutation(ids)
   for role,values in zip(roles,(shuffled[:20],shuffled[20:50],shuffled[50:80],shuffled[80:])):roles[role]+=values.tolist()
  roles={r:np.array(sorted(ids),dtype=np.int64) for r,ids in roles.items()}
  if not np.array_equal(np.sort(np.concatenate(list(roles.values()))),pool):raise ValueError('Complete disjoint released development-only role partition')
  node_inputs={'graph':'graph.pt'};endpoint={};counts={};class_counts={}
  for role,ids in roles.items():
   sub=('split_' if role in ('train','select') else 'sealed/split_')+str(split);folder=directory/sub;folder.mkdir(parents=True,exist_ok=True)
   ys=np.array([labels[int(i)] for i in ids],dtype=np.int64);path=folder/(role+'.pt');torch.save({'ids':torch.from_numpy(ids.copy()),'y':torch.from_numpy(ys)},path)
   if role in ('train','select'):node_inputs[role]=str(path.relative_to(directory))
   else:
    path.chmod(0o600);id_folder=directory/'ids'/('split_'+str(split));id_folder.mkdir(parents=True,exist_ok=True);id_path=id_folder/(role+'_ids.pt');torch.save(torch.from_numpy(ids.copy()),id_path)
    endpoint[role+'_ids']=str(id_path.relative_to(directory));endpoint[role+'_targets']=str(path.relative_to(directory))
   counts[role]=len(ids);class_counts[role]=dict(Counter(map(int,ys)))
  splits.append({'split_id':split,'role_seed':seed,'node_inputs':node_inputs,'endpoint_inputs':endpoint,'role_counts':counts,'role_class_counts':class_counts,'evaluate_purpose':'development',
   'unassigned_target_nodes_without_released_dev_labels':4057-1217})
 files=[receipt(directory,p) for p in sorted(directory.rglob('*.pt'))]
 custody={'schema':'typed-HGB-DBLP-development-role-custody-v2','status':'prepared','dataset_id':'HGB-DBLP','model_graph':schema,'development_pool':1217,'target_nodes':4057,
  'source_archive':{**metadata['archive'],'path':str(Path(metadata['archive']['path']).relative_to(root))},'source_development_labels':{**metadata['development_labels'],'path':str(Path(metadata['development_labels']['path']).relative_to(root))},
  'source_label_member_sha256':metadata['source_label_member_sha256'],'new_to_original_paper':True,'test_label_member_opened':False,
  'request':{'role_rule':'10 fresh private class-stratified masks20TRAIN/30SELECT/30CAL perclass; remaining releaseddevelopment evaluate',
   'role_seed_base':99001,'role_seed_stride':104729,'target_type':'0','feature_type':2,'native_graph':'full raw-ID directed coalesced support; no synthetic self/reverse; ignore weights as nativeHGT; non-target sparse identities',
   'evaluation_exposure':{'evaluation_access_status':'encountered_development','previously_evaluated':True,'protected_confirmation':False,'whole_project_access_audit_complete':False,
    'interpretation':'DBLP is new to original five-dataset paper benchmark; this is fresh role preparation on encountered development data, without historical fitted-state reuse'}},
  'splits':splits,'files':files,'benchmark_fits':0,'predictive_scores':0,'models_constructed':0,'optimizers_constructed':0,'GO':False}
 (directory/'CUSTODY.json').write_text(json.dumps(custody,indent=2,sort_keys=True)+'\n');print(json.dumps(custody),flush=True)

if __name__=='__main__':main()
