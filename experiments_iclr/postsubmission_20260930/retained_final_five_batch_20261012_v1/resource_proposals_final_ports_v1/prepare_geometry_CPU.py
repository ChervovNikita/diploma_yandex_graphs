"""Bounded GRAPH/TRAIN/label-free graph custody;sourcecapacity preview,0 maps."""
import copy,gzip,hashlib,io,json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];name=sys.argv[1];batch=P.parent
PACK={'unimp':'unimp_successor_v1','bsnn':'bsnn_ready_v1','lowrank':'lowrank_gine_ready_v1'};S=batch/PACK[name];sys.path.insert(0,str(S/'runtime'))

def receipt(q):raw=q.read_bytes();return {'path':str(q.relative_to(R)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();entry=json.loads((S/'MASTER_INPUT_ENTRY.json').read_text());plan=json.loads((R/entry['plan']['path']).read_text());studies=[json.loads(gzip.decompress((R/r['path']).read_bytes())) for r in plan['studies']];study=next(s for s in studies if any(name=='lowrank' and 'source_GINE_contract' in r['config'] or name=='bsnn' and 'bsnn_source_contract' in r['config'] or name=='unimp' and 'unimp_source_contract' in r['config'] for r in s['runs']));run=next(r for r in study['runs'] if r['kind']=='model_fit' and (name=='lowrank' and r['config'].get('source_GINE_arm')=='lowrank_private' or name=='bsnn' and r['config'].get('bsnn_arm')=='native_M1' or name=='unimp' and 'unimp_source_contract' in r['config']));c=copy.deepcopy(run['config']);bound=[];start=time.monotonic()
 def load(row):
  q=R/row['path'];assert receipt(q)==row;bound.append(row);return torch.load(q,map_location='cpu',weights_only=True)
 previews=[]
 if name=='lowrank':
  train=load(c['molecular_inputs']['train']);ep=study['data']['evidence']['endpoint_inputs'];packs={'TRAIN':train,'CAL_graphs':load(ep['cal_graphs']),'evaluation_graphs':load(ep['evaluate_graphs'])};rows=[]
  for role,pack in packs.items():
   n=pack['node_ptr'].diff();e=pack['edge_ptr'].diff();rows.append({'role':role,'graphs':len(n),'maximum_graph_nodes':int(n.max()),'maximum_graph_edges':int(e.max()),'any_shuffle_batch128_nodes_upper_bound':int(n.sort(descending=True).values[:128].sum()),'any_shuffle_batch128_edges_upper_bound':int(e.sort(descending=True).values[:128].sum()),'deterministic_serving_batch128_nodes_max':max(int(z.sum()) for z in n.split(128)),'deterministic_serving_batch128_edges_max':max(int(z.sum()) for z in e.split(128))})
  qa=json.loads((S/'LOWRANK_PIPELINE_CPU_QA.json').read_text());geometry={'atom_fields':9,'bond_fields':3,'source_virtual_nodes':0,'hidden':128,'layers':3,'FFN_multiplier':2,'members':4,'batch_size':128,'role_geometry':rows};previews=[{'role':'lowrank_private','actual_registered_parameters':qa['parameter_scalars'],'preview_bytes':qa['checkpoint_serialized_preview_bytes'],'preview_sha256':qa['checkpoint_serialized_preview_sha256'],'preview_source':receipt(S/'LOWRANK_PIPELINE_CPU_QA.json'),'scope':'existing new synthetic1map source checkpoint serialization;datafree rankfactory not rerun'}];new_constructors=0
 else:
  graph=load(c['node_inputs']['graph']);train=load(c['node_inputs']['train']);empty={'ids':torch.empty(0,dtype=torch.long),'y':torch.empty(0,dtype=torch.long)}
  if name=='unimp':
   from adapters import unimp_reference as port
   model,opt,topology=port.construct(torch,R,c['unimp_sources'],x=graph['x'],edges=graph['edge_index'],classes=8,seed=c['initialization_seed'],learning_rate=.001,weight_decay=.0005,device='cpu');adapter=port.attach(torch,model,opt,topology,train_ids=train['ids'],train_y=train['y'],select_ids=empty['ids'],select_y=empty['y'],seed=c['initialization_seed'],provenance={'study':study,'run':run,'capacity_preview_only':True});geometry={'factual_nodes':len(graph['x']),'source_total_nodes':len(topology.x),'features':graph['x'].shape[1],'classes':8,'source_topology_edges':topology.edge_index.shape[1],'source_attention_edges':len(topology.source),'hidden':300,'final_attention_heads':4,'APPNP_attention_passes':10,'TRAIN_context_labels':len(train['ids']),'source_visible_labels':int(.65*len(train['ids'])),'source_query_labels':len(train['ids'])-int(.65*len(train['ids']))};objects=[('native_M1',model,opt,adapter)];new_constructors=1
  else:
   from adapters import bsnn_deterministic as port
   geometry={'nodes':len(graph['x']),'features':graph['x'].shape[1],'classes':2,'directed_incidence_edges':graph['edge_index'].shape[1],'stalk_dim':2,'hidden_channels':32,'effective_hidden_width':64,'layers':2,'source_perlayer_edgeweight_learners':2,'TRAIN_rows':len(train['ids'])};objects=[]
   for arm in ('native_M1','original_GNNM'):
    model,opt,h,args=port.construct(torch,R,c['bsnn_sources'],graph,arm,c['initialization_seed'],'cpu');body=model if arm=='native_M1' else model.body;half=len(body.laplacian_builder.left_right_idx[0]);geometry['plain_saved_tril_maps_shape_per_layer']=[half,2,2]
    # Source builder saved_tril_maps is detached [reversepaired_edges,d,d].
    # Materialize the capacity shape only; no nativeforward or update is taken.
    for learner in body.sheaf_learners:learner.L=torch.zeros(half,2,2)
    adapter=port.attach(torch,model,opt,h,args,graph,train,empty,arm,c['initialization_seed'],{'study':study,'run':run,'capacity_preview_only':True,'arm':arm});objects.append((arm,model,opt,adapter))
   new_constructors=2
  from engine import capture_checkpoint
  for arm,model,opt,adapter in objects:
   for p in model.parameters():opt.state[p].update(step=torch.tensor(0.,dtype=torch.float32),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
   payload=capture_checkpoint(torch,adapter,fit_id='capacity-preview-'+name+'-'+arm,step=0,validation=None,decision=None);buf=io.BytesIO();torch.save(payload,buf);raw=buf.getvalue();q=P/(name+'-'+arm+'-capacity-preview.pt');q.write_bytes(raw);previews.append({'arm':arm,'actual_registered_parameters':sum(p.numel() for p in model.parameters()),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'fully_materialized_named_Adam_moment_capacity':True,'preview_bytes':len(raw),'preview_sha256':hashlib.sha256(raw).hexdigest(),'preview_receipt':receipt(q),'scope':'serialized capacity envelope with actual declared inputdimensions/model/optimizer schema and source RNG/roles;zero-valued full moment capacities;0updates;not a trained or admissible fit checkpoint'})
 report={'schema':'closed-source-port-bounded-CPU-geometry-and-checkpoint-capacity-proposal-v1','port':name,'status':'CPU_PROPOSAL_GEOMETRY_AND_SERIALIZATION_PREPARED_GPU_ADMISSION_PENDING','entry':receipt(S/'MASTER_INPUT_ENTRY.json'),'plan_hash':entry['plan_hash'],'input_receipts':bound,'actual_input_scope':'GRAPH/TRAIN and label-free CAL/evaluation molecular graph packs only;no SELECT/CAL/evaluationtarget authority opened','geometry':geometry,'checkpoint_serialized_previews':previews,'new_model_constructors':new_constructors,'new_optimizer_maps':0,'source_parameter_and_optimizer_storage':'float32;actual source operator/topology dtypes preserved;AMPrequires separate root qualification','CPU_preparation_seconds':time.monotonic()-start,'CUDA_compute':0,'actual_predictive_scores':0,'scientific_fits':0,'GO':False,'root_GPU_peak_wall_RSS_provider_AMP_and_three_GPU_assignments_pending':True};(P/(name.upper()+'_CPU_RESOURCE_PROPOSAL.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('port','status','geometry','checkpoint_serialized_previews','new_model_constructors','new_optimizer_maps','CPU_preparation_seconds','GO')}))
if __name__=='__main__':main()
