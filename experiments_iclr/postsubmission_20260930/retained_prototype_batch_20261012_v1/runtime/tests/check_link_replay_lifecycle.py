"""Fresh source replay writer/readback/support checks plus full tiny epoch."""
import ast,copy,json,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parents[2];ROOT=P.parent;sys.path.insert(0,str(P/'runtime'))
from adapters import link_prototypes as port
from adapters.common_native_wrapper import receipt

def main():
 import torch,numpy
 from torch_geometric.utils import negative_sampling
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();bindings={k:receipt(ROOT,P/'sources'/v) for k,v in port.PATHS.items()};rows=[];steps=0
 for arm in ('endpoint_private4','F_P','C_mu'):
  model,opt,modules,details=port.construct(torch,ROOT,bindings,arm=arm,seed=73101,body=0,lr_multiplier=1.,dropout=.3,device='cpu');_,utils=port.bridge.native_modules(ROOT,{'model':bindings['native'],'utils':bindings['utils']})
  torch.manual_seed(194);nodes=20;pairs=torch.tensor([[i,(i+1)%nodes] for i in range(13)],dtype=torch.long);raw=torch.cat((pairs,pairs.flip(1)),0).T.contiguous();x=torch.randn(nodes,128);select={'positive':torch.tensor([[0,3],[2,9],[3,8]]),'negative':torch.tensor([[0,9],[1,11],[2,10],[6,15]]).repeat(16,1)}
  config={'link_prototype_arm':arm,'training':{'batch_size':3,'eval_batch_size':16},'train_minibatches_per_epoch':4,'link_inputs':{'fixture':True},'prototype_sources':bindings};run={'run_id':'fresh_CPU_source_replay_'+arm,'config':config};study={'data':{'synthetic':True},'source':{'fixed_source_only':True}}
  with tempfile.TemporaryDirectory(prefix='source_replay_fixture_',dir=P) as temporary:
   adapter=port.LinkPrototypeAdapter(torch,numpy,model,opt,modules,details,root=ROOT,run=run,study=study,graph={'x':x,'raw_edge_index':raw},train={'positive':pairs},select=select,replay_directory=temporary,utils=utils,sampler=negative_sampling)
   # Bind only fixture cardinality; actual adapter remains source65536/17.
   learning=modules['learning'];tree=ast.parse(Path(learning.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_replay_batch');space=dict(vars(learning),TRAIN_BATCH=3);exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),'<toy_source_batch_cardinality>','exec'),space);adapter.modules={**modules,'learning':type('FixtureSourceLearning',(),{'train_replay_batch':staticmethod(space['train_replay_batch'])})}
   before=copy.deepcopy(adapter._rng);value=adapter.train_step(1);steps+=4;assert value['operation_counts']['optimizer_updates']==4 and value['operation_counts']['support_context_checks']==4
   epoch_receipt=json.loads((Path(temporary)/'epoch_1/EPOCH.json').read_text());assert epoch_receipt['master_seed']==2026100401 and epoch_receipt['full_batches']==4 and epoch_receipt['dropped_tail_records']==1 and epoch_receipt['stored_teacher_labels'] is False and epoch_receipt['coordinates_stored'] is False
   current=copy.deepcopy(adapter.capture_state());adapter._epoch=-1;adapter.restore_state(current);assert adapter._epoch==1 and all(torch.equal(a,b) for a,b in zip(adapter._rng['torch_cuda'],current['fit_rng']['torch_cuda']))
   metrics=adapter.validate(1);assert metrics['positive_count']==3 and metrics['negative_count']==64;adapter.observe_validation(1,metrics)
   rows.append({'arm':arm,'source_full_negative_permutation_tail_and_context_receipts':True,'source_epoch_authenticated_before_teacher_lookup_and_updates':True,'source_four_full_batches_one_dropped_tail_record':True,'official_strict_Hits50':True,'source_RNG_and_epoch_checkpoint_restore':True});print(json.dumps({'case':rows[-1],'completed_toy_steps':steps}),flush=True)
 assert len(rows)==3 and steps==12 and not torch.cuda.is_initialized();(P/'LINK_REPLAY_LIFECYCLE_CPU_QA.json').write_text(json.dumps({'schema':'source-link-replay-new-provider-lifecycle-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_steps':steps,'scientific_fits':0,'actual_input_arrays':0,'CUDA_compute':0,'existing_kernel_tests_repeated':False},indent=2)+'\n');print(json.dumps({'status':'passed','cases':3,'toy_steps':12,'scientific_fits':0}))

if __name__=='__main__':main()
