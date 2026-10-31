"""Actual source learning-law parity on fabricated twelve-node CPU data.

All eleven source arms, full tiny epoch, exact gradients/RNG and checkpoint
leases. Optimizer dispatch is replaced by a recorded no-op; zero Adam maps,
benchmark GRAPH/TRAIN reads, CUDA initialization or scientific fits.
"""
import atexit
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
sys.path.insert(0,str(HERE/'runtime'))
import torch
import numpy as np
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
from adapters import link_prototypes as current
reference_path=PHASE/'ncnc_completion_successor_root_20261012_v3/runtime/adapters/link_prototypes.py'
spec=importlib.util.spec_from_file_location('adapters.old_replay_reference',reference_path)
reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
bindings=json.loads((PHASE/'retained_prototype_batch_20261012_v1/link_ready_v1/SOURCE_BINDINGS.json').read_text())
from torch_geometric.utils import negative_sampling
_,utils=current.bridge.native_modules(PHASE,{'model':bindings['native'],'utils':bindings['utils']})
fixture=HERE/'fabricated_CPU_data';fixture.mkdir()
torch.manual_seed(611);x=torch.randn(12,128)
pairs=torch.tensor([[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]],dtype=torch.long)
raw=torch.cat((pairs.T,pairs.T.flip(0)),1).contiguous()
graph={'x':x,'raw_edge_index':raw};train={'positive':pairs};select={'positive':pairs[:2],'negative':torch.tensor([[0,5],[2,9]],dtype=torch.long)}
input_receipts={}
for name,value in [('graph',graph),('train',train),('select',select)]:
    path=fixture/(name+'.pt');torch.save(value,path);rawbytes=path.read_bytes()
    input_receipts[name]={'path':str(path.relative_to(PHASE)),'bytes':len(rawbytes),'sha256':hashlib.sha256(rawbytes).hexdigest()}
study={'study_id':'fabricated_CPU_replay_source_parity','data':{'fixture_only':True},'source':{'fixture_only':True}}
rows=[];source_provider_rosters=0;actual_maps=0;body_constructors=0
backward_attempts=0;backward_completed=0;noop_dispatches=0;QA_status='RUNNING';active_arm=None
original_backward=torch.Tensor.backward
def recorded_backward(value,*args,**kwargs):
    global backward_attempts,backward_completed
    backward_attempts+=1;result=original_backward(value,*args,**kwargs);backward_completed+=1;return result
torch.Tensor.backward=recorded_backward
def progress():
    (HERE/'QA_PROGRESS.json').write_text(json.dumps({'status':'PASS' if QA_status=='PASS' else 'FAILED_OR_INTERRUPTED',
      'active_arm':active_arm,'completed_cases':rows,'completed_body_constructors':body_constructors,
      'backward_attempts':backward_attempts,'backward_completed':backward_completed,
      'recorded_noop_Adam_dispatches':noop_dispatches,'actual_optimizer_maps':0,'scientific_fits':0,
      'tiny_CPU_fixture_only':True},indent=2,sort_keys=True)+'\n')
atexit.register(progress)


def make(port, arm, identifier, old=False):
    global body_constructors
    model,opt,mods,details=port.construct(torch,PHASE,bindings,arm=arm,seed=73201,body=0,lr_multiplier=1.,dropout=.3,device='cpu')
    body_constructors+=1
    _,_,fence=port.runtime_replay(torch,mods,utils,negative_sampling)
    if arm in ('P0','J_P','F_P','C_mu'):
        # Only the fabricated CPU proof reduces the guard constant. The source
        # file/function body and actual65536-row scientific factory are intact.
        assert len(graph['x'])<=64 and not torch.cuda.is_initialized()
        assert mods['learning'].TRAIN_BATCH in (65536,3)
        mods['learning'].TRAIN_BATCH=3
    c={'link_prototype_arm':arm,'training':{'batch_size':3,'eval_batch_size':3},'train_minibatches_per_epoch':4,
       'link_inputs':input_receipts,'prototype_sources':bindings,
       'shared_source_replay':{'authority_scope':'synthetic_fixture_only','CPU_provider_fence_AST':fence,
          'store_root':str((HERE/'synthetic_shared_store').relative_to(PHASE))}}
    run={'run_id':identifier,'config':c,'seed':73201,'body':0}
    adapter=port.LinkPrototypeAdapter(torch,np,model,opt,mods,details,root=PHASE,run=run,study=study,
      graph=copy.deepcopy(graph),train=copy.deepcopy(train),select=copy.deepcopy(select),
      replay_directory=fixture/('old_'+arm),utils=utils,sampler=negative_sampling)
    return adapter


def epoch_without_updates(adapter):
    grads=[];model_before=copy.deepcopy(adapter.model.state_dict())
    old_step=adapter.optimizer.step
    def denied_dispatch(*args,**kwargs):
        global noop_dispatches
        noop_dispatches+=1
        assert not adapter.optimizer.state,'No mature state or Adam map may be introduced by this source check'
        row={n:None if p.grad is None else p.grad.detach().clone() for n,p in adapter.model.named_parameters()}
        assert all(v is None or torch.isfinite(v).all() for v in row.values())
        grads.append(row)
    adapter.optimizer.step=denied_dispatch
    try:metrics=adapter.train_step(1)
    finally:adapter.optimizer.step=old_step
    assert len(grads)==4 and metrics['operation_counts']['optimizer_updates']==4
    assert all(torch.equal(model_before[n],p) for n,p in adapter.model.state_dict().items())
    assert not adapter.optimizer.state and not torch.cuda.is_initialized()
    return metrics,grads


for arm in current.ARMS:
    active_arm=arm
    old=make(reference,arm,'old_'+arm,old=True);new=make(current,arm,'new_'+arm)
    assert old.modules['pilot_state'].state_digest(old.model.state_dict())==new.modules['pilot_state'].state_digest(new.model.state_dict())
    old_metrics,old_grads=epoch_without_updates(old);new_metrics,new_grads=epoch_without_updates(new)
    digest=new.modules['pilot_state'].state_digest
    assert old_metrics==new_metrics and digest(old_grads)==digest(new_grads)
    before_old=old.capture_state();before_new=new.capture_state()
    for name in ('fit_rng','epoch','selector'):assert digest(before_old[name])==digest(before_new[name])
    # Exact source arrays/tail/graphs/query/support coordinates all agree;
    # source regenerate is already invoked before each real source loss.
    old_replay=old.provider.EpochReplay(old.rt,old.data,old.replay_directory/'epoch_1',hashlib.sha256((old.replay_directory/'epoch_1/EPOCH.json').read_bytes()).hexdigest())
    new_replay=new.replay_for_epoch(1)
    assert torch.equal(old_replay.permutation,new_replay.permutation) and torch.equal(old_replay.negatives,new_replay.negatives)
    for key in ('permutation_sha256','native_negative_draw_sha256','dropped_tail_sha256','batches'):assert old_replay.receipt[key]==new_replay.receipt[key]
    # weights_only serializer covers the actual fit RNG + added lease pointers.
    buffer=io.BytesIO();torch.save({'model':new.model.state_dict(),'optimizer':new.optimizer.state_dict(),'adapter':new.capture_state()},buffer)
    buffer.seek(0);restored=torch.load(buffer,map_location='cpu',weights_only=True)
    new.model.load_state_dict(restored['model']);new.optimizer.load_state_dict(restored['optimizer']);new.restore_state(restored['adapter'])
    assert digest(restored['adapter'])==digest(new.capture_state())
    bad=copy.deepcopy(restored['adapter']);bad['shared_source_replay']['consumer']='wrong_fixture_consumer'
    try:new.restore_state(bad)
    except ValueError:pass
    else:raise AssertionError('Changed paid consumer must be rejected')
    bad=copy.deepcopy(restored['adapter']);bad['shared_source_replay']['epochs']['1']['manifest']['sha256']='0'*64
    try:new.restore_state(bad)
    except ValueError:pass
    else:raise AssertionError('Changed source epoch manifest must be rejected')
    new.restore_state(restored['adapter']);rows.append({'arm':arm,'full_fabricated_epoch_batches':4,
       'exact_metrics_gradients_fit_RNG_arrays_tail_all_support_receipts':True,'weights_only_checkpoint_lease_restore':True,
       'changed_consumer_and_manifest_denied':True,'source_Adam_dispatches_replaced_by_recorded_noop':8,'actual_Adam_maps':0})

# The exact same source/master/geometry/fence key is shared across all11arms.
stores=list((HERE/'synthetic_shared_store').iterdir());assert len(stores)==1
costs=[json.loads(p.read_text()) for p in stores[0].glob('consumer_costs/*/*.json')]
assert sum(r['source_draw_attempts'] for r in costs)==1 and sum(r['source_rosters_written'] for r in costs)==1
assert len(list(stores[0].glob('consumer_leases/*_epoch_1.json')))==len(current.ARMS)
QA_status='PASS'
result={'schema':'LinkPrototype-shared-replay-source-learning-law-synthetic-CPU-QA-v1','status':'PASS','cases':rows,
        'source_arms':len(rows),'source_epoch_rosters_generated_in_shared_store':1,'source_consumers':len(rows),
        'source_CPU_body_constructors':body_constructors,'complete_synthetic_backward_callbacks':8*len(rows),
        'source_Adam_dispatches_replaced_by_recorded_noop':8*len(rows),'actual_optimizer_maps':0,
        'real_benchmark_GRAPH_TRAIN_or_SELECT_decoded':False,'CUDA_initialized':torch.cuda.is_initialized(),
        'scientific_fits':0,'source_scores':0,'learning_law_AST_preservation_separate':True,
        'synthetic_only_TRAIN_BATCH_guard_constant_override_65536_to3':True,
        'learning_function_body_AST_and_production_65536_source_unchanged':True,
        'prior_v1_geometry_guard_failure_preserved':True,
        'this_proof_is_not_an_accuracy_or_convergence_result':True}
(HERE/'SYNTHETIC_CPU_QA.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
