"""Literal one-body eight-view source controls on custodied node providers.

The exact public single constructor/HPO/controller is inherited from the closed
Polynormer packet. Four TRAIN streams carry two views each; serving is one body.
"""
import copy
from types import SimpleNamespace
from . import common_native_wrapper as common
from . import polynormer_mechanisms as capable
try:
 from ..engine import AdapterStepError
except ImportError:
 from engine import AdapterStepError

CONTROLS=('single8_own','single8_alignment','single8_virtual_contrast')
CONTRACT={'schema':'literal-public-Single8-node-source-port-v1',
 'canonical_method_id':'internal_be_single8_single8_attribution',
 'source_body':'Exact public single WikiBackbone;7local/2global/H512/heads1 Polynormer. Previous SAGE alias is a census alias error.',
 'view_law':'literal Single8Session; four persistent training streams, twice each; all eight full own CE views; one native body/Adam/one backward per update',
 'controls':list(CONTROLS),'auxiliary':'source alignment.05 or alignment.05+residual.05; temperature.2; max512 evenly spaced TRAIN objects',
 'serving':'deterministic native single eval; one route; no stochastic view ensemble',
 'prospective_controller':'inherited5000 full updates/local100,strict-first SELECT accuracy; local model/Adam handoff preserves all live four TRAIN streams',
 'checkpoint_amendment':'explicitly capture/restore all four TRAIN streams, source serving stream, source clocks, counts and local handoff; no source claim of historical exact resume',
 'capable_controls':'exact preserved closed Polynormer native M1/I4 run contracts and common literal originalGNNM controls; exclusive successor, no new control acquisition or constructor equivalence claim'}

def construct(torch,root,*,sources,single8_source,control,features,classes,seed,dropout,learning_rate,train_count,device):
 if control not in CONTROLS:raise ValueError('Declared source eight-view control required')
 session,details=capable.construct(torch,root,sources,arm='native_M1',features=features,classes=classes,
  seed=seed,body=0,dropout=dropout,learning_rate=learning_rate,train_count=train_count,device=device)
 implementation=common.load_module(root,single8_source)
 # Use literal source _recipe under the same prospectively bound public recipe
 # already used by the source Session; only provider/HPO/horizon were bound.
 facade=SimpleNamespace(recipe=lambda task:copy.deepcopy(session.config))
 wrapper=implementation.Single8Session(session,control,implementation._recipe(facade,'wikics'))
 assert wrapper.model.members==1 and len(wrapper.optimizers)==1 and len(wrapper.training_view_streams)==4
 details.update(single8_source=single8_source,control_spec=implementation.control_spec(control),
  four_training_streams_saved=True,source_body_alias_correction='Polynormer,not SAGE')
 return wrapper,details

def attach(torch,session,details,*,x,edges,train_ids,train_y,select_ids,select_y,provenance):
 adapter=capable.attach(torch,session,details,x=x,edges=edges,train_ids=train_ids,train_y=train_y,
  select_ids=select_ids,select_y=select_y,arm='native_M1',provenance={**provenance,'single8_contract':CONTRACT},local_updates=100)
 original_capture,original_restore=adapter.external_capture,adapter.external_restore
 def capture():
  return {'source_controller':original_capture(),'training_view_streams':copy.deepcopy(session.training_view_streams),
   'control_counts':copy.deepcopy(session.control_counts),'control':session.control,'source_single8_contract':CONTRACT}
 def restore(value):
  if value['control']!=session.control or value['source_single8_contract']!=CONTRACT:
   raise ValueError('Different source eight-view control state')
  original_restore(value['source_controller']);session.training_view_streams=copy.deepcopy(value['training_view_streams'])
  session.control_counts=copy.deepcopy(value['control_counts'])
 def train(owner,step):
  before=copy.deepcopy(session.control_counts);counts={'optimizer_updates':0,'optimizer_step_attempts':0}
  optimizer=owner.optimizer;original=optimizer.step
  def tracked(*args,**kwargs):
   counts['optimizer_step_attempts']+=1;value=original(*args,**kwargs);counts['optimizer_updates']+=1;return value
  optimizer.step=tracked
  try:
   result=session.train_step({'x':x,'edge_index':edges,'ids':train_ids},train_y)
   counts.update({k:session.control_counts[k]-v for k,v in before.items()})
   counts.update(backward_calls=1,fullgraph_native_member_trajectories=8)
   if counts['training_body_forwards']!=8 or counts['Adam_step_calls']!=1 or counts['optimizer_updates']!=1:
    raise ValueError('Source Single8 actual view/Adam counters disagree')
   return {'objective':float(result['loss']),'ce':float(result['own_mean']),
    'regularization':float(result['auxiliary']),'operation_counts':counts}
  except BaseException as error:
   counts.update({k:session.control_counts[k]-v for k,v in before.items()})
   raise AdapterStepError(str(error),operation_counts=counts) from error
  finally:optimizer.step=original
 adapter.external_capture=capture;adapter.external_restore=restore;adapter.custom_train=train
 return adapter

def make_adapter(*,torch,project_root,study,run,device):
 c=run['config'];assert study['dataset'] in common.DATASETS and c['backbone']=='native_WikiPolynormer'
 def read(row):return torch.load(common.bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=(read(c['node_inputs'][key]) for key in ('graph','train','select'))
 x,edges=graph['x'].to(device),graph['edge_index'].to(device)
 if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT overlap')
 session,details=construct(torch,project_root,sources=c['mechanism_sources'],single8_source=c['single8_source'],
  control=c['single8_control'],features=x.shape[1],classes=c['classes'],seed=run['seed'],dropout=c['dropout'],
  learning_rate=c['learning_rate'],train_count=len(train['ids']),device=device)
 return attach(torch,session,details,x=x,edges=edges,train_ids=train['ids'].to(device),train_y=train['y'].to(device),
  select_ids=select['ids'].to(device),select_y=select['y'].to(device),provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c})
