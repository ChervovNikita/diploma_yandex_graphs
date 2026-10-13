"""Tiny fabricated source-constructor/bridge QA; never project arrays or scores."""
import copy,gzip,json,sys,tempfile
from pathlib import Path
S=Path(__file__).resolve().parent.parent;P=S.parent;sys.path.insert(0,str(S/'runtime'))
import torch,numpy
from adapters import structural_interaction as structural,cardinality,conditional_pattern as cb
from engine import capture_checkpoint,restore_checkpoint
from structural_cardinality_semantics import FACTORS
Bst=json.loads((S/'STRUCTURAL_SOURCE_BINDINGS.json').read_text());Bca=json.loads((S/'CARDINALITY_SOURCE_BINDINGS.json').read_text());Bcb=json.loads((S/'CONDITIONAL_SOURCE_BINDINGS.json').read_text())
torch.set_num_threads(2);torch.manual_seed(190131);pairs=torch.tensor([(i,j) for i in range(12) for j in range(i+1,12) if (i+j)%3!=0],dtype=torch.long);negative=torch.tensor([(i,j) for i in range(12) for j in range(i+1,12) if (i+j)%3==0],dtype=torch.long);negative=negative.repeat((len(pairs)+len(negative)-1)//len(negative),1)[:len(pairs)];x=torch.randn(12,128);rows=[];maps=0
# Structural source geometry uses absent query edges and real internal witnesses.
spairs=torch.tensor([[0,2],[0,3],[1,2],[1,3],[2,3],[4,2],[4,3],[5,2],[5,3],[6,8],[7,8]],dtype=torch.long);sx=torch.randn(12,3703);qpos=torch.tensor([[0,1],[4,5]],dtype=torch.long).T;qneg=torch.tensor([[6,7],[9,10]],dtype=torch.long).T
for arm in ('structural_private','structural_tied','informed_single','informed_independent4','structure_blind','count_only','native_M1','original_GNNM'):
 state=structural.construct(torch,P,Bst,arm,0,FACTORS[0],600000,'cpu');m,model,opt,shared,private,partition=state;a=structural.Adapter(torch,*state,{'x':sx},{'positive':spairs},{},arm,0,0,{'synthetic':True,'arm':arm});support=m.custody.support_tensor(torch,spairs,range(len(spairs)),12,torch.device('cpu'));inner=[(qpos,qneg)]*4;v,counts=a.episode(support,inner,(qpos,qneg));assert counts['optimizer_updates']==3 and all(torch.isfinite(p).all() for p in model.parameters());maps+=3
 for group in opt.param_groups:
  assert group['foreach'] is False and group['fused'] is False and group['lr']==.001
 before=copy.deepcopy(a.capture_state());payload=capture_checkpoint(torch,a,fit_id='synthetic-structural-'+arm,step=1,validation={},decision={})
 with tempfile.TemporaryDirectory(dir=S/'engineering_tmp') as td:
  q=Path(td)/'state.pt';torch.save(payload,q);restore_checkpoint(torch,q,a,expected_fit_id='synthetic-structural-'+arm)
 assert a._best==before['selector']['best'];rows.append({'family':'structural','arm':arm,'members':model.member_count,'parameters':sum(p.numel() for p in model.parameters()),'optimizer_maps':3,'work':v.get('structural_work',{}),'checkpoint_roundtrip':True})
 del a,model,opt,state
for seed in (0,1,2):
 m,native,utils,proof=cardinality.sources(torch,P,Bca,seed);model,opt=m['model'].make_single(native,'cpu');teacher=m['teacher'].TrainObservationTeacher.from_fabricated_train(pairs,12)
 with cardinality.prior.aliases({'cardinality_'+k:v for k,v in m.items()}):work=m['training'].train_batch(model,opt,x,pairs,negative,torch.tensor([0,7],dtype=torch.long),teacher,epoch=1,batch=1)
 assert work['optimizer_steps']==1 and sum(p.numel() for p in model.parameters())==42772;maps+=1
 data={'x':x,'pairs':pairs};a=cardinality.Adapter(torch,numpy,model,opt,m,utils,None,data,teacher,{'synthetic':True,'seed':seed});keys=m['counter'].CompletionKeys('eval');value=keys.log_uniform((1,2),0,0,'slot',(3,4));assert value<0
 for step in range(1,101):a.observe_validation(step,{'selection_metric':.25 if step<99 else .5})
 assert a.selector.best_epoch==99 and a.selector.states==100
 payload=capture_checkpoint(torch,a,fit_id='synthetic-C64-'+str(seed),step=1,validation={},decision={})
 with tempfile.TemporaryDirectory(dir=S/'engineering_tmp') as td:
  q=Path(td)/'state.pt';torch.save(payload,q);restore_checkpoint(torch,q,a,expected_fit_id='synthetic-C64-'+str(seed))
 rows.append({'family':'cardinality','seed':seed,'parameters':42772,'optimizer_maps':1,'source_loss_uses_all_four_draw_logsigmoids':True,'stream_AST_proof':proof,'checkpoint_roundtrip':True,'fixed_selector_first_tie':True,'eval_key_value':value})
 del a,model,opt,m
assert len({row['eval_key_value'] for row in rows if row['family']=='cardinality'})==3
# No CB arithmetic oracle or full-input kernel is repeated: only new native call/teacher/loss wiring.
for arm in ('J_K','J_K_sep','W_K'):
 m=cb.sources(P,Bcb);model,opt=m['pattern_model'].make_pattern({'prototype':m['prototype'],'design':m['design_spec']},0,'cpu');teacher=m['pattern_teacher'].ObservationTeacher.from_train(pairs,12);data={'pairs':pairs,'x':x}
 with cb.prior.aliases(m):value=cb.batch_step(torch,m,model,opt,data,teacher,negative,torch.tensor([0,7],dtype=torch.long),arm)
 maps+=1;assert torch.isfinite(torch.tensor(value['total_loss']));rows.append({'family':'conditional_pattern','arm':arm,'parameters':sum(p.numel() for p in model.parameters()),'optimizer_maps':1,'complete_native_forward_teacher_loss_bridge':True,'inherited_CB_math_oracles_not_repeated':True})
result={'schema':'literal-sourceports-tiny-fabricated-CPU-QA-v1','status':'passed','cases':rows,'synthetic_optimizer_maps':maps,'actual_input_arrays_opened':0,'actual_input_models_constructed':0,'benchmark_scores':0,'scientific_fits':0,'GO':False,'resource_profile_qualified':False};(S/'SYNTHETIC_CPU_QA.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'passed','cases':len(rows),'synthetic_optimizer_maps':maps,'actual_input_arrays_opened':0,'GO':False}))
