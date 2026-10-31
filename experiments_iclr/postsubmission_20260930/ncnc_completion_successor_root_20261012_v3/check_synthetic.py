
import copy,json,sys
from pathlib import Path
packet=Path(__file__).resolve().parent;phase=packet.parent
sys.path.insert(0,str(packet/'runtime'))
from adapters import link_prototypes as port
from adapters.common_native_wrapper import receipt
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
bindings=json.loads((phase/'retained_prototype_batch_20261012_v1/link_ready_v1/SOURCE_BINDINGS.json').read_text())
torch.manual_seed(611)
x=torch.randn(12,128);pairs=torch.tensor([[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]])
records=torch.tensor([0,4,7]);negative=torch.tensor([[0,4],[1,6],[2,7]])
rows=[]
for arm in ('completion_private4','completion_pooled_after_clamp4','completion_native_M1'):
 model,opt,mods,details=port.construct(torch,phase,bindings,arm=arm,seed=73201,body=0,lr_multiplier=1.,dropout=.3,device='cpu')
 from engine import inspect_ownership
 from types import SimpleNamespace
 inspect_ownership(SimpleNamespace(model=model,optimizer=opt))
 before=copy.deepcopy(model.state_dict()); mode=details['completion_mode']
 loss=mods['prototype'].training_batch_loss(model,x,pairs,records,negative,mode)
 assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
 assert not opt.state, 'Backward check must not commit Adam maps'
 model.eval();graph=mods['graph_ops'].Graph.from_pairs(pairs,12)
 with torch.no_grad():z=model(x,graph,pairs[:3],mode)
 assert z.shape==(3,1 if arm=='completion_native_M1' else 4) and torch.isfinite(z).all()
 if arm=='completion_native_M1':
  assert all(torch.equal(v,torch.ones_like(v)) and not v.requires_grad for n,v in model.named_parameters() if n.endswith('.r') or n.endswith('.s'))
 else:
  assert all(set(v.detach().unique().tolist())<=set((-1.,1.)) for n,v in model.named_parameters() if n.endswith('.r') or n.endswith('.s'))
  if arm=='completion_private4':private_state=copy.deepcopy(model.state_dict())
  else:assert all(torch.equal(private_state[k],v) for k,v in model.state_dict().items()), 'Only declared decoder mode may differ'
 rows.append({'arm':arm,'finite_backward':True,'raw_bank_shape':list(z.shape),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'all_registered_parameters':sum(p.numel() for p in model.parameters()),'optimizer_commits':0,'benchmark_arrays':0})
result={'status':'PASS','cases':rows,'synthetic_only':True,'CUDA_initialized':False,'scientific_fits':0,'optimizer_commits':0,'source_scores':0}
(packet/'SYNTHETIC_CPU_QA.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
