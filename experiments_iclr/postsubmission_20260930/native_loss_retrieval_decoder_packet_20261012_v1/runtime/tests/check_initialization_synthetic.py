"""Supplementary source initial-state checks, CPU constructors only, zero steps."""
import json,sys
from pathlib import Path
RUNTIME=Path(__file__).resolve().parents[1];PACKET=RUNTIME.parent;ROOT=PACKET.parent;sys.path.insert(0,str(RUNTIME))
def main():
 import torch
 from adapters import native_combinations as port
 from adapters import common_native_wrapper as common
 from engine import inspect_ownership
 torch.set_num_threads(1);bindings=json.loads((PACKET/'SOURCE_BINDINGS.json').read_text());factors=common.load_module(ROOT,bindings['factors']);rows=[]
 for features,classes in ((300,10),(745,8)):
  for family,roster in port.FAMILIES.items():
   for backbone in roster['backbones']:
    for arm in roster['arms']:
     source,model,opt,streams,rendered=port.construct_source(torch,ROOT,bindings,family=family,arm=arm,features=features,classes=classes,train_count=2*classes,
      seed=88101,body=0,backbone=backbone,dropout=.2,learning_rate=.001,device='cpu')
     inspect_ownership(type('Fixture',(),{'model':model,'optimizer':opt})())
     assert not opt.state and len(streams)==port.arm_shape(family,arm)[0]
     if family=='shared_recurrent_class_decoder':
      B=model.class_decoder_B;assert torch.equal(B,torch.zeros_like(B));assert tuple(B.shape)==((4,classes,classes) if arm=='shared4_graphP_privateB4' else (classes,classes))
     bodies=list(model) if arm=='joint_untied4_graphP_commonB' else [model.body if hasattr(model,'body') else model]
     for body_id,body in enumerate(bodies):
      for module in body.modules():
       if isinstance(module,factors.FactorLinear):
        assert torch.equal(module.s,torch.ones_like(module.s))
        if module is body.input_linear:
         for member in range(len(module.r)):
          generator=torch.Generator().manual_seed(source.seeds(88101,body_id+member)[1]);signs=torch.randint(0,2,(1,features),generator=generator)*2-1
          assert torch.equal(module.r[member],signs[0].float())
        else:assert torch.equal(module.r,torch.ones_like(module.r))
     rows.append({'family':family,'arm':arm,'backbone':backbone,'features':features,'classes':classes,'fast_row_initialization':True,'zero_B_and_declared_shape':family=='shared_recurrent_class_decoder','optimizer_empty_and_exact_ownership':True})
 assert len(rows)==64 and not torch.cuda.is_initialized()
 result={'schema':'native-combinations-source-initialization-CPU-QA-v1','status':'passed','cases':rows,'toy_optimizer_steps':0,'benchmark_fits':0,'real_tensor_reads':0,'CUDA_compute':0,'runtime_pin_claimed':False}
 (PACKET/'VALIDATION_INITIALIZATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','initialization_cases':len(rows),'toy_optimizer_steps':0}))
if __name__=='__main__':main()
