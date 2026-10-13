"""Independent native-library Xavier/stream proof for the sole Torch2.1 API bridge."""
import json,sys
from pathlib import Path
S=Path(__file__).resolve().parent.parent;P=S.parent;sys.path.insert(0,str(S/'runtime'))
import torch
from adapters.cardinality import sources
B=json.loads((S/'CARDINALITY_SOURCE_BINDINGS.json').read_text());rows=[]
for seed in (0,1,2):
 m,_,_,_=sources(torch,P,B,seed);g=torch.Generator(device='cpu').manual_seed(m['counter'].separated_seed('count-init'));initial=g.get_state().clone();shapes=[(16,270),(16,16),(1,16)];actual=[]
 for shape in shapes:
  value=torch.empty(shape,dtype=torch.float32);m['model']._generator_Xavier_primitive(value,g);actual.append(value.clone())
 final=g.get_state().clone()
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(initial);expected=[]
  for shape in shapes:
   value=torch.empty(shape,dtype=torch.float32);torch.nn.init.xavier_uniform_(value);expected.append(value.clone())
  assert torch.equal(torch.get_rng_state(),final)
 assert all(torch.equal(a,b) for a,b in zip(actual,expected));rows.append({'seed':seed,'matrix_shapes':shapes,'native_library_values_bitwise_identical':True,'native_generator_final_state_bitwise_identical':True})
v={'schema':'C64-generator-API-independent-native-Xavier-proof-v1','status':'passed','cases':rows,'model_constructors':0,'optimizer_objects':0,'actual_input_arrays_decoded':0,'scientific_fits':0,'CUDA':False};(S/'GENERATOR_API_NATIVE_CPU_PROOF.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'status':'passed','seed_cases':3,'all_native_weights_and_generatorstates_bitwise_equal':True,'scientific_fits':0}))
