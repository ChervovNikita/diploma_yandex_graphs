"""Synthetic CPU checks for lossless shared checkpoint storage."""
import io
import importlib.util
import json
from pathlib import Path
import sys
import torch
torch.set_num_threads(2)

spec = importlib.util.spec_from_file_location('owned_alias_engine', sys.argv[1])
engine = importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
checks = []
def passed(name):checks.append(name)

shared = torch.nn.Linear(8, 4)
model = torch.nn.ModuleList([shared, shared, shared, shared])
state = model.state_dict()
clone = engine._clone_tree(torch, state)
assert engine._equal_tree(torch, clone, state)
assert clone['0.weight'].data_ptr() == clone['3.weight'].data_ptr()
assert clone['0.weight'].data_ptr() != state['0.weight'].data_ptr()
assert clone._metadata == state._metadata and clone._metadata is not state._metadata
passed('shared_module_exact_values_aliases_metadata_independent_snapshot')

first = clone['0.weight'].clone()
with torch.no_grad():shared.weight.add_(1)
assert torch.equal(clone['0.weight'], first)
second = engine._clone_tree(torch, model.state_dict())
assert not torch.equal(second['0.weight'], first)
assert second['0.weight'].data_ptr() != clone['0.weight'].data_ptr()
passed('source_mutation_and_later_snapshot_do_not_mutate_saved_state')

x = torch.arange(24, dtype=torch.float64).reshape(4, 6)
views = {'a':x.T, 'b':x.T, 'c':x[:,1:3], 'd':x[:,2:4], 'independent':x.T.clone()}
y=engine._clone_tree(torch, views)
assert engine._equal_tree(torch, views, y)
assert y['a'].data_ptr() == y['b'].data_ptr()
assert y['a'].data_ptr() != y['independent'].data_ptr()
assert y['c'].data_ptr() != y['d'].data_ptr()
passed('identical_noncontiguous_views_reused_other_views_and_bodies_separate')

c=torch.tensor([1+2j,3-4j]);s=torch.sparse_coo_tensor(torch.tensor([[0,1],[1,0]]),torch.tensor([2.,3.]),(2,2))
mixed={'positive':c,'conjugate':c.conj(),'sparse':s,'empty':torch.empty(0),'integer':torch.tensor([2]),'tuple':(1,None),'list':[True,2.]}
rest=engine._clone_tree(torch,mixed)
assert engine._equal_tree(torch,mixed,rest)
assert rest['positive'].data_ptr()!=rest['conjugate'].data_ptr()
passed('complex_conjugate_sparse_empty_integer_and_containers_preserved')

snapshot=io.BytesIO();torch.save(clone,snapshot);snapshot.seek(0)
loaded=torch.load(snapshot,map_location='cpu',weights_only=False)
assert engine._equal_tree(torch,clone,loaded)
assert loaded['0.weight'].data_ptr()==loaded['3.weight'].data_ptr()
model.load_state_dict(loaded,strict=True)
assert torch.equal(model[0].weight,first)
assert model[0].weight is model[3].weight
passed('torch_serializer_and_shared_model_restore_preserve_values_and_aliases')

optimizer=torch.optim.Adam(model.parameters(),lr=.01)
model[0](torch.ones(2,8)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
opt=engine._clone_tree(torch,optimizer.state_dict())
original=engine._clone_tree(torch,optimizer.state_dict())
for p in model.parameters():p.grad=torch.ones_like(p)
optimizer.step();optimizer.load_state_dict(opt)
assert engine._equal_tree(torch,optimizer.state_dict(),original)
passed('synthetic_paid_Adam_state_restore_is_independent_and_exact')

print(json.dumps({'checks_passed':checks,'synthetic_Adam_maps':2,'scientific_fits':0,'CUDA_initialized':torch.cuda.is_initialized(),'GO':False}))
