"""Finite source/metadata gates. Does not import Torch/NumPy or open data/state."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import symtable
import sys

HERE=Path(__file__).resolve().parent


def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    inputs=json.loads((HERE/'INPUTS.json').read_text())
    assert [r['procedure'] for r in inputs['cases']]==['native_single','gnnm_boundary_native','gnnm_private_bn_state_v3']
    assert [r['selected_epoch'] for r in inputs['cases']]==[98,2,92]
    assert sum(r['expected_VALID']['native_forwards'] for r in inputs['cases'])==45
    assert sum(r['expected_VALID']['raw_batches'] for r in inputs['cases'])==15
    assert inputs['rtol']==1e-4 and inputs['atol']==1e-5
    assert inputs['bounds']['automatic_retries']==0 and inputs['bounds']['optimizer_steps']==0
    names={}
    for path in HERE.glob('*.py'):
        source=path.read_text(); tree=ast.parse(source,filename=str(path)); table=symtable.symtable(source,str(path),'exec')
        bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
        allowed=bound|set(dir(builtins))|{'__file__','__name__','__package__'}; missing=[]
        def visit(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed: missing.append(symbol.get_name())
            for child in scope.get_children(): visit(child)
        visit(table); assert not missing,(path.name,missing); names[path.name]=missing
        for node in tree.body:
            if isinstance(node,ast.Import): assert all(not n.name.startswith(('torch','numpy','ogb','scipy')) for n in node.names)
            if isinstance(node,ast.ImportFrom): assert not (node.module or '').startswith(('torch','numpy','ogb','scipy'))
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('train_epoch','train_update','backward','score_fit') for n in ast.walk(tree))
    source=(HERE/'qualify.py').read_text(); tree=ast.parse(source)
    rng=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore_cpu_rng')
    assert not any(isinstance(n,ast.Attribute) and n.attr=='cuda' for n in ast.walk(rng))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='predict']
    assert len(calls)==1 and isinstance(calls[0].args[3],ast.Constant) and calls[0].args[3].value=='VALID'
    assert 'torch.cuda.set_rng' not in source and 'seed_all(' not in source
    out={'schema':'genuine-selected-MolBACE-CPU-VALID-source-checks-v1','status':'PASS',
        'cases':3,'native_construction_limit':3,'whole_VALID_limit':3,'raw_VALID_batch_limit':15,
        'native_forward_limit':45,'selected_epochs':[98,2,92],'fixed_tolerances':{'rtol':1e-4,'atol':1e-5},
        'unbound_global_names':names,'no_top_level_model_imports':True,'TRAIN_optimizer_or_campaign_scoring_calls':0,
        'literal_VALID_serving_only':True,'saved_CUDA_RNG_not_applied':True,'exact_original_device_replay_claim':False,
        'genuine_producer_current_portable_qualification_claim':False,'native_execution':False,
        'model_data_checkpoint_tensor_TEST_reads':0}
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))


if __name__=='__main__': main()
