"""Exercise the exact relative helper against the genuine strict scorer contract."""
import ast, copy, hashlib, importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
def load_helper(path):
 tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','relative')]
 scope={'E':E,'Path':Path}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
 return scope['relative']
def reject(f):
 try:f()
 except ValueError:return
 raise AssertionError('Expected refusal')
def main():
 path=E/'composable_normalization_final_scoring_v3/contracts.py'
 spec=importlib.util.spec_from_file_location('_strict_serving_contract',path)
 contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
 source=json.loads((HERE/'INPUTS.json').read_text())['scorer'];before=copy.deepcopy(source)
 old=load_helper(HERE.with_name('composed_BUDDY_selected_VALID_v1')/'replay.py')
 current=load_helper(HERE/'replay.py')
 reject(lambda:contract.verify(old(source),E))
 receipt=current(source)
 assert set(receipt)=={'path','bytes','sha256'} and contract.verify(receipt,E)==E/source['path_relative_to_expansion']
 assert source==before
 reject(lambda:current({**source,'extra':0}))
 reject(lambda:current({k:v for k,v in source.items() if k!='bytes'}))
 reject(lambda:current({**source,'path_relative_to_expansion':'../escape'}))
 reject(lambda:current({**source,'path_relative_to_expansion':str(E/'absolute')}))
 original=ast.parse((HERE.with_name('composed_BUDDY_selected_VALID_v1')/'replay.py').read_text())
 repaired=ast.parse((HERE/'replay.py').read_text())
 for tree in (original,repaired):tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='relative']
 assert ast.dump(original)==ast.dump(repaired)
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules
 return {'status':'PASS','genuine_strict_scorer_accepts_exact_receipt':True,'old_failure_reproduced':True,'input_unchanged':True,'malformed_and_escaping_receipts_rejected':True,'all_other_replay_AST_unchanged':True,'TRAIN_updates':0,'native_executions':0}
if __name__=='__main__':print(json.dumps(main(),indent=2,sort_keys=True))
