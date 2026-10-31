from pathlib import Path
import importlib.util,argparse,json,os,sys,traceback
print('START_G0_METADATA_ADMISSION',flush=True)
os.environ['CUDA_VISIBLE_DEVICES']='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
root=Path(__file__).resolve().parents[1]
path=root/'amazon_G0_complete_pool_comparison_preparation_20261006_v2/compare_complete.py'
spec=importlib.util.spec_from_file_location('_exact_G0_preT_metadata_admission',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
try:
 spec.loader.exec_module(m)
 print('LOADED_STANDARD_LIBRARY_ONLY_SOURCE',flush=True)
 scope=root/'amazon_G0_complete_comparison_activation_root_20261006_v2/COMPARISON_SCOPE.json'
 import hashlib
 context=m.authenticate(argparse.Namespace(source_root=str(root),scope=str(scope),scope_sha256=hashlib.sha256(scope.read_bytes()).hexdigest()))
 assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
 print(json.dumps({'full_pre_Torch_admission':True,'scoring_or_prediction':False,'label_payload_access':False}),flush=True)
except BaseException as e:
 print(json.dumps({'admission_error':str(e),'type':type(e).__name__,'traceback':traceback.format_exc()}),flush=True)
 raise
