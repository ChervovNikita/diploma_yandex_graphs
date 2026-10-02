
import hashlib,json,os,pathlib,platform,resource,sys,time,unittest
packet=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
manifest=json.loads((packet/'MANIFEST.json').read_text())
for row in manifest['files']:
 p=packet/row['path']
 if not p.resolve().is_relative_to(packet) or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
  raise ValueError('Immutable source changed')
sys.path.insert(0,str(packet))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU-only visibility required')
started=time.monotonic()
import torch,torch_geometric
torch.set_num_threads(1); torch.set_num_interop_threads(1)
if torch.__version__!='2.1.2+cu118' or torch_geometric.__version__!='2.7.0':
 raise ValueError('Retained runtime versions differ')
from torch_geometric.nn import GATConv
import inspect
gat_source=pathlib.Path(inspect.getsourcefile(GATConv))
if hashlib.sha256(gat_source.read_bytes()).hexdigest() != next(r['sha256'] for r in manifest['files'] if r['path']=='sources/gat_conv_2_7_pinned.py'):
 raise ValueError('Imported GAT source differs from preparation pin')
events=[]
class Result(unittest.TextTestResult):
 def startTest(self,test):
  self.start=time.monotonic(); super().startTest(test)
 def note(self,test,state,detail=None):
  events.append({'test':test.id(),'state':state,'seconds':time.monotonic()-self.start,'detail':detail})
 def addSuccess(self,test):
  self.note(test,'passed'); super().addSuccess(test)
 def addSkip(self,test,reason):
  self.note(test,'unsupported_backend_skip',reason); super().addSkip(test,reason)
 def addFailure(self,test,err):
  self.note(test,'failed',self._exc_info_to_string(err,test)); super().addFailure(test,err)
 def addError(self,test,err):
  self.note(test,'error',self._exc_info_to_string(err,test)); super().addError(test,err)
suite=unittest.defaultTestLoader.loadTestsFromName('test_ports')
with (output/'unittest.txt').open('x') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
receipt={'schema':'efficient-graph-ports-cpu-qualification-v1','tests_run':result.testsRun,
 'passed':sum(r['state']=='passed' for r in events),'skips':len(result.skipped),
 'failures':len(result.failures),'errors':len(result.errors),'events':events,
 'seconds':time.monotonic()-started,'torch':torch.__version__,'pyg':torch_geometric.__version__,
 'python':platform.python_version(),'GAT_source_sha256':hashlib.sha256(gat_source.read_bytes()).hexdigest(),
 'torch_source_path':torch.__file__,'threads':torch.get_num_threads(),
 'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'cuda_initialized':torch.cuda.is_initialized(),'GPU_computation':False,
 'inputs':'six-node synthetic engineering fixtures only; no dataset or role files',
 'predictive_or_efficiency_result':False,'skips_never_qualify_a_backend':True,
 'all_tests_passed_without_skips':result.wasSuccessful() and not result.skipped}
with (output/'QUALIFICATION.json').open('x') as stream:
 json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(receipt,allow_nan=False))
raise SystemExit(0 if result.wasSuccessful() else 1)
