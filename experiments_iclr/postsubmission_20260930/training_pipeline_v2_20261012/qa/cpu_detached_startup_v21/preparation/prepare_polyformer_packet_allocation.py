"""Freeze PolyFormer sources/data, run CPU fixtures, compile disabled studies."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE = Path(__file__).resolve().parents[1]
HERE = PACKAGE/"preparation"
DESTINATION = "massive_polyformer_packet_20261012_v1"
LABEL = "polyformer_allocation_20261012_v1"

TRANSFER = GUARD+r'''
transport=phase/(sys.argv[1]+'_transport');target=transport/'payload.b64';offset=int(sys.argv[2])
if offset==0:assert not transport.exists();transport.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as stream:stream.write(sys.argv[3].encode())
print(target.stat().st_size)
'''

REMOTE = GUARD+r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
package=stage/'sources/training_pipeline_v2_20261012';package.mkdir(parents=True)
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes']
 p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[{k:v for k,v in row.items() if k!='data'} for row in payload['files']])
hosts={'allocation':{'status':'verified_live','hostname':socket.gethostname(),'gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],
 'scientific_training_runs':0,'verification_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}}
atomic(stage/'HOST_BINDINGS.json',hosts)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GNNM_REFERENCE_PROJECT=str(phase))
tmp=stage/'engineering_tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
records=[]
def run(command,label,timeout):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'scientific_training_runs':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate();stderr+='\nBounded engineering/preparation timeout retained.\n'
 value={'phase':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,
  'scientific_training_runs':0,'benchmark_scores':0}
 atomic(stage/(label+'_RESULT.json'),value);records.append(value);atomic(stage/'PROGRESS.json',records);print(json.dumps(value),flush=True)
 return value
qa=run([str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p','test_polyformer*.py','-v'],'POLYFORMER_CPU_QA',300)
if qa['exit_code']!=0:
 atomic(stage/'PREPARATION_SUMMARY.json',{'status':'blocked_CPU_QA','records':records,'scientific_fits':0,'GO_created':False})
 sys.exit(1)
custodies=[];ancestry=[]
for dataset in ['citeseer','pubmed','coauthor-cs','coauthor-physics']:
 old=phase/'training_pipeline_v2_20261012/prepared_data'/dataset
 source=old/'CUSTODY.json';custody=json.loads(source.read_text())
 assert custody['dataset_id']==dataset and custody['status']=='prepared' and custody['benchmark_fits']==0 and custody['predictive_scores']==0
 destination=stage/'data'/dataset;destination.mkdir(parents=True)
 for row in custody['files']:
  relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
  original=old/relative;observed=receipt(original)
  assert observed['bytes']==row['bytes'] and observed['sha256']==row['sha256'] and not original.is_symlink()
  target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
  assert receipt(target)['sha256']==row['sha256'];target.chmod(0o444)
 target=destination/'CUSTODY.json';shutil.copyfile(source,target);target.chmod(0o444)
 ancestry.append({'dataset':dataset,'original_custody':receipt(source),'copied_custody':receipt(target),
  'heldout_target_tensors_decoded':False,'prior_data_modified':False})
 custodies.append(str(target))
atomic(stage/'DATA_CUSTODY_ANCESTRY.json',ancestry)
profiles={}
for dataset in ['citeseer','pubmed','coauthor-cs','coauthor-physics']:
 profiles[dataset+'/PolyFormer']={'reservation_mib':65536,'wall_seconds':315360000,
  'estimated_seconds_per_update':.1,'evidence':{'status':'PROVISIONAL_UNMEASURED_REQUIRES_ROOT_QUALIFICATION',
   'assumption':'M1-equivalent scheduling placeholder; route/view multiplier explicit; broad geometry/token-bank/CUDA peaks unqualified',
   'no_optimizer_or_GPU_qualification_performed':True}}
atomic(stage/'RESOURCE_ESTIMATES.json',{'profiles':profiles,'scientific_launch_enabled':False})
result=run([str(repo/'.venv/bin/python'),'-B','-u',str(package/'compile_polyformer_matrix.py'),
 '--project-root',str(phase),'--custody',*custodies,'--resources',str(stage/'RESOURCE_ESTIMATES.json'),
 '--host-bindings',str(stage/'HOST_BINDINGS.json'),'--output',str(stage/'matrix'),'--scalar-coefficients','neighbors'],
 'DISABLED_POLYFORMER_COMPILER',600)
value={'schema':'polyformer-allocation-preparation-v1','status':'prepared' if result['exit_code']==0 else 'blocked_compiler',
 'destination':str(stage.relative_to(phase)),'CPU_QA_passed':True,'copied_real_custodies':custodies,
 'scientific_fits':0,'scores':0,'GO_created':False,'sources_outside_live_pipeline':True,
 'resource_status':'PROVISIONAL_UNMEASURED_REQUIRES_ROOT_QUALIFICATION',
 'records':[{k:v for k,v in row.items() if k not in ['stdout','stderr']} for row in records]}
atomic(stage/'PREPARATION_SUMMARY.json',value);print(json.dumps({'final_summary':value}),flush=True)
sys.exit(result['exit_code'])
'''


def main():
    out = HERE/LABEL
    if out.exists():
        raise FileExistsError("Retain prior receipts; use a fresh explicit label")
    out.mkdir()
    files=[]
    for path in sorted(PACKAGE.rglob("*.py")):
        if "qa" in path.relative_to(PACKAGE).parts:
            continue
        raw=path.read_bytes()
        files.append({"path":str(path.relative_to(PACKAGE)),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"data":base64.b64encode(raw).decode()})
    for relative in ("docs/MATRIX_AXES_DRAFT.json","docs/POLYFORMER_PACKET.md","docs/POLYFORMER_MATRIX.md","ENGINE_README.md"):
        raw=(PACKAGE/relative).read_bytes()
        files.append({"path":relative,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"data":base64.b64encode(raw).decode()})
    encoded=base64.b64encode(zlib.compress(json.dumps({"destination":DESTINATION,"files":files}).encode())).decode()
    (out/"SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in row.items() if k!='data'} for row in files],indent=2)+"\n")
    for offset in range(0,len(encoded),24000):
        result=invoke(TRANSFER,[DESTINATION,str(offset),encoded[offset:offset+24000]],out,"TRANSFER_"+str(offset),45)
        if result["exit_code"]!=0:
            print(json.dumps(result));raise SystemExit(1)
    result=invoke(REMOTE,[DESTINATION,hashlib.sha256(encoded.encode()).hexdigest()],out,"TRANSPORT",1000)
    print(result["stdout"])
    if result["exit_code"]!=0:
        print(result["stderr"]);raise SystemExit(1)


if __name__ == "__main__":
    main()
