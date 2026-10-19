"""Stage additive scheduler repairs and resume existing queues without replacing science."""
from pathlib import Path
import base64, hashlib, json, sys, zlib
HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
sys.path[:0]=[str(E),str(E/'planning/three_gpu_sync_20261015_v1')]
import remote_control,remote51

def call(host,code,timeout=45):
    compile(code, '<guarded-repair-command>', 'exec')
    remote=remote_control if host=='allocation' else remote51
    text='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(code.encode())).decode())+')))'
    assert len(text)<5000,len(text)
    return json.loads(remote.call(text,timeout=timeout))

def root(host):
    return remote_control.REMOTE_ROOT if host=='allocation' else remote51.ROOT

def expansion(host):
    return root(host)+'/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'

def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2)
    return value

def stage(host,package):
    folder=E/package
    manifest=json.loads((folder/'MANIFEST.json').read_text())
    files=[row['path'] for row in manifest['files']]+['MANIFEST.json']
    records=[]
    for relative in files:
        data=(folder/relative).read_bytes()
        packed=zlib.compress(data);digest=hashlib.sha256(data).hexdigest()
        chunks=[packed[i:i+1600] for i in range(0,len(packed),1600)]
        for i,chunk in enumerate(chunks):
            code="""from pathlib import Path
import json,base64
p=Path(ROOT)/'.repair_uploads_20261019'/DIGEST/PART;data=base64.b64decode(DATA)
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():assert p.read_bytes()==data
else:
 with p.open('xb') as f:f.write(data)
print(json.dumps({'bytes':len(data)}))
""".replace('ROOT',repr(expansion(host))).replace('DIGEST',repr(digest)).replace('PART',repr(str(i))).replace('DATA',repr(base64.b64encode(chunk).decode()))
            call(host,code)
        code="""from pathlib import Path
import hashlib,json,zlib
root=Path(ROOT);p=root/RELATIVE;parts=root/'.repair_uploads_20261019'/DIGEST
data=zlib.decompress(b''.join((parts/str(i)).read_bytes() for i in range(COUNT)));assert hashlib.sha256(data).hexdigest()==DIGEST
assert p.resolve().is_relative_to(root) and not p.is_symlink()
if p.exists():assert p.read_bytes()==data,'Refuse source replacement'
else:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
for i in range(COUNT):(parts/str(i)).unlink()
parts.rmdir()
print(json.dumps({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))
""".replace('ROOT',repr(expansion(host))).replace('RELATIVE',repr(package+'/'+relative)).replace('DIGEST',repr(digest)).replace('COUNT',str(len(chunks)))
        records.append(call(host,code))
    return save('REPAIR_STAGE_'+host+'_'+package+'.json',records)

def stop_allocation():
    return save('ALLOCATION_GRACEFUL_STOP_20261019.json',call('allocation', '''from pathlib import Path
import json,time
p=Path(ROOT)/'campaign_execution_20261016_v7/runs/allocation_gpu0'
s=json.loads((p/'STATUS.json').read_text());assert not s['active_fit_ids']
with (p/'STOP').open('x') as f:json.dump({'requested':time.time(),'mode':'finish children; scheduler RAM repair'},f)
print(json.dumps({'status_before':s,'children_signalled':False}))'''.replace('ROOT',repr(expansion('allocation')))))

def launch(host,worker,package,entrypoint,source_sha,plan_sha,extra=None):
    code='''from pathlib import Path
import json,hashlib,subprocess,shlex,time,os
r=Path(ROOT);e=Path(EXPANSION);worker=WORKER;key=worker.replace('/','_');q=e/'campaign_execution_20261016_v7';p=q/('PLAN-'+key+'.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()==PLAN_SHA
v=json.loads(p.read_text());s=Path(v['state_root']);session='gnnm-allfit-'+worker.replace('/','-')+'-20261016'
panes=subprocess.check_output(['tmux','list-panes','-t',session,'-F','#{pane_dead}'],text=True).splitlines();assert panes and all(x=='1' for x in panes),panes
old=json.loads((s/'STATUS.json').read_text());assert not old['active_fit_ids'];assert not Path(v['global_smoke_pass']).exists()
source=e/PACKAGE/'MANIFEST.json';assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA
config=e/'server51_campaign_port_v6/hosts'/(key+'.json');cfg=json.loads(config.read_text())
env={'BENCHMARK_HOST_CONFIG':str(config),'BENCHMARK_HOST_CONFIG_SHA256':hashlib.sha256(config.read_bytes()).hexdigest(),'CUDA_VISIBLE_DEVICES':cfg['physical_gpu_uuid'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
args=[str(r/'.venv/bin/python'),'-B','-u',str(e/PACKAGE/ENTRYPOINT),'--worker',worker,'--plan',str(p),'--plan-sha256',PLAN_SHA,'--source-sha256',SOURCE_SHA]+EXTRA
if (s/'STOP').exists():(s/'STOP').rename(s/('STOP-acknowledged-'+str(time.time_ns())+'.json'))
subprocess.check_call(['tmux','kill-session','-t',session])
subprocess.check_call(['tmux','new-session','-d','-s',session,'-c',str(r)])
subprocess.check_call(['tmux','set-option','-t',session,'remain-on-exit','on'])
log=s/('OWNER-'+str(time.time_ns())+'.log');command=['env']+[k+'='+val for k,val in env.items()]+args
script='exec '+shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1'
subprocess.check_call(['tmux','send-keys','-t',session,'-l',script]);subprocess.check_call(['tmux','send-keys','-t',session,'Enter'])
receipt={'session':session,'worker':worker,'argv':args,'environment':env,'owner_log':str(log),'prior_status':old,'source_sha256':SOURCE_SHA,'all_prior_attempts_preserved':True,'launched_at':time.time()}
with (s/('REPAIR_LAUNCH-'+str(time.time_ns())+'.json')).open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    for key,value in {'ROOT':root(host),'EXPANSION':expansion(host),'WORKER':worker,'PLAN_SHA':plan_sha,'SOURCE_SHA':source_sha,'PACKAGE':package,'ENTRYPOINT':entrypoint,'EXTRA':extra or []}.items():code=code.replace(key,repr(value))
    return save('REPAIR_LAUNCH_'+worker.replace('/','_')+'_20261019.json',call(host,code))

if __name__=='__main__':
    if sys.argv[1]=='stage':print(json.dumps(stage(*sys.argv[2:])))
    elif sys.argv[1]=='stop-allocation':print(json.dumps(stop_allocation()))
