"""Root-reviewed finite gate launch. No campaign or automatic retries."""
from pathlib import Path
import argparse,json,sys
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from remote_control import call,REMOTE_ROOT,REMOTE_EXPANSION
p=argparse.ArgumentParser();p.add_argument('gate',choices=['buddy','ncnc']);a=p.parse_args()
commit=json.loads((HERE/'COMMIT_RECEIPT_07.json').read_text())['commit']
code='''import json,subprocess,os,time,hashlib
from pathlib import Path
r=Path(__REPO__);e=Path(__EXP__);mode=__MODE__;commit=__COMMIT__
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()==commit
push=json.loads((e/'PUSH_RECEIPT_07.json').read_text())
# Exact push is checked separately by the publication helper; preserve its receipt.
assert commit in json.dumps(push)
pid=640428;p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
assert fields[19]=='6055039165' and fields[0]!='Z'
peer={'pid':pid,'start_ticks':fields[19],'peak_host_RSS_bytes':4*1024**3,'peak_GPU_bytes':32*1024**3}
rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE');hwm=int(next(v.split()[1]for v in(p/'status').read_text().splitlines()if v.startswith('VmHWM:')))*1024;assert max(rss,hwm)<=peer['peak_host_RSS_bytes']
c=Path('/sys/fs/cgroup');stats=dict(line.split()for line in(c/'memory.stat').read_text().splitlines());current=int((c/'memory.current').read_text());maximum=(c/'memory.max').read_text().strip()
host=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024
hard=host if maximum=='max'else max(int(maximum)-current,0);reclaim=max(int(stats.get('inactive_file',0))-int(stats.get('shmem',0)),0)
available=min(host,hard+reclaim)-(peer['peak_host_RSS_bytes']-rss)
resources={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'peer':peer,'peer_current_RSS_bytes':rss,'peer_lifetime_HWM_bytes':hwm,'cache_aware_available_after_peer_growth_bytes':available,'memory_events_before':(c/'memory.events').read_text(),'peer_envelope_basis':'Host4GiB estimated from this process lifetime2.51GiB observed peak over28completed fit banks and same remaining native data shapes. GPU32GiB uses the prior admitted hard limit. Host estimate is not a proven bound.'}
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
if mode=='buddy':
 assert available>=12*1024**3
 q=e/'qualification/buddy_production_conditional_seed1_cpu_v2';out=e/'results/buddy_production_conditional_seed1_cpu_v2_attempt01'
 admission={'schema':'root-finite-BUDDY-admission-v1','publication_commit':commit,'manifest_sha256':'cd49d8eb19881f195ac5eabe2af11580756d8defb0bb4470f0ae255a057c3013','output':str(out),'bounds':{'native_constructions':2,'optimizer_updates':2,'full_VALID_calls':3,'native_forwards':32,'reverse_collections':2},'resources':resources,'wall_seconds':600,'TEST_accessed':False,'full_campaign_GO':False,'automatic_retries':0}
 assert hashlib.sha256((q/'MANIFEST.json').read_bytes()).hexdigest()==admission['manifest_sha256']
 command=[str(r/'.venv/bin/python'),'-B','-u',str(q/'supervise.py'),'--root',str(r),'--output',str(out)];env['CUDA_VISIBLE_DEVICES']=''
else:
 prior=e/'results/buddy_production_conditional_seed1_cpu_v2_attempt01/SUPERVISOR.json';assert prior.exists()and json.loads(prior.read_text())['status']in('passed','failed')
 assert available>=24*1024**3
 q=e/'qualification/runtime_link_v6_native_lifecycle_driver_v1';admission=json.loads((q/'ADMISSION_TEMPLATE.json').read_text());out=e/'results/runtime_link_v6_native_lifecycle_attempt01'
 admission.update(published=True,publication_commit=commit,driver_manifest_sha256='7c15edd9b3659aed9fb90829823b76e3b1e627b0916225ca84ba2bb2e5764637',peers=[peer],NCNC_CUDA_resource_evidence_applicable=True,template_is_not_admission=False,resources=resources)
 admission['applicability_reason']='Same unchanged native NCNC backbone/source and shared four-route boundary as authentic resource receipt. Conditional-J adds input correction and20parameter readout.40GiB GPU is an explicit conservative estimate, not a measured conditional-J peak.'
 command=[str(r/'.venv/bin/python'),'-B','-u',str(q/'supervise.py'),'--output',str(out),'--admission',str(e/'qualification/NCNC_ROOT_ADMISSION_07.json')];env['CUDA_VISIBLE_DEVICES']='0'
assert not out.exists()
admp=e/'qualification'/('BUDDY_ROOT_ADMISSION_07.json'if mode=='buddy'else'NCNC_ROOT_ADMISSION_07.json')
with admp.open('x')as f:json.dump(admission,f,indent=2)
log=e/'qualification'/(mode+'_release07_supervisor.log')
with log.open('xb')as f:
 child=subprocess.Popen(command,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
result={'supervisor_pid':child.pid,'command':command,'output':str(out),'admission':admission,'TEST_accessed':False,'full_campaign_GO':False,'automatic_retries':0}
with(e/'qualification'/(mode+'_release07_LAUNCH.json')).open('x')as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
for token,value in [('__REPO__',REMOTE_ROOT),('__EXP__',REMOTE_EXPANSION),('__MODE__',a.gate),('__COMMIT__',commit)]:code=code.replace(token,repr(value))
value=json.loads(call(code,timeout=45).splitlines()[-1]);out=HERE/'qualification'/(a.gate+'_release07_LAUNCH.json')
with out.open('x')as f:json.dump(value,f,indent=2)
print(json.dumps({k:value[k]for k in ['supervisor_pid','output','full_campaign_GO','automatic_retries']}))
