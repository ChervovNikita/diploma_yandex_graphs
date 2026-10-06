"""Own one detached CPU D2 child, immutable identity, separate logs and wait4 accounting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,socket,subprocess,time
HERE=Path(__file__).resolve().parent
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
def now():return datetime.now(timezone.utc).isoformat()
def write(n,v):
 with (HERE/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 (HERE/n).chmod(0o444)
def stat(pid):
 raw=(Path('/proc')/str(pid)/'stat').read_text();f=raw[raw.rfind(')')+2:].split();return {'PID':pid,'start_ticks':int(f[19]),'state':f[0]}
cfg=json.loads((HERE/'CONFIG.json').read_text());argv=cfg['argv'];env=dict(os.environ);env.update(cfg['child_environment'])
write('SUPERVISOR_STARTED.json',{'UTC':now(),'identity':stat(os.getpid()),'pgid':os.getpgid(0),'sid':os.getsid(0),'child_invocation_limit':1})
start=time.monotonic();signals=[];samples=[];peak=0
with (HERE/'child.stdout').open('xb') as out,(HERE/'child.stderr').open('xb') as err:
 child=subprocess.Popen(argv,cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 owner=stat(child.pid);owner.update({'pgid':os.getpgid(child.pid),'sid':os.getsid(child.pid),'argv':argv,'cwd':str(REPO)})
 def exact():
  p=Path('/proc')/str(owner['PID'])
  try:s=stat(owner['PID'])
  except FileNotFoundError:return False
  actual=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  return s['start_ticks']==owner['start_ticks'] and actual==argv and str((p/'cwd').resolve())==str(REPO) and os.getpgid(child.pid)==owner['pgid'] and os.getsid(child.pid)==owner['sid']
 assert owner['pgid']==owner['sid']==child.pid and exact()
 write('CHILD_STARTED.json',{'UTC':now(),'identity':owner,'runtime_spec':cfg['runtime_spec'],'environment':cfg['child_environment'],'limits':{'seconds':600,'RSS_bytes':8*1024**3},'signals_sent':[]})
 reason=None
 while True:
  waited,status,usage=os.wait4(child.pid,os.WNOHANG)
  if waited:break
  elapsed=time.monotonic()-start;rss=0
  if exact():
   for line in (Path('/proc')/str(child.pid)/'status').read_text().splitlines():
    if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
  peak=max(peak,rss);samples.append({'elapsed_seconds':elapsed,'RSS_bytes':rss})
  if reason is None and (elapsed>=600 or rss>8*1024**3):
   reason='600_SECOND_LIMIT' if elapsed>=600 else '8_GiB_RSS_LIMIT'
   if exact():
    os.killpg(owner['pgid'],signal.SIGKILL);signals.append({'UTC':now(),'signal':'SIGKILL','registered_identity_verified':True,'reason':reason})
   else:signals.append({'UTC':now(),'signal_sent':False,'reason':'OWNERSHIP_MISMATCH_SIGNAL_REFUSED'})
  time.sleep(0.25)
 rc=os.waitstatus_to_exitcode(status);child.returncode=rc
write('RESOURCES.json',{'sampling_period_seconds':0.25,'max_sampled_RSS_bytes':peak,'samples':samples,'wait4':{'ru_maxrss_bytes':usage.ru_maxrss*1024,'user_seconds':usage.ru_utime,'system_seconds':usage.ru_stime}})
write('EXIT.json',{'UTC':now(),'child_identity':owner,'exit_code':rc,'raw_wait4_status':status,'exit_code_authority':'os.wait4','terminal_wait_observed':True,
 'elapsed_seconds':time.monotonic()-start,'reason':reason,'signals_sent':signals,'attempts':1,'retry_or_resume':False,'source_sha256':cfg['source_sha256'],
 'release_sha256':cfg['release_sha256'],'max_sampled_RSS_bytes':peak,'wait4_ru_maxrss_bytes':usage.ru_maxrss*1024,'stdout_bytes':(HERE/'child.stdout').stat().st_size,'stderr_bytes':(HERE/'child.stderr').stat().st_size})
for n in ('child.stdout','child.stderr'):(HERE/n).chmod(0o444)
