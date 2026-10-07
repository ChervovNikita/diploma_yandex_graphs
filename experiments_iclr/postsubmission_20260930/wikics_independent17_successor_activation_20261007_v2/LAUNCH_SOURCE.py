import base64,hashlib,json,socket,subprocess,sys,time,zlib
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
a=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
for row in a['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
root=P/a['activation'];assert not (root/'DETACHED_LAUNCH.json').exists();cfg=root/'CONFIG.json';assert hashlib.sha256(cfg.read_bytes()).hexdigest()==a['config_sha256'];config=json.loads(cfg.read_text())
assert not (P/config['execution_root']).exists()
for b in config['bindings']:assert hashlib.sha256((P/b['path']).read_bytes()).hexdigest()==b['sha256']

# Real metadata-only admission validates binding objects before any quality fit.
bank=P/'wikics_private_graph_controls_execution_root_20261007_v2/runs/DONOR_INDEPENDENT_17'
freeze=json.loads((bank/'DONOR_FREEZE.json').read_text());assert freeze['complete'] and len(freeze['members'])==4
checked=[]
for i,row in enumerate(freeze['members']):
 f=P/row['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'];d=json.loads(f.read_text());assert d['epochs']==1100 and d['seed']==17+1009*i
 for artifact in (d['selected'],d['end']):
  p=(P/artifact['path']).resolve(strict=True);assert p.is_relative_to(P);assert hashlib.sha256(p.read_bytes()).hexdigest()==artifact['sha256'];checked.append({'path':artifact['path'],'sha256':artifact['sha256']})
(root/'ARTIFACT_BINDING_SMOKE.json').write_text(json.dumps({'passed':True,'checked_artifacts':checked,'new_training_updates':0,'quality_values_read':False},indent=2)+'\n')

argv=['/usr/bin/python3','-I','-S','-B',str(root/'execute.py'),'--config-sha256',a['config_sha256']]
with (root/'owner.stdout.log').open('xb') as out,(root/'owner.stderr.log').open('xb') as err:
 child=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
time.sleep(.3);r={'PID':child.pid,'argv':argv,'exit_code':child.poll(),'waiter_only_at_launch':True,'fixed_successor_arms':['I_native17','U_stage17'],'dependency_restarted':False,'config_sha256':a['config_sha256']}
if child.poll() is None:
 s=(Path('/proc')/str(child.pid)/'stat').read_text();f=s[s.rfind(')')+2:].split();r.update(start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]))
(root/'DETACHED_LAUNCH.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
