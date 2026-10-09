from pathlib import Path
import hashlib,json,os,socket,subprocess,time,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'geometry_only_core_full_input_qualification_activation_root_20261009_v1';D=P/'geometry_only_core_full_input_qualification_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not(A/'LAUNCH.json').exists() and not(A/'OWNER.log').exists() and not D.exists()
program=A/'SUPERVISOR.py';assert hashlib.sha256(program.read_bytes()).hexdigest()=='cca89b66d105fa932ad0469f42aa76715f3d12af8d29b73541f187f87424568e'
release=A/'RELEASE.json';assert hashlib.sha256(release.read_bytes()).hexdigest()=='fdbf2d4343c4a1ab9a206718874f06fc0ebaf401df16e9821498d153a9cac3d2'
head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip();assert head=='9a8ede4395d9b9d0eb7215b7c61f8d28211f82c7'
with(A/'OWNER.log').open('xb') as log:
 child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(program)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
s=Path('/proc',str(child.pid),'stat').read_text();v=s[s.rfind(')')+2:].split();identity={'pid':child.pid,'start_ticks':int(v[19]),'group':int(v[2]),'session':int(v[3]),'state':v[0],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()};assert identity['group']==identity['session']==child.pid
record={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':identity,'source_commit':head,'supervisor_sha256':'cca89b66d105fa932ad0469f42aa76715f3d12af8d29b73541f187f87424568e','release_sha256':'fdbf2d4343c4a1ab9a206718874f06fc0ebaf401df16e9821498d153a9cac3d2','engineering_TRAIN_forwards':8,'engineering_Adam_steps':2,'scientific_metric_access':False,'normal_host':True,'automatic_retry':False,'scientific_results_opened':False,'TEST_truth_accessed':False}
with(A/'LAUNCH.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
