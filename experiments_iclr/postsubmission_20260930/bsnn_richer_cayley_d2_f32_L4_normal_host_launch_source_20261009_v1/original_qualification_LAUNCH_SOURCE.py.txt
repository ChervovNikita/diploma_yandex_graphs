from pathlib import Path
import hashlib,json,os,socket,subprocess,time,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'bsnn_full_input_qualification_activation_root_20261009_v2';D=P/'bsnn_full_input_qualification_execution_root_20261009_v2'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not(A/'LAUNCH.json').exists() and not(A/'OWNER.log').exists() and not D.exists()
program=A/'SUPERVISOR.py';assert hashlib.sha256(program.read_bytes()).hexdigest()=='9c37e9a73a3b4268a72695f9876ec2cd3e5980f22b9121c47abed1c423f3a067'
release=A/'RELEASE.json';assert hashlib.sha256(release.read_bytes()).hexdigest()=='c834f66088e72fafbececce11a66cb58b48111872669c594abbded69c381995c'
head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip();assert head=='817bdd4f02df596e6f572938c91ade40e826d700'
with(A/'OWNER.log').open('xb') as log:
 child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(program)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
s=Path('/proc',str(child.pid),'stat').read_text();v=s[s.rfind(')')+2:].split();identity={'pid':child.pid,'start_ticks':int(v[19]),'group':int(v[2]),'session':int(v[3]),'state':v[0],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()};assert identity['group']==identity['session']==child.pid
record={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':identity,'source_commit':head,'supervisor_sha256':'9c37e9a73a3b4268a72695f9876ec2cd3e5980f22b9121c47abed1c423f3a067','release_sha256':'c834f66088e72fafbececce11a66cb58b48111872669c594abbded69c381995c','engineering_updates':1,'full_model_eval_calls':8,'normal_host':True,'automatic_retry':False,'scientific_results_opened':False,'TEST_truth_accessed':False}
with(A/'LAUNCH.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
