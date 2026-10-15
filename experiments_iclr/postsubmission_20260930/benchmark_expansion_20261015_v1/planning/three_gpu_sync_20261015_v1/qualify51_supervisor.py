import subprocess,os,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1';Q=E/'research/pubmed_three_gpu_queue_v1';children=[];started=time.time()
for index,kind,uuid in [(0,'native','GPU-9c6c20fa-a620-479d-f72f-163ac74deef2'),(1,'structured','GPU-30283973-140d-9e7b-e780-0fcc487dec54')]:
 out=E/('qualification/pubmed51_'+kind+'_gpu'+str(index));log=out.with_suffix('.log');assert not out.exists() and not log.exists()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 cmd=['timeout','--signal=TERM','--kill-after=10s','960s',str(R/'.venv/bin/python'),'-B','-u',str(Q/'qualify51.py'),'--gpu-index',str(index),'--kind',kind,'--output',str(out),'--source-sha256',{'qualify51.py': '839a4d8575f7e54247522934ffec68acd22270d5c7b6f18b479407ac79f93d31', 'binding.py': '88373842378dd79d1242c067331409541547132fd806796f04285085b5861096'}['qualify51.py'],'--binding-sha256',{'qualify51.py': '839a4d8575f7e54247522934ffec68acd22270d5c7b6f18b479407ac79f93d31', 'binding.py': '88373842378dd79d1242c067331409541547132fd806796f04285085b5861096'}['binding.py'],'--root-resource-admitted']
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('xb')as f:p=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 children.append((p,{'kind':kind,'gpu_index':index,'uuid':uuid,'pid':p.pid,'command':cmd,'output':str(out)}))
(R/'QUALIFY51_LAUNCHES.json').write_text(json.dumps([r for _,r in children],indent=2)+'\n')
for p,r in children:r['returncode']=p.wait()
(R/'QUALIFY51_TERMINAL.json').write_text(json.dumps({'elapsed_seconds':time.time()-started,'children':[r for _,r in children]},indent=2)+'\n')
