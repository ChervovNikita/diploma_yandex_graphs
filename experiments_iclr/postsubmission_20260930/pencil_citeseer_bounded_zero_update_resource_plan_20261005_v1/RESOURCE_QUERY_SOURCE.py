from pathlib import Path
from datetime import datetime,timezone
import csv,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,check=True,timeout=30)
 return dict(args=args,stdout=r.stdout,stderr=r.stderr)
uuid=call(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
assert uuid['stdout'].split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
gpu=call(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
rows=list(csv.reader(gpu['stdout'].splitlines()));assert len(rows)==1
row=[x.strip() for x in rows[0]]
apps=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
cpu=call(['free','-b','--wide'])
values=next(x.split() for x in cpu['stdout'].splitlines() if x.startswith('Mem:'))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),cwd=str(Path.cwd()),GPU_UUID=row[0],gpu_name=row[1],gpu_total_bytes=int(row[2])*2**20,gpu_used_bytes=int(row[3])*2**20,gpu_free_bytes=int(row[4])*2**20,gpu_utilization_percent=int(row[5]),CPU_total_bytes=int(values[1]),CPU_available_bytes=int(values[-1]),raw_queries=dict(uuid=uuid,gpu=gpu,compute_apps=apps,cpu=cpu),scope='one_authorized_host_readonly_system_memory_and_process_metadata_only',models_or_data_imported=False,CUDA_operator_calls=0,host_or_repo_changes=False)
print(json.dumps(result,sort_keys=True))
