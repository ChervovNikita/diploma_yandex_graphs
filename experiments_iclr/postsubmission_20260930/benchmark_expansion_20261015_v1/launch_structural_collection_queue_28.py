"""Launch one published, conditionally admitted collection queue; no training."""
import json
from pathlib import Path
from remote_control import call, REMOTE_ROOT, REMOTE_EXPANSION

code = r'''import datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@)
def pin(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
package=E/'qualification/structural_complete_collection_queue_v1'
manifest=pin(package/'MANIFEST.json')
assert manifest['sha256']=='071ee8c0d137c241536b239346d3786bb4d24c80ded3bd4405e9aa68c1a47f54'
commit=json.loads((E/'COMMIT_RECEIPT_28.json').read_text())['commit']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==commit
for row in json.loads((package/'MANIFEST.json').read_text())['files']:
 p=package/row['path'];assert pin(p)['sha256']==row['sha256']
 assert subprocess.check_output(['git','show',commit+':'+str(p.relative_to(R))],cwd=R)==p.read_bytes()
review=E/'STRUCTURAL_COLLECTION_QUEUE_REVIEW_28.json'
assert json.loads(review.read_text())['reviewed'] is True
sys.path.insert(0,str(E/'qualification/portablev4_link_queued_parent_authority_repair_v1'))
from supervise import preflight,process
peers=[]
old=process(640428)
if old:
 assert old['start_ticks']=='6055039165'
 peers.append({'pid':640428,'start_ticks':6055039165,'peak_host_RSS_bytes':16*1024**3})
for row in json.loads((E/'results/structural_private_risk_v1/OUTCOMES.json').read_text()):
 if row['status']=='started' and process(row['worker_pid']):
  peers.append({'pid':row['worker_pid'],'start_ticks':row['worker_start_ticks'],'peak_host_RSS_bytes':12*1024**3})
resource=preflight({'device':'cpu','peers':peers},{'minimum_host_headroom_after_peer_growth_bytes':8*1024**3,
                  'host_RSS_reservation_bytes':8*1024**3,'disk_reservation_bytes':2*1024**3})
resource.update(scope='CPU-only waiting owner; original queue rechecks CPU/GPU after both predecessors succeed',
                root_source_resource_review=pin(review))
resource_path=E/'STRUCTURAL_COLLECTION_QUEUE_RESOURCES_28.json'
assert not resource_path.exists()
resource_path.write_text(json.dumps(resource,indent=2)+'\n')
admission=json.loads((package/'ROOT_ADMISSION_TEMPLATE.json').read_text())
admission.update(approved=True,harness_manifest_sha256=manifest['sha256'],INPUTS_sha256=pin(package/'INPUTS.json')['sha256'],
 delegates_terminal_hash_binding_after_all6_and_all7_success=True,resources_and_counts_exactly_INPUTS=True,
 published_source_review=pin(E/'COMMIT_RECEIPT_28.json'),qualified_python_review=pin(E/'RESIDUAL_COLLECTION_OUTCOME_21.json'),
 peer_resource_review=pin(resource_path),root_review=pin(review))
path=E/'STRUCTURAL_COLLECTION_QUEUE_ADMISSION_28.json';assert not path.exists()
path.write_text(json.dumps(admission,indent=2)+'\n')
output=E/'qualification/structural_complete_collection_queue_run_v1'
assert not output.exists()
watchdog=shutil.which('timeout');assert watchdog,'Need existing ordinary timeout executable'
command=[watchdog,'--signal=TERM','--kill-after=20s','26h',str(R/'.venv/bin/python'),'-B','-u',str(package/'queue.py'),
         '--admission',str(path),'--admission-sha256',pin(path)['sha256']]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
log=E/'qualification/structural_complete_collection_queue_28.log'
with log.open('xb') as handle:
 child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outer_watchdog':process(child.pid),
         'command':command,'publication_commit':commit,'admission':pin(path),'resources':pin(resource_path),
         'source_manifest':manifest,'scientific_fits':0,'new_collections_after_prerequisites':6,'full_campaign_GO':False}
deadline=time.monotonic()+10
while time.monotonic()<deadline and not (output/'QUEUE.json').exists() and child.poll() is None:time.sleep(.2)
if (output/'QUEUE.json').exists():
 q=json.loads((output/'QUEUE.json').read_text());receipt.update(owner=q['owner'],queue_status=q['status'],queue_output=str(output))
else:receipt.update(queue_status='startup_not_confirmed',exit_code=child.poll(),log_tail=log.read_text()[-4000:])
receipt_path=E/'STRUCTURAL_COLLECTION_QUEUE_LAUNCH_28.json';assert not receipt_path.exists()
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
code = code.replace('@ROOT@', repr(REMOTE_ROOT)).replace('@EXP@', repr(REMOTE_EXPANSION))
receipt = json.loads(call(code, timeout=55))
here = Path(__file__).resolve().parent
with (here/'STRUCTURAL_COLLECTION_QUEUE_LAUNCH_28.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt))
