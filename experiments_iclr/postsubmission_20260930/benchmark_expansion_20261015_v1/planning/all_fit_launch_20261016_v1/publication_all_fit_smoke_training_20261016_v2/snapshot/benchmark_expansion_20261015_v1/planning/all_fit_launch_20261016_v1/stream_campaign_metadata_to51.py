"""Bounded generated campaign metadata stream through the linked Mac; no dataset copy on either Mac."""
from pathlib import Path
import sys,json,shlex,subprocess,importlib.util,hashlib,time
HERE=Path(__file__).resolve().parent
W=HERE.parents[3]
old=W/'postsubmission_research_20260930/gpu77_connection_recovery_v1/stream_exact_allocation_to77_guarded_PTY_20261009_v2.py'
spec=importlib.util.spec_from_file_location('old_stream',old);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCE_REPO=m.SOURCE_REPO;TARGET_REPO=m.TARGET_REPO

def main():
 plan_path=Path(sys.argv[1]).resolve();receipt_path=Path(sys.argv[2]).resolve()
 assert plan_path.is_relative_to(HERE) and receipt_path.is_relative_to(HERE) and not receipt_path.exists()
 plan=json.loads(plan_path.read_text());assert plan
 for row in plan:
  assert set(row)=={'source_relative','target_relative','bytes','sha256'}
  for field in ('source_relative','target_relative'):
   p=Path(row[field]);assert not p.is_absolute() and '..' not in p.parts and p.parts[0] in ('.tools','experiments_iclr')
  assert row['bytes']>0 and len(row['sha256'])==64
 recv=m.RECEIVER.replace("=='peptide'","=='olig'").replace('GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-9c6c20fa-a620-479d-f72f-163ac74deef2').replace('GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced','GPU-30283973-140d-9e7b-e780-0fcc487dec54')
 # Install only complete verified transfers; retain partial files visibly on failure.
 recv=recv.replace("sink=None if present else p.open('xb')", "partial=p.with_name(p.name+'.transfer-part');sink=None if present else partial.open('xb')")
 recv=recv.replace("if not present:p.chmod(0o444)", "if not present:\n  partial.chmod(0o444);partial.replace(p)")
 source=['/usr/bin/ssh','-tt','-e','none','-p','2222','-i','/Users/shmelev/.maclink-reverse/anogena_identity','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',m.remote(SOURCE_REPO,plan,m.SENDER)]
 target=['/usr/bin/ssh','-S','/Users/shmelev/Downloads/reverse_maclink_daily/.gnnm51-control.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','shmelev@192.168.18.51',m.remote(TARGET_REPO,plan,recv)]
 pipeline=shlex.join(source)+' | '+shlex.join(target)
 start=time.time();r=subprocess.run([str(W/'reverse_maclink/maclink-env/bin/python'),'-B',str(W/'reverse_maclink/maclink.py'),'run','--','/bin/bash','-o','pipefail','-c',pipeline],capture_output=True,text=True,timeout=14400)
 result={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'elapsed_seconds':time.time()-start,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'bytes':sum(x['bytes'] for x in plan),'Mac_payload_copy':False,'seven_GPU_role':'forwarding relay only'}
 receipt_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(r.returncode)
if __name__=='__main__':main()
