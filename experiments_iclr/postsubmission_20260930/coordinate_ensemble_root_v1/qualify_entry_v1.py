"""Run only the admitted prescribed-fixture CPU model qualification."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys, time

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--request',required=True,type=Path)
 ap.add_argument('--supervisor',required=True,type=Path)
 a=ap.parse_args()
 phase=Path(__file__).resolve().parents[1]
 request=a.request.resolve()
 if not request.is_relative_to(phase): raise ValueError('Request escapes phase')
 r=json.loads(request.read_text())
 if r['operation']!='prescribed_fixture_cpu_qualification' or r['data_download_or_empirical_training_admitted']: raise ValueError('Wrong admission')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CUDA must be hidden')
 supervisor=a.supervisor.resolve()
 if not supervisor.is_relative_to(phase) or not (supervisor/'environment.json').is_file(): raise ValueError('Require live confined supervisor')
 for f in r['sources']:
  q=(phase/f['path']).resolve()
  if not q.is_relative_to(phase) or sha(q)!=f['sha256']: raise ValueError('Source hash mismatch: '+f['path'])
 output=phase/r['output_relative']
 if output.exists(): raise ValueError('New output required')
 output.mkdir(parents=True)
 begin=time.monotonic()
 child=subprocess.run([sys.executable,'-B',str(phase/r['test_entry']),'--verbose'],cwd=phase,env=os.environ.copy(),capture_output=True,text=True,timeout=r['cpu_test_timeout_seconds'])
 (output/'unittest_stdout.txt').write_text(child.stdout)
 (output/'unittest_stderr.txt').write_text(child.stderr)
 value={'UTC':datetime.now(timezone.utc).isoformat(),'operation':r['operation'],'returncode':child.returncode,'seconds':time.monotonic()-begin,'request_sha256':sha(request),'source_hashes_verified':True,'cuda_hidden':True,'real_dataset_read':False,'benchmark_fit':False,'fixture_only':True,'stdout_sha256':sha(output/'unittest_stdout.txt'),'stderr_sha256':sha(output/'unittest_stderr.txt')}
 (output/'RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(value),flush=True)
 if child.returncode: print(child.stderr,flush=True)
 raise SystemExit(child.returncode)

if __name__=='__main__': main()
