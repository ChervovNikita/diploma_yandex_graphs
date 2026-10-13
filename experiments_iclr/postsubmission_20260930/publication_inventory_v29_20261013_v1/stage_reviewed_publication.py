"""Verify the reviewed finite publication list; stage/commit only by explicit flags."""
import argparse,hashlib,json,socket,subprocess
from pathlib import Path
EXPECTED_REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
EXPECTED_HEAD='02eaafe6d3b262add061264c2d294f84bca0ddaa'
EXPECTED_BRANCH='codex/postsubmission-research-20260930'

def git(*args,input=None):
 return subprocess.check_output(['git','-C',str(EXPECTED_REPO),*args],input=input)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
 # Allocation identity is checked before any project file/Git operation.
 if socket.gethostname()!='anogena-2-0':raise ValueError('Allocation hostname mismatch')
 if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:raise ValueError('Sole allocation GPU UUID mismatch')
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',action='store_true');ap.add_argument('--control-sha256');ap.add_argument('--commit-message-file',type=Path);args=ap.parse_args()
 if args.commit_message_file and not args.stage:raise ValueError('Commit requires --stage in this same reviewed invocation')
 if git('rev-parse','HEAD').decode().strip()!=EXPECTED_HEAD:raise ValueError('Expected base HEAD changed')
 if git('branch','--show-current').decode().strip()!=EXPECTED_BRANCH:raise ValueError('Existing branch changed')
 here=Path(__file__).resolve().parent;control_path=here/'PUBLICATION_CONTROL_BINDING.json';control_raw=control_path.read_bytes();control_hash=digest(control_raw)
 if args.stage and args.control_sha256!=control_hash:raise ValueError('Staging requires the separately reviewed control binding SHA256')
 if args.control_sha256 and args.control_sha256!=control_hash:raise ValueError('Control binding SHA256 mismatch')
 control=json.loads(control_raw);inventory_path=here/'PUBLICATION_INVENTORY.json'
 inventory=json.loads(inventory_path.read_text());rows=inventory['publication_files']+control['control_files'];rows.append({'repo_relative_path':str(control_path.relative_to(EXPECTED_REPO)),'sha256':control_hash,'bytes':len(control_raw)})
 paths={r['repo_relative_path'] for r in rows}
 if len(paths)!=len(rows):raise ValueError('Duplicate publication/control path')
 listed={s for s in (here/'PUBLICATION_ALLOWLIST.txt').read_text().splitlines() if s}
 if listed!=paths:raise ValueError('Human-readable allowlist differs from reviewed literal rows')
 expected_blobs={}
 for row in rows:
  rel=Path(row['repo_relative_path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Nonliteral repository path')
  target=EXPECTED_REPO/rel
  if target.is_symlink() or not target.resolve().is_relative_to(EXPECTED_REPO):raise ValueError('Unsafe publication path')
  raw=target.read_bytes()
  if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:raise ValueError('Reviewed bytes changed: '+str(rel))
  expected_blobs[str(rel)]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
 staged={s.decode() for s in git('diff','--cached','--name-only','-z').split(b'\0') if s}
 if staged:raise ValueError('Preexisting staged changes must be reviewed separately; index required empty')
 if args.stage:
  # Only reviewed literal file paths; no directory/glob/broad staging.
  payload=b''.join(p.encode()+b'\0' for p in sorted(paths))
  subprocess.run(['git','-C',str(EXPECTED_REPO),'--literal-pathspecs','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=payload,check=True)
  staged={s.decode() for s in git('diff','--cached','--name-only','-z').split(b'\0') if s}
  if not staged.issubset(paths):raise ValueError('Staged path outside reviewed list')
  index={}
  for chunk in git('ls-files','-s','-z').split(b'\0'):
   if not chunk:continue
   left,path=chunk.split(b'\t',1);mode,oid,stage=left.decode().split()
   if path.decode() in paths:index[path.decode()]=(oid,stage)
  for path,oid in expected_blobs.items():
   if index.get(path)!=(oid,'0'):raise ValueError('Index bytes do not match reviewed file: '+path)
  if args.commit_message_file:
   subprocess.run(['git','-C',str(EXPECTED_REPO),'commit','--file',str(args.commit_message_file)],check=True)
 print(json.dumps({'mode':'stage and commit' if args.commit_message_file else 'stage only' if args.stage else 'verification only','reviewed_paths':len(paths),'reviewed_bytes':sum(r['bytes'] for r in rows),'control_sha256':control_hash,'staged_paths':len(staged),'push_performed':False,'scientific_fits':0,'GO':False}))
if __name__=='__main__':main()
