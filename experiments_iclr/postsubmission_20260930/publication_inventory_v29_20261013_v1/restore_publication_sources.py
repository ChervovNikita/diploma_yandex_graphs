"""Verify or exclusively restore sealed source imports and exact master metadata.

Default is verification only. No model code is imported or executed.
"""
import argparse,gzip,hashlib,json,os,tarfile
from pathlib import Path

def sha(raw):return hashlib.sha256(raw).hexdigest()
def confined(repo,relative):
 p=Path(relative)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe repository-relative path: '+relative)
 target=repo/p
 for parent in [target,*target.parents]:
  if parent==repo.parent:break
  if parent.is_symlink():raise ValueError('Symlink in restoration path: '+str(parent))
 if not target.resolve().is_relative_to(repo):raise ValueError('Path escapes repository')
 return target

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo-root',type=Path,required=True);ap.add_argument('--restore',action='store_true');args=ap.parse_args()
 repo=args.repo_root.resolve();here=Path(__file__).resolve().parent
 manifest=json.loads((here/'COPIED_PROJECT_SOURCE_IMPORTS_MANIFEST.json').read_text());archive=here/'COPIED_PROJECT_SOURCE_IMPORTS.tar.gz'
 if sha(archive.read_bytes())!=manifest['archive']['sha256']:raise ValueError('Source archive SHA256 mismatch')
 pending=[];equal=0
 with tarfile.open(archive,'r:gz') as tf:
  infos=tf.getmembers()
  if len(infos)!=len(manifest['members']):raise ValueError('Archive member count mismatch')
  for info,row in zip(infos,manifest['members']):
   if not info.isfile() or info.name!=row['repo_relative_path']:raise ValueError('Archive member identity/type mismatch')
   raw=tf.extractfile(info).read()
   if len(raw)!=row['bytes'] or sha(raw)!=row['sha256']:raise ValueError('Source member bytes mismatch: '+info.name)
   target=confined(repo,info.name)
   if target.exists():
    if not target.is_file() or target.read_bytes()!=raw:raise ValueError('Existing unequal target rejected: '+info.name)
    equal+=1
   else:pending.append((target,raw,info.mode & 0o777))
 phase=Path('experiments_iclr/postsubmission_20260930')
 compressed_rows=[]
 for name,key in [('PUBLICATION_PAYLOAD_PREPARATION_RECEIPT.json','compressed_master_metadata'),('RESOURCE_EVIDENCE_METADATA_FREEZE_RECEIPT.json','compressed_metadata')]:
  receipt_path=here/name
  if receipt_path.exists():compressed_rows+=json.loads(receipt_path.read_text())[key]
 for row in compressed_rows:
  source=repo/phase/row['path'];packed=source.read_bytes()
  if len(packed)!=row['bytes'] or sha(packed)!=row['sha256']:raise ValueError('Compressed metadata receipt mismatch')
  raw=gzip.decompress(packed)
  if len(raw)!=row['restored_bytes'] or sha(raw)!=row['restored_sha256']:raise ValueError('Restored metadata bytes mismatch')
  target=confined(repo,str(phase/row['restores_path']))
  if target.exists():
   if not target.is_file() or target.read_bytes()!=raw:raise ValueError('Existing unequal metadata rejected')
   equal+=1
  else:pending.append((target,raw,0o644))
 if args.restore:
  for target,raw,mode in pending:
   confined(repo,str(target.relative_to(repo)));target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as fd:fd.write(raw)
   os.chmod(target,mode)
 print(json.dumps({'mode':'exclusive restoration' if args.restore else 'verification only','source_members':len(manifest['members']),'existing_equal_files':equal,'missing_files':len(pending),'created_files':len(pending) if args.restore else 0,'models_imported':0,'scientific_fits':0,'GO':False}))
if __name__=='__main__':main()
