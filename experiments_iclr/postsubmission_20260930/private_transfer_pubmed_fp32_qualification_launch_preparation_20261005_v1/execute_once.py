"""Disabled prepared one-attempt discarded Pubmed FP32 allocation client."""
from datetime import datetime,timezone
import argparse,ast,base64,hashlib,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
SOURCE=PHASE/'private_transfer_pubmed_disabled_port_preparation_20261005_v2'
SOURCE_SHA='ba897c6588c0ac5d7d2e8473e1582b34c67986c23f265f0588f61c55c747aa6f';REMOTE_SHA='e5c2a205acbae223367d38c3cfddc33e8d9ba369516d344b817dbda171540cc5';EXECUTION_NAME='private_transfer_pubmed_fp32_qualification_execution_root_20261005_v1'
SOURCE_PROGRAMS=('custody.py', 'episode_geometry.py', 'heads.py', 'models.py', 'native_bodies.py', 'native_episode_cycle.py', 'native_family_custody.py', 'native_ncnc.py', 'native_pool_replay.py', 'native_state.py', 'private_adam.py', 'qualify_training_step.py', 'recursive_adjoint.py', 'run.py', 'transfer_step.py', 'vendor/baseline_models/NCN/__init__.py', 'vendor/baseline_models/NCN/model.py', 'vendor/baseline_models/NCN/util.py', 'vendor/baseline_models/__init__.py', 'vendor/evalutors.py', 'vendor/gnn_model.py', 'vendor/scoring.py')
FIXED_METADATA={'ROOT_INPUT_ADOPTION.json': {'path': 'private_transfer_pubmed_allocation_available_execution_root_20261005_v3/ROOT_INPUT_ADOPTION.json', 'sha256': '82cda6a1e9821bc43ccee9e46f336093dda40a64434523766d5f62b17230f186', 'bytes': 1497}, 'CPU_RUNTIME_RECEIPT.json': {'path': 'private_transfer_pubmed_allocation_available_execution_root_20261005_v3/AVAILABLE_INSPECTION.json', 'sha256': 'fffa724f10e423830145cb4cec03ca9e72b0f8562a05a4b8c59e5f856380856e', 'bytes': 2950}, 'INPUT_EXECUTION_MANIFEST.json': {'path': 'private_transfer_pubmed_allocation_available_execution_root_20261005_v3/MANIFEST.json', 'sha256': 'af45fc90fedba99c06348ad474ccbcdaadc177e11294602bc84fa657bdd72bf0', 'bytes': 2328}, 'TRAIN_GEOMETRY_ADOPTION.json': {'path': 'private_transfer_pubmed_train_geometry_execution_root_20261005_v1/ROOT_ADOPTION.json', 'sha256': '08b00806f46a206d1760bee231ffadd5d34e52d5131fdeecb228cd716fe322ab', 'bytes': 970}, 'PORT_SOURCE_ADOPTION.json': {'path': 'private_transfer_pubmed_port_v2_root_adoption_20261005_v1/ROOT_ADOPTION.json', 'sha256': '2641bfce9e54ffa827a884b46598c1e5a36a34ff14e252e3d7b790fa0cb9a26e', 'bytes': 2081}, 'NUMERICAL_SOURCE_REVIEW.md': {'path': 'private_transfer_pubmed_port_independent_static_review_20261005_v2/REPORT.md', 'sha256': 'e9ba1b26d353595858996cf248d486e5dc93b85a13a91ef0d10ffb9b78b14a43', 'bytes': 17025}}
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stage(path,name,expected=None):
 if path.is_symlink() or not path.is_file():raise ValueError('Ordinary exact staging file required')
 raw=path.read_bytes();pin=hashlib.sha256(raw).hexdigest()
 if expected is not None and pin!=expected:raise ValueError('Bound source/metadata changed')
 return {'path':name,'bytes':len(raw),'sha256':pin,'base64':base64.b64encode(raw).decode()}
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-directory',type=Path,required=True);args=parser.parse_args()
 release=args.release_directory.resolve(strict=True)
 if release.parent!=PHASE or release.name!=EXECUTION_NAME:raise ValueError('Exact fresh local execution identity required')
 intent=release/'LOCAL_INTENT.json'
 if intent.exists():raise ValueError('One attempt already exists; inspect exact root, never relaunch')
 remote=(HERE/'owned_supervisor.py.txt').read_bytes()
 if hashlib.sha256(remote).hexdigest()!=REMOTE_SHA:raise ValueError('Exact reviewed supervisor changed')
 ast.parse(remote)
 admission=json.loads((release/'OWNED_SUPERVISION_ADMISSION.json').read_text())
 if admission.get('root_approved') is not True or admission.get('client_independent_review_approved') is not True:raise ValueError('Actual root and launcher review admission absent')
 if admission.get('remote_source_sha256')!=REMOTE_SHA or admission.get('client_sha256')!=sha(Path(__file__)):raise ValueError('Exact independently reviewed launcher/supervisor required')
 if admission.get('TEST_access') is not False or admission.get('fits_authorized') is not False:raise ValueError('Discarded TRAIN-only qualification, no TEST or fits')
 if not admission.get('review_evidence'):raise ValueError('Actual review bytes required')
 for row in admission['review_evidence']:
  path=PHASE/row['path']
  if path.is_symlink() or not path.resolve().is_relative_to(PHASE) or sha(path)!=row['sha256']:raise ValueError('Actual review changed')
 files=[]
 manifest_raw=(SOURCE/'SOURCE_MANIFEST.json').read_bytes()
 if hashlib.sha256(manifest_raw).hexdigest()!=SOURCE_SHA:raise ValueError('Numerical source manifest changed')
 rows=json.loads(manifest_raw)['files']
 if tuple(row['path'] for row in rows)!=SOURCE_PROGRAMS or len({row['path'] for row in rows})!=len(SOURCE_PROGRAMS):raise ValueError('Exact ordered executable closure required')
 for row in rows:
  path=SOURCE/row['path']
  if not path.resolve().is_relative_to(SOURCE) or path.stat().st_size!=row['bytes']:raise ValueError('Bound executable source changed')
  files.append(stage(path,'source/'+row['path'],row['sha256']))
 files.append({'path':'source/SOURCE_MANIFEST.json','bytes':len(manifest_raw),'sha256':SOURCE_SHA,'base64':base64.b64encode(manifest_raw).decode()})
 for name in ('JOB.json','ROOT_REVIEW.md','INDEPENDENT_LAUNCH_REVIEW.md','OWNED_SUPERVISION_ADMISSION.json'):files.append(stage(release/name,name))
 for name,row in FIXED_METADATA.items():files.append(stage(PHASE/row['path'],name,row['sha256']))
 files.append(stage(HERE/'owned_supervisor.py.txt','owned_supervisor.py.txt',REMOTE_SHA))
 if admission.get('job_sha256')!=sha(release/'JOB.json') or admission.get('root_review_sha256')!=sha(release/'ROOT_REVIEW.md') or admission.get('launcher_review_sha256')!=sha(release/'INDEPENDENT_LAUNCH_REVIEW.md'):raise ValueError('Actual job/review binding absent')
 payload={'files':files,'source_manifest_sha256':SOURCE_SHA,'remote_source_sha256':REMOTE_SHA,'root_review_sha256':sha(release/'ROOT_REVIEW.md'),'launcher_review_sha256':sha(release/'INDEPENDENT_LAUNCH_REVIEW.md')}
 with intent.open('x') as stream:json.dump({'UTC':datetime.now(timezone.utc).isoformat(),'single_attempt':True,'client_sha256':sha(Path(__file__)),'remote_source_sha256':REMOTE_SHA,'job_sha256':sha(release/'JOB.json'),'relaunch_after_disconnect':False,'fits':0,'VALID_TEST_values_access':False},stream,indent=2);stream.write('\n')
 ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20',LOGIN]
 command='cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -B -c '+shlex.quote(remote.decode())
 try:result=subprocess.run([*ssh,command],input=json.dumps(payload),capture_output=True,text=True,timeout=4500)
 except subprocess.TimeoutExpired as error:
  def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
  (release/'CONSOLE.jsonl').write_text(decoded(error.stdout))
  (release/'TRANSPORT.json').write_text(json.dumps({'transport_timeout':True,'remote_terminal_status':'UNKNOWN_REPOLL_EXACT_ROOT_DO_NOT_RELAUNCH','stderr':decoded(error.stderr),'sole_authorized_route':LOGIN},indent=2)+'\n');raise
 (release/'CONSOLE.jsonl').write_text(result.stdout)
 (release/'TRANSPORT.json').write_text(json.dumps({'exit_code':result.returncode,'stderr':result.stderr,'remote_source_sha256':REMOTE_SHA,'sole_authorized_route':LOGIN},indent=2)+'\n')
 if result.returncode:raise RuntimeError(result.stderr[-4000:])
 value=json.loads(result.stdout.splitlines()[-1])
 if value.get('completed') is not True:raise ValueError('No complete terminal; preserve intent and never relaunch')
 for key,name in (('qualification_receipt','QUALIFICATION_RECEIPT.json'),('execution_receipt','EXECUTION_RECEIPT.json')):
  with (release/name).open('x') as stream:json.dump(value[key],stream,indent=2);stream.write('\n')
 print(json.dumps({'completed':True,'qualification_passed':value['qualification_receipt']['passed'],'fits':0,'VALID_TEST_values_access':False,'raw_model_or_input_payloads_fetched':False}))
if __name__=='__main__':main()
