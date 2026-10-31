"""Guarded, immutable molecular data/CPU-QA/disabled-compiler preparation.

No benchmark fit, score, GO, old-data overwrite or live-pipeline write occurs.
Every SSH attempt, including failures/timeouts, receives its own local receipt.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
LABEL = "molecular_allocation_20261012_v1"
DESTINATION = "massive_molecular_packet_20261012_v1"
PREFIX = ["ssh", "-tt", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
    "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
    "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"]

GUARD = r'''
import base64,hashlib,json,os,shutil,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
'''

TRANSFER = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport')
target=transport/'payload.b64';offset=int(sys.argv[2])
if offset==0:
 assert not transport.exists();transport.mkdir()
else:assert target.stat().st_size==offset
with target.open('ab') as stream:stream.write(sys.argv[3].encode())
print(json.dumps({'bytes':target.stat().st_size,'fits':0,'scores':0}))
'''

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport')
encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
package=stage/'sources/training_pipeline_v2_20261012';package.mkdir(parents=True)
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 target=package/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
atomic(stage/'SOURCE_INVENTORY.json',[{k:v for k,v in row.items() if k!='data'} for row in payload['files']])
for path in package.rglob('*'):
 if path.is_file():path.chmod(0o444)
host={'allocation':{'status':'verified_live','hostname':socket.gethostname(),
 'gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],
 'verification_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scientific_training_runs':0}}
atomic(stage/'HOST_BINDINGS.json',host)
exposure={'schema':'molecular-development-exposure-decision-v1','datasets':{
 'ogbg-molhiv':{'prior_exposure':'encountered_development; whole-project evidence-access audit incomplete',
  'evaluate_purpose':'development','protected_confirmation':False,'whole_project_access_audit_complete':False,
  'evidence':['research_dataset_experiment_census_20261011_v1/REPORT.md'],
  'author_replication':False},
 'ogbg-molbace':{'prior_exposure':'unknown_whole_project_exposure; never_claim_unused',
  'evaluate_purpose':'development','protected_confirmation':False,'whole_project_access_audit_complete':False,
  'evidence':['root preparation instruction: MolBACE is not certified unused'],
  'author_replication':False}}}
atomic(stage/'EXPOSURE_DECISIONS.json',exposure)
profiles={}
for dataset,epoch_estimate,memory in [('ogbg-molhiv',10.,32768),('ogbg-molbace',1.,16384)]:
 for epochs in [100,300]:
  profiles[dataset+'/GINE/'+str(epochs)]={'reservation_mib':memory,
   'wall_seconds':int((19520+60)*epochs*epoch_estimate*1.5)+86400,
   'estimated_seconds_per_epoch':epoch_estimate,'estimated_export_seconds':epoch_estimate,
   'estimated_CAL_seconds':5.,'estimated_evaluation_seconds':5.,
   'evidence':{'status':'PROVISIONAL_UNMEASURED_NOT_RUNTIME_QUALIFICATION',
    'assumption':'Conservative M1-equivalent scheduling placeholder; shared BE scales by four actual routes; broad width512/depth6 and graph sizes not qualified',
    'root_resource_qualification_required':True,'backbone_optimizer_updates_measured':0}}
atomic(stage/'RESOURCE_ESTIMATES.json',{'schema':'molecular-provisional-resources-v1','profiles':profiles,
 'scientific_launch_enabled':False,'requires_root_qualified_profiles_before_GO':True})
cache=stage/'provider_cache';cache.mkdir()
old=phase/'new_tasks/data/molhiv/provider/ogbg_molhiv'
ancestry={'schema':'molecular-provider-cache-ancestry-v1','historical_cache_modified':False,
 'cached_processing_producer_version':'unknown','current_provider_versions_are_not_historical_processing_provenance':True,
 'copied_dataset_caches':[]}
if payload.get('official_archive'):
 archive=payload['official_archive'];raw=base64.b64decode(archive['data'])
 assert hashlib.sha256(raw).hexdigest()==archive['sha256'] and len(raw)==archive['bytes']
 archive_dir=stage/'official_archives';archive_dir.mkdir()
 saved=archive_dir/'bace.zip';saved.write_bytes(raw);saved.chmod(0o444)
 shutil.copyfile(saved,cache/'bace.zip')
 ancestry['retained_official_archives']=[receipt(saved)]
 ancestry['official_archive_origin']=archive['origin']
 ancestry['transport']='same_official_HTTPS_archive_downloaded_on_Mac_then_SHA256_verified_on_allocation'
if old.is_dir() and not payload.get('reuse_molhiv_manifest'):
 originals=[]
 for path in sorted(old.rglob('*')):
  assert not path.is_symlink()
  if path.is_file():originals.append(receipt(path))
 shutil.copytree(old,cache/'ogbg_molhiv')
 ancestry['copied_dataset_caches'].append({'source':str(old.relative_to(phase)),
  'destination':str((cache/'ogbg_molhiv').relative_to(phase)),'source_files':originals})
atomic(stage/'CACHE_ANCESTRY.json',ancestry)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
 MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GNNM_REFERENCE_PROJECT=str(phase))
tmp=stage/'engineering_tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
records=[]
def run(command,label,timeout):
 tick=time.monotonic()
 child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,
  'engineering_or_dataprep_only':True,'scientific_training_runs':0})
 try:stdout,stderr=child.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  child.kill();stdout,stderr=child.communicate();stderr+='\nPreparation timeout retained.\n'
 value={'phase':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,
  'stdout':stdout,'stderr':stderr,'scientific_training_runs':0,'benchmark_scores':0}
 atomic(stage/(label+'_RESULT.json'),value);records.append(value)
 atomic(stage/'PROGRESS.json',records);print(json.dumps(value),flush=True)
 return value
qa=run([str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p',payload['qa_pattern'],'-v'],
 'MOLECULAR_CPU_QA',300)
if payload.get('inherit_numeric_QA'):
 prior=phase/payload['inherit_numeric_QA'];old_package=prior/'sources/training_pipeline_v2_20261012'
 checked={}
 for rel in ['engine.py','adapters/molecular.py','adapters/molecular_endpoints.py','tests/test_molecular_endpoints.py']:
  assert (package/rel).read_bytes()==(old_package/rel).read_bytes()
  checked[rel]=receipt(package/rel)
 prior_result=prior/'TARGETED_CPU_EQUIVALENCE_QA_RESULT.json'
 assert json.loads(prior_result.read_text())['exit_code']==0
 atomic(stage/'NUMERIC_QA_INHERITANCE.json',{'prior_QA_result':receipt(prior_result),
  'identical_numerical_sources':checked,'extra_numerical_fits_or_scores':0})
completed=[]
if payload.get('reuse_molhiv_manifest'):
 sys.path.insert(0,str(package))
 from compile_molecular_matrix import bind_custody
 reused=phase/payload['reuse_molhiv_manifest']
 custody,bindings=bind_custody(phase,reused)
 assert custody['dataset']=='ogbg-molhiv'
 atomic(stage/'REUSED_MOLHIV_CUSTODY.json',{'manifest':receipt(reused),'verified_bound_files':bindings,
  'heldout_target_tensors_decoded':False,'additional_backbone_fits':0,'additional_scores':0})
 completed.append(str(reused))
for dataset in (['ogbg-molbace'] if completed else ['ogbg-molhiv','ogbg-molbace']):
 destination=stage/'data'/dataset
 command=[str(repo/'.venv/bin/python'),'-B','-u',str(package/'prepare_molecular_data.py'),
  '--project-root',str(phase),'--dataset',dataset,'--dataset-root',str(cache),'--output',str(destination),
  '--split-seed','190131','--select-fraction','1/2','--allow-download','--cache-ancestry',str(stage/'CACHE_ANCESTRY.json')]
 row=run(command,'DATAPREP_'+dataset.replace('-','_'),1200)
 if row['exit_code']==0:completed.append(str(destination/'MANIFEST.json'))
if completed and qa['exit_code']==0:
 command=[str(repo/'.venv/bin/python'),'-B','-u',str(package/'compile_molecular_matrix.py'),
  '--project-root',str(phase),'--custody',*completed,'--resources',str(stage/'RESOURCE_ESTIMATES.json'),
  '--exposure-decisions',str(stage/'EXPOSURE_DECISIONS.json'),'--host-bindings',str(stage/'HOST_BINDINGS.json'),
  '--engine-prefix',str(package.relative_to(phase)),'--output',str(stage/'matrix'),
  '--epochs','100','300','--scalar-coefficients','neighbors']
 compile_result=run(command,'DISABLED_COMPILER',300)
else:compile_result={'exit_code':None,'blocker':'Targeted CPU QA or every real dataset preparation failed'}
value={'schema':'molecular-allocation-preparation-v1','destination':str(stage.relative_to(phase)),
 'CPU_QA_passed':qa['exit_code']==0,'qa_pattern':payload['qa_pattern'],'completed_data_manifests':completed,
 'compiler_completed':compile_result['exit_code']==0,'scientific_training_runs':0,'benchmark_scores':0,
 'GO_receipt_created':False,'source_bundle_outside_live_pipeline':True,
 'resource_qualification':'PROVISIONAL_ESTIMATES_REQUIRE_ROOT_QUALIFICATION',
 'records':[{k:v for k,v in row.items() if k not in ['stdout','stderr']} for row in records]}
atomic(stage/'PREPARATION_SUMMARY.json',value);print(json.dumps({'final_summary':value}),flush=True)
sys.exit(0 if value['compiler_completed'] else 1)
'''


def invoke(code, args, out, label, timeout):
    started = datetime.now(timezone.utc).isoformat()
    command = PREFIX + [shlex.join(["/usr/bin/python3", "-I", "-S", "-B", "-c", code, *args])]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        value = {"UTC": started, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        value = {"UTC": started, "exit_code": None, "transport_timeout": True,
                 "stdout": decoded(error.stdout), "stderr": decoded(error.stderr)}
    (out/(label+".json")).write_text(json.dumps(value, indent=2)+"\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--reuse-molhiv-manifest", help="Existing closed project-relative custody; verified by hashes without decoding targets")
    parser.add_argument("--qa-pattern", default="test_molecular_endpoints.py")
    parser.add_argument("--official-archive", help="Exact downloaded official BACE zip, preserved and hashed")
    parser.add_argument("--archive-origin-receipt", help="Successful official URL/download receipt")
    parser.add_argument("--inherit-numeric-QA", help="Project-relative prior QA snapshot; numerical source bytes must match")
    args = parser.parse_args()
    for value in (args.label, args.destination):
        if not value.replace("_", "").isalnum():
            raise ValueError("Use a single explicit immutable name")
    if args.reuse_molhiv_manifest:
        rel = Path(args.reuse_molhiv_manifest)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Use project-relative custody")
    archive = None
    if args.official_archive:
        raw = Path(args.official_archive).read_bytes()
        origin = json.loads(Path(args.archive_origin_receipt).read_text())
        if (origin.get("exit_code") != 0 or origin.get("official_url") != "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bace.zip"
                or origin.get("sha256") != hashlib.sha256(raw).hexdigest() or origin.get("bytes") != len(raw)):
            raise ValueError("Official archive differs from its successful download receipt")
        archive = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                   "data": base64.b64encode(raw).decode(), "origin": origin}
    if args.inherit_numeric_QA:
        relative = Path(args.inherit_numeric_QA)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Use project-relative prior QA")
    out = HERE/args.label
    if out.exists():
        raise FileExistsError("Retain prior attempt receipts; use a new explicit preparation label")
    out.mkdir()
    files = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "qa" in path.relative_to(PACKAGE).parts:
            continue
        raw = path.read_bytes()
        files.append({"path": str(path.relative_to(PACKAGE)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    for relative in ("docs/MATRIX_AXES_DRAFT.json", "docs/MOLECULAR_CONFIG.schema.json",
                     "docs/MOLECULAR_PACKET.md", "docs/MOLECULAR_ENDPOINTS_MATRIX.md", "ENGINE_README.md"):
        raw = (PACKAGE/relative).read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"destination": args.destination, "files": files,
        "reuse_molhiv_manifest": args.reuse_molhiv_manifest, "qa_pattern": args.qa_pattern,
        "official_archive": archive, "inherit_numeric_QA": args.inherit_numeric_QA}).encode())).decode()
    (out/"SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in row.items() if k != "data"} for row in files], indent=2)+"\n")
    for offset in range(0, len(encoded), 24000):
        result = invoke(TRANSFER, [args.destination, str(offset), encoded[offset:offset+24000]], out, "TRANSFER_"+str(offset), 45)
        if result["exit_code"] != 0:
            print(json.dumps(result), flush=True)
            raise SystemExit(1)
    result = invoke(REMOTE, [args.destination, hashlib.sha256(encoded.encode()).hexdigest()], out, "TRANSPORT", 2800)
    print(result["stdout"], flush=True)
    if result["exit_code"] != 0:
        print(result["stderr"], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
