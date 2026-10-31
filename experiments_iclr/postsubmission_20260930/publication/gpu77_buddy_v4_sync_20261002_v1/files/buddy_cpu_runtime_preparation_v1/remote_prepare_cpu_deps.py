"""Repo-local optional CPU dependencies; active virtual environment unchanged."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
from importlib import metadata
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
ROOT=PHASE/'buddy_cpu_runtime_preparation_v1'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
if Path.cwd().resolve()!=REPO or os.environ.get('GNNM_SSH_DESTINATION')!=LOGIN:
    raise RuntimeError('Wrong repository/route')
r=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
if r.stdout.splitlines()!=[UUID]:raise RuntimeError('Wrong allocation')
if metadata.version('torch')!='2.1.2+cu118':raise RuntimeError('Unqualified base Torch ABI')
state=ROOT/'root_setup_v1';state.mkdir(exist_ok=False)
target=state/'site';target.mkdir();temp=state/'tmp';temp.mkdir();cache=state/'pip_cache';cache.mkdir()
env=os.environ.copy();env.update(TMPDIR=str(temp),TMP=str(temp),TEMP=str(temp),PIP_CACHE_DIR=str(cache),
 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
packages=['datasketch==1.6.5','torch-sparse==0.6.18+pt21cu118']
try:scatter=metadata.version('torch-scatter')
except metadata.PackageNotFoundError:scatter=None
if scatter is None:packages.append('torch-scatter==2.1.2+pt21cu118')
base={k:metadata.version(k) for k in ('torch','torch-geometric','numpy','scipy','pandas','ogb')}
argv=[sys.executable,'-B','-m','pip','install','--disable-pip-version-check','--no-deps','--only-binary=:all:',
 '--target',str(target),'--find-links','https://data.pyg.org/whl/torch-2.1.0+cu118.html',*packages]
started=time.monotonic();value={'schema':'repo-local-buddy-cpu-dependency-setup-v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_destination':LOGIN,'gpu_uuid':UUID,'python':sys.version,'base_packages':base,'existing_scatter':scatter,
 'packages_requested':packages,'target':str(target),'active_virtualenv_modified':False,'GPU_compute':False,'datasets_models_or_scores':False}
try:
    proc=subprocess.run(argv,cwd=REPO,env=env,capture_output=True,text=True,timeout=240,check=False)
    (state/'pip_stdout.txt').write_text(proc.stdout);(state/'pip_stderr.txt').write_text(proc.stderr)
    value.update(exit_code=proc.returncode,status='DEPENDENCIES_INSTALLED_UNQUALIFIED' if proc.returncode==0 else 'SETUP_FAILED')
    if proc.returncode==0:
        value['target_distributions']={d.metadata['Name']:d.version for d in metadata.distributions(path=[str(target)])}
        value['installed_record_sha256']={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.dist-info/RECORD')}
except BaseException as error:value.update(status='SETUP_EXCEPTION',error_type=type(error).__name__,error=str(error))
value['seconds']=time.monotonic()-started
(state/'SETUP_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value));raise SystemExit(0 if value.get('exit_code')==0 else 1)
