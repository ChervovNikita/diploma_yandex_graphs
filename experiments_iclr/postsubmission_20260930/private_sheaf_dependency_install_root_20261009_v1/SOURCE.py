from pathlib import Path
import subprocess,socket,json,os,hashlib,datetime,time
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'private_sheaf_dependency_install_root_20261009_v1';T=P/'private_sheaf_dependency_overlay_20261009_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0' and not A.exists() and not T.exists()
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
A.mkdir();rt=json.loads((P/'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json').read_text());env=dict(os.environ,PYTHONPATH=os.pathsep.join(rt['PYTHONPATH']),PYTHONDONTWRITEBYTECODE='1');start=time.monotonic()
cmd=[rt['python'],'-B','-m','pip','install','--disable-pip-version-check','--index-url','https://pypi.org/simple','--no-deps','--target',str(T),'torch-householder==1.0.1']
r=subprocess.run(cmd,cwd=R,env=env,capture_output=True,text=True,timeout=120)
v=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,elapsed_seconds=time.monotonic()-start,installed_only_in_project=T.is_relative_to(R),scientific_execution=False,other_environment_modified=False)
(A/'INSTALL_RECEIPT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
