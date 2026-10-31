"""Freeze environment recipe tooling; syntax/capture/dry-run only, no installs."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from prepare_molecular_packets_allocation import GUARD, TRANSFER, invoke

PACKAGE = Path(__file__).resolve().parents[1]
DESTINATION = "master_sync_environment_tooling_20261012_v1"
LABEL = "master_sync_environment_allocation_20261012_v1"

REMOTE = GUARD + r'''
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/payload['destination'];assert not stage.exists();stage.mkdir()
package=stage/'runtime';package.mkdir()
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def receipt(path):
 return {'path':str(path.relative_to(phase)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
helper=phase/'master_sync_tooling_20261012_v4/runtime/prepare_sync_bundle.py'
shutil.copyfile(helper,package/'prepare_sync_bundle.py');(package/'prepare_sync_bundle.py').chmod(0o444)
atomic(stage/'SOURCE_INVENTORY.json',[receipt(p) for p in sorted(package.rglob('*')) if p.is_file()])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
records=[]
def run(command,label,timeout):
 tick=time.monotonic();child=subprocess.Popen(command,cwd=package,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
 atomic(stage/(label+'_START.json'),{'pid':child.pid,'start_ticks':int(fields[19]),'command':command,'package_installs':0})
 stdout,stderr=child.communicate(timeout=timeout)
 result={'label':label,'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'stdout':stdout,'stderr':stderr,
  'package_installs':0,'wheel_downloads':0,'scientific_fits':0,'GO_created':False}
 atomic(stage/(label+'_RESULT.json'),result);records.append(result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['stdout','stderr']}),flush=True)
 if result['exit_code']!=0:print(stdout,stderr,flush=True);sys.exit(1)
 return result
run([str(repo/'.venv/bin/python'),'-B','-c','import ast;from pathlib import Path;[ast.parse(Path(p).read_text()) for p in ["prepare_sync_environment.py","prepare_sync_bundle.py"]]'],
 'ENVIRONMENT_TOOL_SYNTAX',60)
environment='master_sync_tooling_20261012_v4/ENVIRONMENT.json'
result=run([str(repo/'.venv/bin/python'),'-B',str(package/'prepare_sync_environment.py'),'capture','--project-root',str(phase),
 '--recipe',str((stage/'RECIPE.json').relative_to(phase)),'--environment',environment],'SOURCE_DEPENDENCY_METADATA_CLOSURE',120)
sys.path.insert(0,str(package));from prepare_sync_bundle import fingerprint
recipe=json.loads((stage/'RECIPE.json').read_text());assert recipe['environment_hash']=='96ca1fc422b22e985e1ac0ee47af919140087eb4034ad898911d9b681bf955b7'
result=run([str(repo/'.venv/bin/python'),'-B',str(package/'prepare_sync_environment.py'),'dry-run','--project-root',str(phase),
 '--recipe',str((stage/'RECIPE.json').relative_to(phase)),'--recipe-hash',fingerprint(recipe),'--base-python',str(repo/'.venv/bin/python')],
 'EXACT_RECREATION_DRY_RUN',120)
proof={'schema':'environment-recreation-dry-run-proof-v1','recipe_hash':fingerprint(recipe),'environment_hash':recipe['environment_hash'],
 'pinned_distributions':len(recipe['packages']),'dependency_edges':len(recipe['dependency_edges']),
 'base_platform_recreated_or_target_qualified':False,'running_allocation_env_changed':False,'package_installs':0,'wheel_downloads':0,
 'server77_access':False,'scientific_fits':0,'GO_created':False}
atomic(stage/'FINAL_PROOF.json',proof)
paths=[stage/'RECIPE.json',stage/'FINAL_PROOF.json',stage/'SOURCE_INVENTORY.json',stage/'EXACT_RECREATION_DRY_RUN_RESULT.json']
print(json.dumps({'proof':proof,'mirror_files':[{'receipt':receipt(p),'data':base64.b64encode(p.read_bytes()).decode()} for p in paths]}))
'''


def main():
    out = PACKAGE / "preparation" / LABEL
    if out.exists(): raise FileExistsError("Preserve earlier attempt")
    out.mkdir(); files = []
    for relative in ("prepare_sync_environment.py", "docs/MASTER_SYNC_ENVIRONMENT.md"):
        raw = (PACKAGE / relative).read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    encoded = base64.b64encode(zlib.compress(json.dumps({"destination": DESTINATION, "files": files}).encode())).decode()
    result = invoke(TRANSFER, [DESTINATION, "0", encoded], out, "TRANSFER", 60)
    if result["exit_code"] != 0: print(json.dumps(result)); raise SystemExit(1)
    result = invoke(REMOTE, [DESTINATION, hashlib.sha256(encoded.encode()).hexdigest()], out, "TRANSPORT", 600)
    for line in result["stdout"].splitlines():
        if not line.startswith('{'): continue
        try: payload = json.loads(line)
        except json.JSONDecodeError: continue
        if "mirror_files" in payload:
            for row in payload["mirror_files"]:
                raw = base64.b64decode(row["data"]); assert hashlib.sha256(raw).hexdigest() == row["receipt"]["sha256"]
                p = out / "allocation_artifacts" / row["receipt"]["path"]; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
            print(json.dumps(payload["proof"],indent=2))
        else: print(json.dumps(payload),flush=True)
    if result["exit_code"] != 0: print(result["stdout"],result["stderr"]); raise SystemExit(1)


if __name__ == "__main__": main()
