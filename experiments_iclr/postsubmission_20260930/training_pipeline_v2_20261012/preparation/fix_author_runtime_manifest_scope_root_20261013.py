"""Prepare a sync-tool successor respecting author runtime receipt authority."""
import base64,hashlib,json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
P=Path(__file__).resolve().parents[2]
SOURCE=P/'final_master_sync_tooling_20261012_v2/runtime/generic_manifest.py'
OUT=P/'final_master_sync_author_runtime_scope_20261013_v3'

def amend(text):
 old='    files, reasons, specs, author, empty, aliases = {}, {}, [], [], set(), {}'
 assert text.count(old)==1
 text=text.replace(old,old+'\n    allocation_runtime_providers = {}',1)
 old='            for row in cli["source_files"] + cli["raw_files"] + cli.get("runtime_provider_files", []): add(row, "author_all_host_hash_validation_only")'
 assert text.count(old)==1
 new='''            for row in cli["source_files"] + cli["raw_files"]: add(row, "author_all_host_hash_validation_only")
            # run_ogb_author_cli verifies these receipts against the declared
            # repository, not the research phase. They describe its existing
            # allocation-only environment and must never become phase paths.
            repository = Path(cli["execution_repository"]).resolve()
            if not Path(root).resolve().is_relative_to(repository):
                raise ValueError("Author runtime repository must contain the authorized phase")
            for provider in cli.get("runtime_provider_files", []):
                row = checked_receipt(provider)
                if not row["path"].startswith(".venv/lib/"):
                    raise ValueError("Only declared repository venv library providers are external to phase")
                if receipt(repository, row["path"]) != row:
                    raise ValueError("Allocation author runtime provider changed")
                prior = allocation_runtime_providers.get(row["path"])
                if prior is not None and prior["receipt"] != row:
                    raise ValueError("Conflicting allocation author runtime provider")
                allocation_runtime_providers[row["path"]] = {"receipt": row,
                    "authority": "allocation_author_execution_repository",
                    "scientific_execution_host": "allocation",
                    "phase_relative_file": False, "bundled": False,
                    "actual_author_preflight_rechecks_repository_receipt": True}'''
 text=text.replace(old,new,1)
 old='        "all_study_bindings_needed_on_every_host": True, "tensor_deserialization_performed": False,'
 assert text.count(old)==1
 text=text.replace(old,'        "allocation_only_author_runtime_providers": [allocation_runtime_providers[k] for k in sorted(allocation_runtime_providers)],\n'+old,1)
 return text

REMOTE=GUARD+r'''
out=phase/sys.argv[1];assert not out.exists();out.mkdir()
payload=json.loads(base64.b64decode(sys.argv[2]));assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==sys.argv[3]
source=phase/'final_master_sync_tooling_20261012_v2/runtime';runtime=out/'runtime';runtime.mkdir()
for q in source.iterdir():
 if q.is_file():
  raw=q.read_bytes()
  if q.name=='generic_manifest.py':
   assert hashlib.sha256(raw).hexdigest()==payload['prior_sha256'];raw=base64.b64decode(payload['new'])
  (runtime/q.name).write_bytes(raw)
for q in runtime.glob('*.py'):compile(q.read_text(),str(q),'exec')
rows=[]
for q in sorted(runtime.iterdir()):
 if q.is_file():rows.append({'path':str(q.relative_to(phase)),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()});q.chmod(0o444)
value={'schema':'author-runtime-authority-sync-tooling-successor-v1','files':rows,'prior_tooling':'final_master_sync_tooling_20261012_v2/runtime','actual_author_runner_unchanged':True,'allocation_repo_venv_provider_receipts_not_reclassified_as_phase_files':True,'venv_not_bundled':True,'all_source_and_raw_role_files_remain_manifest_checked':True,'scientific_fits':0,'GO':False}
(out/'MANIFEST.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''

def main():
 OUT.mkdir();old=SOURCE.read_bytes();new=amend(old.decode());compile(new,str(SOURCE),'exec')
 (OUT/'generic_manifest.py').write_text(new);(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
 payload={'prior_sha256':hashlib.sha256(old).hexdigest(),'new':base64.b64encode(new.encode()).decode()}
 raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode();r=invoke(REMOTE,[OUT.name,base64.b64encode(raw).decode(),hashlib.sha256(raw).hexdigest()],OUT,'STAGE',90);assert r['exit_code']==0
 v=json.loads(r['stdout']);(OUT/'MANIFEST.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'files':len(v['files']),'science':0,'scope_corrected':True}))
if __name__=='__main__':main()
