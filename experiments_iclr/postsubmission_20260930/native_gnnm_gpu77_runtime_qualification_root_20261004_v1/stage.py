from remote_transport import run,HERE,REPO,PHASE,PYTHON
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,zlib
P=HERE.parent;SOURCE=P/'accuracy_first_native_gnnm_reference_source_preparation_20261004_v2';BANK=P/'accuracy_first_graph_view_source_preparation_20261004_v2';REMOTE=PHASE/HERE.name
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(name,obj):
 with (HERE/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
files={}
def add(f,relative):
 f=Path(f);raw=f.read_bytes()
 row=dict(path=relative,bytes=len(raw),sha256=sha(f),data=base64.b64encode(raw).decode())
 if relative in files:assert files[relative]==row
 else:files[relative]=row
for package in [SOURCE,BANK]:
 manifest=json.loads((package/'MANIFEST.json').read_text())
 for row in manifest['payload']:
  f=package/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'];add(f,'staged_phase/'+str(f.relative_to(P)))
 for name in ['MANIFEST.json','SEAL.json']:
  f=package/name;add(f,'staged_phase/'+str(f.relative_to(P)))
 binding=json.loads((package/'SOURCE_BINDINGS.json').read_text())
 for desc in [*binding['immutable_neural_files'],binding['v6_manifest'],binding['v6_seal'],binding['native_recipe_source'],binding['block_source'],*binding['context']]:
  f=P/desc['path'];assert f.stat().st_size==desc['bytes'] and sha(f)==desc['sha256'];add(f,'staged_phase/'+desc['path'])
coverage=P/'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json';add(coverage,'staged_phase/'+str(coverage.relative_to(P)))
for name in ['check_one.py','supervisor.py','INPUT_BINDINGS.json','DETERMINISTIC_RUNTIME_PROSPECTIVE_POLICY.json','CHECKER_REVIEW.json']:add(HERE/name,name)
rows=list(files.values());save('STAGE_INVENTORY.json',[{k:v for k,v in r.items() if k!='data'} for r in rows])
compressed=zlib.compress(json.dumps(rows).encode(),9);encoded=base64.b64encode(compressed).decode();chunks=[encoded[i:i+40000] for i in range(0,len(encoded),40000)]
for n,chunk in enumerate(chunks,1):
 code='from pathlib import Path\nimport json,os,hashlib\n'
 code+='repo=Path('+repr(str(REPO))+');root=Path('+repr(str(REMOTE/'stage_chunks'))+')\n'
 code+='assert Path.cwd()==repo and os.uname().nodename=="peptide";root.mkdir(parents=True,exist_ok=True)\n'
 code+='raw='+repr(chunk)+'.encode();path=root/'+repr('part%03d.b64'%n)+'\n'
 code+='with path.open("xb") as f:f.write(raw)\nprint(json.dumps({"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}))\n'
 value=run('native_gnnm_gpu77_stage_chunk%03d_20261004_v1'%n,code);assert value['sha256']==hashlib.sha256(chunk.encode()).hexdigest()
code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport json,os,base64,zlib,hashlib\n'
code+='repo=Path('+repr(str(REPO))+');root=Path('+repr(str(REMOTE))+');parts=root/"stage_chunks"\n'
code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code+='raw=base64.b64decode(b"".join((parts/("part%03d.b64"%n)).read_bytes() for n in range(1,'+str(len(chunks)+1)+')),validate=True)\n'
code+='assert hashlib.sha256(raw).hexdigest()=='+repr(hashlib.sha256(compressed).hexdigest())+'\nrows=json.loads(zlib.decompress(raw))\n'
code+='for r in rows:\n f=root/r["path"];assert f.resolve().is_relative_to(root) and not f.is_symlink();raw=base64.b64decode(r["data"],validate=True);assert len(raw)==r["bytes"] and hashlib.sha256(raw).hexdigest()==r["sha256"]\n if f.exists():assert f.read_bytes()==raw\n else:\n  f.parent.mkdir(parents=True,exist_ok=True)\n  with f.open("xb") as out:out.write(raw)\n'
code+='print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="EXACT_NATIVE_REFERENCE_PAYLOADS_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),model_execution=False)))\n'
value=run('native_gnnm_gpu77_stage_join_20261004_v1',code);save('STAGE_RECEIPT.json',value);print(json.dumps(value))
