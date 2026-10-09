import base64,hashlib,json,pathlib,subprocess,time
WORKSPACE=pathlib.Path('/Users/alex/Documents/ChatGPT/anogena allocation');P=WORKSPACE/'postsubmission_research_20260930';D=P/'pre_sigmoid_qk36_original_route_phase2_closure_and_assembly_preparation_root_20261010_v1';W=P/'gpu77_connection_recovery_v1';IDENT='QK36_NORMAL77_PHASE2_ATTESTED_COMPACT_HARVEST_ROOT_20261010_v1'
started=time.monotonic();r=subprocess.run(['python3','-B',str(W/'run_gpu77_v3.py'),'--id',IDENT,'--command-file',str(W/(IDENT+'_COMMAND.txt'))],cwd=WORKSPACE,capture_output=True,text=True)
transport=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,transport_elapsed_seconds=time.monotonic()-started)
(D/'ROOT_SAVED_HARVEST_TRANSPORT.json').write_text(json.dumps(transport,indent=2)+'\n')
wrapper=json.loads(next(line for line in r.stdout.splitlines() if line.startswith('{')));remote=json.loads(next(line for line in wrapper['stdout'].splitlines() if line.startswith('{')))
(D/'ROOT_SAVED_HARVEST_RECEIPT.json').write_text(json.dumps({k:v for k,v in remote.items() if k!='payloads'},indent=2,sort_keys=True)+'\n')
assert r.returncode==wrapper['exit_code']==0 and remote['metadata']['complete']
for row in remote['payloads']:
 b=base64.b64decode(row['base64']);binding=row['binding'];assert len(b)==binding['bytes'] and hashlib.sha256(b).hexdigest()==binding['sha256'];p=D/'compact_metadata'/binding['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as stream:stream.write(b)
print(json.dumps(dict(complete=True,exact_compact_files=len(remote['payloads']),remote_receipt=remote['receipt'],no_arrays_or_scores_decoded=True)))
