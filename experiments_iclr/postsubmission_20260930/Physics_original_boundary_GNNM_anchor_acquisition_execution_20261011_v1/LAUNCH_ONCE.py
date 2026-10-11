import pathlib,subprocess,shlex,json,base64,hashlib
here=pathlib.Path(__file__).resolve().parent
remote=(here/'REMOTE_LAUNCH.py').read_text()
proc=subprocess.run(['ssh','-tt','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-p','2222','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(remote)],capture_output=True,text=True)
if proc.returncode:
 print(proc.stdout);print(proc.stderr);raise SystemExit(proc.returncode)
line=next(x for x in proc.stdout.splitlines() if x.startswith('{"status": "RUNNING"'))
data=json.loads(line)
actual=here/'actual_launch_receipts';actual.mkdir(exist_ok=False)
receipts=[]
for r in data.pop('files'):
 b=base64.b64decode(r.pop('base64'))
 assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 with (actual/r['name']).open('xb') as f:f.write(b)
 receipts.append(r)
with (here/'FETCH_LAUNCH_RECEIPTS.json').open('x') as f:f.write(json.dumps(dict(files=receipts,quality_scores_read=False,CAL_TEST_access=False),indent=2)+'\n')
print(json.dumps(data))
