"""Stage exact reviewed V3 sources, then evaluate the complete closed family once."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
REMOTE = r'''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,sys
repo=Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930';os.chdir(repo)
assert subprocess.run(['git','rev-parse','--show-toplevel'],capture_output=True,text=True,check=True).stdout.strip()==str(repo)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
a=json.load(sys.stdin);prefixes={'graph_mixed_block_closed_family_evaluation_preparation_20261003_v3','mixed_closed_family_evaluation_source_review_20261003_v3','graph_mixed40_complete_evaluation_root_command_20261003_v3'}
staged=[]
for row in a['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] in prefixes
 p=phase/relative;assert p.resolve().is_relative_to(phase) and not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase))
 b=base64.b64decode(row['base64']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==b;state='identical_existing'
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
  state='created'
 staged.append(dict(path=row['path'],bytes=len(b),sha256=row['sha256'],state=state))
root=phase/'graph_mixed40_complete_evaluation_root_command_20261003_v3'
with (root/'SOURCE_CUSTODY.json').open('x') as f:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),files=staged,authorized_route=LOGIN,other_jobs_mutated=False),f,indent=2);f.write('\n')
command=phase/a['root_command'];assert command.is_relative_to(root)
result=subprocess.run(['/usr/bin/python3','-B',str(command),'--execute'],cwd=repo)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),root_command_exit=result.returncode,source_files_verified=len(staged),other_jobs_mutated=False)),flush=True)
sys.exit(result.returncode)
'''.replace('authorized_route=LOGIN', 'authorized_route=' + repr(LOGIN))


def main():
    started = datetime.now(timezone.utc).isoformat()
    input_path = HERE / 'SOURCE_TRANSPORT_INPUT.json'
    payload = input_path.read_bytes()
    packet = json.loads(payload)
    phase = HERE.parent
    for row in packet['files']:
        p = phase / row['path']
        data = p.read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, REPO])
    receipt = HERE / 'EXECUTION_TRANSPORT_RECEIPT.json'
    assert not receipt.exists()
    with (HERE / 'TRANSPORT.stdout.log').open('x') as stdout, (HERE / 'TRANSPORT.stderr.log').open('x') as stderr:
        result = subprocess.run([*SSH, command], input=payload, stdout=stdout, stderr=stderr)
    record = dict(start_UTC=started, terminal_UTC=datetime.now(timezone.utc).isoformat(),
                  exit_code=result.returncode, authorized_route=LOGIN,
                  input_sha256=hashlib.sha256(payload).hexdigest(),
                  client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  remote_code_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
                  exact_source_files=len(packet['files']), source_evaluator_modified_after_review=False,
                  no_new_predictive_fits=True, automatic_retry=False, heldout_labels_closed=True)
    with receipt.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print(json.dumps(record), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
