"""Recover five pinned paper objects, without restoring the historical tree."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = r'''
import socket,subprocess,pathlib,json,zlib,base64,sys,tarfile,hashlib,time
assert socket.gethostname()=='anogena-2-0'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1])
archive=repo/'author_archives/mac_cleanup_20261002_v1/desktop_evidence.objects.tar.gz'
assert archive.is_file() and archive.stat().st_size==5206338271
wanted={row['archive_member']:row for row in payload['objects']}
result={};start=time.monotonic();members_seen=0
with tarfile.open(archive,'r|gz') as stream:
 for member in stream:
  members_seen+=1
  if member.name not in wanted:continue
  row=wanted[member.name]
  assert member.isfile() and member.size==row['bytes']
  data=stream.extractfile(member).read()
  assert hashlib.sha256(data).hexdigest()==row['sha256']
  result[row['local_name']]={'sha256':row['sha256'],'bytes':len(data),'base64':base64.b64encode(data).decode()}
  if len(result)==len(wanted):break
assert len(result)==len(wanted),'Not every pinned source member was found'
print(json.dumps({'complete':True,'objects':result,'archive_bytes':archive.stat().st_size,'archive_full_hash_repeated':False,'members_seen':members_seen,'seconds':time.monotonic()-start}))
'''

def main():
    custody = json.loads((PHASE/'manuscript_custody_local_audit_20261011_v1/AUDIT.json').read_text())
    rows = custody['exact_historical_source_archive']['selected_objects']
    names = ['main.tex', 'appendix.tex', 'references.bib',
             'FINAL_DELIVERY_SOURCE_BINDING.json', 'MANUSCRIPT_STATIC_AUDIT.json']
    assert len(rows) == len(names) == 5
    objects = [dict(row, local_name=name) for row, name in zip(rows, names)]
    payload = base64.b64encode(zlib.compress(json.dumps({'objects':objects}).encode())).decode()
    transport_path = PHASE/'publication/publish_exact_inventory_v19.py'
    spec = importlib.util.spec_from_file_location('review_custody_transport',transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    args = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',
            '/Users/alex/.ssh/mlspace__private_key_anogena.txt','-p','2222',
            'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,
                        REPO,UUID,'historical-paper-object-recovery'])]
    try:
        response = transport.terminal_transport(args,input=payload,capture_output=True,
                                                text=True,timeout=240)
        assert response.returncode == 0, response.stderr[:500]
        result = json.loads(response.stdout)
        assert result['complete'] and set(result['objects']) == set(names)
        target = HERE/'historical_round18_source'
        target.mkdir(exist_ok=False)
        for name, item in result['objects'].items():
            data=base64.b64decode(item.pop('base64'))
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            (target/name).write_bytes(data)
        result.update(UTC=datetime.now(timezone.utc).isoformat(),
                      source_scope='Historical round18; not equated with final Desktop source',
                      source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (HERE/'HISTORICAL_SOURCE_RECOVERY_RECEIPT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except Exception as exc:
        receipt={'UTC':datetime.now(timezone.utc).isoformat(),'complete':False,
                 'error_type':type(exc).__name__,'error':str(exc)[:1500]}
        (HERE/'HISTORICAL_SOURCE_RECOVERY_FAILURE_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))
        raise

if __name__ == '__main__':
    main()
