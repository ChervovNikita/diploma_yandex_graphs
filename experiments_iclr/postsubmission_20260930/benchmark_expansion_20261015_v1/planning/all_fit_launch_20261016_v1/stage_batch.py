"""Hash-checked immutable project source transfer in small compressed batches."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
import stage_source


def stage(host, paths):
    remote = stage_source.remote_control if host == 'allocation' else stage_source.remote51
    root = remote.REMOTE_ROOT if host == 'allocation' else remote.ROOT
    phase = str(Path(root)/'experiments_iclr/postsubmission_20260930')
    records = []
    for source in paths:
        source = Path(source).resolve()
        assert source.is_relative_to(stage_source.E.parent) and not source.is_symlink()
        raw = source.read_bytes()
        assert len(raw) < 2_000_000
        records.append({'path': str(source.relative_to(stage_source.E.parent)),
                        'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'content': base64.b64encode(raw).decode()})
    assert len(records) == len({r['path'] for r in records})
    def encode(rows):
        return base64.b64encode(zlib.compress(json.dumps(rows).encode())).decode()
    batches, pending = [], []
    for row in records:
        if pending and len(encode(pending+[row])) > 16000:
            batches.append(pending);pending=[]
        pending.append(row)
        assert len(encode(pending)) <= 30000, row['path']
    if pending:batches.append(pending)
    receipts = []
    for batch in batches:
        payload = zlib.compress(json.dumps(batch).encode())
        payload_sha = hashlib.sha256(payload).hexdigest()
        temporary = str(Path(phase)/'benchmark_expansion_20261015_v1/planning/all_fit_launch_20261016_v1/transport_staging'/('source-'+payload_sha+'.zlib'))
        for offset in range(0,len(payload),3000):
            chunk=payload[offset:offset+3000]
            upload='''import base64,hashlib
from pathlib import Path
p=Path(TEMP_VALUE);root=Path(PHASE_VALUE);assert p.resolve().is_relative_to(root) and not p.is_symlink()
b=base64.b64decode(CHUNK_VALUE);offset=OFFSET_VALUE
p.parent.mkdir(parents=True,exist_ok=True)
size=p.stat().st_size if p.exists() else 0
if size>offset:
 with p.open('rb') as f:f.seek(offset);assert f.read(len(b))==b
else:
 assert size==offset
 with p.open('ab') as f:f.write(b)
print('OK')'''.replace('TEMP_VALUE',repr(temporary)).replace('PHASE_VALUE',repr(phase)).replace('CHUNK_VALUE',repr(base64.b64encode(chunk).decode())).replace('OFFSET_VALUE',str(offset))
            assert remote.call(upload,timeout=60).strip()=='OK'
        code = '''import base64,hashlib,json,zlib
from pathlib import Path
root=Path(PHASE_VALUE);temporary=Path(TEMP_VALUE);payload=temporary.read_bytes();assert hashlib.sha256(payload).hexdigest()==PAYLOAD_SHA_VALUE
rows=json.loads(zlib.decompress(payload));out=[]
for row in rows:
 p=root/row['path'];assert p.resolve().is_relative_to(root) and not p.is_symlink()
 raw=base64.b64decode(row['content']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==raw,'Refuse source overwrite: '+str(p)
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
 out.append({'path':str(p),'bytes':len(raw),'sha256':row['sha256']})
temporary.unlink()
print(json.dumps(out))'''.replace('PHASE_VALUE',repr(phase)).replace('TEMP_VALUE',repr(temporary)).replace('PAYLOAD_SHA_VALUE',repr(payload_sha))
        receipts.extend(json.loads(remote.call(code,timeout=60)))
    return {'host':host,'files':receipts,'compressed_batches':len(batches)}


if __name__=='__main__':
    print(json.dumps(stage(sys.argv[1],[Path(x) for x in sys.argv[2:]])))
