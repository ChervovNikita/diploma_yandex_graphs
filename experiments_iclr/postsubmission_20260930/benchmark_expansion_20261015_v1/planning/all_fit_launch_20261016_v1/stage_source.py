"""Copy only selected new project source files to the verified remote host."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
sys.path.insert(0, str(E))
sys.path.insert(0, str(E/'planning/three_gpu_sync_20261015_v1'))
import remote_control
import remote51


def stage(host, paths):
    remote = remote_control if host == 'allocation' else remote51
    root = remote.REMOTE_ROOT if host == 'allocation' else remote.ROOT
    phase = Path(root)/'experiments_iclr/postsubmission_20260930'
    records = []
    for source in paths:
        source = Path(source).resolve()
        assert source.is_relative_to(E.parent) and not source.is_symlink()
        raw = source.read_bytes()
        records.append({'path': str(source.relative_to(E.parent)), 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'content': base64.b64encode(zlib.compress(raw)).decode()})
    created = []
    # Limit each forwarded command's size; do not stage datasets/checkpoints.
    for record in records:
        assert record['bytes'] < 120000, record['path']
        code = '''
import base64,hashlib,json,zlib
from pathlib import Path
root=Path(PHASE_VALUE)
row=ROW_VALUE
p=root/row['path']; assert p.resolve().is_relative_to(root) and not p.is_symlink()
raw=zlib.decompress(base64.b64decode(row['content']));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
if p.exists():
 assert p.read_bytes()==raw, 'Refuse frozen source replacement: '+str(p)
else:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
print(json.dumps({'path':str(p),'bytes':len(raw),'sha256':row['sha256']}))
'''.replace('PHASE_VALUE', repr(str(phase))).replace('ROW_VALUE', repr(record))
        created.append(json.loads(remote.call(code, timeout=45)))
    return {'host': host, 'files': created}


if __name__ == '__main__':
    print(json.dumps(stage(sys.argv[1], [Path(x) for x in sys.argv[2:]])))
