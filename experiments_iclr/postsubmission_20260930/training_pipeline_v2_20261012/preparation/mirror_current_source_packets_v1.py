"""Mirror compact exact source/metadata bytes for Git publication, no arrays."""
import base64
import hashlib
import json
from pathlib import Path

from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE=Path(__file__).resolve().parents[1]
PHASE=PACKAGE.parent
HERE=Path(__file__).resolve().parent/'current_packet_source_mirror_20261012_v1'
REMOTE=GUARD+r'''
files=[]
names=['master_controller_source_20261012_v3','massive_continuous_affinity_packet_20261012_v2',
       'gru_memory_packet_20261012_v3','landmark_memory_packet_20261012_v3']
for name in names:
 directory=phase/name;assert directory.is_dir()
 for path in sorted(directory.rglob('*')):
  if not path.is_file() or path.suffix not in {'.py','.json','.md','.csv','.sh'}:continue
  assert not path.is_symlink()
  size=path.stat().st_size
  if size>=2000000:continue
  raw=path.read_bytes()
  files.append({'path':str(path.relative_to(phase)),'bytes':size,
     'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()})
assert sum(row['bytes'] for row in files)<20000000
print(json.dumps({'files':files,'numerical_arrays_opened':False,'scientific_fits':0}))
'''


def main():
    assert not HERE.exists();HERE.mkdir()
    result=invoke(REMOTE,[],HERE,'MIRROR_TRANSPORT',60)
    assert result['exit_code']==0
    value=json.loads(result['stdout']);receipts=[]
    for row in value['files']:
        raw=base64.b64decode(row.pop('data'))
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        path=PHASE/row['path']
        assert path.resolve().is_relative_to(PHASE.resolve())
        if path.exists():
            assert path.read_bytes()==raw, 'Do not replace a different local frozen byte authority'
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        receipts.append(row)
    (HERE/'EXACT_FILE_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'exact_source_metadata_files':len(receipts),
        'total_bytes':sum(r['bytes'] for r in receipts),'bulk_arrays_fetched':False}))


if __name__=='__main__':main()
