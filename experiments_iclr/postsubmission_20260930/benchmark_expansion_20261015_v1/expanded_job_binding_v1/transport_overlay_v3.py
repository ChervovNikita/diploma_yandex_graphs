"""Explicit guarded source/metadata transport for overlay v3 only."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
E, P = HERE.parent, HERE.parent.parent
spec = importlib.util.spec_from_file_location('_v3_reviewed_remote_control', E / 'remote_control.py')
remote = importlib.util.module_from_spec(spec); spec.loader.exec_module(remote)
REMOTE = remote.REMOTE_EXPANSION + '/' + HERE.name


def stage():
    files = [HERE / n for n in ('build_BUDDY_production_overlay_v3.py', 'compile_overlay_v3.py', 'OVERLAY_v3.json', 'README_v3.md', 'transport_overlay_v3.py')]
    qpath = E / 'qualification/BUDDY_PRODUCTION_SEED1_QUALIFICATION_07.json'
    q = json.loads(qpath.read_text()); files += [qpath] + [E / r['path'] for r in q['files']]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path in files:
            assert path.suffix in ('.py', '.json', '.md') and path.is_relative_to(P)
            archive.add(path, arcname=str(path.relative_to(P)), recursive=False)
    packet = stream.getvalue(); names = [str(p.relative_to(P)) for p in files]
    code = remote.GUARD + '''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
print('READY',flush=True)
data=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(data)==SIZE_LITERAL and hashlib.sha256(data).hexdigest()==HASH_LITERAL
changed=[]
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 members=archive.getmembers();assert {m.name for m in members}==set(NAMES_LITERAL)
 for m in members:
  assert m.isfile() and Path(m.name).suffix in ('.py','.json','.md')
  target=root/m.name;assert target.resolve().is_relative_to(root) and not target.is_symlink()
  raw=archive.extractfile(m).read()
  if target.exists():
   assert target.read_bytes()==raw,'Preserve staged source and use a new generation: '+str(target)
   continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:out.write(raw)
  changed.append(m.name)
print(json.dumps({'files':len(members),'packet_bytes':len(data),'changed':changed,'native_work':0,'full_fit_descriptors_staged':0}),flush=True)
'''
    for key, val in [('ROOT_LITERAL', repr(remote.REMOTE_PHASE)), ('SIZE_LITERAL', str(len(packet))),
                     ('HASH_LITERAL', repr(hashlib.sha256(packet).hexdigest())), ('NAMES_LITERAL', repr(names))]: code = code.replace(key, val)
    process = subprocess.Popen(remote.SSH + ['stty raw -echo; python3 -B -c ' + shlex.quote(code)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first = process.stdout.readline()
    if first.strip() != b'READY':
        out, err = process.communicate(timeout=20)
        raise RuntimeError((first + out + err).decode(errors='replace'))
    out, err = process.communicate(input=packet, timeout=90)
    if process.returncode: raise RuntimeError((out + err).decode(errors='replace'))
    value = json.loads(out.decode().strip())
    with (HERE / 'OVERLAY_STAGE_RECEIPT_v3.json').open('x') as handle: json.dump(value, handle, indent=2); handle.write('\n')
    return value


def compile_remote():
    expected = hashlib.sha256((HERE / 'compile_overlay_v3.py').read_bytes()).hexdigest()
    code = 'import hashlib,subprocess,sys,json\nfrom pathlib import Path\np=Path(' + repr(REMOTE + '/compile_overlay_v3.py') + ')\n'
    code += 'assert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(expected) + '\n'
    code += 'result=subprocess.run([sys.executable,"-B",str(p)],capture_output=True,text=True)\nprint(json.dumps({"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}))'
    result = json.loads(remote.call(code, timeout=480))
    with (HERE / 'OVERLAY_COMPILE_RECEIPT_v3.json').open('x') as handle: json.dump(result, handle, indent=2); handle.write('\n')
    if result['returncode']: raise RuntimeError(json.dumps(result))
    return json.loads(result['stdout'])


def collect():
    code = '''import json,hashlib
from pathlib import Path
d=Path(OUTPUT_LITERAL)
r=json.loads((d/'READINESS.json').read_text());index=json.loads((d/'DESCRIPTOR_INDEX.json').read_text())['fits']
files={}
for name in ['READINESS.json','BLOCKED_FITS.json','SOURCE_RECEIPTS.json','DELTA.json','VERIFICATION.json']:
 b=(d/name).read_bytes();files[name]={'text':b.decode(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
files['COMPACT_INDEX.json']={'value':{'schema':'expanded-compact-fit-index-v3','full_index':r['artifacts']['index'],
 'columns':['fit_id','status','ancestor_contract_hash','expanded_contract_hash','descriptor_sha256'],
 'fits':[[x['id'],x['status'],x['ancestor_contract_hash'],x['expanded_contract_hash'],x['descriptor']['sha256']] for x in index]}}
print(json.dumps(files))
'''.replace('OUTPUT_LITERAL', repr(REMOTE + '/generated_v3'))
    files = json.loads(remote.call(code, timeout=180))
    output = HERE / 'compact_generated_v3'; output.mkdir(exist_ok=False)
    receipts = {}
    for name, row in files.items():
        raw = row['text'].encode() if 'text' in row else (json.dumps(row['value'], separators=(',', ':')) + '\n').encode()
        if 'text' in row: assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        with (output / name).open('xb') as handle: handle.write(raw)
        receipts[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    receipt = {'remote_output': REMOTE + '/generated_v3', 'files': receipts, 'full_fit_descriptors_copied': 0,
               'data_or_tensor_files_copied': 0, 'native_work': 0, 'campaign_GO': False}
    (output / 'COLLECTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('stage', 'compile', 'collect'))
    action = parser.parse_args().action
    print(json.dumps({'stage': stage, 'compile': compile_remote, 'collect': collect}[action](), sort_keys=True))
