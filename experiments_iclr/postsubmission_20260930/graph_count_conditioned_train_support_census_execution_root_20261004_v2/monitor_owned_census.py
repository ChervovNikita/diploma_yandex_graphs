"""Read only the original owned census handle and its bounded metadata."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owned_census_driver', HERE / 'stage_admit_census.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=int, required=True)
args = parser.parse_args()
assert 1 <= args.sequence <= 99
out = HERE / ('owned_monitor%02d' % args.sequence)
out.mkdir()
expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
code = driver.context() + 'expected=' + repr(expected) + '\n'
code += '''
proc=Path('/proc')/str(expected['supervisor_PID']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 assert int(f[19])==expected['supervisor_start_ticks']
 argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 assert f[0]=='Z' or argv==expected['command']
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(f[19]),state=f[0])
files=[];large=[];result=None;seed_summaries=[];terminal=None;progress=None
names=['supervision/run01/TERMINAL.json','supervision/run01/PHYSICAL_TERMINAL.json',
 'supervision/run01/SUPERVISOR_CUSTODY.json','supervision/run01/PROGRESS.json',
 'supervision/run01/CHILD_STDERR.txt','run01/CENSUS.json','run01/PROGRESS.json',
 'run01/FILE_CUSTODY.json','run01/FINAL_CUSTODY.json','run01/FAILURE.json',
 'run01/seed0_COUNTS.json','run01/seed1_COUNTS.json','run01/seed2_COUNTS.json','DETACHED_STDERR.txt']
for relative in names:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<=64*1024**2
 b=p.read_bytes();b.decode('utf8');row=dict(path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if len(b)<2_000_000:files.append(dict(row,data=base64.b64encode(b).decode()))
 else:large.append(row)
 if relative=='run01/CENSUS.json':result=json.loads(b)
 if relative=='run01/PROGRESS.json':progress=json.loads(b)
 if relative=='supervision/run01/TERMINAL.json':terminal=json.loads(b)
 if relative.startswith('run01/seed'):
  q=json.loads(b);seed_summaries.append(dict(seed=q['seed'],batches=len(q['rows']),descriptor=row,
   population_fields={name:sorted(q['rows'][0]['populations'][name]) for name in ['positive','negative']}))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,
 files=files,large_server_retained_metadata=large,result=result,progress=progress,
 seed_metadata=seed_summaries,physical_terminal=terminal,signals_sent=False,retries=False,
 metadata_only=True,models_or_scores_read=False)))
'''
value = driver.transport('monitor%02d' % args.sequence, code)
value['fetched_descriptors'] = []
for row in value.pop('files'):
    raw = base64.b64decode(row.pop('data'), validate=True)
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    p = out / row['path']
    assert p.resolve().is_relative_to(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as stream:
        stream.write(raw)
    value['fetched_descriptors'].append(row)
with (out / 'OBSERVATION.json').open('x') as stream:
    json.dump(value, stream, indent=2); stream.write('\n')
terminal = value.get('physical_terminal')
summary = dict(UTC=value['UTC'], supervisor_identity=value['supervisor_identity'],
    progress=value['progress'], result=value['result'], seed_metadata=value['seed_metadata'],
    physical_status=terminal.get('status') if terminal else None,
    stop_reason=terminal.get('stop_reason') if terminal else None,
    child_exit_code=terminal.get('child_exit_code') if terminal else None,
    linked=terminal.get('linked') if terminal else None)
with (out / 'SUMMARY.json').open('x') as stream:
    json.dump(summary, stream, indent=2); stream.write('\n')
print(json.dumps(summary, indent=2))
