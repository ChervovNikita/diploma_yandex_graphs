"""Observe only the admitted CPU supervisor and exact owned output metadata."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('count_cpu_owned_transport', HERE / 'stage_admit_cpu.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
parser = argparse.ArgumentParser()
parser.add_argument('--sequence',type=int,required=True)
args = parser.parse_args()
assert 1 <= args.sequence <= 99
out = HERE / ('owned_monitor%02d' % args.sequence)
out.mkdir()
expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
code = driver.context() + 'expected=' + repr(expected) + '\n'
code += '''
proc=Path('/proc')/str(expected['supervisor_PID']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split();assert int(f[19])==expected['supervisor_start_ticks']
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(f[19]),state=f[0])
files=[];large=[];summary=None
for relative in ['owned_supervisor/run01/SUPERVISOR_TERMINAL.json','owned_supervisor/run01/SUPERVISOR_CUSTODY.json','owned_supervisor/run01/PHYSICAL_PROGRESS.json','owned_supervisor/run01/CHILD_STDERR.txt','fabricated_cpu/run01/QUALIFICATION.json','fabricated_cpu/run01/FINAL_CUSTODY.json','fabricated_cpu/run01/SOURCE_CUSTODY.json','fabricated_cpu/run01/RUNTIME_SCOPE.json','fabricated_cpu/run01/FAILURE.json','fabricated_cpu/run01/PROGRESS.json','DETACHED_STDERR.txt']:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<=32*1024**2
 b=p.read_bytes();b.decode('utf8');row=dict(path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if len(b)<2_000_000:files.append(dict(row,data=base64.b64encode(b).decode()))
 else:large.append(row)
 if relative.endswith('/QUALIFICATION.json'):
  q=json.loads(b);r=q['result'];summary=dict(status=q['status'],source_manifest_sha256=q['source_manifest_sha256'],root_release_sha256=q['root_release_sha256'],runtime_profile=q['runtime_profile'],cases=r['cases'],qa_manifest_sha256=q['qa_manifest_sha256'],comparison_reports=len(r['comparison_reports']),native_full_batch_resource_qualification=q['native_full_batch_resource_qualification'],inclusive_wall_seconds=q['inclusive_wall_seconds'],peak_RSS_bytes=q['peak_RSS_bytes'],qualification_descriptor=row,scientific_fit_admitted=q['scientific_fit_admitted'],GPU_data_access=q['GPU_data_access'],TEST_supported=q['TEST_supported'])
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,files=files,large_server_retained_metadata=large,qualification_summary=summary,signals_sent=False,retries=False,metadata_only=True)))
'''
value = driver.transport('monitor%02d' % args.sequence,code)
rows = value.pop('files')
terminal = None
value['fetched_descriptors'] = []
for row in rows:
    raw = base64.b64decode(row.pop('data'),validate=True)
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    p=out/row['path']
    assert p.resolve().is_relative_to(out)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream:
        stream.write(raw)
    value['fetched_descriptors'].append(row)
    if row['path'].endswith('SUPERVISOR_TERMINAL.json'):
        terminal=json.loads(raw)
with (out/'OBSERVATION.json').open('x') as stream:
    json.dump(value,stream,indent=2);stream.write('\n')
summary=dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],qualification=value['qualification_summary'],physical_status=terminal.get('physical_status') if terminal else None,overall_status=terminal.get('overall_status') if terminal else None,stop_reason=terminal.get('stop_reason') if terminal else None,child_exit_code=terminal.get('child_exit_code') if terminal else None,collected_result=terminal.get('collected_oracle_result') if terminal else None)
with (out/'SUMMARY.json').open('x') as stream:
    json.dump(summary,stream,indent=2);stream.write('\n')
print(json.dumps(summary,indent=2))
