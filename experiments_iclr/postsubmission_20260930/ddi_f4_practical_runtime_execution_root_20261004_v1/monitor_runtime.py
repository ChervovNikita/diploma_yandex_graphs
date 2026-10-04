"""Fetch only owned runtime receipts; never fetch DDI tensors or checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/' + HERE.name
TRANSPORT = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=int, required=True)
    args = parser.parse_args()
    assert args.sequence > 0
    destination = HERE / ('monitor%02d' % args.sequence)
    assert not destination.exists()
    launch = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
    spec = importlib.util.spec_from_file_location('ddi_runtime_monitor_transport', TRANSPORT)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.HERE = HERE
    code = '''from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json,os
root=Path(REMOTE)
assert os.uname().nodename=='peptide'
def identity(pid):
 try:
  raw=(Path('/proc')/str(pid)/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
  return dict(PID=pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),state=f[0])
 except FileNotFoundError:return None
actual=identity(launch['PID'])
if actual is not None:assert actual['start_ticks']==launch['start_ticks'] and actual['group']==actual['session']==launch['PID']
names=['QUEUE_TERMINAL.json','DETACHED.stderr']
for arm in ('target_only','joint','separate'):
 names.extend(arm+'/'+p for p in ('OWNED_CHILD.json','PHYSICAL_TERMINAL.json','STDERR.txt','run01/runtime_summary.json','run01/runtime_updates.jsonl','run01/paired_stream.jsonl'))
files=[]
for name in names:
 p=root/name
 if not p.exists():continue
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<3000000
 raw=p.read_bytes()
 files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor=actual,files=files,VALID_scoring=False,TEST_access=False,model_or_dataset_payload_fetched=False)))
'''
    value = transport.run('ddi_f4_practical_monitor%02d_20261004' % args.sequence,
                          'REMOTE=' + repr(REMOTE) + '\nlaunch=' + repr(launch) + '\n' + code)
    destination.mkdir()
    import base64
    for row in value['files']:
        path = destination / row['path']
        assert path.resolve().is_relative_to(destination)
        raw = base64.b64decode(row['data'], validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        del row['data']
    (destination / 'OBSERVATION.json').write_text(json.dumps(value, indent=2) + '\n')
    report = dict(UTC=value['UTC'], supervisor=value['supervisor'], runtime_arms=[], VALID_scoring=False, TEST_access=False)
    for arm in ('target_only', 'joint', 'separate'):
        path = destination / arm / 'run01/runtime_summary.json'
        if path.is_file():
            r = json.loads(path.read_text())
            report['runtime_arms'].append({k: r[k] for k in ('arm', 'status', 'completed_native_updates',
                'completed_native_records', 'epoch_wall_seconds', 'error_type', 'error') if k in r})
    terminal = destination / 'QUEUE_TERMINAL.json'
    if terminal.is_file():
        r = json.loads(terminal.read_text())
        report['queue_status'] = r['status']
        report['failure'] = r['failure']
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
