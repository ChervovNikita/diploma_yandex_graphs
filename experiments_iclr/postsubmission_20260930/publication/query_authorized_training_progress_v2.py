"""Live initializer progress with canonical phase custody, not coordinator counts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.snapshot):
        raise ValueError('Fresh simple snapshot required')
    output = PHASE / 'coordination_snapshots' / args.snapshot
    output.mkdir(exist_ok=False)
    code = (PHASE / 'coordination_snapshots/20261002_photo_live_progress_v3/REMOTE_CODE.txt').read_text()
    code += '''
import sys
from pathlib import Path
helper=phase/'graph_init_precision_continuation_v3_cap_binding'
sys.path.insert(0,str(helper))
import continuation_support as custody
registry_path=phase/'graph_init_precision_execution_root_v2/study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json'
registry_record=custody.descriptor(registry_path)
registry=custody.registry_guard(registry_record)
anchor=Path(registry['anchor_directory'])
counts={};pending=[];failures=[]
for row in registry['attempts']:
 terminal=anchor/'terminals'/(row['key']+'.json')
 if not terminal.is_file():
  pending.append({'phase':row['phase'],'arm':row['arm'],'output':row['output'],'claim_present':(anchor/'claims'/(row['key']+'.json')).is_file()})
  continue
 try:
  custody.completed_phase(registry_record,registry,row)
  counts[row['phase']]=counts.get(row['phase'],0)+1
 except Exception as error:
  failures.append({'key':row['key'],'phase':row['phase'],'error_type':type(error).__name__,'reason':str(error)})
print(json.dumps({'canonical_phase_counts':counts,'valid_canonical_phases':sum(counts.values()),'required_phases':72,
 'pending':pending,'custody_failures':failures,'registry_sha256':registry_record['sha256'],
 'validation_metrics_disclosed':False,'final_labels_read':False,
 'scope':'canonical claim/terminal/freeze and text custody; tensor payload audit remains required before scoring'}))
'''
    (output / 'REMOTE_CODE.txt').write_text(code)
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', LOGIN]
    result = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])],
                            capture_output=True, text=True, timeout=45)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), destination=LOGIN,
                   exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   source_sha256=hashlib.sha256(code.encode()).hexdigest(), outcome_metrics_requested=False)
    with (output / 'RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
