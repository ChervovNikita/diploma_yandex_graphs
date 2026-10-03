"""Recover complete small receipts from the read-only MacLink transport, with hash checks."""
from pathlib import Path
import argparse
import base64
import hashlib
import json

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=int,required=True);args=parser.parse_args()
assert 1<=args.sequence<=99
transport_path=PHASE/f'gpu77_connection_recovery_v1/commands/buddy_v4_heldout_evaluation_monitor_20261004_v{args.sequence}/RECEIPT.json'
transport=json.loads(transport_path.read_text());assert transport['exit_code']==0
value=json.loads(transport['stdout'].strip());assert value['sequence']==args.sequence
assert value['remote_writes'] is False and value['evaluation_relaunched'] is False
assert value['scientific_binary_payloads_fetched'] is False and value['prediction_export_executed'] is False
assert value['scalar_results_fetched_only_after_all15_physical_and_stage_closure'] is True
destination=HERE/f'fetched_registered_outputs_v{args.sequence}'
destination.mkdir(exist_ok=False)
rows=[]
for row in value['files']:
    data=base64.b64decode(row['data_base64'],validate=True)
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
    relative=path.relative_to('experiments_iclr/postsubmission_20260930')
    target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as handle:handle.write(data)
    item={key:row[key] for key in row if key!='data_base64'}
    item['local_path']=str(target.relative_to(HERE));rows.append(item)
summary={key:value[key] for key in value if key!='files'}
summary['files']=rows
summary['transport']=dict(path=str(transport_path.relative_to(PHASE)),bytes=transport_path.stat().st_size,
                          sha256=hashlib.sha256(transport_path.read_bytes()).hexdigest())
with (HERE/f'MONITOR_RESULT_{args.sequence:02d}.json').open('x') as handle:
    json.dump(summary,handle,indent=2);handle.write('\n')
print(json.dumps({key:summary[key] for key in ('status','UTC','sequence','all15_closure_established',
    'final_scalar_files_present','physical_exit_code','stage_status','owned_handles','missing','oversized')}))
