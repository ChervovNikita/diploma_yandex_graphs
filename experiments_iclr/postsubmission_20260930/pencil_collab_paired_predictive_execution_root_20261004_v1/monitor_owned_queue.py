"""Read original PENCIL queue handles and progress counters, without scores."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pencil_scientific_client', HERE / 'prepare_and_launch.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=int, required=True)
    args = parser.parse_args()
    assert args.sequence > 0
    receipt = HERE / ('MONITOR_%04d.json' % args.sequence)
    assert not receipt.exists()
    original = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
    code = client.base.context() + 'original=' + repr(original) + '\n'
    code += '''def read(relative):
 p=root/relative
 if not p.exists():return None
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<2000000
 return json.loads(p.read_text())
def handle(expected):
 p=Path('/proc')/str(expected['PID'])/'stat'
 try:
  raw=p.read_text();f=raw[raw.rfind(')')+2:].split()
 except FileNotFoundError:return dict(expected=expected,actual=None)
 actual=dict(PID=expected['PID'],start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),state=f[0])
 assert actual['start_ticks']==expected['start_ticks'] and actual['group']==actual['session']==expected['PID']
 return dict(expected=expected,actual=actual)
progress=read('QUEUE_PROGRESS.json');complete=read('QUEUE_COMPLETE.json');failure=read('QUEUE_FAILURE.json');error=read('QUEUE_EXCEPTION.json')
result=dict(UTC=datetime.now(timezone.utc).isoformat(),queue=handle(original),progress=progress,
 queue_complete=complete,queue_failure=failure,queue_exception=error,seeds=[],
 metadata_only=True,predictive_values_read=False,signals_sent=False,automatic_retry=False)
for seed in (0,1,2):
 launch=read('seed%d_SUPERVISOR_LAUNCH.json'%seed)
 if launch is None:continue
 item=dict(seed=seed,supervisor=handle(launch))
 worker=read('seed%d/supervision/run01/LAUNCH.json'%seed)
 if worker is not None:
  identity=worker['identity'];expected=dict(PID=identity['pid'],start_ticks=identity['start_ticks'])
  item['worker']=handle(expected)
 p=read('seed%d/run01/PROGRESS.json'%seed)
 if p is not None:item['progress']={k:p[k] for k in ('status','stage','epoch_index','TRAIN_batches','TRAIN_queries','VALID_batches','VALID_queries','optimizer_updates','elapsed_seconds') if k in p}
 terminal=read('seed%d/supervision/run01/TERMINAL.json'%seed)
 if terminal is not None:item['terminal']={k:terminal[k] for k in ('status','physical_exit_code','physical_session_closed','direct_child_reaped','stop','completed_scientific_fits','final_parent_budget_check') if k in terminal}
 result['seeds'].append(item)
print(json.dumps(result))
'''
    result = client.transport('monitor%04d' % args.sequence, code)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
