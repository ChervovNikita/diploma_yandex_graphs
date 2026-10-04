"""Read only queue/ownership/capacity metadata; excludes all scientific files."""
import argparse
import json
from pathlib import Path

from execution_common import ROOT, save
import prepare_and_launch as client


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=int,required=True)
    args=parser.parse_args();assert args.sequence>0
    destination=ROOT/('MONITOR_%04d.json'%args.sequence)
    assert not destination.exists()
    launch=json.loads((ROOT/'DETACHED_LAUNCH.json').read_text())
    queue=client.plan()['queue']
    code=client.context()+'launch='+repr(launch)+'\nqueue='+repr(queue)+'\n'
    code+='''def read(name):
 p=root/name
 if not p.exists():return None
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<2000000
 return json.loads(p.read_text())
def handle(expected):
 try:
  raw=(Path('/proc')/str(expected['PID'])/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 except FileNotFoundError:return dict(expected=expected,actual=None)
 actual=dict(PID=expected['PID'],start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),state=f[0])
 assert actual['start_ticks']==expected['start_ticks'] and actual['group']==actual['session']==expected['PID']
 return dict(expected=expected,actual=actual)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),queue=handle(launch),
 progress=read('QUEUE_PROGRESS.json'),terminal=read('QUEUE_TERMINAL.json'),exception=read('QUEUE_EXCEPTION.json'),
 cells=[],metadata_only=True,predictive_values_read=False,logs_read=False,checkpoints_read=False,
 signals_sent=False,scientific_completion_adopted=False,automatic_retry=False)
for cell in queue:
 prefix='supervision/'+cell['cell_id']+'/'
 ownership=read(prefix+'OWNED_CHILD.json');terminal=read(prefix+'PHYSICAL_TERMINAL.json')
 if ownership is None and terminal is None:continue
 item=dict(cell=cell,physical_terminal=terminal)
 if ownership is not None:
  identity=ownership['identity'];item['worker']=handle(dict(PID=identity['pid'],start_ticks=identity['start_ticks']))
 value['cells'].append(item)
print(json.dumps(value))
'''
    result=client.run('monitor_%04d'%args.sequence,code)
    save(destination,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
