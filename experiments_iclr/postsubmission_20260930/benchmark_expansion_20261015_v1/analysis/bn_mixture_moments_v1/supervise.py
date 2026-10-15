"""Run the three frozen CPU diagnostics sequentially; preserve any failure."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import selected_molecular
    selected_molecular.require_allocation()
    root, output = (Path(p).resolve() for p in sys.argv[1:])
    assert here.is_relative_to(root) and output.is_relative_to(root)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows=[]
    for index in range(3):
        argv=[sys.executable,'-B','-u',str(here/'diagnose.py'),str(root),str(index),str(output/f'seed_{index}')]
        with (output/f'seed_{index}.log').open('x') as log:
            child=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
            row={'index':index,'PID':child.pid,'argv':argv,'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (output/f'START_{index}.json').write_text(json.dumps(row,indent=2)+'\n')
            try:
                row['returncode']=child.wait(timeout=330)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                row.update(returncode=child.returncode,timeout=True)
        rows.append(row)
        if row['returncode']: break
    value={'status':'complete' if len(rows)==3 and all(r['returncode']==0 for r in rows) else 'failed',
           'children':rows,'wall_seconds':time.monotonic()-started,'automatic_retries':0,
           'TRAIN_calls':0,'TEST_accessed':False}
    (output/'END.json').write_text(json.dumps(value,indent=2)+'\n')
    return 0 if value['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
