"""Run independent read-only checks via existing guarded clients; save only locally."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64, hashlib, json, sys, time, zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent; E=HERE.parents[2]
sys.path.insert(0,str(E)); import remote_control
sys.path.insert(0,str(E/'planning/three_gpu_sync_20261015_v1')); import remote51
bindings=json.loads((HERE.parent/'GLOBAL_SMOKE_BINDINGS_PORT6.json').read_text())
source=(HERE/'audit_remote.py').read_text()
def run(worker):
    host=worker.split('/')[0]; rc=remote_control if host=='allocation' else remote51
    root=rc.REMOTE_ROOT if host=='allocation' else rc.ROOT
    code='AUDIT_HOST='+repr(host)+'\nAUDIT_ROOT='+repr(root)+'\nAUDIT_WORKER='+repr(worker)+'\nBINDINGS='+repr(bindings)+'\n'+source
    command='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(code.encode(),9)).decode())+')))'
    started=time.time()
    try:
        raw=rc.call(command,timeout=55)
        value=json.loads(zlib.decompress(base64.b64decode(raw)))
        value['audit_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
        (HERE/(worker.replace('/','_')+'.json')).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'worker':worker,'elapsed':time.time()-started,'counts':value['workers'][worker]['counts'],'pass_validated':value['workers'][worker]['pass_validated'],'audit_errors':value['workers'][worker]['audit_errors'],'pid_matches':len(value['pid_matches'])}),flush=True)
    except Exception as exc:
        (HERE/(worker.replace('/','_')+'_TRANSPORT_FAILURE.json')).write_text(json.dumps({'error':str(exc),'type':type(exc).__name__,'elapsed':time.time()-started},indent=2)+'\n')
        print(json.dumps({'worker':worker,'error':str(exc),'type':type(exc).__name__}),flush=True)
workers=sys.argv[1:] or ['allocation/gpu0','server51/gpu0','server51/gpu1']
with ThreadPoolExecutor(max_workers=3) as pool:
    for future in as_completed([pool.submit(run,w) for w in workers]):future.result()
