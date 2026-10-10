"""Fixed one-export, full equal-control fusion study; no base-model training."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,sys,time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    frozen=json.loads((H/'FREEZE.json').read_text())
    for row in frozen['bound_files']:
        assert hashlib.sha256((P/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    source=P/'common_wrapper_graph_reliability_source_20261010_v1'
    sys.path.insert(0,str(source))
    import export_neighbors,study
    start=time.monotonic()
    exported=export_neighbors.run(source/'FROZEN_SUPPORT.json',H/'actual_export_v1')
    assert exported['complete'] and exported['banks']==45
    fitted=study.run(source/'FROZEN_SUPPORT.json',H/'actual_export_v1',H/'actual_study_v1')
    result=dict(export=exported,study=fitted,seconds=time.monotonic()-start,TEST_access=False,new_base_fits=0)
    with (H/'COMPLETE.json').open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
