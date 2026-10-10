"""One fixed closed-bank serving diagnostic, no new backbone training."""
from pathlib import Path
import hashlib,importlib.util,json,socket,subprocess,sys
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    for row in json.loads((H/'FREEZE.json').read_text())['bound_files']:
        assert hashlib.sha256((P/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    path=P/'bootstrap_graph_serving_combination_source_20261010_v1/source.py'
    spec=importlib.util.spec_from_file_location('closed_bank_combination',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.run(H/'FROZEN_SUPPORT.json',H/'actual_study_v1')
if __name__=='__main__':main()
