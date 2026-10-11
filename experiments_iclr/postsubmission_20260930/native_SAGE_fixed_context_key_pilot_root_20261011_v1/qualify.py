"""Finite actual-TRAIN qualification only; no VALID/TEST quality or scientific fit."""
from pathlib import Path
import hashlib,importlib.util,json,socket,subprocess,sys
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=Path(__file__).resolve().parent

def main():
    assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    frozen=json.loads((H/'QUALIFICATION_FREEZE.json').read_text())
    for row in frozen['bound_files']:
        assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    assert frozen['independent_source_review_passed'] and frozen['TRAIN_only']
    assert not (H/'actual_family_v1').exists() and not (H/'actual_qualification_v1').exists()
    source=P/'native_SAGE_fixed_context_key_source_20261011_v1/qualify_full_train.py'
    spec=importlib.util.spec_from_file_location('admitted_fixed_key_TRAIN_qualification',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=H/'CONFIG.json';receipt=module.qualify(json.loads(config.read_text()),H/'actual_qualification_v1',hashlib.sha256(config.read_bytes()).hexdigest(),wall_seconds=180,cuda_bytes=32*1024**3,rss_bytes=16*1024**3)
    assert receipt['status']=='PASS' and len(receipt['units'])==16 and receipt['native_TRAIN_trajectories']==34
    assert receipt['VALID_quality_calls']==0 and not receipt['TEST_access'] and receipt['scientific_fits']==0
    import resource,torch
    receipt.update(qualified=True,configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),VALID_quality_scored=False,root_CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),root_CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),root_process_lifetime_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (H/'ACTUAL_QUALIFICATION_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'qualified':True,'units':len(receipt['units']),'native_TRAIN_trajectories':34,'elapsed_seconds':receipt['elapsed_seconds'],'CUDA_peak_allocated_bytes':receipt['root_CUDA_peak_allocated_bytes'],'VALID_quality_scored':False,'TEST_access':False,'scientific_fits':0}))
if __name__=='__main__':main()
