"""Narrow resource observation for a prospective co-resident DDI check.

No model import, process inventory, quality read or launch is performed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
TRANSPORT = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
REPO = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
UUID = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'


def main():
    spec = importlib.util.spec_from_file_location('ddi_practical_runtime_transport', TRANSPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    code = '''from datetime import datetime,timezone
from pathlib import Path
import json,os,subprocess
assert os.uname().nodename=='peptide' and str(Path.cwd())==REPO
q=subprocess.run(['nvidia-smi','-i',UUID,'--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
items=[v.strip() for v in q.stdout.strip().split(',')]
assert items[0]==UUID and len(items)==4
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_UUID=UUID,memory_total_MiB=int(items[1]),memory_used_MiB=int(items[2]),memory_free_MiB=int(items[3]),numerical_execution=False,TEST_read=False,unrelated_process_inventory=False)))
'''
    value = module.run('ddi_f4_practical_gpu_window_20261004_01',
                       'REPO=' + repr(REPO) + '\nUUID=' + repr(UUID) + '\n' + code)
    result = dict(observation=value, root_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation='Instantaneous free memory only; not full-fit or combined-update admission.',
                  root_UTC=datetime.now(timezone.utc).isoformat())
    with (HERE / 'GPU_WINDOW_01.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
