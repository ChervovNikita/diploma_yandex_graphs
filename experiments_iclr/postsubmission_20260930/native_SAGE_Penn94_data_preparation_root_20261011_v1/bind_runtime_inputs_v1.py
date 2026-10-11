"""Bind existing public tensors, id-only roles and runtime sources; no fit."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=PHASE/'native_SAGE_Penn94_data_preparation_root_20261011_v1'
SOURCE=PHASE/'native_SAGE_Penn94_confirmation_integrated_source_20261011_v1'

def item(path):
    path=Path(path).resolve()
    assert path.is_relative_to(REPO)
    return {'path':str(path),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    import numpy as np
    cfg=json.loads((SOURCE/'CONFIG_TEMPLATE.json').read_text())
    for role in ('cal','confirm'):
        target=HERE/'prepared_v1'/(role+'_ids.npz')
        assert not target.exists()
        with np.load(HERE/'prepared_v1'/(role+'_targets.npz'),allow_pickle=False) as source:
            ids=source['ids']
            assert ids.dtype==np.int64 and ids.ndim==1
            # Deliberately do not open the target array.
            np.savez(target,ids=ids)
        cfg['inputs'][role+'_ids']=item(target)
    cfg['provider_modules']={name:item(importlib.import_module(name).__file__)
                             for name in cfg['provider_modules']}
    cfg['provider_receipts']=list(cfg['provider_modules'].values())
    cfg['status']='ROOT_CONFIG_BOUND_FOR_REPRESENTATIVE_QUALIFICATION_ONLY'
    cfg['root_pending']=['Complete-graph qualification, measured schedule and separate acquisition/CAL/CONFIRM admissions.']
    cfg['owner_seconds']={'qualify':600,'acquire':86400,'calibrate':1800,'confirm':1800}
    config=HERE/'QUALIFICATION_CONFIG_V1.json'
    assert not config.exists()
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    admission={'stage':'qualify','admitted':True,'config_sha256':item(config)['sha256'],
               'integrated_source_seal_sha256':item(SOURCE/'SEAL.json')['sha256'],
               'source_and_provider_custody_complete':True,'CONFIRM_truth_access':False,
               'input_custody_receipt_sha256':item(HERE/'PUBLIC_ROLES_PREPARATION_RECEIPT_V1.json')['sha256'],
               'scope':'Actual full graph, native ten optimizer bodies; disposable engineering qualification, not predictive evidence.',
               'UTC':datetime.now(timezone.utc).isoformat()}
    target=HERE/'QUALIFICATION_ADMISSION_V1.json'
    assert not target.exists()
    target.write_text(json.dumps(admission,indent=2)+'\n')
    result={'complete':True,'UTC':admission['UTC'],'config':item(config),'admission':item(target),
            'membership_exports':{k:cfg['inputs'][k+'_ids'] for k in ('cal','confirm')},
            'provider_modules':cfg['provider_modules'],'model_fits':0,'predictive_scores_read':False,
            'CONFIRM_targets_read':False,'qualification_admitted':True,'scientific_execution_admitted':False}
    (HERE/'RUNTIME_INPUT_BINDING_RECEIPT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
