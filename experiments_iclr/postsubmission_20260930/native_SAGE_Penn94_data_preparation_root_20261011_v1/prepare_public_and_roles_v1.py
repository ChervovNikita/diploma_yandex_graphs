"""Create full public Penn94 inputs and separate target roles; never score."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=PHASE/'native_SAGE_Penn94_data_preparation_root_20261011_v1'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()

def describe(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    start=time.monotonic()
    receipt=json.loads((HERE/'PINNED_PAYLOAD_RECEIPT_V1.json').read_text())
    for row in receipt['files']:
        path=REPO/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    adapter=PHASE/'native_SAGE_Penn94_confirmation_preparation_20261011_v1/data_adapter.py'
    seal=json.loads((adapter.parent/'MANIFEST.json').read_text())
    # The exact adapter source is bound independently in the output receipt.
    import numpy as np
    import scipy
    from scipy.io import loadmat
    import torch
    import torch_geometric
    from torch_geometric.utils import one_hot
    spec=importlib.util.spec_from_file_location('penn94_public_adapter',adapter)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=HERE/'prepared_v1'
    assert not out.exists()
    mat=loadmat(HERE/'raw/Penn94.mat')
    a=mat['A'].tocsr().tocoo()
    assert a.shape==(41554,41554)
    raw_edges=np.stack((a.row,a.col)).astype(np.int64)
    public_metadata,private_y=module.custodian_metadata(np,mat['local_info'])
    splits=np.load(HERE/'raw/fb100-Penn94-splits.npy',allow_pickle=True)
    assert len(splits)>0 and isinstance(splits[0],dict)
    eligible=np.nonzero(private_y>=0)[0].astype(np.int64)
    roles,role_record=module.custodian_roles(np,splits[0],eligible)
    x,edges,public_record=module.public_tensors(torch,one_hot,public_metadata,raw_edges)
    assert torch.isfinite(x).all()
    out.mkdir()
    torch.save({'x':x,'edge_index':edges},out/'public.pt')
    # Roles and targets are not bundled with public model inputs.
    for name,ids in roles.items():
        file_name='confirm' if name=='evaluation_ids' else name
        np.savez(out/(file_name+'_targets.npz'),ids=ids,y=private_y[ids])
    public_record.update(preparation_seconds=time.monotonic()-start,
                         public_tensor_bytes=x.numel()*x.element_size()+edges.numel()*edges.element_size())
    result={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
            'dataset':'Penn94','published_split_index':0,'public':public_record,
            'roles':role_record,'files':{p.name:describe(p) for p in out.iterdir()},
            'adapter_source':describe(adapter),'preparer_source':describe(Path(__file__)),
            'raw_payload_receipt':describe(HERE/'PINNED_PAYLOAD_RECEIPT_V1.json'),
            'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,
                       'torch':torch.__version__,'torch_geometric':torch_geometric.__version__},
            'custodian_release_targets_decoded':True,
            'CONFIRM_targets_in_public_bundle':False,'targets_exposed_to_model_or_selector':False,
            'model_fits':0,'prediction_or_TEST_metrics_read':False,
            'statistical_or_novelty_confirmation':False}
    (HERE/'PUBLIC_ROLES_PREPARATION_RECEIPT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
