"""Bind the already projected VALID-only file. No model or TEST input."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert HERE.is_relative_to(PHASE) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    prep = PHASE/'masked_context_pubmed_allocation_preparation_20261010_v1'
    custody_file = prep/'DATA_V4_ABSOLUTE_EXPORTER_CUSTODY.json'
    train = json.loads(custody_file.read_text())
    valid = prep/'data_v4/VALID_ONLY.npz'
    with np.load(valid, allow_pickle=False) as archive:
        assert set(archive.files) == {'valid_ids','valid_y'}
        arrays = {key:archive[key].copy() for key in archive.files}
    assert all(v.dtype == np.int64 and v.shape == (3942,) for v in arrays.values())
    assert np.bincount(arrays['valid_y'],minlength=3).tolist() == [820,1547,1575]
    def fp(value):
        header=json.dumps(dict(dtype=value.dtype.str,shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(header+b'\n'+value.tobytes(order='C')).hexdigest()
    sha = lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    value = dict(schema='masked-context-PubMed-VALID-role-custody-v1',
        split_protocol=train['split_protocol'],split_seed=train['split_seed'],split_identity=train['split_identity'],
        VALID_count=3942, VALID_class_counts=[820,1547,1575],
        train_custody_sha256=sha(custody_file),valid_bundle_sha256=sha(valid),
        array_fingerprints={key:fp(v) for key,v in arrays.items()},
        TEST_labels_loaded=False,model_constructed=False,actual_custody=True,
        custodian_source=dict(path=str(Path(__file__)),sha256=sha(Path(__file__))))
    out=HERE/'VALID_CUSTODY.json'
    assert not out.exists()
    out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
