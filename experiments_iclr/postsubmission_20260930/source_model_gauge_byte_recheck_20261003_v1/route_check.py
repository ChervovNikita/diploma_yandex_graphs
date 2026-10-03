"""Read-only route/repository/source verification before a CPU fixture recheck."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'graph_ncNC_member_completion_qualification_preparation_20261003_v2'
INTERPRETER = Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
SOURCE_SEAL = 'a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9'
WITNESS_SHA = '9a6a738fa48816300b574cba57851d31831004867b1a6f9acf24346e79255bef'
INTERPRETER_SHA = '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'


def main():
    top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert top == str(REPO)
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert set(uuids) == {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
                          'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
    assert Path(sys.executable).resolve() == INTERPRETER.resolve()
    interpreter_sha = hashlib.sha256(INTERPRETER.read_bytes()).hexdigest()
    assert interpreter_sha == INTERPRETER_SHA
    witness = PHASE / 'source_model_gauge_probe_20261003_v1/WITNESS.json'
    assert hashlib.sha256(witness.read_bytes()).hexdigest() == WITNESS_SHA
    sys.path.insert(0, str(SOURCE))
    import native_reference
    assert native_reference.verify_seal() == SOURCE_SEAL
    import torch
    assert torch.__version__ == '2.7.1'
    assert torch.get_default_dtype() == torch.float32
    assert str(torch.get_default_device()) == 'cpu'
    assert not torch.cuda.is_initialized()
    out = PHASE / 'source_model_gauge_byte_recheck_20261003_v1'
    assert not out.exists(), 'Successor remote path must be fresh'
    print(json.dumps(dict(schema='gauge-byte-recheck-route-v1',
                         UTC=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                         target='shmelev@192.168.18.77', repo=top, GPU_UUIDs=uuids,
                         interpreter=str(INTERPRETER), interpreter_sha256=interpreter_sha,
                         source_manifest_sha256=SOURCE_SEAL, original_witness_sha256=WITNESS_SHA,
                         torch_version=torch.__version__, default_dtype=str(torch.get_default_dtype()),
                         default_device=str(torch.get_default_device()),
                         CUDA_initialized=torch.cuda.is_initialized(), successor_path_fresh=True,
                         remote_writes=False, dataset_checkpoint_or_label_access=False)))


if __name__ == '__main__':
    main()
