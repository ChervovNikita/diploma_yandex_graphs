"""Source-only defaults and exact singleton/runtime release guards."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PYTHON = PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
VERSIONS = dict(torch='2.1.2+cu118', PyG='2.7.0', numpy='1.26.4',
                torch_scatter='2.1.2+pt21cu118', torch_sparse='0.6.18+pt21cu118', ogb='1.3.6', CUDA='11.8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    for row in manifest['files']:
        path = ROOT/row['path']
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Source seal differs: '+row['path'])
    return sha(ROOT/'MANIFEST.json')


def allocation(cpu_check=False):
    if socket.gethostname() != 'anogena-2-0':
        raise ValueError('Authorized singleton hostname required')
    if subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() != [GPU]:
        raise ValueError('Authorized singleton physical GPU required')
    if str(Path(sys.executable).absolute()) != str(PYTHON):
        raise ValueError('Existing interpreter required')
    if not ROOT.is_relative_to(PHASE) or Path.cwd().resolve() != REPO:
        raise ValueError('Exact project repository and phase only')
    expected = '' if cpu_check else GPU
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise ValueError('CPU check must hide CUDA; training must expose exact UUID')


def runtime_versions():
    import importlib.metadata
    import inspect
    import numpy, torch, torch_geometric, torch_sparse, torch_scatter
    found = dict(torch=torch.__version__, PyG=torch_geometric.__version__, numpy=numpy.__version__,
        torch_sparse=torch_sparse.__version__, torch_scatter=torch_scatter.__version__,
        ogb=importlib.metadata.version('ogb'), CUDA=torch.version.cuda)
    if found != VERSIONS:
        raise ValueError('Pinned installed runtime differs: '+str(found))
    from torch_geometric.nn import GATConv,GCNConv,GINEConv
    from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
    from torch_geometric.utils import negative_sampling
    from ogb.linkproppred import Evaluator as LinkEvaluator
    from ogb.graphproppred import Evaluator as GraphEvaluator
    providers={x.__module__+'.'+x.__qualname__:dict(path=inspect.getfile(x),sha256=sha(inspect.getfile(x)))
        for x in (GATConv,GCNConv,GINEConv,AtomEncoder,BondEncoder,negative_sampling,LinkEvaluator,GraphEvaluator)}
    if providers != json.loads((ROOT/'RUNTIME_PIN.json').read_text())['providers']:
        raise ValueError('Installed operator/evaluator source identity changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return found


def admit(job_path):
    job = json.loads(Path(job_path).read_text())
    # Before tensor/package import or label loading.
    if job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True:
        raise ValueError('Source preparation is disabled pending root release and independent review')
    if job.get('fixed_protocol_adopted') is not True:
        raise ValueError('Root adoption of this fixed prospective protocol required')
    if job.get('TEST_access') is not False or job.get('automatic_retry') is not False:
        raise ValueError('TEST closed and no automatic retry')
    allocation()
    if job['source_manifest_sha256'] != verify_manifest():
        raise ValueError('Reviewed exact source differs')
    from data import bound
    if not job.get('source_review_evidence') or not job.get('resource_qualification_evidence'):
        raise ValueError('Independent source review and task-complete resource checks required')
    for row in job['source_review_evidence']:
        review=json.loads(bound(PHASE,row).read_text())
        if review.get('source_manifest_sha256') != job['source_manifest_sha256'] or review.get('approved') is not True:
            raise ValueError('Exact source review approval receipt required')
    for row in job['resource_qualification_evidence']:
        receipt=json.loads(bound(PHASE,row).read_text())
        if receipt.get('schema') != 'internal-be-resource-qualification-v1' or receipt.get('passed') is not True or receipt.get('task') != job['task'] or receipt.get('source_manifest_sha256') != job['source_manifest_sha256']:
            raise ValueError('Task/resource receipt required for exact source')
        if receipt.get('complete_TRAIN_update') is not True or receipt.get('complete_VALID_evaluation') is not True or receipt.get('predictive_scores_closed') is not True:
            raise ValueError('Complete representative source/resource work required')
    config = json.loads(bound(PHASE, job['config']).read_text())
    if config['task'] != job['task'] or job['arm'] not in config['arms'] or job['seed'] not in config['pilot_seeds']:
        raise ValueError('Exact fixed pilot cell required')
    if job['config']['path'] != ROOT.name+'/configs/'+job['task']+'.json':
        raise ValueError('Fixed source configuration only')
    if job['soft_seconds'] != config['budget']['cell_soft_seconds'] or job['hard_seconds'] != config['budget']['cell_hard_seconds'] or job.get('external_hard_bound_confirmed') is not True:
        raise ValueError('Unchanged finite budget and external supervisor required')
    output = Path(job['output_directory']).resolve()
    if not output.is_relative_to(PHASE) or output.exists() or not output.parent.is_dir():
        raise ValueError('Fresh phase-owned output required')
    return job, config, output
