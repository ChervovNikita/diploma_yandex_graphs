"""Existing allocation context, immutable closure and full-data custody guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REPO = ROOT.parents[2]
ALLOCATION = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
RUNTIME = {'torch':'2.1.2+cu118','numpy':'1.26.4','PyG':'2.7.0',
           'torch_scatter':'2.1.2+pt21cu118','torch_sparse':'0.6.18+pt21cu118','CUDA':'11.8'}
TOLERANCES = {'logits':(1e-5,1e-6),'gradients_moments':(1e-4,2e-6),'parameters':(1e-5,1e-6)}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def bound(row):
    path = (PHASE/row['path']).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file() or sha(path)!=row['sha256']:
        raise ValueError('Exact phase-relative file binding changed')
    return path


def guard(program,kind):
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();job=json.loads(args.job.read_text());output=args.output.resolve()
    if job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True:
        raise ValueError('Disabled source requires independent root review and release')
    if job.get('TEST_access') is not False or job.get('retry') is not False or job['kind']!=kind:
        raise ValueError('Exact declared work kind; TEST closed and no retry')
    if str(REPO)!=ALLOCATION or Path.cwd().resolve()!=REPO.resolve() or socket.gethostname()!='anogena-2-0':
        raise ValueError('Existing one-GPU allocation repository only')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:
        raise ValueError('Exact authorized physical GPU and visible singleton required')
    if str(Path(sys.executable).absolute())!=str(PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'):
        raise ValueError('Existing native interpreter required')
    if output.exists() or not output.parent.is_dir() or not output.is_relative_to(PHASE.resolve()) or str(output)!=job['output_directory']:
        raise ValueError('Fresh exact root-owned output required')
    if sha(program)!=job['program_sha256'] or sha(ROOT/'MANIFEST.json')!=job['source_manifest_sha256']:
        raise ValueError('Reviewed program/manifest changed')
    for row in json.loads((ROOT/'MANIFEST.json').read_text())['files']:
        path=ROOT/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Source closure changed')
    if not job['source_review_evidence']:raise ValueError('Exact independent source-review evidence required')
    for row in job['source_review_evidence']:bound(row)
    if not isinstance(job['soft_seconds'],int) or job['soft_seconds']<=0 or not isinstance(job['hard_seconds'],int) or job['hard_seconds']<=job['soft_seconds'] or job.get('external_hard_bound_confirmed') is not True:
        raise ValueError('Root-bound finite soft/hard limits required')
    if kind in ('numerical','cost'):
        if job.get('fits_authorized') is not False or job.get('VALID_values_access') is not False:
            raise ValueError('Optional discarded full-graph TRAIN diagnostics only')
    elif kind=='fit':
        plan=json.loads(bound(job['plan']).read_text())
        if plan.get('root_adopted') is not True or plan['seeds']!=[17,29,43] or job['seed'] not in plan['seeds']:
            raise ValueError('Exact fixed three-seed source plan must be adopted')
        if job['arm'] not in plan['arms']+['DONOR_COMMON','DONOR_INDEPENDENT']:
            raise ValueError('Fixed arm/phase required')
        if job.get('fits_authorized') is not True or job.get('VALID_values_access') is not True:
            raise ValueError('Root-authorized ordinary fitting and prescribed selector roles required')
        # Explicit user engineering waiver: tiny numeric reference agreement is
        # not a fitting prerequisite. Actual source/data/runtime/finite-gradient
        # checks and owned finite caps remain mandatory. No failed result is passed.
        if plan.get('engineering_qualification_policy')!='ordinary_finite_smoke_no_strict_parity_prerequisite':
            raise ValueError('Exact reviewed engineering waiver policy required')
        if bound(plan['engineering_waiver'])!=ROOT/'ENGINEERING_WAIVER.json':
            raise ValueError('Exact sealed residual engineering policy required')
        native_policy=plan['native_training_engineering_policy']
        if native_policy['sha256']!=plan['native_training_engineering_policy_sha256']:
            raise ValueError('Native training policy hash differs')
        bound(native_policy)
    else:raise ValueError('Only fixed fitting or optional TRAIN diagnostics')
    return args,job,output


def runtime():
    import numpy as np
    import torch
    import torch_geometric
    import torch_scatter
    import torch_sparse
    versions={'torch':str(torch.__version__),'numpy':np.__version__,'PyG':torch_geometric.__version__,
              'torch_scatter':torch_scatter.__version__,'torch_sparse':torch_sparse.__version__,'CUDA':torch.version.cuda}
    if versions!=RUNTIME or torch.cuda.device_count()!=1:raise ValueError('Pinned full native operator/runtime differs')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return torch,versions


def load_data(torch,job,train_only):
    authority=json.loads(bound(job['data_manifest']).read_text())
    data=torch.load(bound(authority['available']),map_location='cpu',weights_only=True)
    keys={'x','edge_index','train_ids','train_y'}
    if not train_only:keys|={'valid_ids','valid_y'}
    if set(data)!=keys or tuple(data['x'].shape)!=(11701,300) or data['x'].dtype!=torch.float32 or tuple(data['edge_index'].shape)!=(2,442907) or len(data['train_ids'])!=580:
        raise ValueError('Full official WikiCS graph/roles required; no toy or TEST payload')
    if train_only and authority.get('qualified_reader_deserializes_VALID_TEST_values') is not False:
        raise ValueError('Exact TRAIN-only projection authority required')
    if not train_only and (len(data['valid_ids'])!=5274 or authority.get('TEST_labels_available_to_trainer') is not False):
        raise ValueError('Fixed official split0 complete VALID only')
    if data['edge_index'].dtype!=torch.long or data['train_ids'].dtype!=torch.long or data['train_y'].dtype!=torch.long:
        raise ValueError('Source integer graph/supervision types required')
    if len(data['train_ids'].unique())!=580 or int(data['train_y'].min())<0 or int(data['train_y'].max())>=10:
        raise ValueError('Exact complete TRAIN supervision geometry required')
    return {k:(v.to('cuda:0') if k not in ('valid_ids','valid_y') else v) for k,v in data.items()},authority


def deadline(started,job):
    if time.monotonic()-started>job['soft_seconds']:raise TimeoutError('Fixed cap; no partial fit, shortening or retry')


def compare(torch,left,right,kind):
    if left.shape!=right.shape or left.dtype!=right.dtype or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError('Reference shape/type/finiteness differs')
    rtol,atol=TOLERANCES[kind]
    error=float((left-right).abs().max()) if left.numel() else 0.
    if not torch.allclose(left,right,rtol=rtol,atol=atol):raise ValueError(kind+' reference mismatch, max_abs='+str(error))
    return error
