"""Disabled prospective public-edge-only custodian. No models or role labels."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from source import HERE,PHASE,bind,inside,sha,verify_manifest,fingerprint,write
from rewire import swap_graph
from stage_plan import REWIRE,SERVER_HOST,SERVER_GPU,SERVER_PHASE


def admit(path,digest):
    path=inside(path)
    if sha(path)!=digest:raise ValueError('Exact separate root graph-generation release required')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='masked-context-public-rewire-generation-release-v1':raise ValueError('Public graph custodian release required')
    for key in ('enabled','root_generation_authorized','source_review_approved','finite_owner_bound','public_edge_custody_verified'):
        if spec.get(key) is not True:raise ValueError('Graph generation remains disabled: '+key)
    if any(spec.get(k) is not False for k in ('automatic_retry','labels_used','model_construction','TEST_access')):
        raise ValueError('No retry, labels, model or TEST input')
    bindings=verify_manifest(spec['source_manifest_sha256'])
    if spec.get('recipe')!=REWIRE or spec.get('maximum_active_seconds')!=600:
        raise ValueError('One frozen rewire recipe and600-second finite generation envelope required')
    if spec['train_bundle']['sha256']!=bindings['TRAIN_bundle_server_only']['sha256']:
        raise ValueError('Only the existing exact public factual edges from frozen TRAIN bundle')
    bind(spec['train_bundle']);bind(spec['existing_owner_release'])
    if socket.gethostname()!=SERVER_HOST or str(PHASE)!=SERVER_PHASE:
        raise ValueError('Exact authorized allocation project only')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()!=[SERVER_GPU] or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Exact route, CPU-only graph construction')
    python_path=Path(os.path.abspath(spec['runtime']['python']['path']))
    if not python_path.is_relative_to(PHASE) or sha(python_path)!=spec['runtime']['python']['sha256'] or Path(sys.executable).resolve()!=python_path.resolve() or os.environ.get('PYTHONPATH','')!=spec['runtime']['PYTHONPATH']:
        raise ValueError('Exact existing root-bound interpreter/environment for the frozen RNG recipe')
    output=inside(spec['output'],existing=False)
    if output.exists() or output.is_relative_to(HERE):raise ValueError('Fresh separate server graph-generation output required')
    return spec,bindings,output


def generate(spec,bindings,output,started):
    output.mkdir(parents=True,exist_ok=False)
    try:
        import numpy as np
        if str(np.__version__)!=bindings['frozen_qualified_providers']['numpy']:
            raise ValueError('Original bound NumPy provider required')
        with np.load(bind(spec['train_bundle']),allow_pickle=False) as archive:
            if set(archive.files)!={'x','edge_index','train_ids','train_y'}:raise ValueError('Exact original bundle header required')
            # npz is lazy: only the public edge array is read. No feature/role/label value is loaded.
            original=archive['edge_index'].copy()
        if original.dtype!=np.int64 or original.shape!=(2,88648) or fingerprint(original)!=bindings['original_edge_fingerprint']:
            raise ValueError('Exact complete original public graph required')
        edges,metadata=swap_graph(original.T.tolist(),19717,deadline=started+600)
        rewritten=np.ascontiguousarray(np.asarray(edges,dtype=np.int64).T)
        path=output/'AUXILIARY_EDGES_ONLY.npz'
        np.savez_compressed(path,edge_index=rewritten)
        if time.monotonic()-started>=600:raise TimeoutError('Frozen generation envelope exhausted')
        report=dict(schema='masked-context-degree-preserving-rewire-custody-v1',algorithm='double_edge_swap',
            **metadata,graph_fingerprint=fingerprint(rewritten),original_edge_fingerprint=fingerprint(original),
            auxiliary_bundle_sha256=sha(path),auxiliary_bundle_bytes=path.stat().st_size,
            generator_source_sha256=sha(__file__),train_bundle_sha256=spec['train_bundle']['sha256'],
            source_manifest_sha256=spec['source_manifest_sha256'],one_common_graph_for_all_optimizer_seeds=True,
            inclusive_seconds=time.monotonic()-started,numpy_version=str(np.__version__),
            python_version=sys.version,python_launcher_sha256=spec['runtime']['python']['sha256'],
            feature_values_loaded=False,role_id_or_label_values_loaded=False,TEST_access=False,
            model_constructed=False,automatic_retry=False,server_only=True)
        write(output/'REWIRE_CUSTODY.json',report)
        return report
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            inclusive_seconds=time.monotonic()-started,automatic_retry=False,no_alternate_seed_or_graph=True,
            model_constructed=False,TEST_access=False))
        raise


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('describe','generate'),default='describe')
    parser.add_argument('--release',type=Path);parser.add_argument('--release-sha256')
    args=parser.parse_args()
    if args.mode=='describe':print(json.dumps(dict(enabled=False,recipe=REWIRE,no_model_or_label_input=True)));return
    if args.release is None or args.release_sha256 is None:parser.error('Separate exact root release required')
    spec,bindings,output=admit(args.release,args.release_sha256)
    report=generate(spec,bindings,output,started)
    print(json.dumps(dict(complete=True,output=str(output),server_only=True,swaps=report['swap_count'])))


if __name__=='__main__':main()
