"""TRAIN-only count geometry; no features, prediction, training or heldout read."""
import argparse
import ast
import codecs
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    if array.size:
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def gate(release_path, release_sha):
    plan = json.loads((HERE / 'PLAN.json').read_text())
    repo = Path(plan['repository'])
    execution = PHASE / plan['execution_directory']
    require(Path.cwd() == repo and os.uname().nodename == 'peptide', 'Exact project host/root required')
    require(release_path.resolve() == execution / 'ROOT_RELEASE.json' and not release_path.is_symlink()
            and sha(release_path) == release_sha, 'Exact external release required')
    release = json.loads(release_path.read_text())
    require(release['status'] == 'APPROVED' and release['root_authorization_reference']
            and release['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
            and release['plan_sha256'] == sha(HERE / 'PLAN.json'), 'Exact source approval required')
    require(release['authorized_stage'] == 'TRAIN_support_census' and release['fits'] == 0
            and release['VALID_TEST_reads'] is False and release['automatic_retry'] is False
            and release['caps'] == plan['caps'], 'TRAIN-only non-scientific scope differs')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = HERE / row['path']
        require(path.resolve().is_relative_to(HERE) and not path.is_symlink()
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Source changed')
    for row in plan['source_pins']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and not path.is_symlink()
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Producer changed')
    runtime = json.loads((PHASE / plan['runtime_authority']).read_text())
    require(Path(sys.executable).resolve() == Path(runtime['interpreter_path']).resolve()
            and sha(Path(runtime['interpreter_path'])) == runtime['interpreter_sha256'], 'Interpreter differs')
    for name, version in runtime['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Runtime distribution differs')
    require(os.environ['CUDA_VISIBLE_DEVICES'] == plan['GPU_UUID'], 'Owned GPU route differs')
    require(os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '2', 'Thread profile differs')
    return plan, runtime, execution


def summarize(torch, queries, neighbors, full_keys, nodes):
    lrows, lnodes = neighbors.left
    rrows, rnodes = neighbors.right
    counts, selected = [], []
    from graph_ops import membership
    for side, (rows, candidates) in enumerate(((lrows, lnodes), (rrows, rnodes))):
        keys = queries[rows, 1-side] * nodes + candidates
        bits = membership(full_keys, keys).to(torch.int64)
        n = torch.bincount(rows, minlength=len(queries))
        k = torch.zeros_like(n).index_add(0, rows, bits)
        require(bool(((0 <= k) & (k <= n)).all()), 'Invalid observed count')
        counts.append(n)
        selected.append(k)
    n = torch.stack(counts, dim=1)
    k = torch.stack(selected, dim=1)
    r = torch.minimum(k, n-k)
    group = torch.where(r == 0, 0, torch.where(r == 1, 1, 2))
    joint = torch.bincount(group[:,0]*3 + group[:,1], minlength=9).reshape(3,3)
    sides = []
    for side in range(2):
        table, frequency = torch.unique(torch.stack((n[:,side],k[:,side]),dim=1), dim=0, return_counts=True)
        sides.append(dict(n_k_frequency=[list(map(int, pair)) + [int(freq)] for pair, freq in zip(table.cpu().tolist(),frequency.cpu().tolist())],
            empty=int((n[:,side] == 0).sum()), zero_count=int((k[:,side] == 0).sum()), full_count=int(((k[:,side] == n[:,side]) & (n[:,side] > 0)).sum()),
            categorical_or_complement=int((r[:,side] == 1).sum()), genuine_subset=int((r[:,side] > 1).sum()),
            total_slots=int(n[:,side].sum()), total_observed_bits=int(k[:,side].sum()), max_n=int(n[:,side].max()), max_k=int(k[:,side].max()), max_r=int(r[:,side].max()),
            n_times_r_sum=int((n[:,side]*r[:,side]).sum())))
    return dict(queries=len(queries), sides=sides, joint_group_counts=joint.cpu().tolist(),
        both_sides_nonconstant=int((r > 0).all(1).sum()), both_sides_genuine_subset=int((r > 1).all(1).sum()),
        either_side_genuine_subset=int((r > 1).any(1).sum()),
        support_count_digest=tensor_sha(torch.cat((n,k),dim=1)),
        query_digest=tensor_sha(queries))


def final_file_custody(plan, runtime, release_path, release_sha, data):
    """File/metadata checks only; no GPU/RNG probe, array load or new input role."""
    require(sha(release_path) == release_sha, 'Final release changed')
    release = json.loads(release_path.read_text())
    require(sha(HERE/'MANIFEST.json') == release['source_manifest_sha256']
            and sha(HERE/'PLAN.json') == release['plan_sha256'], 'Final source/plan changed')
    rows = []
    for scope,path,pin in (('source_manifest',HERE/'MANIFEST.json',release['source_manifest_sha256']),
                           ('root_release',release_path,release_sha),
                           ('interpreter',Path(runtime['interpreter_path']),runtime['interpreter_sha256'])):
        require(path.is_file() and sha(path) == pin, 'Final control/interpreter file changed')
        rows.append(dict(scope=scope,path=str(path),bytes=path.stat().st_size,sha256=pin))
    for base, pins, scope in ((HERE,json.loads((HERE/'MANIFEST.json').read_text())['files'],'source'),
                              (PHASE,plan['source_pins'],'producer_or_authority')):
        for pin in pins:
            path = base/pin['path']
            require(path.resolve().is_relative_to(base) and not path.is_symlink()
                    and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Final source pin changed')
            rows.append(dict(scope=scope,path=str(path),bytes=path.stat().st_size,sha256=pin['sha256']))
    require(Path(sys.executable).resolve() == Path(runtime['interpreter_path']).resolve()
            and sha(Path(runtime['interpreter_path'])) == runtime['interpreter_sha256'], 'Final interpreter changed')
    for name, version in runtime['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Final distribution changed')
    runtime_files = runtime['runtime_source_pins'] + runtime['runtime_binary_files'] + [runtime['negative_sampler']]
    for pin in runtime_files:
        path = Path(pin['path'])
        require(path.is_file() and not path.is_symlink() and sha(path) == pin['sha256']
                and ('bytes' not in pin or path.stat().st_size == pin['bytes']), 'Final runtime file changed')
        rows.append(dict(scope='runtime_file',path=str(path),bytes=path.stat().st_size,sha256=pin['sha256']))
    train_files = []
    if data is not None:
        authority = json.loads((PHASE/plan['data_authority']).read_text())
        require(data == Path(authority['dataset_root']).resolve() and data.is_relative_to(PHASE), 'Final TRAIN root differs')
        # Same two files already authorized in the original numerical body.
        # No feature, VALID, TEST or checkpoint path is enumerated or opened.
        for name in ('split/time/train.pt','raw/edge.csv.gz'):
            path = data/name; pin = authority['files'][name]
            require(path.resolve().is_relative_to(data) and not path.is_symlink()
                    and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Final TRAIN file changed')
            train_files.append(dict(relative_path=name,path=str(path),bytes=pin['bytes'],sha256=pin['sha256']))
    return dict(schema='TRAIN-census-final-file-custody-v2',status='MATCH',
        source_manifest_sha256=release['source_manifest_sha256'],plan_sha256=release['plan_sha256'],
        release_sha256=release_sha,interpreter_sha256=runtime['interpreter_sha256'],
        distribution_versions=runtime['distribution_versions'],files=rows,TRAIN_files=train_files,
        GPU_operations=False,array_loads=False,additional_scientific_input_roles=False,
        actual_profile_fields_beyond_original_guards_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    plan, runtime, execution = gate(args.release, args.release_sha256)
    output = execution / 'run01'
    require(not output.exists(), 'Fresh census required; no restart')
    output.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    try:
        import numpy as np
        import pandas as pd
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.__version__ == '2.7.1' and torch.cuda.device_count() == 1
                and not torch.are_deterministic_algorithms_enabled(), 'Existing ordinary numerical profile required')
        props = torch.cuda.get_device_properties(0)
        require(props.name == 'NVIDIA A100 80GB PCIe', 'Expected A100 profile required')
        require(props.total_memory == runtime['gpu_total_memory_bytes'], 'Authority GPU capacity differs')
        for pin in runtime['runtime_source_pins'] + runtime['runtime_binary_files'] + [runtime['negative_sampler']]:
            require(sha(Path(pin['path'])) == pin['sha256'], 'Numerical runtime source/binary changed')
        authority = json.loads((PHASE / plan['data_authority']).read_text())
        data = Path(authority['dataset_root']).resolve()
        require(data.is_relative_to(PHASE), 'TRAIN dataset leaves project')
        for name in ('split/time/train.pt','raw/edge.csv.gz'):
            path = data / name
            pin = authority['files'][name]
            require(path.resolve().is_relative_to(data) and not path.is_symlink()
                    and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'TRAIN file changed')
        allowed = [np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype(np.int64)),codecs.encode]
        with torch.serialization.safe_globals(allowed):
            stored = torch.load(data / 'split/time/train.pt', map_location='cpu', weights_only=True)
        require(type(stored) is dict and set(stored) == set(authority['expected_arrays']['train']), 'TRAIN schema differs')
        for name, spec in authority['expected_arrays']['train'].items():
            array = stored[name]
            require(type(array) is np.ndarray and list(array.shape) == spec['shape'] and str(array.dtype) == spec['dtype']
                    and tensor_sha(torch.from_numpy(array)) == spec['sha256'], 'TRAIN arrays differ')
        pairs = torch.from_numpy(stored['edge']).to('cuda:0')
        nodes = plan['nodes']
        raw = pd.read_csv(data/'raw/edge.csv.gz', compression='gzip', header=None).values.T.astype(np.int64)
        expanded = np.repeat(raw,2,axis=1)
        expanded[0,1::2] = expanded[1,0::2]
        expanded[1,1::2] = expanded[0,0::2]
        raw_edges = torch.from_numpy(expanded).to('cuda:0')
        require(tensor_sha(raw_edges) == authority['train_raw_tensor_digests']['ordered_raw_graph'], 'Raw TRAIN graph differs')
        prototype = PHASE / plan['graph_source_directory']
        sys.path.insert(0,str(prototype))
        from graph_ops import Graph, enumerate_neighbors
        require(Path(sys.modules['graph_ops'].__file__).resolve() == prototype/'graph_ops.py', 'Graph source shadowed')
        full = Graph.from_pairs(pairs,nodes)
        full_keys = full.row*nodes + full.col
        from torch_geometric.utils import negative_sampling
        require(Path(inspect.getsourcefile(negative_sampling)).resolve() == Path(runtime['negative_sampler']['path']).resolve()
                and hashlib.sha256(inspect.getsource(negative_sampling).encode()).hexdigest() == runtime['negative_sampler']['function_sha256'], 'Sampler changed')
        iterator_path = PHASE / plan['iterator_source']
        definitions = [node for node in ast.parse(iterator_path.read_text()).body if isinstance(node,ast.ClassDef) and node.name == 'PermIterator']
        require(len(definitions) == 1, 'Original iterator source absent')
        scope = dict(torch=torch)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(iterator_path),'exec'),scope)
        for seed in plan['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            negative = negative_sampling(raw_edges,nodes).T
            require(negative.dtype == torch.long and negative.shape[1] == 2 and len(negative) >= len(pairs), 'Native negative geometry differs')
            iterator = scope['PermIterator'](pairs.device,len(pairs),65536)
            require(len(iterator) == 17 and len(iterator.idx) == len(pairs), 'Original full batch/tail policy differs')
            stream = dict(seed=seed,negative_draw_sha256=tensor_sha(negative),permutation_sha256=tensor_sha(iterator.idx),
                dropped_tail_sha256=tensor_sha(iterator.idx[17*65536:]),rows=[])
            for batch, record_ids in enumerate(iterator,1):
                graph = Graph.mask_train_batch(pairs,record_ids,nodes)
                populations = {}
                for name,query in (('positive',pairs[record_ids]),('negative',negative[record_ids])):
                    neighbors = enumerate_neighbors(graph,query)
                    populations[name] = summarize(torch,query,neighbors,full_keys,nodes)
                    del neighbors
                stream['rows'].append(dict(batch=batch,record_ids_sha256=tensor_sha(record_ids),populations=populations))
                torch.cuda.synchronize(0)
                require(torch.cuda.max_memory_allocated() <= plan['caps']['cuda_allocated_bytes']
                        and torch.cuda.max_memory_reserved() <= plan['caps']['cuda_reserved_bytes'], 'Observed CUDA cap exceeded')
                write(output/'PROGRESS.json',dict(completed_streams=len(rows),seed=seed,batches=batch,elapsed_seconds=time.monotonic()-started))
                del graph,populations
            require(len(stream['rows']) == 17, 'Incomplete TRAIN mask stream')
            rows.append(stream)
            write(output/('seed%d_COUNTS.json'%seed),stream)
        write(output/'CENSUS.json',dict(status='COMPLETE_TRAIN_CENSUS_ONLY',UTC=datetime.now(timezone.utc).isoformat(),
            source_manifest_sha256=sha(HERE/'MANIFEST.json'),release_sha256=args.release_sha256,
            streams=len(rows),full_batches_per_stream=17,positive_records_per_stream=1114112,negative_records_per_stream=1114112,
            wall_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            data_files_opened=['split/time/train.pt','raw/edge.csv.gz'],features_read=False,VALID_TEST_read=False,learned_models=False,optimizer_updates=0,
            interpretation='TRAIN geometry and prospective cost evidence only. Seeds/masks are not independent benchmark graphs. No prediction, novelty or quality claim.'))
    except BaseException as error:
        write(output/'FAILURE.json',dict(status='FAILED',type=type(error).__name__,condition=str(error),completed_streams=len(rows),automatic_retry=False))
        raise
    finally:
        try:
            custody = final_file_custody(plan,runtime,args.release,args.release_sha256,locals().get('data'))
        except BaseException as error:
            custody = dict(schema='TRAIN-census-final-file-custody-v2',status='FAILED',
                type=type(error).__name__,condition=str(error),GPU_operations=False,
                array_loads=False,additional_scientific_input_roles=False)
        write(output/'FILE_CUSTODY.json',custody)
        write(output/'FINAL_CUSTODY.json',dict(completed=(output/'CENSUS.json').exists() and custody['status']=='MATCH',
            file_custody_sha256=sha(output/'FILE_CUSTODY.json'),
            files=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(output.iterdir()) if f.is_file() and not f.name.endswith('.tmp')]))


if __name__ == '__main__':
    main()
