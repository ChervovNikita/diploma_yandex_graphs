"""Disabled, selective official DDI TRAIN/VALID artifact acquisition.

Root supplies an exact external release. Only five literal archive members
are decoded; the retained OGB sources are evidence and are never imported.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
MEMBERS = (
    'ddi/split/target/train.pt',
    'ddi/split/target/valid.pt',
    'ddi/raw/num-node-list.csv.gz',
    'ddi/raw/num-edge-list.csv.gz',
    'ddi/raw/edge.csv.gz',
)
SCHEMA = 'hlgnn-ddi-train-valid-v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def budget(plan, started):
    require(time.monotonic() - started <= plan['wall_seconds_cap'], 'Worker wall cap exceeded')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= plan['host_RSS_bytes_cap'],
            'Observed Linux worker peak RSS cap exceeded')


def source_controls(plan, release_path, release_sha):
    require(sha(release_path) == release_sha, 'Root release changed')
    release = json.loads(release_path.read_text())
    require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256']
            and sha(HERE / 'PLAN.json') == release['plan_sha256'], 'Source controls changed')
    for pin in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = HERE / pin['path']
        require(path.resolve().is_relative_to(HERE) and not path.is_symlink()
                and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Source payload changed')


def gate(args):
    plan = json.loads((HERE / 'PLAN.json').read_text())
    execution = PHASE / plan['execution_directory']
    release_path = args.root_release.resolve()
    require(release_path == execution / 'ROOT_RELEASE.json' and not args.root_release.is_symlink()
            and sha(release_path) == args.root_release_sha256, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    require(release['status'] == 'APPROVED' and release['root_authorization_reference']
            and release['stage'] == 'DDI_HLGNN_TRAIN_VALID_artifact', 'Root approval required')
    require(plan['status'] == 'DISABLED_SOURCE_PREPARATION' and plan['schema'] == SCHEMA
            and tuple(plan['selected_members']) == MEMBERS, 'Frozen artifact contract differs')
    source_controls(plan, release_path, args.root_release_sha256)
    output = execution / 'run01'
    require(not output.exists(), 'Fresh run01 required; no restart or retry')
    output.mkdir(parents=True)
    return plan, release_path, output


def input_path(relative):
    path = PHASE / relative
    require(path.resolve().is_relative_to(PHASE) and not path.is_symlink() and path.is_file(), 'Invalid input path')
    return path


def unzip_selected(archive, central_path, output, plan, started, evidence):
    require(central_path.stat().st_size <= plan['central_metadata_bytes_cap'], 'Central metadata cap exceeded')
    recorded = json.loads(central_path.read_text())
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        require(len(infos) <= plan['central_entries_cap'], 'Too many central entries')
        observed = [dict(name=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size, CRC=x.CRC) for x in infos]
        require(recorded == observed, 'Acquisition central-directory metadata differs from frozen archive')
        names = [x.filename for x in infos]
        require(len(names) == len(set(names)), 'Duplicate/ambiguous archive names')
        target_files = {n for n in names if n.startswith('ddi/split/target/') and not n.endswith('/')}
        require(target_files <= {'ddi/split/target/train.pt', 'ddi/split/target/valid.pt', 'ddi/split/target/test.pt'},
                'Ambiguous or combined target split storage; no substitute accessor permitted')
        require(not any(Path(n).name == 'split_dict.pt' for n in names), 'Combined split member forbidden')
        require(not any(n in names for n in ('ddi/raw/node-feat.csv.gz', 'ddi/raw/edge-feat.csv.gz',
                                             'ddi/raw/edge_weight.csv.gz', 'ddi/raw/edge-weight.csv.gz')),
                'Unexpected native attributes/weight source')
        selected = output / 'selected'
        selected.mkdir()
        total = 0
        for name in MEMBERS:
            matches = [x for x in infos if x.filename == name]
            require(len(matches) == 1 and not matches[0].is_dir() and not matches[0].flag_bits & 1,
                    'Literal selected member missing/ambiguous/encrypted: ' + name)
            info = matches[0]
            cap = plan['member_bytes_caps'][name]
            total += info.file_size
            require(info.file_size <= cap and total <= plan['selected_member_bytes_cap'], 'Selected bytes cap exceeded')
            path = selected / Path(name).name
            h, count = hashlib.sha256(), 0
            with z.open(info) as stream, path.open('xb') as target:
                while True:
                    part = stream.read(1048576)
                    if not part:
                        break
                    count += len(part)
                    require(count <= cap, 'Member streaming cap exceeded')
                    budget(plan, started)
                    target.write(part)
                    h.update(part)
                target.flush()
                os.fsync(target.fileno())
            require(count == info.file_size, 'Selected member length differs')
            evidence['decoded_members'].append(dict(name=name, bytes=count, sha256=h.hexdigest()))
            if name in plan['previous_selected_member_sha256']:
                require(h.hexdigest() == plan['previous_selected_member_sha256'][name], 'Prior selected member differs')
        return selected


def gunzip_bounded(path, cap, plan, started):
    parts, total = [], 0
    with gzip.open(path, 'rb') as stream:
        while True:
            part = stream.read(1048576)
            if not part:
                break
            total += len(part)
            require(total <= cap, 'Decompressed CSV cap exceeded')
            budget(plan, started)
            parts.append(part)
    return b''.join(parts)


def single_count(raw):
    rows = list(csv.reader(io.StringIO(raw.decode('ascii'))))
    require(len(rows) == 1 and len(rows[0]) == 1 and rows[0][0].strip().isdigit(), 'Single-graph count metadata required')
    return int(rows[0][0])


def edge_tensor(value, name, torch, np, nodes, max_rows):
    # Same NumPy-to-Torch conversion as OGB, with no dtype or row-order cast.
    original_type = type(value).__name__
    if type(value) is np.ndarray:
        require(value.dtype == np.int64, 'Native int64 required without coercion: ' + name)
        value = torch.from_numpy(value)
    require(type(value) is torch.Tensor and value.dtype == torch.int64 and value.device.type == 'cpu'
            and value.ndim == 2 and value.shape[1] == 2 and 0 < len(value) <= max_rows,
            'Native CPU int64 [rows,2] required: ' + name)
    require(int(value.min()) >= 0 and int(value.max()) < nodes, 'Native node ID range differs: ' + name)
    return value, original_type


def binding(value):
    raw = value.contiguous().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), device='cpu',
                tensor_bytes=len(raw), tensor_bytes_sha256=hashlib.sha256(raw).hexdigest(), row_order_preserved=True)


def canonical_keys(value, nodes, torch):
    return torch.sort(torch.minimum(value[:, 0], value[:, 1]) * nodes
                      + torch.maximum(value[:, 0], value[:, 1])).values


def overlap_count(sorted_keys, query_keys, torch):
    index = torch.searchsorted(sorted_keys, query_keys)
    return int(((index < len(sorted_keys)) & (sorted_keys[index.clamp(max=len(sorted_keys) - 1)] == query_keys)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', type=Path, required=True)
    parser.add_argument('--root-release-sha256', required=True)
    args = parser.parse_args()
    plan, release_path, output = gate(args)
    started = time.monotonic()
    evidence = dict(UTC_started=datetime.now(timezone.utc).isoformat(), decoded_members=[], phase='inputs',
                    TEST_payload_decoded=False, models_initialized=False, training_or_scoring=False, automatic_retry=False)
    try:
        require(str(Path(sys.executable).resolve()) == str(Path(plan['interpreter']).resolve())
                and sha(Path(sys.executable)) == plan['interpreter_sha256'], 'Existing RAPIDS interpreter differs')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only route requires hidden GPUs')
        archive = input_path(plan['archive_relative_path'])
        central = input_path(plan['central_metadata_relative_path'])
        train_only = input_path(plan['TRAIN_ONLY_relative_path'])
        require(archive.stat().st_size == plan['archive_bytes'] and sha(archive) == plan['archive_sha256'], 'Official archive differs')
        require(sha(train_only) == plan['TRAIN_ONLY_sha256'], 'Existing TRAIN_ONLY file differs')
        central_sha = sha(central)
        metadata = json.loads((HERE / 'DDI_OFFICIAL_METADATA.json').read_text())['values']
        require(metadata == plan['official_metadata'], 'Official metadata differs')
        budget(plan, started)
        evidence['phase'] = 'five_literal_members'
        selected = unzip_selected(archive, central, output, plan, started, evidence)
        evidence['phase'] = 'native_split_decode'
        import numpy as np
        import torch
        require(np.__version__ == plan['numpy_version'] and torch.__version__.split('+')[0] == plan['torch_version'],
                'Existing used numerical package differs')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        # Legacy NumPy-containing official split payloads require this native format.
        # Only the exact hash-bound official TRAIN and VALID member bytes reach it.
        train_raw = torch.load(selected / 'train.pt', map_location='cpu', weights_only=False)
        valid_raw = torch.load(selected / 'valid.pt', map_location='cpu', weights_only=False)
        require(type(train_raw) is dict and set(train_raw) == {'edge'}, 'Native TRAIN must have exactly edge, without weight')
        require(type(valid_raw) is dict and set(valid_raw) == {'edge', 'edge_neg'}, 'Native fixed VALID keys differ')
        node_raw = gunzip_bounded(selected / 'num-node-list.csv.gz', 1048576, plan, started)
        count_raw = gunzip_bounded(selected / 'num-edge-list.csv.gz', 1048576, plan, started)
        nodes, records = single_count(node_raw), single_count(count_raw)
        evidence['single_graph'] = dict(num_nodes=nodes, raw_edge_rows=records)
        require(nodes == plan['expected_nodes'] and records == plan['expected_TRAIN_records'], 'Native single-graph size differs from TRAIN')
        train, train_type = edge_tensor(train_raw['edge'], 'train.edge', torch, np, nodes, records)
        valid, valid_type = edge_tensor(valid_raw['edge'], 'valid.edge', torch, np, nodes, plan['VALID_positive_rows_cap'])
        negatives, neg_type = edge_tensor(valid_raw['edge_neg'], 'valid.edge_neg', torch, np, nodes, plan['VALID_negative_rows_cap'])
        require(len(train) == records and len(negatives) >= 100, 'Native TRAIN count or global VALID pool size differs')
        evidence['tensors'] = {name: binding(value) for name, value in
                               (('train.edge', train), ('valid.edge', valid), ('valid.edge_neg', negatives))}
        evidence['native_field_types'] = dict(train_edge=train_type, valid_edge=valid_type, valid_edge_neg=neg_type)
        require(evidence['tensors']['train.edge']['tensor_bytes_sha256'] == plan['TRAIN_tensor_bytes_sha256'],
                'Original TRAIN row order differs from acquired TRAIN_ONLY binding')
        budget(plan, started)
        evidence['phase'] = 'native_raw_graph'
        edge_csv = gunzip_bounded(selected / 'edge.csv.gz', plan['edge_csv_decompressed_bytes_cap'], plan, started)
        evidence['raw_edge_CSV'] = dict(bytes=len(edge_csv), sha256=hashlib.sha256(edge_csv).hexdigest())
        # In the admitted two-column integer domain, CSV row parsing yields the
        # same values/order as pinned pd.read_csv(...).values.T.astype(np.int64).
        raw_edge = np.empty((2, records), dtype=np.int64)
        rows = 0
        for row in csv.reader(io.StringIO(edge_csv.decode('ascii'))):
            require(rows < records and len(row) == 2 and all(x.strip().isdigit() for x in row), 'Raw edge CSV domain/row count differs')
            endpoints = [int(x) for x in row]
            require(all(0 <= x < nodes for x in endpoints), 'Raw graph node IDs differ')
            raw_edge[:, rows] = endpoints
            rows += 1
            if rows % 65536 == 0:
                budget(plan, started)
        require(rows == records, 'Raw edge rows differ from native count metadata')
        del edge_csv
        # Exact pinned OGB inverse-edge operations: interleave, never concatenate
        # all forwards and then all reversals, sort, normalize, or coalesce here.
        duplicated_edge = np.repeat(raw_edge, 2, axis=1)
        duplicated_edge[0, 1::2] = duplicated_edge[1, 0::2]
        duplicated_edge[1, 1::2] = duplicated_edge[0, 0::2]
        graph = torch.from_numpy(duplicated_edge)
        raw_pairs = torch.from_numpy(raw_edge.T)
        require(torch.equal(graph[:, 0::2], raw_pairs.T) and torch.equal(graph[:, 1::2], raw_pairs.flip(1).T), 'Native inverse ordering differs')
        evidence['tensors']['graph_edge_index'] = binding(graph)
        evidence['native_raw_order'] = dict(raw_pairs=binding(raw_pairs), inverse_edges_interleaved=True,
                                          no_sort_or_coalesce=True, official_node_mapping='Identity of native zero-based integer IDs; no remapping.')
        evidence['phase'] = 'membership_and_candidates'
        train_keys = canonical_keys(train, nodes, torch)
        raw_keys = canonical_keys(raw_pairs, nodes, torch)
        valid_keys = canonical_keys(valid, nodes, torch)
        negative_keys = canonical_keys(negatives, nodes, torch)
        train_unique = int(torch.unique_consecutive(train_keys).numel())
        raw_unique = int(torch.unique_consecutive(raw_keys).numel())
        evidence['graph_qualification'] = dict(TRAIN_rows=records, native_raw_rows=rows, directed_graph_entries=graph.shape[1],
             TRAIN_unique_undirected=train_unique, raw_unique_undirected=raw_unique,
             TRAIN_self_loops=int((train[:, 0] == train[:, 1]).sum()), raw_self_loops=int((raw_pairs[:, 0] == raw_pairs[:, 1]).sum()),
             canonical_membership_equal=bool(torch.equal(train_keys, raw_keys)), graph_edge_weight_present=False,
             TRAIN_weight_present=False, graph_edge_weight=None, actual_HLGNN_loss_branch='AUC')
        q = evidence['graph_qualification']
        require(q['TRAIN_unique_undirected'] == q['raw_unique_undirected'] == records and q['TRAIN_self_loops'] == q['raw_self_loops'] == 0
                and q['canonical_membership_equal'] and graph.shape[1] == 2 * records, 'Native raw graph does not equal unique loop-free TRAIN; no substitution')
        candidates = dict(positive_rows=len(valid), negative_rows=len(negatives),
                         positive_unique_undirected=int(torch.unique_consecutive(valid_keys).numel()),
                         negative_unique_undirected=int(torch.unique_consecutive(negative_keys).numel()),
                         TRAIN_VALID_positive_overlap=overlap_count(train_keys, valid_keys, torch),
                         TRAIN_VALID_negative_overlap=overlap_count(train_keys, negative_keys, torch),
                         VALID_positive_negative_overlap=overlap_count(valid_keys, negative_keys, torch),
                         positive_self_loops=int((valid[:, 0] == valid[:, 1]).sum()),
                         negative_self_loops=int((negatives[:, 0] == negatives[:, 1]).sum()),
                         protocol='Fixed native global negative pool shared across positives for OGB Hits@20/50/100',
                         row_order_and_dtype_preserved=True, resampled=False, per_positive_regrouping=False,
                         graph_insertion=False, TEST_disjointness_numerically_checked=False)
        evidence['VALID_qualification'] = candidates
        require(candidates['positive_unique_undirected'] == len(valid) and candidates['TRAIN_VALID_positive_overlap'] == 0
                and candidates['TRAIN_VALID_negative_overlap'] == 0 and candidates['VALID_positive_negative_overlap'] == 0
                and candidates['positive_self_loops'] == candidates['negative_self_loops'] == 0, 'Native TRAIN/VALID disjointness or candidate domain differs')
        budget(plan, started)
        evidence['phase'] = 'strict_artifact'
        payload = dict(schema=SCHEMA, num_nodes=nodes, graph_edge_index=graph, graph_edge_weight=None,
                       train={'edge': train}, valid={'edge': valid, 'edge_neg': negatives})
        temporary = output / 'HLGNN_DDI_TRAIN_VALID.pt.tmp'
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(temporary.stat().st_size <= plan['artifact_bytes_cap'], 'Artifact bytes cap exceeded')
        roundtrip = torch.load(temporary, map_location='cpu', weights_only=True)
        require(type(roundtrip) is dict and set(roundtrip) == {'schema', 'num_nodes', 'graph_edge_index', 'graph_edge_weight', 'train', 'valid'}
                and roundtrip['schema'] == SCHEMA and type(roundtrip['num_nodes']) is int and roundtrip['num_nodes'] == nodes
                and roundtrip['graph_edge_weight'] is None and set(roundtrip['train']) == {'edge'}
                and set(roundtrip['valid']) == {'edge', 'edge_neg'}, 'Strict serialization keys differ')
        for saved, original in ((roundtrip['graph_edge_index'], graph), (roundtrip['train']['edge'], train),
                                (roundtrip['valid']['edge'], valid), (roundtrip['valid']['edge_neg'], negatives)):
            require(type(saved) is torch.Tensor and saved.device.type == 'cpu' and saved.dtype == original.dtype
                    and torch.equal(saved, original), 'Artifact changed native graph/candidate dtype or order')
        source_controls(plan, release_path, args.root_release_sha256)
        require(sha(archive) == plan['archive_sha256'] and sha(train_only) == plan['TRAIN_ONLY_sha256']
                and sha(central) == central_sha, 'Final official input custody differs')
        for member in evidence['decoded_members']:
            require(sha(selected / Path(member['name']).name) == member['sha256'], 'Final selected member custody differs')
        budget(plan, started)
        artifact = output / 'HLGNN_DDI_TRAIN_VALID.pt'
        require(not artifact.exists(), 'Artifact overwrite forbidden')
        temporary.rename(artifact)
        evidence.update(status='COMPLETE_OFFICIAL_HLGNN_DDI_TRAIN_VALID_ARTIFACT', phase='complete',
                        schema=SCHEMA, artifact_file=artifact.name, artifact_sha256=sha(artifact), artifact_bytes=artifact.stat().st_size,
                        source_manifest_sha256=sha(HERE / 'MANIFEST.json'), plan_sha256=sha(HERE / 'PLAN.json'),
                        root_release_sha256=args.root_release_sha256, archive_sha256=plan['archive_sha256'],
                        acquisition_central_metadata_sha256=central_sha, TRAIN_ONLY_sha256=plan['TRAIN_ONLY_sha256'],
                        source_pins_sha256=sha(HERE / 'SOURCE_PINS.json'), native_graph_and_fixed_VALID_contract_qualified=True,
                        training_runtime_qualified=False, full_training_budget_qualified=False,
                        GPU_compute=False, torch_threads=torch.get_num_threads(),
                        host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        wall_seconds=time.monotonic() - started, UTC_completed=datetime.now(timezone.utc).isoformat())
        budget(plan, started)
        write(output / 'QUALIFICATION.json', evidence)
    except BaseException as error:
        evidence.update(status='INCOMPLETE_NONQUALIFYING', error_type=type(error).__name__, error=str(error),
                        wall_seconds=time.monotonic() - started, UTC_completed=datetime.now(timezone.utc).isoformat())
        write(output / 'QUALIFICATION_FAILURE.json', evidence)
        raise


if __name__ == '__main__':
    main()
