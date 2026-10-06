#!/usr/bin/env python3
"""Exhaustive TRAIN-only nonedge witness census. Stdlib, CPU, no learned model."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import struct
import time

ROOT = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
PURPOSE = 'TRAIN_only_complete_nonedge_internal_CN_witness_census'
AVAILABLE_SHA = '1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc'
TRAIN_SHA = '1e97ad3a73ecfeb69489aa6d92d02dc44c2925b4056a8e4cd4b548f1c0ebaa27'
GATE_RELATIVE = 'shared_NCN_structural_TRAIN_only_execution_root_20261006_v2/qualification/QUALIFICATION.json'
GATE_SHA = '4e46256b3e2a2869c1590fa543b4c13571ecadc8cd558952f4073be7b4aa3e07'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def enumerate_candidates(neighbors, edges, *, pair_budget, unique_budget, deadline=None):
    """q=(u,v) has an internal CN edge (w,z) iff u,v belong to C(w,z).

    Enumerate every supported unordered edge, then every unordered pair in its
    triangle co-neighbor group. Reject known TRAIN pairs/self; count each unique
    candidate's exact number of internal edges. Caps abort, never truncate.
    """
    facts = set(edges); candidates = {}; known_with_witness = set()
    work = {'support_edges_processed': 0, 'triangle_incidence': 0, 'triangle_groups_ge2': 0,
            'co_neighbor_pair_occurrences': 0, 'known_TRAIN_pair_occurrences_rejected': 0,
            'self_pair_occurrences_rejected': 0, 'nonedge_witness_occurrences': 0,
            'CN_intersection_probe_upper_bound': 0}
    group_hist = Counter()
    for w, z in edges:
        if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Fixed CPU census soft bound exceeded')
        common = sorted(neighbors[w] & neighbors[z]); work['support_edges_processed'] += 1
        work['CN_intersection_probe_upper_bound'] += min(len(neighbors[w]), len(neighbors[z]))
        work['triangle_incidence'] += len(common); group_hist[len(common)] += 1
        if len(common) < 2: continue
        work['triangle_groups_ge2'] += 1
        for u, v in combinations(common, 2):
            work['co_neighbor_pair_occurrences'] += 1
            if work['co_neighbor_pair_occurrences'] > pair_budget: raise RuntimeError('Pair work cap exceeded; census incomplete, no sampling fallback')
            if u == v:
                work['self_pair_occurrences_rejected'] += 1; continue
            pair = (u, v)  # common is sorted; canonical unordered pair
            if pair in facts:
                work['known_TRAIN_pair_occurrences_rejected'] += 1; known_with_witness.add(pair); continue
            work['nonedge_witness_occurrences'] += 1
            if pair not in candidates and len(candidates) >= unique_budget:
                raise RuntimeError('Unique candidate storage cap exceeded; census incomplete, no truncation')
            candidates[pair] = candidates.get(pair, 0)+1
    if work['triangle_incidence'] % 3: raise ValueError('Undirected triangle incidence is inconsistent')
    return candidates, known_with_witness, work, group_hist


def summarize(neighbors, edges, candidates, known, work, group_hist, native_draws=7740):
    n = len(neighbors); universe = n*(n-1)//2-len(edges)
    digest = hashlib.sha256(); lcl_hist = Counter(); endpoints = Counter()
    cn_hist = Counter(); examples = []
    for (u, v), count in sorted(candidates.items()):
        common = neighbors[u] & neighbors[v]
        # Direct query-centric verification of every emitted candidate's LCL;
        # completeness follows the inverse edge-group enumeration above.
        direct = sum(z > w and z in common for w in common for z in neighbors[w])
        if v in neighbors[u] or u == v or direct != count or count < 1:
            raise ValueError('Candidate failed independent direct internal-edge verification')
        digest.update(struct.pack('<IIQ', u, v, count)); lcl_hist[count] += 1
        endpoints[u] += 1; endpoints[v] += 1; cn_hist[len(common)] += 1
        if len(examples) < 16: examples.append({'u': u, 'v': v, 'CN': len(common), 'LCL': count})
    # A labeled hypothetical uniform-directed draw calculation, not a claim
    # about PyG's exact finite sampling law or any candidate's true label.
    directed_universe, successes = 2*universe, 2*len(candidates)
    k = min(native_draws, directed_universe)
    p_none = 0. if k > directed_universe-successes else math.exp(sum(math.log1p(-successes/(directed_universe-i)) for i in range(k))) if directed_universe else None
    return {'scope': PURPOSE, 'complete': True, 'nodes': n, 'undirected_TRAIN_edges': len(edges),
        'triangle_count': work['triangle_incidence']//3, 'full_TRAIN_nonedge_universe': universe,
        'unique_TRAIN_nonedge_candidates': len(candidates), 'candidate_fraction_of_all_TRAIN_nonedges': len(candidates)/universe if universe else None,
        'known_TRAIN_pairs_with_internal_witness_rejected': len(known), 'candidate_endpoints': len(endpoints),
        'candidate_LCL_histogram': dict(sorted(lcl_hist.items())), 'candidate_CN_histogram': dict(sorted(cn_hist.items())),
        'edge_co_neighbor_group_size_histogram': dict(sorted(group_hist.items())),
        'top_candidate_endpoint_counts': [{'node': node, 'candidate_incidence': count} for node, count in endpoints.most_common(10)],
        'candidate_digest': {'sha256': digest.hexdigest(), 'serialization': 'sorted canonical u,v,LCL; packed little-endian uint32,uint32,uint64'},
        'candidate_preview_first16': examples, 'full_candidate_list_written': False,
        'hypothetical_uniform_directed_sampling': {'draws': k, 'expected_witness_rows': k*len(candidates)/universe if universe else None,
            'probability_no_witness': p_none, 'assumption': 'uniform directed TRAIN-absent nonself pairs without replacement; not a verified PyG law'},
        'enumeration_work': work, 'direct_candidate_checks': len(candidates), 'graph_scope': 'full TRAIN graph; upper bound for union-masked support opportunity',
        'labels': 'TRAIN-absent unlabeled candidates; may contain future/held positives; no truth guarantee',
        'VALID_TEST_values_access': False, 'features_checkpoints_models_access': False, 'fits': 0, 'updates': 0,
        'interpretation': 'Eligibility census only; not method failure, representative-negative certification, ranking result, exact-statistic collision diagnosis, or sharing causality.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--job', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); job = json.loads(args.job.read_text())
    if platform.system() != 'Linux' or socket.gethostname() != 'peptide' or Path.cwd().resolve() != REPO or ROOT.parent != PHASE:
        raise ValueError('Exact authorized 18.77 repository only')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or job.get('purpose') != PURPOSE or job.get('execution_enabled') is not True or job.get('source_review_approved') is not True:
        raise ValueError('Root-adopted CPU-only census required')
    if job.get('VALID_TEST_values_access') is not False or job.get('retry') is not False or job.get('fits') != 0:
        raise ValueError('No held inputs, fits or retry')
    if sha(__file__) != job['program_sha256'] or sha(ROOT/'SOURCE_MANIFEST.json') != job['source_manifest_sha256']:
        raise ValueError('Reviewed source changed')
    for row in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['files']:
        f=(ROOT/row['path']).resolve(strict=True)
        if not f.is_relative_to(ROOT) or sha(f)!=row['sha256'] or f.stat().st_size!=row['bytes']: raise ValueError('Sealed file changed')
    if sha(PHASE/GATE_RELATIVE) != GATE_SHA: raise ValueError('Actual six-control qualifier binding changed')
    gate = json.loads((PHASE/GATE_RELATIVE).read_text())
    if gate.get('passed') is not True or gate.get('TEST_access') is not False or gate.get('VALID_values_access') is not False: raise ValueError('Source gate scope differs')
    manifest_path = PHASE/'citeseer_heart_official_acquisition_server_20261005_v1/AVAILABLE_MANIFEST.json'
    if sha(manifest_path) != AVAILABLE_SHA: raise ValueError('Exact acquisition manifest changed')
    manifest = json.loads(manifest_path.read_text()); record = manifest['files']['train_pos.txt']
    if record['relative_path'] != 'available/citeseer/train_pos.txt' or record['sha256'] != TRAIN_SHA: raise ValueError('TRAIN role changed')
    train_path=(manifest_path.parent/record['relative_path']).resolve(strict=True)
    if not train_path.is_relative_to(PHASE) or sha(train_path)!=TRAIN_SHA or train_path.stat().st_size!=record['bytes']: raise ValueError('Exact TRAIN bytes changed')
    if job.get('nodes')!=3327 or job.get('pair_occurrence_cap')!=10000000 or job.get('unique_candidate_cap')!=1000000 or job.get('worker_soft_seconds')!=40 or job.get('result_bytes_cap')!=65536:
        raise ValueError('Prospective CPU/input/work/output bounds changed')
    output=args.output.resolve()
    if output.exists() or not output.parent.is_dir() or not output.is_relative_to(PHASE): raise ValueError('Fresh authorized output only')
    output.mkdir(); write(output/'START.json', {'PID': os.getpid(), 'purpose': PURPOSE, 'source_manifest_sha256': job['source_manifest_sha256'], 'job_sha256': sha(args.job), 'CPU_only': True})
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3)); resource.setrlimit(resource.RLIMIT_CPU, (45, 50))
        facts=set(); raw=selfs=0
        for line in train_path.read_text().splitlines():
            fields=line.split('\t')
            if len(fields)!=2: raise ValueError('Native tab-separated TRAIN format differs')
            u,v=map(int,fields);raw+=1
            if not 0<=u<3327 or not 0<=v<3327: raise ValueError('TRAIN endpoint outside node universe')
            if u==v:selfs+=1;continue
            pair=tuple(sorted((u,v)))
            if pair in facts:raise ValueError('Duplicate TRAIN fact')
            facts.add(pair)
        if len(facts)!=3870 or (raw,selfs,len(facts)) != tuple(record['counts'][k] for k in ('raw_rows','self_loops','native_nonself_rows')):raise ValueError('Complete TRAIN counts differ')
        edges=sorted(facts);neighbors=[set() for _ in range(3327)]
        for u,v in edges:neighbors[u].add(v);neighbors[v].add(u)
        enumerated=time.monotonic()
        candidates,known,work,groups=enumerate_candidates(neighbors,edges,pair_budget=job['pair_occurrence_cap'],unique_budget=job['unique_candidate_cap'],deadline=started+job['worker_soft_seconds'])
        enumerated_seconds=time.monotonic()-enumerated
        result=summarize(neighbors,edges,candidates,known,work,groups)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result.update(source_manifest_sha256=job['source_manifest_sha256'],job_sha256=sha(args.job),TRAIN_sha256=TRAIN_SHA,structural_qualification_sha256=GATE_SHA,
            enumeration_seconds=enumerated_seconds,inclusive_seconds=time.monotonic()-started,worker_user_seconds=usage.ru_utime,worker_system_seconds=usage.ru_stime,worker_peak_RSS_bytes=usage.ru_maxrss*1024)
        payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
        if len(payload.encode())>job['result_bytes_cap']:raise RuntimeError('Compact output cap exceeded; complete result not admitted')
        if time.monotonic()-started>job['worker_soft_seconds']:raise TimeoutError('Whole-worker CPU soft bound exceeded')
        (output/'CENSUS.json').write_text(payload)
        print(json.dumps({'status':'complete_paid_TRAIN_nonedge_witness_census','unique_candidates':len(candidates),'output':str(output)},sort_keys=True),flush=True)
    except (Exception,KeyboardInterrupt) as error:
        write(output/'FAILURE.json',{'complete':False,'error':type(error).__name__+': '+str(error),'inclusive_seconds':time.monotonic()-started,'partial_work_preserved':True,'retry':False,'VALID_TEST_values_access':False})
        raise


if __name__=='__main__':main()
