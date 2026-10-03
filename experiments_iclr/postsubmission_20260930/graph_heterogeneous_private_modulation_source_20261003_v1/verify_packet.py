#!/usr/bin/env python3
"""Read-only stdlib integrity and source-accounting checks; no model imports."""
import hashlib
import json
import re
from pathlib import Path

PACKET = Path(__file__).resolve().parent
WORKSPACE = PACKET.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((PACKET / name).read_text())


def safe(root, name):
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = root / relative
    assert path.is_file() and not path.is_symlink()
    assert path.resolve().is_relative_to(root)
    return path


def verify():
    seal = read('SEAL.json')
    manifest = PACKET / 'MANIFEST.sha256'
    assert sha(manifest) == seal['manifest_sha256']
    names = []
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        assert match
        expected, name = match.groups()
        assert sha(safe(PACKET, name)) == expected, name
        names.append(name)
    actual = {str(p.relative_to(PACKET)) for p in PACKET.rglob('*')
              if p.is_file() and p.name not in {'MANIFEST.sha256', 'SEAL.json'}}
    assert len(names) == len(set(names)) == seal['files_sealed']
    assert set(names) == actual
    inputs = read('INPUT_BINDINGS.json')
    assert inputs['predecessor_manifest_sha256'] == '4710544dd21c04ad41729895ecbe0dc99a57a2975a396d9899b23711d81cf947'
    assert inputs['consulted_index_sha256'] == '3618a9d22dbbc6369f850f2fb680ccb3e8727344998e0ec5ea945802c58ba2eb'
    for row in inputs['inputs']:
        assert sha(safe(WORKSPACE, row['path'])) == row['sha256'], row['path']
    scopes = read('READ_SCOPES.json')
    units = scopes['units']
    acc = scopes['accounting']
    assert len(units) == acc['source_recipe_license_dependency_retrieval_units'] == 40
    assert sum(r['status'] != 'retrieved_not_semantically_read' for r in units) == acc['units_semantically_inspected'] == 38
    assert sum(r['status'] == 'retrieved_not_semantically_read' for r in units) == acc['units_retrieved_not_semantically_read'] == 2
    assert sum(r['prior_retained_source_scope_reread'] for r in units) == acc['prior_retained_source_units_scoped_reread'] == 4
    assert acc['author_program_source_units_semantically_inspected'] == 25
    assert acc['program_source_units_retrieved'] == 28
    for key in ['new_full_primary_paper_reads', 'new_scoped_primary_method_reads',
                'retained_primary_paper_rereads', 'dataset_payloads_read',
                'published_numeric_results_read', 'model_imports_or_runs',
                'GPU_remote_compute', 'new_agents', 'adopted_drivers_or_launches']:
        assert acc[key] == 0, key
    scope = read('SCOPE_RECEIPT.json')
    assert scope['accounting'] == acc
    assert scope['all_project_writes_within_packet'] and scope['source_qualification_complete']
    for key in ['predecessor_source_protocol_index_ledger_manuscript_changed',
                'dataset_checkpoint_logits_native_results_logs_access',
                'ML_model_import_execution', 'GPU_SSH_remote_compute',
                'new_driver_launch_or_freeze', 'existing_Squirrel_BUDDY_lanes_touched',
                'runtime_release_byte_or_utility_qualification_complete']:
        assert scope[key] is False, key
    expected_blobs = {}
    for key in ['hgb', 'sehgnn', 'pyhgt']:
        tree = read('metadata/' + key + '_tree.json')
        assert tree['truncated'] is False
        for e in tree['entries']:
            if e['type'] == 'blob':
                expected_blobs[(tree['repository'], tree['commit'], e['path'])] = e['sha']
    for row in read('metadata/dgl_v043_blob_bindings.json'):
        expected_blobs[('dmlc/dgl', row['commit'], row['path'])] = row['blob_sha1']
    retrieval = read('SOURCE_RETRIEVAL.json')
    assert len(retrieval) == 40
    assert len({r['saved_path'] for r in retrieval}) == 40
    by_saved = {r['saved_path']: r for r in retrieval}
    for r in retrieval:
        path = safe(PACKET, r['saved_path'])
        assert sha(path) == r['saved_sha256']
        assert r['http_status'] == 200
        assert r['url'] == f"https://raw.githubusercontent.com/{r['repository']}/{r['commit']}/{r['path']}"
        assert r['blob_sha1'] == expected_blobs[(r['repository'], r['commit'], r['path'])]
        if not r['path'].lower().endswith('.md'):
            body = path.read_bytes()
            assert sha(path) == r['raw_sha256']
            gitblob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
            assert gitblob == r['blob_sha1']
        else:
            kept = [int(line.split(':', 1)[0]) for line in path.read_text().splitlines()]
            assert kept == r['retained_lines'] and kept == sorted(set(kept))
    for u in units:
        path = safe(PACKET, u['saved_path'])
        assert sha(path) == u['saved_sha256'] == by_saved[u['saved_path']]['saved_sha256']
        n = len(path.read_text().splitlines())
        for lo, hi in u['semantic_saved_line_ranges_inclusive']:
            assert 1 <= lo <= hi <= n
    hgt = (PACKET / 'source/hgb__NC__benchmark__methods__HGT__model.py').read_text().splitlines()
    train = (PACKET / 'source/hgb__NC__benchmark__methods__HGT__train_hgt.py').read_text().splitlines()
    assert 'v_linear(' in hgt[68] and "data[inp_key]" in hgt[68]
    assert 'relation_msg' in hgt[49] and 'torch.bmm' in hgt[49]
    assert 'F.softmax' in hgt[56] and "cross_reducer = 'mean'" in hgt[73]
    assert "str(meta_path[0]) + '_' + str(meta_path[1])" in train[65]
    assert 'use_norm' in train[37] and 'default=False' in train[37]
    assert 'labels.max()' in train[136]
    site = read('SOURCE_SITE_MAP.json')
    assert site['factor_shapes_per_layer']['c'] == [4]
    assert site['factor_shapes_per_layer']['u'] == ['d']
    decision = read('FEASIBILITY_DECISION.json')
    assert decision['rank'] == 1 and decision['members'] == 4
    assert decision['candidate_changed'] is False
    for key in ['driver_written', 'training_launched', 'speed_claim', 'scientific_hypothesis_rejected',
                'current_compute_availability_used_to_decide_science', 'numerical_author_reproduction_claimed']:
        assert decision[key] is False
    comp = read('MINIMUM_COMPLETE_RELEASE_COMPARISON.json')
    first = comp['smallest_credible_representative_first_stage']
    full = comp['existing_full_test_unchanged']
    assert len(first['arms']) == full['arms'] == 9
    assert first['paired_master_seeds'] == full['master_seeds'] == [131, 137, 139, 149, 151]
    assert first['configurations'] == 9 * 5 == 45
    assert first['total_downstream_pipeline_optimizer_jobs'] == 4*5 + 4*5 + 4*5 == 60
    assert first['complete_encoder_trajectory_run_equivalents'] == 4*5 + 4*5*4 + 5*4 == 120
    assert full['configurations'] == 90
    res = read('RESOURCE_ACCOUNTING.json')['DBLP_source_schema_illustration']
    def core(d):
        return 4*4*(d*d+d) + 2*4*d + 6*8 + 2*6*d*d//8 + 4
    assert core(64) == res['native_layer_core_parameters'] == 73268
    assert 2*4*64 == res['global_layer_fast'] == 512
    assert 2*4*64 + 4 + 6 + 64 == res['CP_layer_fast'] == 586
    assert 4*64 + 4*6*64 == res['unrestricted_layer_fast'] == 1792
    assert 3*(586-512) == res['CP_minus_global_full_HGT'] == 222
    assert core(72) + 2*4*72 == res['global72_plus_native_layer'] == 93076
    data = read('DATASET_RELEASE_QUALIFICATION.json')
    assert data['dataset_payloads_retrieved'] == data['dataset_payloads_read'] == 0
    probes = read('DATA_ACCESS_PROBE.json')
    assert len(probes) == 4 and all(r['dataset_payload_downloaded'] is False for r in probes)
    landing = next(r for r in probes if r['key'] == 'new_HGB_NC_release_landing')
    assert landing['status'] == 200 and {'ACM.zip','DBLP.zip'} <= set(landing['listed_archives'])
    licenses = read('LICENSE_SCOPE.json')
    assert licenses['HGB']['root_or_NC_benchmark_HGT_baseline_GNN_license_found'] is False
    assert licenses['SeHGNN']['license_or_copying_path_found'] is False
    assert licenses['original_pyHGT']['root_license'] == 'MIT'
    assert licenses['DGL_0_4_3']['root_license'] == 'Apache-2.0'
    return {'status':'PASS','files_sealed':len(names),'retrieved_source_units':40,
            'scoped_units_semantically_inspected':38,'prior_retained_source_units_reread':4,
            'primary_paper_reads':0,'data_or_model_execution':0,
            'representative_DBLP_first_configurations':45,
            'complete_two_graph_proposal_configurations':90,
            'manifest_sha256':seal['manifest_sha256'],
            'scope':'Integrity, read accounting, pinned source anchors and analytic parameter/job arithmetic only; no runtime, release byte, utility, reproduction or license permission certificate.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
