"""Standard-library-only custody and completeness audit of the fetched recheck."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def main():
    counts = {'hash_checks': 0, 'strict_comparison_records': 0}

    def check_file(path, expected):
        raw = path.read_bytes()
        assert len(raw) == expected['bytes'], str(path)
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], str(path)
        counts['hash_checks'] += 1

    inputs = json.loads((HERE / 'INPUT_BINDINGS.json').read_text())
    for e in inputs['files']:
        check_file(PHASE / e['path'], e)
    static = json.loads((HERE / 'STATIC_CHECK.json').read_text())
    assert static['all_passed'] and static['check_count'] == 21
    assert all(static['checks'].values())
    assert hashlib.sha256((HERE / 'recheck.py').read_bytes()).hexdigest() == static['new_recheck_sha256']
    counts['hash_checks'] += 1
    fetched = json.loads((HERE / 'FETCH_RECEIPT.json').read_text())
    assert len(fetched['files']) == 5 and fetched['every_fetched_file_hash_verified']
    for e in fetched['files']:
        check_file(HERE / e['path'], e)
    for e in fetched['originals']:
        check_file(PHASE / e['path'], e)
    wrapper_sha = next(e['sha256'] for e in inputs['files'] if e['path'].endswith('/run_gpu77_v3.py'))
    for action in ['ROUTE', 'EXECUTION', 'FETCH']:
        r = json.loads((HERE / (action + '_TRANSPORT_RECEIPT.json')).read_text())
        assert r['exit_code'] == 0 and r['target'] == 'shmelev@192.168.18.77'
        assert not r['credential_value_recorded'] and r['wrapper_sha256'] == wrapper_sha
        command = (HERE / (action + '_COMMAND.txt')).read_bytes()
        assert hashlib.sha256(command).hexdigest() == r['command_sha256']
        counts['hash_checks'] += 1
    route = json.loads((HERE / 'ROUTE.json').read_text())
    assert not route['remote_writes'] and route['successor_path_fresh'] and not route['CUDA_initialized']
    execution = json.loads((HERE / 'EXECUTION.json').read_text())
    assert execution['exit_code'] == 0 and execution['env_overrides'] == {}
    assert not execution['isolation_wrapper'] and not execution['base_environment_changed']
    check_file(HERE / 'RECHECK.json', dict(bytes=execution['result_bytes'], sha256=execution['result_sha256']))
    result = json.loads((HERE / 'RECHECK.json').read_text())
    original = json.loads((PHASE / 'source_model_gauge_probe_20261003_v1/WITNESS.json').read_text())
    assert result['status'] == 'BYTE_IDENTITY_CONFIRMED_ON_FIXED_FIXTURE'
    assert result['strict_checker_tolerance'] is None
    assert result['engineering_seed'] == original['engineering_seed'] == 20261003
    assert result['fixed_scales'] == original['fixed_scales']
    assert result['changed_parameters'] == original['changed_parameters']
    assert result['source_manifest_sha256'] == original['source_manifest_sha256']
    assert result['runtime']['torch_version'] == original['torch_version'] == '2.7.1'
    assert result['runtime']['normal_execution'] and not result['runtime']['CUDA_initialized']
    assert result['all_original_claimed_tensors_numerically_equal']
    assert result['all_original_claimed_tensors_strict_byte_identical']
    assert not result['dataset_checkpoint_or_label_access'] and not result['GPU_compute']
    assert not result['predictive_superiority_or_new_primitive_claim']
    assert not result['universal_floating_point_byte_invariance_claim']

    def strict(r, shape):
        assert r['dtype_identical'] and r['shape_identical'] and r['numerical_equal']
        assert r['dtype_before'] == r['dtype_after'] == 'torch.float32'
        assert r['shape_before'] == r['shape_after'] == shape
        assert r['contiguous_uint8_bytes_identical'] and r['strict_byte_identity']
        assert r['differing_byte_count'] == 0
        assert r['before_sha256'] == r['after_sha256']
        assert r['before_bytes'] == r['after_bytes']
        counts['strict_comparison_records'] += 1

    old_rows = {(r['mode'], r['graph']): r for r in original['comparisons']}
    assert len(result['comparisons']) == 6
    assert {(r['mode'], r['graph']) for r in result['comparisons']} == set(old_rows)
    for row in result['comparisons']:
        strict(row['logits'], [6, 4])
        assert len(row['members']) == 4 and {r['member'] for r in row['members']} == set(range(4))
        for r in row['members']:
            strict(r, [6])
        strict(row['served_mean'], [6])
        assert row['maximum_logit_difference'] == 0
        assert row['hidden_scaling_tolerance'] == dict(rtol=1e-5, atol=1e-6)
        assert row['hidden_scaling_allclose'] and row['original_cosines_reproduced_exactly']
        old = old_rows[(row['mode'], row['graph'])]
        for key in ['mean_squared_hidden_cosine_before', 'mean_squared_hidden_cosine_after']:
            assert row[key] == old[key]
    assert set(result['responses']) == {'private', 'pooled_after_clamp'}
    for response in result['responses'].values():
        strict(response['concatenated'], [12, 4])
        assert response['maximum_response_difference'] == 0
        assert {r['member'] for r in response['members']} == set(range(4))
        for r in response['members']:
            strict(r, [12])
        assert {r['graph'] for r in response['graph_responses']} == {'drop_first_two', 'drop_last_two'}
        for graph in response['graph_responses']:
            strict(graph['responses'], [6, 4])
            assert {r['member'] for r in graph['members']} == set(range(4))
            for r in graph['members']:
                strict(r, [6])
    assert counts['strict_comparison_records'] == 66
    controls = result['comparator_controls']
    assert controls['signed_zero']['numerical_equal'] and not controls['signed_zero']['strict_byte_identity']
    assert controls['signed_zero']['before_sha256'] != controls['signed_zero']['after_sha256']
    assert not controls['dtype']['dtype_identical'] and not controls['dtype']['strict_byte_identity']
    assert not controls['shape']['shape_identical'] and not controls['shape']['strict_byte_identity']
    assert controls['noncontiguous']['strict_byte_identity']
    manifest = HERE / 'MANIFEST.json'
    manifest_verified = False
    if manifest.exists():
        m = json.loads(manifest.read_text())
        assert set(e['path'] for e in m['files']) == {p.name for p in HERE.iterdir() if p.name != 'MANIFEST.json'}
        for e in m['files']:
            check_file(HERE / e['path'], e)
        manifest_verified = True
    record = dict(schema='gauge-byte-recheck-local-verification-v1',
                  UTC=datetime.now(timezone.utc).isoformat(), all_checks_passed=True,
                  **counts, original_bound_inputs_unchanged=True, source_AST_checks=21,
                  claimed_logit_tensors=6, explicit_member_logits=24,
                  claimed_concatenated_responses=2, explicit_member_graph_responses=16,
                  signed_zero_dtype_shape_noncontiguous_controls_passed=True,
                  exact_remote_fetch_verified=True, manifest_verified=manifest_verified,
                  numerical_imports=False)
    if not manifest.exists():
        with (HERE / 'VERIFICATION.json').open('x') as f:
            json.dump(record, f, indent=2)
            f.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
