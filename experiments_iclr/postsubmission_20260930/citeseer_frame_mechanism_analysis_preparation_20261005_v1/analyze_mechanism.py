"""Post-complete, CPU-only description fixed by the frozen mechanism protocol.

No fit, inference replay, TEST loader, member-logit analysis, causal reset, new
selector, coefficient fitting or subgroup model choice. Numerical imports and
all checkpoint/prediction/input payload reads follow both authenticated gates.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'citeseer_frame_mechanism_analysis_preparation_20261005_v1'
COHORT = PHASE / 'citeseer_endpoint_frame_paired_development_20261005_v1'
ANALYSIS = PHASE / 'citeseer_frame_complete_analysis_20261005_v2'
MECHANISM = PHASE / 'citeseer_frame_mechanism_protocol_20261005_v1'
TRAIN_SOURCE = PHASE / 'citeseer_heart_ncn_trainval_runner_source_20261005_v1'
PLAN_SHA = 'fd9fec451a81d0512cd8431ba2a990c580592ff6a6e8e12d784386a99c5e9daa'
HEAD_SHA = '49f2a61825d5dbcbe977926ddd3e2ea35eb02c6603b79833092e4e560fbab52f'
TRAIN_MANIFEST_SHA = 'efa95806d86e3cc261a8506042204d8d32e39506386d6faf90a625d52be12ff9'
MECHANISM_SHA = '71d42989be1c16c4eef423eb1b66abdc321fc759783054faa2b34c9b62b52d28'
ANALYSIS_SOURCE_SHA = '4006920c39200f9eb93ecffaaf55718be3917fe579295e9bbd4ba7045d9f8cee'
ANALYSIS_PROTOCOL_SHA = '25e00f42b6bc5df625ba673a090770c02e9b86b6f9928170a2d77e9d378b9c11'
ANALYSIS_MANIFEST_SHA = 'b51eb68c5dc9ee888cf8aa59cdedb9f2d7e075c4a75411548310c09a16662b31'
AVAILABLE_RELATIVE = 'citeseer_heart_official_acquisition_server_20261005_v1/AVAILABLE_MANIFEST.json'
AVAILABLE_SHA = '1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc'
FAMILIES = ('native_single', 'ordinary_independent4', 'unframed_f4', 'shared_frame_f4',
            'private_frame_f4', 'same_four_frames_single', 'independent_frame4')
PRIMARY = ('native_single', 'ordinary_independent4', 'same_four_frames_single', 'independent_frame4')
SECONDARY = ('shared_frame_f4', 'unframed_f4')
PARTITIONS = (('TRAIN_common_neighbor_support', ('zero', 'positive')),
              ('TRAIN_endpoint_degree', ('0_or_1', '2_to_5', '6_or_more')))
INPUT_NAMES = {'train_pos.txt', 'valid_pos.txt', 'heart_valid_samples.npy', 'gnn_feature'}
T2_975 = 4.302652729696142
ROUND_TOL = 1e-12


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while piece := stream.read(1024 * 1024):
            digest.update(piece)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path))


def save(output, name, value):
    with (output / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def phase_file(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an exact file relative to the authorized phase')
    path = (PHASE / path).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve(strict=True)) or not path.is_file():
        raise ValueError('File leaves the authorized project phase')
    return path


def component_ids(base, family):
    if family == 'ordinary_independent4':
        return [f'b{base}_native_single_seed{base}'] + [
            f'b{base}_independent_member_m{member}_seed{base + 5 * member}' for member in range(1, 4)]
    if family == 'independent_frame4':
        return [f'b{base}_independent_frame_member_m{member}_seed{base + 5 * member}' for member in range(4)]
    return [f'b{base}_{family}_seed{base}']


def seed_summary(values):
    assert len(values) == 3 and all(math.isfinite(value) for value in values)
    mean, sd = statistics.mean(values), statistics.stdev(values)
    return dict(values=values, mean=mean, sd=sd,
                descriptive_t95_seed_interval=[mean - T2_975 * sd / math.sqrt(3),
                                               mean + T2_975 * sd / math.sqrt(3)],
                inference_scope='Descriptive seed variation on one fixed graph/split; normal seed approximation. No query/graph independence or significance claim.')


def contrast(candidate, baseline, empty=False):
    if empty:
        return dict(status='empty_stratum', paired_differences=[None] * 3, mean=None,
                    uncertainty=None, significance_test_performed=False)
    differences = [a - b for a, b in zip(candidate, baseline)]
    return dict(status='descriptive_only', **seed_summary(differences),
                candidate_higher_blocks=sum(value > 0 for value in differences),
                tied_blocks=sum(value == 0 for value in differences), significance_test_performed=False)


def read_positive_rows(path, metadata, expected):
    pairs, raw, self_count = [], 0, 0
    with path.open() as stream:
        for line in stream:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                raise ValueError('Positive file is not native two-column text')
            u, v = map(int, fields)
            assert 0 <= u < 3327 and 0 <= v < 3327
            raw += 1
            if u == v:
                self_count += 1
            else:
                pairs.append((u, v))
    assert raw == metadata['counts']['raw_rows']
    assert self_count == metadata['counts']['self_loops']
    assert len(pairs) == metadata['counts']['native_nonself_rows'] == expected
    assert len({tuple(sorted(pair)) for pair in pairs}) == len(pairs)
    return pairs


def unit_interval(value):
    assert math.isfinite(value) and -ROUND_TOL <= value <= 1 + ROUND_TOL
    clamped = min(1.0, max(0.0, value))
    return clamped, clamped != value


def frame_axes(row):
    if row['arm'] == 'shared_frame_f4':
        return [0]
    if row['arm'] in ('private_frame_f4', 'same_four_frames_single'):
        return list(range(4))
    if row['arm'] == 'independent_frame_member':
        assert row['axis_index'] == row['ensemble_member_index']
        return [row['axis_index']]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path,
                        help='Exact root-reviewed receipt with post-complete artifact hashes; absent means closed gate')
    args = parser.parse_args()
    started, output, authorized = time.monotonic(), None, False
    stage = 'authorized_route_and_two_completion_gates'
    try:
        assert Path.cwd() == REPO and socket.gethostname() == 'anogena-2-0'
        assert Path(__file__).resolve().parent == SOURCE
        assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.splitlines() == [
                                  'GPU-44039938-fd82-41d2-fefd-de71514e2fac']
        authorized = True
        plan_path = COHORT / 'PLAN.json'
        assert sha(plan_path) == PLAN_SHA and sha(TRAIN_SOURCE / 'heads.py') == HEAD_SHA
        assert sha(MECHANISM / 'PROTOCOL.json') == MECHANISM_SHA
        cohort_path, result_path = COHORT / 'COHORT_FREEZE.json', ANALYSIS / 'RESULTS.json'
        if not cohort_path.exists() or not result_path.exists() or args.admission is None:
            print(json.dumps(dict(status='WAIT_FOR_BOTH_AUTHENTICATED_COMPLETIONS_AND_ADMISSION',
                                  cohort_terminal_present=cohort_path.exists(), analysis_result_present=result_path.exists(),
                                  admission_supplied=args.admission is not None,
                                  checkpoint_prediction_history_input_payload_reads=0, TEST_access=False)))
            return
        admission_path = args.admission.resolve(strict=True)
        assert admission_path.is_relative_to(PHASE.resolve(strict=True)) and admission_path.stat().st_size < 65536
        admission = json.loads(admission_path.read_text())
        assert admission['source_review_approved'] is True
        assert admission['script_sha256'] == sha(Path(__file__))
        assert admission['source_manifest_sha256'] == sha(SOURCE / 'SOURCE_MANIFEST.json')
        assert admission['plan_sha256'] == PLAN_SHA and admission['mechanism_protocol_sha256'] == MECHANISM_SHA
        for item in json.loads((SOURCE / 'SOURCE_MANIFEST.json').read_text())['files']:
            assert sha(SOURCE / item['path']) == item['sha256']
        assert sha(cohort_path) == admission['cohort_freeze_sha256']
        cohort = json.loads(cohort_path.read_text())
        assert cohort['complete'] is True and cohort['TEST_access'] is False
        assert cohort['plan_sha256'] == PLAN_SHA and cohort['source_manifest_sha256'] == TRAIN_MANIFEST_SHA
        plan = json.loads(plan_path.read_text())
        rows, registry = plan['physical_fits'], cohort['completed_physical_fits']
        assert len(rows) == len(registry) == 36
        assert [item['id'] for item in registry] == [row['id'] for row in rows]
        registered = {item['id']: item for item in registry}
        assert len(registered) == 36
        assert registry[0]['baseline_reused'] is True
        assert all(item['baseline_reused'] is False for item in registry[1:])
        for row in rows:
            assert sha(COHORT / 'fits' / row['id'] / 'FREEZE.json') == registered[row['id']]['freeze_sha256']
        # Gate two: successful complete analysis is bound before any payload access.
        assert not (ANALYSIS / 'FAILURE.json').exists()
        assert sha(ANALYSIS / 'analyze_valid.py') == ANALYSIS_SOURCE_SHA
        assert sha(ANALYSIS / 'PROTOCOL.json') == ANALYSIS_PROTOCOL_SHA
        assert sha(ANALYSIS / 'ANALYSIS_MANIFEST.json') == ANALYSIS_MANIFEST_SHA
        for item in json.loads((ANALYSIS / 'ANALYSIS_MANIFEST.json').read_text())['files']:
            assert sha(ANALYSIS / item['path']) == item['sha256']
        assert sha(result_path) == admission['analysis_RESULTS_sha256']
        result = json.loads(result_path.read_text())
        assert result['status'] == 'complete_audited_development_comparison'
        assert result['source_plan_sha256'] == PLAN_SHA and result['source_manifest_sha256'] == TRAIN_MANIFEST_SHA
        assert result['cohort_freeze_sha256'] == admission['cohort_freeze_sha256']
        assert result['TEST_access'] is False and result['fit_count'] == result['optimizer_updates'] == 0
        assert result['independent_aggregate_retuning'] is False
        assert set(result['family_summaries']) == set(FAMILIES)
        fit_audits = {item['id']: item for item in result['fit_audits']}
        assert len(result['fit_audits']) == len(fit_audits) == 36 and set(fit_audits) == set(registered)
        assert sha(ANALYSIS / 'START.json') == admission['analysis_START_sha256']
        analysis_start = json.loads((ANALYSIS / 'START.json').read_text())
        assert analysis_start['cohort_freeze_sha256'] == admission['cohort_freeze_sha256']
        assert analysis_start['protocol_sha256'] == ANALYSIS_PROTOCOL_SHA
        assert analysis_start['analysis_source_sha256'] == ANALYSIS_SOURCE_SHA
        assert analysis_start['TEST_access'] is False
        prediction_path = ANALYSIS / 'served_VALID_predictions.pt'
        assert result['served_predictions_sha256'] == admission['served_VALID_predictions_sha256']
        assert sha(prediction_path) == result['served_predictions_sha256']
        assert sha(ANALYSIS / 'PER_QUERY.json') == result['per_query_sha256']
        relative_output = Path(admission['output_relative'])
        assert not relative_output.is_absolute() and '..' not in relative_output.parts
        output = (PHASE / relative_output).resolve()
        assert output.is_relative_to(PHASE.resolve(strict=True)) and not output.exists() and output.parent.is_dir()
        output.mkdir()
        save(output, 'START.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
             admission_reference=reference(admission_path), protocol_reference=reference(MECHANISM / 'PROTOCOL.json'),
             source_manifest_reference=reference(SOURCE / 'SOURCE_MANIFEST.json'),
             cohort_freeze_reference=reference(cohort_path), analysis_RESULTS_reference=reference(result_path),
             gate='Both exact completions authenticated before checkpoint/prediction/history/input payload access',
             fit_count=0, optimizer_updates=0, TEST_access=False))
        stage = 'post_gate_input_and_served_prediction_description'
        # No scientific import occurs before both completion gates above.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        torch.set_num_threads(2)
        freezes, identities, first_job = {}, None, None
        for row in rows:
            folder = COHORT / 'fits' / row['id']
            freeze = json.loads((folder / 'FREEZE.json').read_text())
            assert freeze['TEST_access'] is False and freeze['cohort_plan_sha256'] == PLAN_SHA
            assert freeze['source_manifest_sha256'] == TRAIN_MANIFEST_SHA
            assert all(freeze[key] == row[key] for key in ('arm', 'seed', 'member_count', 'paired_seed_block'))
            assert fit_audits[row['id']]['selected_epoch'] == freeze['selected_epoch']
            assert fit_audits[row['id']]['checkpoint_sha256'] == freeze['checkpoint_sha256']
            assert sha(folder / 'CONFIG.json') == freeze['config_sha256']
            config = json.loads((folder / 'CONFIG.json').read_text())
            assert all(config['job'][key] == value for key, value in row.items())
            if identities is None:
                identities, first_job = freeze['input_identities'], config['job']
            assert freeze['input_identities'] == identities == config['input_identities']
            freezes[row['id']] = freeze
        assert set(identities) == INPUT_NAMES
        assert first_job['available_manifest_relative'] == AVAILABLE_RELATIVE
        assert first_job['available_manifest_sha256'] == AVAILABLE_SHA
        available_path = phase_file(AVAILABLE_RELATIVE)
        assert sha(available_path) == AVAILABLE_SHA
        available = json.loads(available_path.read_text())
        assert available['TEST_available_to_loader'] is False and set(available['files']) == INPUT_NAMES
        for name, item in available['files'].items():
            assert identities[name] == dict(sha256=item['sha256'], bytes=item['bytes'])
            assert tuple(Path(item['relative_path']).parts) == ('available', 'citeseer', name)
        positive_paths = {}
        for name in ('train_pos.txt', 'valid_pos.txt'):
            path = (available_path.parent / available['files'][name]['relative_path']).resolve(strict=True)
            assert path.is_relative_to(PHASE.resolve(strict=True))
            assert sha(path) == identities[name]['sha256'] and path.stat().st_size == identities[name]['bytes']
            positive_paths[name] = path
        train = read_positive_rows(positive_paths['train_pos.txt'], available['files']['train_pos.txt'], 3870)
        valid = read_positive_rows(positive_paths['valid_pos.txt'], available['files']['valid_pos.txt'], 227)
        assert not ({tuple(sorted(pair)) for pair in train} & {tuple(sorted(pair)) for pair in valid})
        neighbors = [set() for _ in range(3327)]
        for u, v in train:
            neighbors[u].add(v)
            neighbors[v].add(u)
        assignments = []
        for index, (u, v) in enumerate(valid):
            common, degree = len(neighbors[u] & neighbors[v]), min(len(neighbors[u]), len(neighbors[v]))
            assignments.append(dict(query_index=index, positive_endpoints=[u, v], TRAIN_common_neighbors=common,
                TRAIN_min_endpoint_degree=degree, TRAIN_common_neighbor_support='zero' if common == 0 else 'positive',
                TRAIN_endpoint_degree='0_or_1' if degree <= 1 else '2_to_5' if degree <= 5 else '6_or_more'))
        predictions = torch.load(prediction_path, map_location='cpu', weights_only=True)
        assert predictions['input_identities'] == identities
        assert predictions['cohort_freeze_sha256'] == admission['cohort_freeze_sha256']
        expected_keys = {f'citeseer_native_b{base}/{family}' for base in range(3) for family in FAMILIES}
        assert set(predictions['served']) == set(predictions['components']) == expected_keys
        reciprocal_ranks = {}
        for base in range(3):
            for family in FAMILIES:
                key = f'citeseer_native_b{base}/{family}'
                assert predictions['components'][key] == component_ids(base, family)
                entry = predictions['served'][key]
                pos, neg = entry['pos'], entry['neg']
                assert pos.shape == (227,) and neg.shape == (227, 500)
                assert pos.dtype == neg.dtype == torch.float32
                assert torch.isfinite(pos).all() and torch.isfinite(neg).all()
                rank = 1 + .5 * ((neg >= pos[:, None]).sum(1) + (neg > pos[:, None]).sum(1))
                rr = 1 / rank.float()
                assert torch.equal(rank, entry['rank']) and torch.equal(rr, entry['reciprocal_rank'])
                assert rr.mean().item() == result['family_summaries'][family]['values'][base]
                reciprocal_ranks[key] = rr
        strata = []
        for partition, bins in PARTITIONS:
            total = 0
            for label in bins:
                indices = [row['query_index'] for row in assignments if row[partition] == label]
                total += len(indices)
                families = {}
                for family in FAMILIES:
                    values = [reciprocal_ranks[f'citeseer_native_b{base}/{family}'][indices].mean().item()
                              for base in range(3)] if indices else [None] * 3
                    families[family] = seed_summary(values) if indices else dict(status='empty_stratum', values=values, mean=None, sd=None)
                strata.append(dict(partition=partition, bin=label, query_count=len(indices), query_indices=indices,
                    families=families, primary_paired_descriptions={reference:contrast(
                        families['private_frame_f4']['values'], families[reference]['values'], empty=not indices) for reference in PRIMARY},
                    secondary_paired_descriptions={reference:contrast(
                        families['private_frame_f4']['values'], families[reference]['values'], empty=not indices) for reference in SECONDARY},
                    ranking_scope='Each positive query retains its complete released 500-negative ranking; bins select positive queries only',
                    subgroup_rescue_or_selection=False))
            assert total == 227
        save(output, 'STRUCTURAL_QUERY_ASSIGNMENTS.json', dict(input_identities=identities, queries=assignments,
             topology='Undirected observed TRAIN only; no self loops; CN and degree are two separate partitions'))
        save(output, 'STRATIFIED_MRR.json', dict(strata=strata, full_cohort_family_summaries=result['family_summaries'],
             interpretation='Descriptive validation-selected development only; no subgroup selects a model or rescues failed aggregate quality',
             significance_test_performed=False, TRAIN_VALID_only=True, TEST_access=False))
        stage = 'selected_checkpoint_frame_map_description'
        frame_records = []
        for row in rows:
            axes = frame_axes(row)
            if axes is None:
                continue
            freeze = freezes[row['id']]
            checkpoint_path = COHORT / 'fits' / row['id'] / 'selected_checkpoint.pt'
            assert sha(checkpoint_path) == freeze['checkpoint_sha256']
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['selected_epoch'] == freeze['selected_epoch']
            assert checkpoint['selected_VALID_MRR'] == freeze['selected_VALID_MRR']
            assert checkpoint['config']['input_identities'] == identities
            assert all(checkpoint['config']['job'][key] == value for key, value in row.items())
            vectors = checkpoint['predictor']['v']
            assert vectors.dtype == torch.float32 and vectors.shape == (len(axes), 256) and torch.isfinite(vectors).all()
            vectors = vectors.to(torch.float64)
            norms = vectors.square().sum(1).sqrt()
            assert torch.isfinite(norms).all() and bool((norms > 0).all())
            unit = vectors / norms[:, None]
            descriptors = []
            for index, axis in enumerate(axes):
                raw = 1 - float(unit[index, axis]) ** 2
                energy, clamped = unit_interval(raw)
                descriptors.append(dict(checkpoint_row=index, prospective_initial_axis=axis,
                    vector_norm=float(norms[index]), off_initial_axis_energy=energy,
                    floating_roundoff_clamped=clamped))
            pairs = []
            if row['arm'] in ('private_frame_f4', 'same_four_frames_single'):
                for left in range(len(axes)):
                    for right in range(left + 1, len(axes)):
                        cosine = float((unit[left] * unit[right]).sum())
                        distance_squared_fraction, clamped = unit_interval(1 - cosine ** 2)
                        pairs.append(dict(left_checkpoint_row=left, right_checkpoint_row=right,
                            reflection_Frobenius_distance=math.sqrt(8 * distance_squared_fraction),
                            sign_invariant=True, floating_roundoff_clamped=clamped,
                            coordinate_scope='Two frames within this one selected model and common encoder coordinates'))
            frame_records.append(dict(id=row['id'], arm=row['arm'], seed=row['seed'], paired_seed_block=row['paired_seed_block'],
                checkpoint_reference=reference(checkpoint_path), selected_epoch=freeze['selected_epoch'],
                prospective_plan_row=row, frames=descriptors, within_checkpoint_map_pairs=pairs,
                pair_scope='No cross-checkpoint/independent-encoder distances. Shared and one-frame checkpoints have no unique map pair.',
                coordinate_limit='Per-fit parameter coordinates; learned bases across architectures/checkpoints are not semantically aligned.',
                norm_limit='Vector norm depends on the scale gauge and is not transform magnitude.'))
            del checkpoint, vectors, norms, unit
        assert len(frame_records) == 21 and sum(len(record['frames']) for record in frame_records) == 39
        save(output, 'FRAME_MAP_DESCRIPTORS.json', dict(checkpoints=frame_records,
             map_diversity_is_prediction_diversity=False, cross_encoder_map_distances_computed=False,
             member_logits_or_causal_reset_performed=False,
             interpretation='Norm/off-axis state and within-model Householder map distance only; no causal or complementary-prediction claim.'))
        result_files = ['STRUCTURAL_QUERY_ASSIGNMENTS.json', 'STRATIFIED_MRR.json', 'FRAME_MAP_DESCRIPTORS.json']
        save(output, 'RESULTS.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
             status='complete_post_gate_prospective_mechanism_description', admission_reference=reference(admission_path),
             protocol_reference=reference(MECHANISM / 'PROTOCOL.json'), source_manifest_reference=reference(SOURCE / 'SOURCE_MANIFEST.json'),
             plan_reference=reference(plan_path), head_reference=reference(TRAIN_SOURCE / 'heads.py'),
             cohort_freeze_reference=reference(cohort_path), analysis_RESULTS_reference=reference(result_path),
             analysis_START_reference=reference(ANALYSIS / 'START.json'), served_prediction_reference=reference(prediction_path),
             available_manifest_reference=reference(available_path), positive_input_references={name:reference(path) for name,path in positive_paths.items()},
             input_identities=identities, completed_fit_registry=registry,
             reuse_contract='One initially completed native baseline plus 35 prospectively fixed remaining fits; each native single is ordinary ensemble member zero.',
             selected_frame_checkpoint_count=21, frame_vector_count=39, partition_count=2, fixed_bin_count=5,
             files={name:reference(output / name) for name in result_files}, fit_count=0, optimizer_updates=0,
             history_payloads_read=0, feature_payloads_loaded=0, negative_pool_payloads_loaded=0, TEST_access=False,
             per_member_logits_analyzed=False, checkpoint_reselection=False, coefficient_or_stratum_tuning=False,
             inference_replay=False, causal_reset=False, subgroup_rescue=False, novelty_or_superiority_established=False,
             limitations=['Validation selected checkpoints; all structural descriptions are development diagnostics.',
                          'Three seeds on one graph/split do not establish statistical significance or graph generality.',
                          'Householder map distances are computed only within a model sharing encoder coordinates.',
                          'Frame-vector norms depend on scale gauge; coordinates across checkpoints are not semantically aligned.',
                          'Map diversity is not prediction diversity; no complementary-error claim follows.',
                          'No favorable subgroup can rescue unsuccessful aggregate quality.'],
             inclusive_seconds=time.monotonic() - started))
        print(json.dumps(dict(status='complete_post_gate_prospective_mechanism_description', output=str(output),
                              fit_count=0, TEST_access=False, subgroup_rescue=False)))
    except (Exception, KeyboardInterrupt) as error:
        failure = dict(UTC=datetime.now(timezone.utc).isoformat(), error=type(error).__name__ + ': ' + str(error),
                       stage=stage, fit_count=0, optimizer_updates=0, TEST_access=False, partial_outputs_preserved=True,
                       retry=False, inclusive_seconds=time.monotonic() - started)
        if authorized and output is not None and output.is_dir() and not (output / 'FAILURE.json').exists():
            save(output, 'FAILURE.json', failure)
        else:
            print(json.dumps(failure))
        raise


if __name__ == '__main__':
    main()
