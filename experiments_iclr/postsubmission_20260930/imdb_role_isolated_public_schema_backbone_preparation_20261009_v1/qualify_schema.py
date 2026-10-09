"""Inactive once-only real-input schema and ten-role qualification; no model."""
import argparse
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import build_roles
from role_loader import HERE, load, read, require, sha, source_gate


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    source_gate()
    cfg = read(args.release)
    require(cfg['enabled'] is True and cfg['action'] == 'qualify_actual_IMDB_schema_and_fixed_roles'
            and cfg['root_source_review_approved'] is True and cfg['development_input_scope_authorized'] is True
            and cfg['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and cfg['seeds'] == list(range(1, 11)), 'One reviewed actual-schema/fixed-role qualification')
    require(str(args.input_root.resolve()) == cfg['input_root'] and str(args.output.resolve()) == cfg['output_directory']
            and not args.output.exists() and isinstance(cfg['expected_input_files'], dict)
            and len(cfg['authority_archive_sha256']) == 64, 'Fresh bound output and exact root input authority')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {'status': 'started', 'schema_qualified': False, 'model_import_or_fit': False,
              'TEST_member_open_stat_hash_or_parse': False, 'TEST_membership_known': False,
              'source_seal_sha256': sha(HERE / 'SEAL.json'), 'release': cfg, 'roles': []}
    report_path = args.output / 'SCHEMA_REPORT.json'
    write(report_path, record)
    try:
        np = importlib.import_module('numpy')
        require(np.__version__ == cfg['numpy_version'], 'Existing recorded native split provider')
        record['NumPy_provider'] = {'version': np.__version__, 'path': str(Path(np.__file__).resolve()), 'sha256': sha(np.__file__)}
        spec = read(HERE / 'SOURCE_EXPECTATIONS.json')
        roles = build_roles.build_all(np, args.input_root, cfg['seeds'], spec['class_dim'], cfg['expected_input_files'])
        role_folder = args.output / 'roles'
        role_folder.mkdir()
        for role in roles:
            path = role_folder / ('seed' + str(role['seed']) + '.json')
            with path.open('x') as stream:
                json.dump(role, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
        # Stream node/link/development inputs once; reuse the same buffers for every split.
        data = load(args.input_root, role_folder / 'seed1.json', spec)
        record['actual_input_schema'] = data.report()
        for role in roles:
            split = data.with_roles(role)
            path = role_folder / ('seed' + str(role['seed']) + '.json')
            record['roles'].append({'seed': role['seed'], 'path': str(path.resolve()), 'bytes': path.stat().st_size,
                                    'sha256': sha(path), 'TRAIN_count': len(split.train_ids), 'VALID_count': len(split.valid_ids),
                                    'TRAIN_positive_counts': split.report()['TRAIN_positive_counts'],
                                    'VALID_positive_counts': split.report()['VALID_positive_counts']})
        # Observe one exact integer partition against native global seed/shuffle semantics.
        state = np.random.get_state()
        try:
            original_ids = np.asarray(sorted(data.train_ids + data.valid_ids), dtype=np.int64)
            np.random.seed(1)
            np.random.shuffle(original_ids)
            boundary = len(original_ids) // 5
            require(sorted(int(node) for node in original_ids[:boundary]) == roles[0]['valid_ids']
                    and sorted(int(node) for node in original_ids[boundary:]) == roles[0]['train_ids'], 'Native global NumPy seed/shuffle partition identity')
        finally:
            np.random.set_state(state)
        record.update(status='complete', schema_qualified=True, native_integer_split_identity_observed=True,
                      all_ten_roles_qualified=True, native_backbone_numerically_qualified=False,
                      official_TEST_membership_and_scoring_deferred=True, scientific_training_authorized=False)
    except BaseException as error:
        record.update(status='failed', schema_qualified=False, failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter() - started, CPU_user_seconds=usage.ru_utime,
                      CPU_system_seconds=usage.ru_stime,
                      process_RSS_high_water_bytes=int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                      complete_inputs_are_hashed_before_and_after_validation=True,
                      all_role_descriptors_and_partial_failure_preserved=True)
        write(report_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.qualify:
        print(json.dumps({'inactive': True, 'NumPy_data_or_model_access': False}))
        return
    require(args.release and args.input_root and args.output, 'Explicit schema-only root release')
    run(args)


if __name__ == '__main__':
    main()
