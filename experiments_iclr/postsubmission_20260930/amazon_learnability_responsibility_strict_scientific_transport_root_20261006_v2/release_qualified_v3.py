"""Create only the reviewed construction-observer successor admission."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

P = Path(__file__).resolve().parent.parent
DISABLED = 'amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v3'
RELEASED = 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2'
PRIOR = 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor(path):
    return {'path': str(path.relative_to(P)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-sha', required=True)
    args = parser.parse_args()
    report = P / args.review
    assert report.resolve().is_relative_to(P) and not report.is_symlink()
    assert sha(report) == args.review_sha and report.stat().st_mode & 0o222 == 0
    source = P / DISABLED
    assert sha(source / 'six_arm_worker.py') == 'f19a94be4102e74f30d2b779ca602e288cf40c446ae367ce9d9ab44091569ad1'
    old = (P / 'amazon_learnability_responsibility_sequential_train_only_execution_preparation_20261006_v2/six_arm_worker.py').read_text()
    new = (source / 'six_arm_worker.py').read_text()
    assert old.replace('str(torch.get_default_device())', 'str(torch.empty(0).device)') == new
    assert (source / 'QUEUE.json').read_bytes() == (P / PRIOR / 'QUEUE.json').read_bytes()
    plan = json.loads((source / 'RELEASE_PLAN.json').read_text())['flag_only_successors']
    binding = json.loads((source / 'SOURCE_BINDINGS.json').read_text())
    out = P / RELEASED
    assert not out.exists()
    out.mkdir()
    for key, name in (('worker', 'six_arm_worker.py'), ('accessor', 'train_only_accessor.py'),
                      ('sequential', 'sequential_vjp.py'), ('warm_response_probe', 'warm_response_probe.py')):
        row = plan[key]
        disabled = P / row['disabled_path']
        assert sha(disabled) == row['disabled_sha256']
        content = disabled.read_bytes()
        assert content.count(b'SOURCE_RELEASED = False') == 1
        content = content.replace(b'SOURCE_RELEASED = False', b'SOURCE_RELEASED = True')
        assert len(content) == row['anticipated_bytes'] and hashlib.sha256(content).hexdigest() == row['anticipated_sha256']
        path = out / name
        path.write_bytes(content)
        path.chmod(0o444)
        binding['files'][key] = descriptor(path)
        binding['future_flag_only_releases'][key].update(actual_release_authorized=True, enabled_file_created=True)
    (out / 'QUEUE.json').write_bytes((source / 'QUEUE.json').read_bytes())
    (out / 'held_a_evaluator.py').write_bytes((source / 'held_a_evaluator.py').read_bytes())
    binding.update(root_default_device_observer_compatibility_successor_only=True,
                   no_scientific_recipe_caps_runtime_or_scores_changed=True)
    (out / 'SOURCE_BINDINGS.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    admission = json.loads((P / PRIOR / 'PREREQUISITES.json').read_text())
    admission['UTC'] = datetime.now(timezone.utc).isoformat()
    admission['source_bindings_sha256'] = sha(out / 'SOURCE_BINDINGS.json')
    for row in admission['independent_review_evidence']:
        if row['role'] == 'scientific_source':
            row.update(evidence=descriptor(report), reviewed_source_sha256=plan['worker']['disabled_sha256'],
                       unresolved_blocking_source_defects=0)
    admission['construction_default_device_observer_only_successor'] = True
    admission['preserved_pre_update_failure'] = descriptor(P / 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v1/monitor_20261005T233653Z/output/scientific_run/FAILURE.json')
    admission['root_explicit_fit_authorized'] = True
    (out / 'PREREQUISITES.json').write_text(json.dumps(admission, indent=2, sort_keys=True) + '\n')
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'REVIEWED_OBSERVER_ONLY_SUCCESSOR_RELEASED_DISTINCT_LAUNCH_REQUIRED',
               'disabled_worker_sha256': plan['worker']['disabled_sha256'], 'released_worker': descriptor(out / 'six_arm_worker.py'),
               'independent_delta_review': descriptor(report), 'prior_admission': descriptor(P / PRIOR / 'PREREQUISITES.json'),
               'new_admission': descriptor(out / 'PREREQUISITES.json'), 'scientific_caps_and_queue_unchanged': True,
               'A_evaluator_authorized': False, 'fit_launched_here': False, 'original_failures_preserved': True}
    (out / 'RELEASE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    rows = [descriptor(x) for x in sorted(out.iterdir())]
    for row in rows:
        row['path'] = Path(row['path']).name
    (out / 'MANIFEST.json').write_text(json.dumps({'schema': 'root_exact_flag_only_observer_successor_v2', 'files': rows}, indent=2) + '\n')
    for path in out.iterdir():
        path.chmod(0o444)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
