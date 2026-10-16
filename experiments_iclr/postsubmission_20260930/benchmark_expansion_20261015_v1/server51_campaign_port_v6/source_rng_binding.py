"""Bind the unchanged OGB iterator audit to exact verified server51 source paths."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
E = HERE.parent
ALLOCATION_ROOT = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
SERVER51_ROOT = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
PHASE_RELATIVE = 'experiments_iclr/postsubmission_20260930'
ORIGINAL_RELATIVE = 'benchmark_expansion_20261015_v1/qualification/runtime_v2_source_rng_v1.json'
ORIGINAL_SHA = 'd8be44fce4ac8d531689f67547e2d4069fd354af8dc571fbf5cffcfe12bbad4c'
AUDIT_RELATIVE = 'benchmark_expansion_20261015_v1/server51_campaign_port_v6/source_rng_audits/server51.json'
AUDIT_SHA = '2bb4f4eb9af39ebc76f95fc08a89f4e75a18dfb60979d21cf05aaf0e1b6402c6'
MODE = 'omitted_test_iterator_base_seed_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def read_pinned(path, sha):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'Exact source RNG audit bytes required')
    return json.loads(raw)


def expected_profile(original):
    """Change only the root of exact source receipts; preserve all content pins."""
    result = copy.deepcopy(original)
    rows = [*result['installed_sources'].values(), *result['native_sources'].values(), result['native_manifest']]
    require(len(rows) == 24, 'Exact audited installed/native source set required')
    for row in rows:
        require(set(row) == {'path', 'bytes', 'sha256'}, 'Exact source receipt required')
        path = PurePosixPath(row['path'])
        require(path.is_relative_to(ALLOCATION_ROOT) and '..' not in path.parts,
                'Source audit path escapes original allocation')
        row['path'] = str(PurePosixPath(SERVER51_ROOT) / path.relative_to(ALLOCATION_ROOT))
    return result


@lru_cache(maxsize=1)
def verify_audits():
    original = read_pinned(E.parent / ORIGINAL_RELATIVE, ORIGINAL_SHA)
    current = read_pinned(HERE / 'source_rng_audits/server51.json', AUDIT_SHA)
    expected = copy.deepcopy(original)
    expected['dependency_profile'] = expected_profile(original['dependency_profile'])
    require({k: v for k, v in current.items() if k != 'host_rebinding'} == expected,
            'Host audit must preserve every assumption and semantic result')
    require(len(original['semantic_check']['cases']) == 36 and original['TEST_accessed'] is False
            and original['model_calls'] == 0, 'Original finite no-TEST audit required')
    evidence = current['host_rebinding']
    require(evidence['all_24_source_bytes_and_dependency_versions_equal'] is True
            and evidence['method_or_rng_draw_changes'] is False
            and evidence['fresh_semantic_execution'] is False, 'Explicit identical-source evidence required')
    return current


def rebind_source_rng(job, root, phase_relative):
    """Change two execution metadata fields only, after normal path relocation."""
    result = copy.deepcopy(job)
    declaration = job.get('source_rng_emulation')
    if declaration is None:
        return result, []
    require(root in (ALLOCATION_ROOT, SERVER51_ROOT) and phase_relative == PHASE_RELATIVE,
            'Source RNG audit only supports the two explicit campaign hosts')
    expected = {'audit': str(PurePosixPath(root) / phase_relative / ORIGINAL_RELATIVE),
                'audit_sha256': ORIGINAL_SHA, 'mode': MODE}
    require(declaration == expected, 'Only the exact original source RNG declaration may rebind')
    require(job.get('family') == 'ogb' and job.get('procedure') == 'native_single'
            and job.get('policy', {}).get('track') == 'source_reference'
            and job.get('TEST_access') is False, 'Closed-TEST native OGB source-reference job required')
    if root == ALLOCATION_ROOT:
        return result, []
    verify_audits()
    replacement = {'audit': str(PurePosixPath(root) / phase_relative / AUDIT_RELATIVE),
                   'audit_sha256': AUDIT_SHA, 'mode': MODE}
    result['source_rng_emulation'] = replacement
    changes = [{'pointer': '/source_rng_emulation/' + name, 'from': declaration[name], 'to': replacement[name]}
               for name in ('audit', 'audit_sha256')]
    return result, changes
