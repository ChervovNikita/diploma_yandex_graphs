"""One immutable allocation of every physical task; no expiring network lease."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from custody import digest, require, same_or_create
from host_binding import WORKERS, normalized_absolute

CAMPAIGN = 'expanded6460_allocation_server51_v1'


def validate(ownership, canonical_tasks):
    require(ownership['schema'] == 'portable-global-ownership-v2' and ownership['campaign'] == CAMPAIGN,
            'Fixed campaign ownership required')
    rows = ownership['assignments']
    index = {t['id']: t for t in canonical_tasks}
    require(len(index) == len(canonical_tasks) == len(rows) == 6730, 'Keep all 6730 unique tasks')
    require(sum(t['kind'] == 'fit' for t in canonical_tasks) == 6460, 'Keep all 6460 fits')
    require(len({r['id'] for r in rows}) == len(rows) and {r['id'] for r in rows} == set(index), 'One owner per ID')
    owners = {r['id']: r['worker'] for r in rows}
    cohorts = {}
    for row in rows:
        task = index[row['id']]
        require(set(row) == {'id', 'worker', 'canonical_runtime_job_hash', 'science_contract_hash', 'dependencies', 'output_relative', 'cohort'}, 'Exact owner fields required')
        require(row['worker'] in WORKERS, 'Unknown physical worker')
        for key in ('science_contract_hash', 'dependencies', 'output_relative'):
            require(row[key] == task[key], 'Ownership changed canonical ' + key)
        require(row['canonical_runtime_job_hash'] == task.get('runtime_job_hash'), 'Canonical job changed')
        require(row['cohort'] == task['group_id'], 'Preserve canonical matched/source cohort')
        require(cohorts.setdefault(row['cohort'], row['worker']) == row['worker'], 'Whole cohorts must stay together')
        for dep in row['dependencies']:
            require(owners[dep] == row['worker'], 'I4 and its four fits must share physical worker')
    require(len({r['output_relative'] for r in rows}) == len(rows), 'Disjoint immutable output names required')
    for host in ('allocation', 'server51'):
        config = ownership['hosts'][host]
        normalized_absolute(config['project_root'])
        phase = PurePosixPath(config['phase_relative'])
        require(not phase.is_absolute() and '..' not in phase.parts, 'Safe root-relative host phase required')
        require(config['host'] == host and config['maximum_children_per_GPU'] == 2, 'Host scope changed')
    return owners


def claim_host(root, ownership, host):
    """Call under the host lock. Reassignment requires a new reviewed study, not timeout."""
    value = {'schema': 'portable-permanent-host-owner-v2', 'campaign': CAMPAIGN,
             'ownership_sha256': digest(ownership), 'host': host,
             'project_root': ownership['hosts'][host]['project_root']}
    same_or_create(Path(root) / 'OWNER.json', value)
    return value


def claim_task(root, ownership_sha256, row, bound_hash, output):
    value = {'schema': 'portable-permanent-task-owner-v2', 'ownership_sha256': ownership_sha256,
             'assignment': row, 'bound_runtime_job_hash': bound_hash, 'output': str(Path(output).resolve())}
    folder = Path(root) / 'claims'
    folder.mkdir(exist_ok=True)
    same_or_create(folder / (digest(row['id']) + '.json'), value)
    return value
