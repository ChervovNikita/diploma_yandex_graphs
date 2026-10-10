"""Future positive-nine and complete27 metadata gates. Stdlib only."""
import json
from source import bind
from stage_plan import SEEDS,STAGE1,CONDITIONS,ALL_CONDITIONS


def positive_nine(row,bindings):
    value=json.loads(bind(row).read_text())
    if value.get('schema')!='masked-context-PubMed-stage1-complete-nine-comparison-v1' or value.get('complete_nine') is not True:
        raise ValueError('Actual complete fixed-nine comparison required')
    if value.get('source_manifest_sha256')!=bindings['stage1_manifest']['sha256']:
        raise ValueError('Exact original Stage1 source outcome required')
    if value.get('continuation_eligible') is not True or not value.get('gates') or any(v is not True for v in value['gates'].values()):
        raise ValueError('Every original frozen accuracy/NLL/member/macro continuation gate must pass')
    expected={f'seed{seed}__{name}' for seed in SEEDS for name in STAGE1}
    if set(value.get('record_custody',{}))!=expected:raise ValueError('All nine immutable outcomes and owner custody required')
    if len(value.get('paired_seeds',[]))!=3 or {r['seed'] for r in value['paired_seeds']}!=set(SEEDS):
        raise ValueError('No favorable seed substitution')
    return value


def require_27(records):
    expected={(seed,name) for seed in SEEDS for name in ALL_CONDITIONS}
    if set(records)!=expected:raise ValueError('No partial-family mechanism comparison: preserve exactly all27 records')
    if any(r.get('complete') is not True for r in records.values()):raise ValueError('Every failed/incomplete record is retained and prevents complete27 interpretation')
    return records
