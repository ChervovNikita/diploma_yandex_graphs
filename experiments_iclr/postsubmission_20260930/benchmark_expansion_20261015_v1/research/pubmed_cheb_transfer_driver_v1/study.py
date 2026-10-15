"""Source/metadata and scalar node diagnostics; no eager scientific imports."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
PHASE = E.parent
EXPANSION = Path('experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1')
SEEDS = [101,202,303]
BANKS = ['M1','I4','G0','uniform','structured','permuted']
COMPOSITION = E/'research/pubmed_cheb_residual_composition_v1'


def require(ok,message):
    if not ok: raise ValueError(message)


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def helpers():
    return load(E/'pilot_20261015_structural_private_risk_v1/run.py','_pubmed_existing_launch_helpers')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def jobs(spec):
    """Original native-body/G0 builder plus exact residual descriptors; no writes."""
    sys.path.insert(0,str(E/'runtime'))
    native = load(E/'runtime/run_pilot.py','_pubmed_original_pilot_builder')
    ext = load(COMPOSITION/'extension.py','_pubmed_transfer_composition')
    original, cohorts = native.build_jobs(spec['native_builder_spec'])
    result = []
    remote_extension = str(Path(spec['project_root'])/EXPANSION/'research/pubmed_cheb_residual_composition_v1/extension.py')
    for seed,cohort in zip(SEEDS,cohorts):
        selected = [j for j in original if j['cohort_id'] == cohort['id']]
        require(len(selected) == 5 and [j.get('body') for j in selected[:4]] == [0,1,2,3], 'Four native bodies and G0 required')
        for j in selected:
            kind = 'native' if j['procedure'] == 'native_single' else 'G0'
            result.append({'cohort_seed':seed,'kind':kind,'job':j})
        for variant in ('uniform','structured','permuted'):
            j = ext.descriptor(spec['project_root'],seed,variant,spec['composition_manifest']['sha256'])
            # Source-only macOS evaluation resolves /home through /System/Volumes/Data.
            # Preserve the exact canonical allocation paths of the frozen CPU descriptor.
            j['recipe']['_repository_root'] = spec['project_root']
            j['session_transform']['path'] = remote_extension
            require(j['cohort_id'] == cohort['id'], 'Matched cohort identity differs')
            result.append({'cohort_seed':seed,'kind':variant,'job':j})
    for index,row in enumerate(result):
        row.update(index=index,id=row['job']['id'],job_hash=digest(row['job']))
        row['token'] = str(index).zfill(3)+'_'+row['job_hash'][:16]
    require(len(result) == len({r['id'] for r in result}) == 24 and len(cohorts) == 3,'Exactly24 cold trajectories required')
    require([r['cohort_seed'] for r in result] == [s for s in SEEDS for _ in range(8)],'Frozen seed order differs')
    return result,cohorts


def terminal_gate(end,outcomes,rows):
    require(end.get('status') == 'complete' and end.get('TEST_accessed') is False and end.get('outcomes') == outcomes and
        len(rows) == len(outcomes) == 24 and [r.get('id') for r in outcomes] == [r['id'] for r in rows] and
        all(r.get('status') == 'complete' and r.get('returncode') == 0 and not r.get('timed_out') and
            isinstance(r.get('RESULT',{}).get('sha256'),str) and len(r['RESULT']['sha256']) == 64 for r in outcomes),
        'All24 terminal fits required before collection/quality analysis; no partial summary')


def collection_gate(end,rows):
    require(end.get('status') == 'complete' and end.get('TEST_accessed') is False and len(rows) == 12 and
        [r.get('id') for r in end.get('outcomes',[])] == [r['id'] for r in rows] and
        all(r.get('returncode') == 0 and r.get('status') in ('complete','complete_with_sanity_warning') and
            not r.get('timed_out') and r.get('RESULT',{}).get('sha256') for r in end['outcomes']),
        'All12 successful selected shared-bank collections required; no partial summary')


def fit(root,output,row,outcome):
    h = helpers(); folder = Path(output)/'fits'/row['token']
    path = h.verified(root,outcome['RESULT'])
    require(path == folder/'RESULT.json','Terminal fit path differs')
    value = json.loads(path.read_text()); job = json.loads((folder/'JOB.json').read_text())
    require(job == row['job'] and digest(job) == value['job_hash'] == row['job_hash'] and value['job_id'] == row['id'] and
        value['status'] == 'complete' and value['TEST_accessed'] is False and value['policy'] == job['policy'] and
        value['seed'] == job['seed'] and value['selected_epoch'] is not None and
        not value.get('failure') and not value.get('close_error'), 'Exact completed own fit/selection required')
    binding = value['runtime_binding']
    for name,receipt in value['runtime_sources'].items():
        require(h.verified(root,receipt) == h.safe(root,'runtime/'+name) and
            binding['runtime_sha256'][name] == receipt['sha256'],'Original runtime differs')
    require(set(value['runtime_sources']) == set(binding['runtime_sha256']) == {'run_fit.py','adapters.py','bookkeeping.py','common.py'},'Runtime closure differs')
    require(binding['callback_sha256'] == job['callback_sha256'] and
        binding['extension_sha256'] == job.get('session_transform',{}).get('sha256'),'Native callback/extension differs')
    h.verified(root,{'path':job['callback_module'],'sha256':job['callback_sha256']})
    for dependency,sha in binding['dependency_sha256'].items():h.verified(root,{'path':dependency,'sha256':sha})
    for name,key in (('selected.pt','selected_checkpoint'),('selected_VALID.pt','selected_VALID_export')):
        require(h.verified(root,value[key]) == folder/name,'Own selected authority differs')
    return value,folder


def accept_export(t,saved,fit,job,provider):
    require(saved['schema'] == 'own-selected-VALID-logits-v1' and saved['role'] == 'VALID' and saved['TEST_accessed'] is False and
        saved['job_id'] == job['id'] and saved['job_hash'] == fit['job_hash'] and saved['policy'] == job['policy'] and
        saved['selected_epoch'] == fit['selected_epoch'] and saved['selected_checkpoint'] == fit['selected_checkpoint'] and
        saved['validation'] == fit['selected_VALID'] and saved['cohort_id'] == job['cohort_id'], 'Own saved VALID authority differs')
    z,y,ids = saved['raw_logits'],saved['labels'],saved['ids']
    require(tuple(z.shape) == (3943,3) and z.dtype == t.float32 and tuple(y.shape) == tuple(ids.shape) == (3943,) and
        ids.dtype == y.dtype == t.long and t.unique(ids).numel() == 3943 and t.isfinite(z).all().item() and
        t.bincount(y,minlength=3).tolist() == [72,1919,1952], 'Complete finite original VALID required')
    for value,key in ((ids,'ids_sha256'),(y,'labels_sha256')):
        require(hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == provider['role_digests']['VALID'][key],
            'Actual VALID role identity/order/labels differ')


def predictions(logits):
    return [max(range(len(z)),key=lambda c:z[c]) for z in logits]  # first argmax tie, matching native source


def metrics(logits,labels,classes=3):
    require(len(logits) == len(labels) > 0 and all(len(z) == classes and all(math.isfinite(v) for v in z) for z in logits),
        'Aligned finite multiclass arrays required')
    pred = predictions(logits); losses = []
    for z,y in zip(logits,labels):
        require(type(y) is int and 0 <= y < classes,'Native class range differs')
        maximum = max(z); losses.append(math.log(sum(math.exp(v-maximum) for v in z))+(maximum-z[y]))
    per_class = []
    for cls in range(classes):
        positions = [i for i,y in enumerate(labels) if y == cls]; n = len(positions)
        per_class.append({'class':cls,'support':n,'correct':sum(pred[i] == cls for i in positions),
            'accuracy':statistics.mean(pred[i] == cls for i in positions) if n else None,
            'NLL':statistics.mean(losses[i] for i in positions) if n else None})
    supported = [r for r in per_class if r['support']]
    return {'support':len(labels),'correct':sum(a == b for a,b in zip(pred,labels)),
        'accuracy':statistics.mean(a == b for a,b in zip(pred,labels)),'NLL':statistics.mean(losses),
        'macro_accuracy':statistics.mean(r['accuracy'] for r in supported),
        'macro_NLL':statistics.mean(r['NLL'] for r in supported),'macro_supported_classes':len(supported),'per_class':per_class}


def errors(members,pool,labels):
    pred = [predictions(z) for z in members]; served = predictions(pool)
    correct = [any(p[i] == y for p in pred) for i,y in enumerate(labels)]
    witness = [any(all(z[i][c] > z[i][y] for z in members) for c in range(3) if c != y) for i,y in enumerate(labels)]
    return {'no_correct_member':sum(not v for v in correct),'pool_errors':sum(p != y for p,y in zip(served,labels)),
        'pool_lost_correct_alternative':sum(p != y and c for p,y,c in zip(served,labels,correct)),
        'strict_common_false_rival':sum(witness),'unanimous_wrong_argmax':sum(
            pred[0][i] != y and all(p[i] == pred[0][i] for p in pred) for i,y in enumerate(labels)),
        'pool_top_ties':sum(sum(v == max(z) for v in z) > 1 for z in pool),
        'member_top_ties':[sum(sum(v == max(z) for v in z) > 1 for z in member) for member in members]}


def repairs(reference,candidate,labels):
    a,b = predictions(reference['pool']),predictions(candidate['pool'])
    old = [predictions(z) for z in reference['members']]; new = [predictions(z) for z in candidate['members']]
    before = [any(p[i] == y for p in old) for i,y in enumerate(labels)]
    after = [any(p[i] == y for p in new) for i,y in enumerate(labels)]
    witness = lambda members:[any(all(z[i][c] > z[i][y] for z in members) for c in range(3) if c != y)
        for i,y in enumerate(labels)]
    old_witness,new_witness = witness(reference['members']),witness(candidate['members'])
    per_class = []
    for cls in range(3):
        positions = [i for i,y in enumerate(labels) if y == cls]
        per_class.append({'class':cls,'support':len(positions),
            'repairs':sum(a[i] != cls and b[i] == cls for i in positions),
            'harms':sum(a[i] == cls and b[i] != cls for i in positions)})
    return {'repairs':sum(x != y and z == y for x,z,y in zip(a,b,labels)),
        'harms':sum(x == y and z != y for x,z,y in zip(a,b,labels)),
        'new_correct_member_alternatives':sum(not x and y for x,y in zip(before,after)),
        'lost_correct_member_alternatives':sum(x and not y for x,y in zip(before,after)),
        'new_alternative_served':sum(not x and c and p == y for x,c,p,y in zip(before,after,b,labels)),
        'new_alternative_still_lost':sum(not x and c and p != y for x,c,p,y in zip(before,after,b,labels)),
        'new_no_correct_member':sum(x and not y for x,y in zip(before,after)),
        'reference_common_rival_acquires_correct_member':sum(w and c for w,c in zip(old_witness,after)),
        'reference_common_rival_served_repairs':sum(w and p == y for w,p,y in zip(old_witness,b,labels)),
        'new_strict_common_false_rivals':sum(not a and b for a,b in zip(old_witness,new_witness)),
        'cleared_strict_common_false_rivals':sum(a and not b for a,b in zip(old_witness,new_witness)),
        'per_class':per_class}


def summarize(rows,group,fields):
    result = []
    for name in dict.fromkeys(r[group] for r in rows):
        selected = sorted((r for r in rows if r[group] == name),key=lambda r:r['seed'])
        require([r['seed'] for r in selected] == SEEDS,'Exactly three complete seed rows required')
        result.append({group:name,'seeds':SEEDS,'summary':{k:{'values':[r[k] for r in selected],
            'mean':statistics.mean(r[k] for r in selected),'sample_SD':statistics.stdev(r[k] for r in selected)} for k in fields}})
    return result
