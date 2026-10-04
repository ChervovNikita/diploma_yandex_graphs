"""Local stdlib receipt/projection audit. Never import or execute target code."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import collections
import hashlib
import json
import re
import shlex

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE / 'graph_count_conditioned_train_support_census_execution_root_20261004_v2'
SOURCE = PHASE / 'graph_count_conditioned_train_support_census_preparation_20261004_v3'
REMOTE_PHASE = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930'
REPO = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
REMOTE_ROOT = REMOTE_PHASE + '/' + ROOT.name
PIN = 'e6c7f6bc64e9d3454f2a7881acb16f493ec00b2126de0397b62c7b64a8b064e2'
inputs = {}
checks = []


def read_bytes(p, role):
    assert not p.is_symlink(), p
    raw = p.read_bytes()
    relative = str(p.relative_to(PHASE))
    inputs[relative] = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), role=role)
    return raw


def read_json(p, role='existing_JSON_receipt'):
    return json.loads(read_bytes(p, role))


def digest(p):
    return hashlib.sha256(read_bytes(p, 'local_source_or_metadata_pin')).hexdigest()


def check(name, condition, detail=None):
    checks.append(dict(check=name, passed=bool(condition), detail=detail))
    assert condition, name


def pin(p, row):
    raw = read_bytes(p, 'declared_local_metadata_pin')
    check('local descriptor: ' + str(p.relative_to(PHASE)), len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'])


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


manifest = read_json(SOURCE / 'MANIFEST.json')
check('reviewed source manifest', digest(SOURCE / 'MANIFEST.json') == PIN)
for row in manifest['files']:
    pin(SOURCE / row['path'], row)
plan = read_json(SOURCE / 'PLAN.json')
release = read_json(ROOT / 'ROOT_RELEASE.json')
launch = read_json(ROOT / 'DETACHED_LAUNCH.json')
admission = read_json(ROOT / 'ROOT_ADMISSION.json')
obs = read_json(ROOT / 'owned_monitor01/OBSERVATION.json')
terminal = read_json(ROOT / 'owned_monitor01/supervision/run01/TERMINAL.json')
physical = read_json(ROOT / 'owned_monitor01/supervision/run01/PHYSICAL_TERMINAL.json')
super_custody = read_json(ROOT / 'owned_monitor01/supervision/run01/SUPERVISOR_CUSTODY.json')
file_custody = read_json(ROOT / 'owned_monitor01/run01/FILE_CUSTODY.json')
final_custody = read_json(ROOT / 'owned_monitor01/run01/FINAL_CUSTODY.json')
census = read_json(ROOT / 'owned_monitor01/run01/CENSUS.json')
projection = read_json(ROOT / 'FULL_CENSUS_COMPACT_PROJECTION.json')
root_aggregate = read_json(ROOT / 'FULL_CENSUS_AGGREGATE.json')
release_sha = digest(ROOT / 'ROOT_RELEASE.json')
terminal_sha = digest(ROOT / 'owned_monitor01/supervision/run01/TERMINAL.json')
check('original supervisor launch identity', launch['supervisor_PID'] == 3219898 and launch['supervisor_start_ticks'] == 1725006651)
check('original command source and release', launch['command'][2].endswith(SOURCE.name + '/supervise.py') and launch['command'][-1] == release_sha)
check('release source and plan', release['source_manifest_sha256'] == PIN and release['plan_sha256'] == digest(SOURCE / 'PLAN.json') and release['status'] == 'APPROVED')
check('authorized streams and masks', admission['seeds'] == [0,1,2] and admission['full_batches_per_seed'] == 17 and admission['query_rows_per_population_per_batch'] == 65536)
for name, receipt in [('physical', physical), ('terminal', terminal), ('census', census), ('file_custody', file_custody)]:
    check(name + ' source/release pins', receipt['source_manifest_sha256'] == PIN and receipt['release_sha256'] == release_sha)
check('clean observed physical closure', physical['physical_exit_code'] == 0 and physical['physical_session_closed'] and physical['direct_child_reaped'] and not physical['unresolved_cleanup'] and not physical['ownership_relinquished_on_supervisor_exit'] and not physical['errors'] and physical['stop'] is None and not physical['received_supervisor_signals'] and not physical['termination_actions'])
check('physical receipt inherited byte-derived fields', all(terminal[k] == v for k,v in physical.items()))
check('child identity and held observed exit', physical['identity']['pid'] == 3219899 and physical['identity']['start_ticks'] == 1725006669 and physical['identity']['group'] == 3219899 and physical['identity']['session'] == 3219899 and physical['exit_observed_unreaped'] == dict(code=1,pid=3219899,status=0))
check('complete collected status', terminal['status'] == 'COMPLETE_TRAIN_METADATA_ONLY' and terminal['linked']['status'] == 'COLLECTED_COMPLETE_TRAIN_METADATA_ONLY' and super_custody['status'] == terminal['status'] and census['status'] == 'COMPLETE_TRAIN_CENSUS_ONLY')
check('physical and terminal custody links', terminal['physical_terminal_sha256'] == digest(ROOT / 'owned_monitor01/supervision/run01/PHYSICAL_TERMINAL.json') and super_custody['terminal_sha256'] == terminal_sha)
check('observation embeds exact collected objects', obs['result'] == census and obs['physical_terminal'] == terminal)
for row in obs['fetched_descriptors']:
    pin(ROOT / 'owned_monitor01' / row['path'], row)
check('final output custody completed', final_custody['completed'] and final_custody['file_custody_sha256'] == digest(ROOT / 'owned_monitor01/run01/FILE_CUSTODY.json'))
for key, filename in [('result','CENSUS.json'),('file_custody','FILE_CUSTODY.json'),('final_custody','FINAL_CUSTODY.json')]:
    check('linked ' + key, terminal['linked'][key+'_sha256'] == digest(ROOT / 'owned_monitor01/run01' / filename))
expected_outputs = {'CENSUS.json','FILE_CUSTODY.json','FINAL_CUSTODY.json','PROGRESS.json','seed0_COUNTS.json','seed1_COUNTS.json','seed2_COUNTS.json'}
check('exact seven worker descriptors', {r['path'] for r in super_custody['child_output_files']} == expected_outputs and len(super_custody['child_output_files']) == 7)
retained = {r['path']:r for r in obs['large_server_retained_metadata']}
for row in super_custody['child_output_files']:
    if row['path'].startswith('seed'):
        check('server-retained raw custody '+row['path'], retained['run01/'+row['path']] == dict(path='run01/'+row['path'],bytes=row['bytes'],sha256=row['sha256']))
    else:
        pin(ROOT / 'owned_monitor01/run01' / row['path'], row)
for row in final_custody['files']:
    check('final/supervisor output descriptor '+row['path'], row == next(r for r in super_custody['child_output_files'] if r['path'] == row['path']))
check('52 custody source/control/runtime descriptors', len(file_custody['files']) == 52 and len({r['path'] for r in file_custody['files']}) == 52 and file_custody['status'] == 'MATCH')
local_custody = []
for row in file_custody['files']:
    if row['path'].startswith(REMOTE_PHASE + '/'):
        relative = row['path'][len(REMOTE_PHASE)+1:]
        pin(PHASE / relative, row)
        local_custody.append(relative)
runtime = read_json(PHASE / plan['runtime_authority'])
runtime_rows = runtime['runtime_source_pins'] + runtime['runtime_binary_files'] + [runtime['negative_sampler']]
runtime_custody = {r['path']:r for r in file_custody['files'] if r['scope'] == 'runtime_file'}
check('runtime metadata descriptors agree with authority', set(runtime_custody) == {r['path'] for r in runtime_rows} and all(runtime_custody[r['path']]['sha256'] == r['sha256'] and ('bytes' not in r or runtime_custody[r['path']]['bytes'] == r['bytes']) for r in runtime_rows) and file_custody['interpreter_sha256'] == runtime['interpreter_sha256'] and file_custody['distribution_versions'] == runtime['distribution_versions'])
data_authority = read_json(PHASE / plan['data_authority'])
check('exact two authorized TRAIN metadata descriptors', len(file_custody['TRAIN_files']) == 2 and {r['relative_path'] for r in file_custody['TRAIN_files']} == {'split/time/train.pt','raw/edge.csv.gz'} and all(r['sha256'] == data_authority['files'][r['relative_path']]['sha256'] and r['bytes'] == data_authority['files'][r['relative_path']]['bytes'] for r in file_custody['TRAIN_files']))
check('TRAIN-only execution receipt', census['data_files_opened'] == ['split/time/train.pt','raw/edge.csv.gz'] and census['optimizer_updates'] == 0 and not census['VALID_TEST_read'] and not census['features_read'] and not census['learned_models'] and not admission['scientific_fit_admitted'])
check('resource observations below original declared caps', physical['wall_seconds'] < release['caps']['wall_seconds'] and physical['peak_observed_session_RSS_bytes'] < release['caps']['host_RSS_bytes'] and census['cuda_peak_allocated_bytes'] < release['caps']['cuda_allocated_bytes'] and census['cuda_peak_reserved_bytes'] < release['caps']['cuda_reserved_bytes'])

# Authenticate projection transport and exact code text, without running code.
projector_text = read_bytes(ROOT / 'project_completed_census.py','statically_reviewed_projector_source').decode()
projector_ast = ast.parse(projector_text)
literal_body = next(ast.literal_eval(n.value) for n in projector_ast.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id == 'code')
context = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\nrepo=Path('+repr(REPO)+');phase=Path('+repr(REMOTE_PHASE)+');root=Path('+repr(REMOTE_ROOT)+')\nassert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
expected_code = context + 'pins=' + repr(retained) + '\n' + literal_body
expected_command = 'cd ' + shlex.quote(REPO) + " && /usr/bin/python3 -I -S -B - <<'QAROOTPY'\n" + expected_code + '\nQAROOTPY\n'
command_path = PHASE / 'gpu77_connection_recovery_v1/count_census_v2_compact_integer_projection01_20261004.txt'
command_bytes = read_bytes(command_path,'original_recorded_projection_command')
check('original projection command matches reviewed source/pins', command_bytes == expected_command.encode())
transport = read_json(ROOT / 'count_census_v2_compact_integer_projection01_20261004_LOCAL_TRANSPORT.json')
outer = json.loads(transport['stdout'])
check('projection transport succeeded and pins command', transport['exit_code'] == 0 and transport['stderr'] == '' and outer['exit_code'] == 0 and outer['stderr'] == '' and transport['command_sha256'] == hashlib.sha256(command_bytes).hexdigest())
check('transported projection exact semantic equality', json.loads(outer['stdout']) == projection)
check('projection terminal and no-fit scope', projection['source_terminal_sha256'] == terminal_sha and projection['histogram_rows_recomputed_and_checked'] is True and projection['original_histograms_retained_server_side'] is True and projection['optimizer_updates'] == 0 and projection['models_or_features_or_VALID_TEST_read'] is False)

# Independent full projection consistency and aggregation.
check('all three ordered projected streams', [s['seed'] for s in projection['streams']] == [0,1,2])
populations = collections.defaultdict(list)
batch_rows = []
sha_re = re.compile(r'^[0-9a-f]{64}$')
for stream in projection['streams']:
    seed = stream['seed']
    check('raw pin to terminal seed '+str(seed), stream['source_descriptor'] == retained['run01/seed%d_COUNTS.json'%seed] and next(r['sha256'] for r in terminal['linked']['seed_files'] if r['path'] == REMOTE_ROOT+'/run01/seed%d_COUNTS.json'%seed) == stream['source_descriptor']['sha256'])
    check('17 ordered batches seed '+str(seed), [r['batch'] for r in stream['rows']] == list(range(1,18)))
    check('seed digest syntax '+str(seed), all(sha_re.fullmatch(stream[k]) for k in ['negative_draw_sha256','permutation_sha256','dropped_tail_sha256']))
    for batch in stream['rows']:
        check('exact populations seed/batch %d/%d'%(seed,batch['batch']), set(batch['populations']) == {'positive','negative'} and sha_re.fullmatch(batch['record_ids_sha256']))
        for name,pop in batch['populations'].items():
            q=pop['queries'];j=pop['joint_group_counts'];tag='%d/%d/%s'%(seed,batch['batch'],name)
            check('query/joint counts '+tag, q == 65536 and len(j)==3 and all(len(r)==3 for r in j) and all(type(v)==int and v>=0 for r in j for v in r) and sum(map(sum,j))==q)
            check('joint derived counters '+tag, pop['both_sides_nonconstant']==sum(j[a][b] for a in (1,2) for b in (1,2)) and pop['both_sides_genuine_subset']==j[2][2] and pop['either_side_genuine_subset']==sum(j[2])+sum(r[2] for r in j)-j[2][2])
            check('query/support digest syntax '+tag, sha_re.fullmatch(pop['query_digest']) and sha_re.fullmatch(pop['support_count_digest']))
            check('exact two sides '+tag,len(pop['sides'])==2)
            for index,side in enumerate(pop['sides']):
                check('side scalar types '+tag+'/'+str(index), all(type(v)==int and v>=0 for k,v in side.items() if k != 'group_counts_verified') and side['group_counts_verified'] is True)
                marg=[sum(j[g]) if index==0 else sum(r[g] for r in j) for g in range(3)]
                check('side partition/marginal '+tag+'/'+str(index), [side['zero_count']+side['full_count'],side['categorical_or_complement'],side['genuine_subset']]==marg and side['empty']<=side['zero_count'] and side['genuine_transition_work']<=side['n_times_r_sum'] and side['max_r']<=side['max_k'] and side['max_k']<=side['max_n'])
            populations[name].append(pop)
            batch_rows.append(dict(seed=seed,batch=batch['batch'],population=name,queries=q,
                both_sides_nonconstant=pop['both_sides_nonconstant'],both_sides_genuine_subset=pop['both_sides_genuine_subset'],
                either_side_genuine_subset=pop['either_side_genuine_subset'],
                total_slots=sum(s['total_slots'] for s in pop['sides']),n_times_r_sum=sum(s['n_times_r_sum'] for s in pop['sides']),
                genuine_groups=sum(s['genuine_groups'] for s in pop['sides']),genuine_group_slot_loops=sum(s['genuine_group_slot_loops'] for s in pop['sides']),
                genuine_transition_work=sum(s['genuine_transition_work'] for s in pop['sides'])))


def describe(values):
    return dict(min=min(values),max=max(values),sum=sum(values),mean=sum(values)/len(values))


aggregates={}
for name,pops in populations.items():
    q=sum(p['queries'] for p in pops)
    j=[[sum(p['joint_group_counts'][a][b] for p in pops) for b in range(3)] for a in range(3)]
    counts=dict(both_sides_nonconstant=sum(j[a][b] for a in (1,2) for b in (1,2)),both_sides_genuine_subset=j[2][2],either_side_genuine_subset=sum(j[2])+sum(r[2] for r in j)-j[2][2],either_side_nonconstant=q-j[0][0],both_categorical=j[1][1],coupling_with_genuine_subset=j[1][2]+j[2][1]+j[2][2],only_one_side_nonconstant=sum(j[0][1:])+j[1][0]+j[2][0],both_unique=j[0][0])
    fractions={k:v/q for k,v in counts.items()}
    side_aggregates=[]
    for index in (0,1):
        sides=[p['sides'][index] for p in pops]
        summed={k:sum(s[k] for s in sides) for k in ['empty','zero_count','full_count','categorical_or_complement','genuine_subset','total_slots','total_observed_bits','n_times_r_sum','genuine_groups','genuine_group_slot_loops','genuine_transition_work']}
        side_aggregates.append(dict(side=['left','right'][index],query_occurrences=q,counts=summed,
            fractions={k:summed[k]/q for k in ['empty','zero_count','full_count','categorical_or_complement','genuine_subset']},
            mean_slots=summed['total_slots']/q,mean_observed_bits=summed['total_observed_bits']/q,
            maximums={k:max(s[k] for s in sides) for k in ['max_n','max_k','max_r','distinct_n_r_groups','genuine_groups','maximum_genuine_group_queries','largest_group_rolling_count_cells']},
            per_batch={k:describe([s[k] for s in sides]) for k in ['total_slots','n_times_r_sum','genuine_groups','genuine_group_slot_loops','genuine_transition_work']}))
    selected=[r for r in batch_rows if r['population']==name]
    aggregates[name]=dict(query_occurrences=q,joint_group_counts=j,counts=counts,fractions=fractions,sides=side_aggregates,
        per_batch={k:describe([r[k] for r in selected]) for k in ['both_sides_nonconstant','both_sides_genuine_subset','either_side_genuine_subset','total_slots','n_times_r_sum','genuine_groups','genuine_group_slot_loops','genuine_transition_work']},
        per_seed=[dict(seed=s['seed'],**{k:sum(r['populations'][name][k] for r in s['rows']) for k in ['queries','both_sides_nonconstant','both_sides_genuine_subset','either_side_genuine_subset']}) for s in projection['streams']])
    recorded=next(r for r in root_aggregate['populations'] if r['population']==name)
    check('root aggregate exact counts '+name, recorded['query_occurrences']==q and all(recorded[k]==counts[k] for k in ['both_sides_nonconstant','both_sides_genuine_subset','either_side_genuine_subset']) and recorded['fractions']=={k:fractions[k] for k in recorded['fractions']})
    check('root aggregate per-batch ranges '+name,recorded['per_batch_both_nonconstant_range']==[min(p['both_sides_nonconstant']/p['queries'] for p in pops),max(p['both_sides_nonconstant']/p['queries'] for p in pops)])
    for recorded_key,side_key in [('per_side_max_genuine_groups','genuine_groups'),('per_side_max_genuine_slot_loops','genuine_group_slot_loops'),('per_side_max_genuine_transition_work','genuine_transition_work'),('per_side_max_slots','total_slots')]:
        check('root aggregate '+recorded_key+' '+name,recorded[recorded_key]==max(s[side_key] for p in pops for s in p['sides']))
check('full projection query totals', all(v['query_occurrences']==3342336 for v in aggregates.values()) and len(batch_rows)==102)
pooled_q=sum(v['query_occurrences'] for v in aggregates.values())
pooled=dict(query_occurrences=pooled_q,side_occurrences=pooled_q*2,
    total_slots=sum(s['counts']['total_slots'] for v in aggregates.values() for s in v['sides']),
    n_times_r_sum=sum(s['counts']['n_times_r_sum'] for v in aggregates.values() for s in v['sides']),
    counts={k:sum(v['counts'][k] for v in aggregates.values()) for k in aggregates['positive']['counts']})
pooled['fractions']={k:v/pooled_q for k,v in pooled['counts'].items()}
pooled['nonconstant_side_occurrences']=sum(s['counts']['categorical_or_complement']+s['counts']['genuine_subset'] for v in aggregates.values() for s in v['sides'])
pooled['nonconstant_side_fraction']=pooled['nonconstant_side_occurrences']/pooled['side_occurrences']
save('INPUTS.json',dict(schema='census-actual-audit-inputs-v1',files=list(inputs.values()),server_retained_raw_descriptors=list(retained.values()),server_raw_bytes_independently_read=False,target_code_executed=False,server_access=False))
save('CHECKS.json',dict(status='PASS_LOCAL_PROJECTION_AND_RECEIPT_CHECKS',checks=checks,check_count=len(checks),failed_checks=0,
    independent_histogram_recomputation=False,projection_source_statically_reviewed=True,projection_transport_and_command_authenticated=True))
save('AGGREGATE.json',dict(schema='independent-census-projection-aggregate-v1',populations=aggregates,pooled=pooled,
    batch_population_rows=batch_rows,query_occurrences_are_not_unique_records=True,independent_benchmark_graphs=1,
    cost_proxies_are_not_measured_training_runtime_or_memory=True))
save('OBSERVATION_SCOPE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS_SCOPED_ACTUAL_METADATA_AUDIT',
    source_manifest_sha256=PIN,release_sha256=release_sha,terminal_sha256=terminal_sha,
    original_supervisor=dict(pid=3219898,start_ticks=1725006651),original_child=dict(pid=3219899,start_ticks=1725006669),
    independent_raw_seed_rehash=False,raw_seed_bytes_retained_server_side=sum(r['bytes'] for r in retained.values()),
    evidence='Local receipt bytes/pins, reviewed exact projector command, successful metadata transport, and all 102 projected population receipts. Original histograms independently rehashed/recomputed by root projector, not reread by this auditor.',
    full_native_schedule_qualified=False,scientific_fit_admitted=False,
    observations=dict(census_wall_seconds=census['wall_seconds'],physical_wall_seconds=physical['wall_seconds'],sampled_child_session_peak_RSS_bytes=physical['peak_observed_session_RSS_bytes'],cuda_peak_allocated_bytes=census['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=census['cuda_peak_reserved_bytes']),
    exclusions=['No tensor/model/features/checkpoint reads','No server access or process probes','No scientific workloads, target execution, retries, or manuscript edits','Source scalar recomputation is independently reviewed, not independently rerun on raw histograms','Missing local STARTED/LAUNCH supervisor payloads are pinned by custody only; original supervisor birth identity is receipt evidence'],
    checked_local_custody_rows=len(local_custody),checked_runtime_metadata_rows=len(runtime_custody)))
print(json.dumps(dict(status='PASS_SCOPED_ACTUAL_METADATA_AUDIT',checks=len(checks),populations={k:dict(counts=v['counts'],fractions=v['fractions'],per_batch=v['per_batch'],sides=[dict(side=s['side'],counts=s['counts'],maximums=s['maximums']) for s in v['sides']]) for k,v in aggregates.items()},pooled=pooled),indent=2))
