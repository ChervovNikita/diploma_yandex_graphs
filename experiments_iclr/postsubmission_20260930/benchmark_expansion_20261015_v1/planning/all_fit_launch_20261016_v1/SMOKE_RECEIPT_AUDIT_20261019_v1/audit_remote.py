"""Read-only host-side metadata audit. Invoked in memory behind the guarded route."""
import base64, collections, hashlib, json, sys, time, zlib
from pathlib import Path
START = time.time()
ROOT = Path(AUDIT_ROOT)
E = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
X = E / 'campaign_execution_20261016_v7'
cache = {}
errors = []
def require(ok, message):
    if not ok: raise ValueError(message)
def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
def fetch(path):
    path = Path(path).resolve(); require(path.is_relative_to(ROOT), 'Outside authorized project: '+str(path))
    key = str(path)
    if key not in cache:
        raw = path.read_bytes()
        cache[key] = ({'path':key,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},raw)
    return cache[key]
def pin(path): return fetch(path)[0]
def read(path): return json.loads(fetch(path)[1])
def verified(row):
    actual = pin(row['path'])
    require(all(actual[k] == row[k] for k in ('bytes','sha256')), 'Hash mismatch: '+row['path'])
    return read(row['path'])
def source(relative, expected):
    p=E/relative/'MANIFEST.json'; manifest=read(p)
    require(all(pin(p)[k]==expected[k] for k in ('bytes','sha256')), 'Frozen manifest mismatch '+relative)
    for row in manifest['files']:
        verified({**row,'path':str(p.parent/row['path'])}) if row['path'].endswith('.json') else require(all(pin(p.parent/row['path'])[k]==row[k] for k in ('bytes','sha256')), 'Source member mismatch '+row['path'])
    return {'manifest':pin(p),'members_verified':len(manifest['files'])}
source_checks={k:source(v, BINDINGS['sources'][k]) for k,v in [('execution','campaign_execution_20261016_v7'),('smoke','qualification/all_fit_smoke_20261016_v6'),('port','server51_campaign_port_v6')]}
sys.path.insert(0,str(E/'qualification/all_fit_smoke_20261016_v6'))
from validate import validate_result, serialized_pin
from support import CANONICAL_SHA
out={'host':AUDIT_HOST,'started':START,'sources':source_checks,'workers':{},'remote_mutations':0,'scientific_accuracy_claim':False,'pid_matches':[]}
for binding in BINDINGS['workers']:
    worker=binding['worker']
    if not worker.startswith(AUDIT_HOST+'/'): continue
    if AUDIT_WORKER and worker!=AUDIT_WORKER: continue
    plan=verified(binding['plan']); state_root=Path(plan['state_root'])
    require(plan['worker']==worker and plan['TEST_access'] is False and plan['full_training_uses_smoke_checkpoint'] is False and plan['smoke_updates']==2,'Plan scope mismatch')
    require(plan['source_manifest']==source_checks['execution']['manifest'],'Plan execution source differs')
    require(plan['smoke_source_manifest']==source_checks['smoke']['manifest'],'Plan smoke source differs')
    inventory=verified(plan['inventory'])
    require(inventory['source_manifest']==source_checks['smoke']['manifest'] and inventory['producer_manifest']==source_checks['port']['manifest'],'Inventory sources differ')
    require(inventory['canonical_plan']['sha256']==CANONICAL_SHA and inventory['class_substitution'] is False,'Canonical input mismatch')
    verified(inventory['canonical_plan'])
    canonical=inventory['canonical_fit_ids']
    require(len(canonical)==len(set(canonical))==6460 and canonical==sorted(canonical) and digest(canonical)==inventory['canonical_fit_ids_sha256']==plan['canonical_fit_ids_sha256'],'Canonical universe malformed')
    fits=[t for t in plan['tasks'] if t['kind']=='fit']; exports=[t for t in plan['tasks'] if t['kind']!='fit']
    rows={r['id']:r for r in inventory['fits'] if r['worker']==worker}
    require(len(fits)==len(set(t['id'] for t in fits))==len(rows) and set(rows)==set(t['id'] for t in fits),'Plan inventory coverage mismatch')
    status_path=state_root/'STATUS.json'; status=read(status_path)
    w={'plan':binding['plan'],'inventory':plan['inventory'],'assigned_fit_count':len(fits),'assigned_export_count':len(exports),'assigned_id_tokens':sorted(hashlib.sha256(t['id'].encode()).hexdigest() for t in fits),'assigned_export_tokens':sorted(hashlib.sha256(t['id'].encode()).hexdigest() for t in exports),'canonical_tokens':sorted(hashlib.sha256(x.encode()).hexdigest() for x in canonical),'canonical_fit_ids_sha256':digest(canonical),'counts':{},'nonpass':[],'audit_errors':[],'pass_validated':0,'pass_updates':0,'retained_prior_attempts':0,'status':{k:status.get(k) for k in ('phase','counts','updated','active_fit_ids','TEST_accessed')},'status_receipt':pin(status_path),'summary_present':(state_root/'SMOKE_SUMMARY.json').exists(),'global_marker_present':Path(plan['global_smoke_pass']).exists(),'training_barrier_present':(state_root/'TRAINING_BARRIER_ACCEPTED.json').exists()}
    counts=collections.Counter(); evidence=[]; known_dirs=set(); pass_native={}
    for task in fits:
        fid=task['id']; row=rows[fid]; token=hashlib.sha256(fid.encode()).hexdigest(); folder=state_root/'smokes'/token; known_dirs.add(token)
        try:
            require(task['worker']==worker,'Task worker mismatch')
            for k in ('runtime_job_hash','canonical_runtime_job_hash','science_contract_hash','descriptor','runtime_entrypoint'):
                require(task[k]==row[k],'Task/inventory differs: '+k)
            for k in ('descriptor','runtime_entrypoint'): verified(row[k]) if k=='descriptor' else require(all(pin(row[k]['path'])[q]==row[k][q] for q in ('bytes','sha256')),'Runtime entrypoint differs')
            state_path=folder/'STATE.json'
            if not state_path.exists():
                counts['pending']+=1; w['nonpass'].append({'id':fid,'status':'pending','state_present':False}); continue
            state=read(state_path); st=state['status']; counts[st]+=1
            require(state.get('id')==fid and state.get('worker')==worker and state.get('phase')=='smoke','State identity mismatch')
            if st!='PASS':
                item={'id':fid,'status':st,'attempt':state.get('attempt'),'state_receipt':pin(state_path),'failure':state.get('failure')}
                if state.get('native_result'):
                    n=verified(state['native_result']); f=n.get('failure') or {}
                    item.update(native_result=state['native_result'],native_failure={k:f.get(k) for k in ('type','message')} if isinstance(f,dict) else f,TRAIN_updates=n.get('TRAIN_updates'),child_pid=n.get('child_pid'))
                w['nonpass'].append(item)
            if st=='PASS':
                attempt=verified(state['attempt_result'])
                require({k:v for k,v in state.items() if k!='attempt_result'}==attempt,'STATE/ATTEMPT_RESULT mismatch')
                require(attempt['status']=='PASS' and attempt['exit_code']==0 and not attempt.get('unknown_partial_work') and not attempt.get('failure'),'Attempt not clean PASS')
                native=verified(attempt['native_result']); terminal=verified(attempt['terminal'])
                require(terminal.get('exit_code')==0 and not terminal.get('forced_kill') and not terminal.get('signals_forwarded'),'Terminal not clean exit zero')
                require(read(Path(attempt['output'])/'RESULT.json')==native,'Native archived/output result differ')
                validate_result(row,native,plan['inventory'],BINDINGS['sources']['smoke']['sha256'])
                require(native['source_manifest']==source_checks['smoke']['manifest'] and native['producer_manifest']==source_checks['port']['manifest'],'Native frozen source differs')
                require(native['host_config']==plan['host_config'],'Native host config mismatch')
                require(native.get('scientific_accuracy_claim') is False and native.get('full_training_admitted') is False and native.get('VALID_subsampled') is False and native['VALID'].get('TEST_accessed') is False,'Engineering scope mismatch')
                for key in ('actual_callback','actual_extension','host_config'):
                    if native.get(key):
                        require(all(pin(native[key]['path'])[q]==native[key][q] for q in ('bytes','sha256')),'Actual source receipt mismatch: '+key)
                evidence.append([fid,pin(state_path)['sha256'],state['attempt_result']['sha256'],attempt['native_result']['sha256'],attempt['terminal']['sha256']])
                pass_native[fid]=(native,state)
                w['pass_validated']+=1; w['pass_updates']+=native['TRAIN_updates']
            attempts=list((folder/'attempts').glob('[0-9][0-9][0-9][0-9]'))
            w['retained_prior_attempts']+=max(0,len(attempts)-1)
            for a in attempts:
                nr=a/'NATIVE_RESULT.json'
                if not nr.exists(): continue
                n=read(nr)
                if n.get('child_pid')==1495596:
                    match={'path':str(nr),'receipt':pin(nr),'native':n}
                    for name in ('TERMINAL.json','INVOCATION.json','ATTEMPT_RESULT.json','CHILD.json','GUARDIAN.json'):
                        if (a/name).exists():match[name]=read(a/name)
                    out['pid_matches'].append(match)
        except Exception as exc:
            w['audit_errors'].append({'id':fid,'error':type(exc).__name__+': '+str(exc)})
    unknown=[p.parent.name for p in (state_root/'smokes').glob('*/STATE.json') if p.parent.name not in known_dirs]
    require(not unknown,'Unknown smoke state directories')
    w['counts']=dict(counts); w['pass_receipt_chain_digest']=digest(sorted(evidence))
    w['status_counts_match_state_scan']=dict(counts)==status['counts']
    if w['summary_present']:
        summary=read(state_root/'SMOKE_SUMMARY.json'); w['summary_receipt']=pin(state_root/'SMOKE_SUMMARY.json')
        try:
            require(summary['status']=='PASS' and summary['worker']==worker and summary['inventory']==plan['inventory'],'Summary binding mismatch')
            require(summary['fit_count']==len(fits) and summary['TEST_accessed'] is False and summary['expected_fit_ids']==sorted(rows),'Summary scope mismatch')
            require(len(summary['results'])==len(rows) and len({r['id'] for r in summary['results']})==len(rows),'Summary duplicate/missing results')
            for r in summary['results']:
                n,s=pass_native[r['id']]
                require(r['result']==n and r['result_receipt']==s['native_result'] and r['attempt_result']==s['attempt_result'],'Summary receipt mismatch')
            w['summary_verified']=True
        except Exception as exc: w['audit_errors'].append({'summary_error':type(exc).__name__+': '+str(exc)})
    out['workers'][worker]=w
out['finished']=time.time(); out['elapsed_seconds']=out['finished']-START
out['files_hashed']=len(cache); out['bytes_hashed']=sum(p[0]['bytes'] for p in cache.values())
print(base64.b64encode(zlib.compress(json.dumps(out,separators=(',',':'),allow_nan=False).encode(),9)).decode())
