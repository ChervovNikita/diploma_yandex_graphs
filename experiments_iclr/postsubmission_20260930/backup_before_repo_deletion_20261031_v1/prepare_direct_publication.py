"""Prepare a browsable Git tree of compact source and scientific records.

Scans only selected new/changed plaintext. Does not publish, change the live
index, delete source files, inspect tensors, or launch scientific jobs.
"""
from collections import Counter
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import preserve_records_v7 as scan
from prepare_direct_manifest_v2 import ROOT,HERE,EX


def git(*args,data=None,env=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=data,env=env)


def write(name,value):
    path=HERE/name
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return path


def digest(raw):return hashlib.sha256(raw).hexdigest()


def blob_id(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def main():
    assert socket.gethostname()=='anogena-2-0'
    assert ROOT.resolve()==ROOT and (ROOT.stat().st_dev,ROOT.stat().st_ino)==(49,9257095286857617644)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (HERE/'DIRECT_FILES_MANIFEST.json').exists()
    for i in range(3):
        value=getpass.getpass('Private scan literal %d/3 (not saved): '%(i+1))
        if value:scan.PRIVATE_LITERALS.append(value.encode())
    candidate=json.loads((HERE/'DIRECT_CANDIDATE.json').read_text())
    head=git('rev-parse','HEAD').decode().strip()
    assert head==candidate['base_commit']
    index_path=ROOT/'.git/index'
    index_before=digest(index_path.read_bytes())
    symbolic=git('symbolic-ref','HEAD').decode().strip()
    tracked=set(git('ls-files','-z').decode().rstrip('\0').split('\0'))
    rows={r['path']:r for r in candidate['files']}
    smoke=ROOT/EX/'campaign_execution_20261016_v7/runs/allocation_gpu0/SMOKE_SUMMARY.json'
    rows[str(smoke.relative_to(ROOT))]={'path':str(smoke.relative_to(ROOT)),'bytes':smoke.stat().st_size,'category':'current_campaign_verification'}
    # Add compact late shutdown status and scientific invocation records. The
    # original inventory predates the explicitly requested shutdown.
    full=ROOT/EX/'campaign_execution_20261016_v7/runs/allocation_gpu0/full'
    for pattern in ('*/STATE.json','*/JOB.json','*/attempts/*/JOB.json','*/attempts/*/INVOCATION.json','*/attempts/*/ATTEMPT_RESULT.json','*/attempts/*/TERMINAL.json','*/attempts/*/output/RESULT.json','*/attempts/*/output/CONFIG.json'):
        for p in full.glob(pattern):
            rel=str(p.relative_to(ROOT))
            rows[rel]={'path':rel,'bytes':p.stat().st_size,'category':'current_campaign_score_config_and_status'}
    for name in ('USER_SHUTDOWN.json','DIRECT_GIT_SCOPE_STOP.json','prepare_direct_manifest_v2.py','prepare_direct_publication.py','preserve_records_v7.py','GIT_REMOTE_PREFLIGHT.json'):
        p=HERE/name;rel=str(p.relative_to(ROOT));rows[rel]={'path':rel,'bytes':p.stat().st_size,'category':'direct_publication_provenance'}
    records=[];changes=[];coverage=[];unique={};read_started=time.time()
    redacted_dir=HERE/'direct_redacted';redacted_dir.mkdir()
    materialization=[]
    for i,r in enumerate(sorted(rows.values(),key=lambda x:x['path'])):
        p=ROOT/r['path']
        stream,info=scan.open_record(p)
        with stream:raw=stream.read()
        clean,rules=scan.clean_bytes(raw,r['path'])
        source=digest(raw);sha=digest(clean);oid=blob_id(clean)
        mode='100755' if info.st_mode & 0o111 else '100644'
        published_source=p
        if rules:
            published_source=redacted_dir/sha
            if not published_source.exists():published_source.write_bytes(clean)
            changes.append({'path':r['path'],'rules':rules,'source_sha256':source,'published_sha256':sha})
        item={'path':r['path'],'category':r['category'],'source_bytes':len(raw),'source_sha256':source,'bytes':len(clean),'sha256':sha,'git_blob':oid,'mode':mode,'redaction_rules':rules}
        records.append(item);unique[oid]=len(clean);materialization.append((str(published_source),oid))
        if r['path'].startswith(EX+'campaign_execution_20261016_v7/runs/allocation_gpu0/full/') and p.name=='STATE.json':
            state=json.loads(raw);relbase=p.parent.relative_to(ROOT)
            result_paths=sorted(str(q.relative_to(ROOT)) for q in p.parent.glob('attempts/*/output/RESULT.json'))
            config_paths=sorted(str(q.relative_to(ROOT)) for q in p.parent.glob('attempts/*/JOB.json'))
            if not config_paths:config_paths=sorted(str(q.relative_to(ROOT)) for q in p.parent.glob('attempts/*/INVOCATION.json'))
            coverage.append({'job_hash':p.parent.name,'job_id':state.get('id'),'state':r['path'],'status':state.get('status'),'failure':state.get('failure'),'score_records':result_paths,'configuration_records':config_paths,'attempt':state.get('attempt'),'exit_code':state.get('exit_code'),'finished':state.get('finished')})
        if (i+1)%1000==0:print(json.dumps({'scanned_files':i+1,'elapsed_seconds':round(time.time()-read_started,1)}),flush=True)
    preserved=set(rows)|tracked
    for row in coverage:
        assert all(p in preserved for p in row['score_records']+row['configuration_records'])
        if row['status']=='complete':assert row['score_records'] and row['configuration_records'],row['job_hash']
    # Batch Git object storage uses ordinary filenames, never executable source
    # interpretation. Redacted files are separate; originals remain untouched.
    input_paths='\n'.join(p for p,oid in materialization)+'\n'
    assert all('\n' not in p and '"' not in p for p,oid in materialization)
    oids=git('hash-object','-w','--stdin-paths',data=input_paths.encode()).decode().splitlines()
    assert oids==[oid for p,oid in materialization]
    refs=[]
    for line in git('for-each-ref','--format=%(objectname) %(objecttype) %(refname)').decode().splitlines():
        oid,kind,name=line.split(' ',2);assert kind=='commit';refs.append({'ref':name,'commit':oid})
    tips=sorted({r['commit'] for r in refs})
    parents=git('merge-base','--independent',*tips).decode().splitlines()
    manifest={'status':'PREPARED_FOR_DIRECT_PUBLICATION','schema':'direct-source-results-git-manifest-v1','base_commit':head,'live_symbolic_head':symbolic,'live_index_sha256':index_before,'original_refs':refs,'preservation_parents':parents,'source_files_unchanged':True,'selected_files':len(records),'selected_bytes':sum(r['bytes'] for r in records),'unique_blob_count':len(unique),'unique_blob_bytes':sum(unique.values()),'tracked_paths_retained_by_parent_tree':candidate['tracked_paths_retained_by_parent_tree'],'categories':dict(Counter(r['category'] for r in records)),'redacted_files':len(changes),'credential_values_saved':False,'excluded_counts':candidate['excluded_counts'],'excluded_bytes':candidate['excluded_bytes'],'campaign_scope':'allocation_gpu0 partition only; not global campaign totals','campaign_status_counts':dict(Counter(r['status'] for r in coverage)),'campaign_coverage':coverage,'files':records,'limitations':['Raw model states, prediction arrays, data, installed runtimes, archives, generic logs and per-iteration JSONL are excluded at user request.','Historical Git commits and unchanged tracked files are inherited without rewriting.','Large derived planning dumps and geometry arrays encoded in JSON are excluded; compact scores, configurations and conclusions are selected.']}
    write('DIRECT_REDACTIONS.json',changes)
    write('DIRECT_FILES_MANIFEST.json',manifest)
    env=dict(os.environ,GIT_INDEX_FILE=str(HERE/'direct-publication.index'))
    assert not (HERE/'direct-publication.index').exists()
    git('read-tree',head,env=env)
    for name in ('DIRECT_FILES_MANIFEST.json','DIRECT_REDACTIONS.json'):
        p=HERE/name;raw=p.read_bytes();oid=git('hash-object','-w','--',str(p)).decode().strip();records.append({'path':str(p.relative_to(ROOT)),'git_blob':oid,'mode':'100644'})
    entries='\n'.join(r['mode']+' '+r['git_blob']+'\t'+r['path'] for r in records)+'\n'
    git('update-index','--index-info',env=env,data=entries.encode())
    tree=git('write-tree',env=env).decode().strip()
    assert git('rev-parse','HEAD').decode().strip()==head and digest(index_path.read_bytes())==index_before
    write('DIRECT_PREPARED_TREE.json',{'status':'PASS','tree':tree,'files':len(records),'manifest_sha256':digest((HERE/'DIRECT_FILES_MANIFEST.json').read_bytes()),'live_head_unchanged':True,'live_index_unchanged':True})
    print(json.dumps({'status':'READY_FOR_PARENT_REVIEW','files':len(records),'unique_blob_bytes':sum(unique.values()),'campaign_status_counts':manifest['campaign_status_counts'],'redacted_files':len(changes),'tree':tree}),flush=True)


if __name__=='__main__':main()
