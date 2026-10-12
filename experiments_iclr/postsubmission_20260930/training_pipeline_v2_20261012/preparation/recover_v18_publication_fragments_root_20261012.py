"""Build small publication fragments from the completed source snapshot; no SSH rerun."""
import base64,gzip,hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[2]
OLD=P/'publication/Master_v18_source_storage_20261012_v1'
OUT=P/'publication/Master_v18_source_storage_20261012_v2'
def main():
    observation=json.loads((OLD/'SNAPSHOT.json').read_text());assert observation['exit_code']==0
    value=json.loads(observation['stdout']);OUT.mkdir();archives=OUT/'archives';archives.mkdir();rows=[]
    for row in value['files']:
        packed=base64.b64decode(row['payload']);raw=gzip.decompress(packed)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        chunks=[packed[n:n+1_000_000] for n in range(0,len(packed),1_000_000)]
        target={'original_project_relative_path':row['path'],'original_bytes':row['bytes'],'original_sha256':row['sha256'],'fragments':[]}
        prefix=hashlib.sha256(row['path'].encode()).hexdigest()[:24]
        for n,chunk in enumerate(chunks):
            wrapper={k:v for k,v in target.items() if k!='fragments'}|{'schema':'exact-project-source-fragment-v1','encoding':'base64-gzip-fragment','part_index':n,'part_count':len(chunks),'fragment_sha256':hashlib.sha256(chunk).hexdigest(),'payload':base64.b64encode(chunk).decode()}
            path=archives/(prefix+'.'+str(n)+'.archive.json');path.write_text(json.dumps(wrapper,separators=(',',':'))+'\n')
            assert path.stat().st_size<2_000_000
            target['fragments'].append({'archive_project_relative_path':str(path.relative_to(P)),'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append(target)
    (OUT/'RESTORATION_INDEX.json').write_text(json.dumps({'archives':rows},indent=2)+'\n')
    (OUT/'ARCHIVE_INDEX.json').write_text(json.dumps({'original_sources':len(rows),'fragment_files':sum(len(r['fragments']) for r in rows),'source_bytes':sum(r['original_bytes'] for r in rows),'excluded_non_source_files':value['excluded_non_source_files'],'datasets_or_checkpoints_included':False,'all_original_paths_bytes_digests_preserved':True},indent=2)+'\n')
    shutil.copyfile(P/'publication/rehydrate_fragment_archives.py',OUT/'rehydrate_source_archives.py')
    (OUT/'README_PREDECESSOR.md').write_text(value['README'])
    (OUT/'SNAPSHOT_SUMMARY.json').write_text(json.dumps({'head':value['head'],'sources':len(rows),'source_bytes':sum(r['original_bytes'] for r in rows),'reused_snapshot':str((OLD/'SNAPSHOT.json').relative_to(P)),'no_remote_reads_or_hash_repeats':True,'scientific_fits':0,'GO':False},indent=2)+'\n')
    print(json.dumps({'sources':len(rows),'fragments':sum(len(r['fragments']) for r in rows),'source_bytes':sum(r['original_bytes'] for r in rows),'scientific_fits':0}))
if __name__=='__main__':main()
