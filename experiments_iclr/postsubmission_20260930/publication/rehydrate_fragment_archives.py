"""Verify exact archived source bytes, or restore without overwriting changes."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path,PurePosixPath

def confined(root,relative):
    rel=PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Project-relative path required')
    path=root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes project')
    cursor=root
    for part in rel.parts:
        cursor/=part
        if cursor.is_symlink():raise ValueError('Source symlink prohibited')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--restore',action='store_true');args=parser.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent.parent
    index=json.loads((here/'RESTORATION_INDEX.json').read_text());prepared=[];paths=set();parts=0
    for row in index['archives']:
        relative=row['original_project_relative_path']
        if relative in paths:raise ValueError('Duplicate original source')
        paths.add(relative);fragments=[]
        for number,fragment in enumerate(row['fragments']):
            raw=confined(root,fragment['archive_project_relative_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=fragment['archive_sha256']:raise ValueError('Archive digest mismatch')
            wrapper=json.loads(raw)
            if wrapper['schema']!='exact-project-source-fragment-v1' or wrapper['encoding']!='base64-gzip-fragment':raise ValueError('Unsupported archive schema')
            if wrapper['part_index']!=number or wrapper['part_count']!=len(row['fragments']):raise ValueError('Fragment order/count mismatch')
            for key in ('original_project_relative_path','original_bytes','original_sha256'):
                if wrapper[key]!=row[key]:raise ValueError('Original source binding mismatch')
            payload=base64.b64decode(wrapper['payload'],validate=True)
            if hashlib.sha256(payload).hexdigest()!=wrapper['fragment_sha256']:raise ValueError('Fragment payload mismatch')
            fragments.append(payload);parts+=1
        value=gzip.decompress(b''.join(fragments))
        if len(value)!=row['original_bytes'] or hashlib.sha256(value).hexdigest()!=row['original_sha256']:raise ValueError('Original source digest mismatch')
        target=confined(root,relative)
        exists=target.exists()
        if args.restore and exists and (not target.is_file() or target.read_bytes()!=value):raise ValueError('Refuse to overwrite different source: '+relative)
        prepared.append((target,value,exists))
    restored=0
    if args.restore:
        for target,value,exists in prepared:
            if not exists:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(value)
                restored+=1
    print(json.dumps({'verified_original_sources':len(prepared),'verified_fragments':parts,'exact_source_bytes':sum(len(v) for _,v,_ in prepared),'restored_files':restored,'scientific_jobs_started':0}))
if __name__=='__main__':main()
