"""Select ordinary compact Git additions from existing inventory metadata.

No archives, tensors, training, deletion, or live-index changes. Existing tracked
files are retained by the parent tree; only new/changed selected files are read.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
HERE = ROOT / 'experiments_iclr/postsubmission_20260930/backup_before_repo_deletion_20261031_v1'
EX = 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/'
SOURCE = {'.py','.pyi','.sh','.bash','.yaml','.yml','.toml','.ini','.cfg','.r','.jl','.c','.cpp','.h','.hpp','.js','.ts','.sbatch','.patch','.diff'}
NOTES = {'.md','.rst','.tex','.bib','.sty','.cls','.bst'}
TEXT = SOURCE | NOTES | {'.json','.csv','.tsv','.txt','.jsonl'}
BAD_PARTS = {'.git','.venv','.tools','.cache','.gnnm_runtime','__pycache__','site-packages','site_packages','node_modules','industrial_uv','envs','provider','provider_cache','raw','processed','datasets','working_data','node_bases','withheld_inputs','development_inputs','transport_audit','deletion_transport','payloads','payload','chunks','dependencies','.tmp','tmp'}
SEMANTIC = re.compile(r'result|metric|score|summary|decision|conclusion|config|protocol|hparam|cost|converg|terminal|attempt|job|plan|failure|interrupt|selected|history|curve|validation|setting|readme|requirement|manifest|report|ledger|inventory|stop|state',re.I)


def classify(row):
    p=Path(row['path']); parts=set(p.parts); name=p.name.lower(); suffix=p.suffix.lower()
    if row.get('preservation_category')=='reproducible_runtime_or_vendor_git' or parts & BAD_PARTS:
        return None,'runtime_dataset_transport_or_bulk_payload'
    if 'industrial_runtime_image_root' in row['path'] or 'dependency_overlay' in row['path'] or '/providers/dgl_cpu/' in row['path']:
        return None,'runtime_image'
    if suffix not in TEXT and p.name not in {'LICENSE','Makefile','Dockerfile','.gitignore','STOP','STOP_REQUEST'}:
        return None,'binary_archive_log_or_other_nonselected_format'
    if name in {'.env','.netrc','credentials'} or 'credential' in name:
        return None,'private_data'
    # Preserve the final dispatch configurations; older generated jobs are
    # reproducible duplicates, while their source and research notes remain.
    if re.search(r'/generated[^/]*/',row['path']) and not row['path'].startswith(EX+'server51_campaign_port_v6/'):
        return None,'superseded_generated_planning_duplicate'
    if any(x in row['path'].lower() for x in ('/transport/','/command_audit/','/ssh_audit/')):
        return None,'transport_record'
    if suffix=='.jsonl':
        return None,'per_iteration_or_bulk_trace'
    size_exception=(row['path'].startswith(EX+'campaign_execution_20261016_v7/') and name.startswith('plan-') and row['bytes']<25*1024**2) or (name in {'result.json','results.json','final_results.json'} and row['bytes']<20*1024**2)
    if row['bytes']>2*1024**2 and not size_exception:
        return None,'oversized_generated_or_raw_record'
    if suffix in SOURCE:
        return 'source_and_configuration',None
    if suffix in NOTES:
        return 'notes_and_manuscript',None
    if suffix in {'.csv','.tsv'}:
        return 'score_or_analysis_table',None
    if suffix in {'.json','.jsonl','.txt'} and SEMANTIC.search(name):
        return 'scientific_settings_results_and_status',None
    if p.name in {'LICENSE','Makefile','Dockerfile','.gitignore','STOP','STOP_REQUEST'}:
        return 'project_metadata',None
    return None,'nonselected_generic_record'


def main():
    assert socket.gethostname()=='anogena-2-0'
    assert ROOT.resolve()==ROOT and (ROOT.stat().st_dev,ROOT.stat().st_ino)==(49,9257095286857617644)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().rstrip('\0').split('\0'))
    changed=set(subprocess.check_output(['git','diff','--name-only','HEAD','-z'],cwd=ROOT).decode().rstrip('\0').split('\0'))
    selected=[]; excluded=Counter(); excluded_bytes=Counter(); retained=0; oversized=[]
    for line in (HERE/'REFINED_INVENTORY.jsonl').open():
        row=json.loads(line); path=row['path']
        if path in tracked and path not in changed:
            retained+=1;continue
        category,reason=classify(row)
        if category:
            selected.append({'path':path,'bytes':row['bytes'],'category':category})
        else:
            excluded[reason]+=1;excluded_bytes[reason]+=row['bytes']
            if reason=='oversized_generated_or_raw_record' and SEMANTIC.search(Path(path).name):
                oversized.append({'path':path,'bytes':row['bytes']})
    selected.sort(key=lambda x:x['path'])
    result={'status':'CANDIDATE_REVIEW','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'tracked_paths_retained_by_parent_tree':retained,'selected_files':len(selected),'selected_bytes':sum(r['bytes'] for r in selected),'categories':dict(Counter(r['category'] for r in selected)),'excluded_counts':dict(excluded),'excluded_bytes':dict(excluded_bytes),'oversized_scientific_named_records':oversized,'files':selected}
    (HERE/'DIRECT_CANDIDATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'files','oversized_scientific_named_records'}},indent=2))


if __name__=='__main__':main()
