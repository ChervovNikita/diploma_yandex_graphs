"""Catalogue actual final-master geometries; metadata, no resource callbacks."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from source_geometry import extract,source_branch


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def fingerprint(value):return hashlib.sha256(canonical(value)).hexdigest()
def pin(path):
    b=path.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def local(root,relative):
    q=Path(relative);p=(root/q).resolve()
    if q.is_absolute() or '..' in q.parts or not p.is_relative_to(root):raise ValueError('Project-relative metadata path required')
    return p


def catalogue(root,plan):
    records={};factory_pins={};fit_count=0;original_newdata=0
    placement={key:g['gpu_assignment'] for g in plan['groups'] for key in g['run_keys']}
    for bound in plan['studies']:
        p=local(root,bound['path']);raw=p.read_bytes();study=json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
        if fingerprint(study)!=bound['spec_hash']:raise ValueError('Current study spec changed')
        compact=''.join(c for c in study['dataset'].lower() if c.isalnum())
        if compact in {'romanempire','amazonratings','minesweeper','questions','tolokers'}:
            raise ValueError('Original paper datasets excluded for every method')
        for run in study['runs']:
            if run['kind']!='model_fit':continue
            fit_count+=1;original_newdata+=int(run['config'].get('is_original_gnnm',False))
            factory=run['adapter']
            if factory not in factory_pins:factory_pins[factory]={'path':factory.split(':',1)[0],**pin(local(root,factory.split(':',1)[0]))}
            branch=source_branch(run['config']);key=(study['study_id'],run['method'],factory,fingerprint(branch))
            if key not in records:
                records[key]={'study_id':study['study_id'],'dataset':study['dataset'],'method':run['method'],
                    'factory':factory,'factory_pin':factory_pins[factory],'source_branch':branch,
                    'fit_count':0,'actual_declared_geometries':{},'workers':set(),'budgets':set(),
                    'has_paid_ancestors':False,'resource_envelope_qualified_by_this_catalogue':False}
            row=records[key];row['fit_count']+=1;row['workers'].add(placement[study['study_id']+'/'+run['run_id']])
            geometry=extract(run['config']);gkey=fingerprint(geometry)
            if gkey not in row['actual_declared_geometries']:
                row['actual_declared_geometries'][gkey]={'geometry_id':gkey,'geometry':geometry,'representative_run_id':run['run_id'],'fits':0}
            row['actual_declared_geometries'][gkey]['fits']+=1;row['budgets'].add(canonical(run['budget']).decode())
            row['has_paid_ancestors']|=bool(run['parents'])
    output=[]
    for _,row in sorted(records.items()):
        row['actual_declared_geometries']=list(row['actual_declared_geometries'].values())
        row['workers']=sorted(row['workers']);row['budgets']=[json.loads(v) for v in sorted(row['budgets'])];output.append(row)
    return {'schema':'final-master-actual-source-geometries-v1','planned_fit_groups':fit_count,
            'original_GNNM_fit_groups_on_new_data':original_newdata,'source_branch_groups':len(output),'groups':output,
            'missing_common_dimensions_are_not_assumed1':True,'nested_native_dimensions_included':True,
            'source_data_runtime_dependency_state_equivalence_and_maximum_envelopes_need_separate_evidence':True,
            'scientific_fits_executed':0,'new_actual_input_callbacks':0,'whole_master_physical_admission':False,'GO':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--canonical-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation metadata route required')
    root=Path(a.project_root).resolve()
    if root!=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'):
        raise ValueError('Exact authorized research project root required')
    master=local(root,a.plan);plan=json.loads(master.read_text())
    if fingerprint(plan)!=a.canonical_sha256:raise ValueError('Final master canonical hash changed')
    result=catalogue(root,plan);result.update(plan={'path':a.plan,**pin(master)},plan_hash=a.canonical_sha256)
    output=local(root,a.output)
    if output.exists():raise ValueError('Fresh metadata result required')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups',)}))

if __name__=='__main__':main()
