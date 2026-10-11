"""Small stdlib source/custody helpers. No runtime admission at import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=P/'native_SAGE_Physics_confirmation_root_20261011_v1'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCKS=(8101,8219,8363,8501,8663,8819,9001,9199,9403,9623)
CORE=('BASE','E4','N4','equipped_M1','equipped_I4')

def require(condition,message):
    if not condition:raise ValueError(message)

def route():
    require(socket.gethostname()=='anogena-2-0' and Path.cwd()==R,'Literal authorized hostname/repository before project reads')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID],
            'Sole authorized GPU UUID required')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return h.hexdigest()

def scoped(path):
    path=Path(path).resolve();require(path.is_relative_to(P),'Pipeline project paths stay in authorized phase');return path

def item(path):
    path=Path(path).resolve();require(path.is_relative_to(R),'Receipt stays in authorized scientific repository')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}

def checked(value):
    require(isinstance(value,dict) and set(value)=={'path','bytes','sha256'},'Exact immutable file receipt')
    require(item(value['path'])==value,'Pinned bytes changed: '+str(value.get('path')))
    return Path(value['path']).resolve()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=scoped(path);require(not path.exists(),'One-shot fresh record only; no overwrite/retry')
    temp=path.with_suffix(path.suffix+'.tmp');require(not temp.exists(),'Partial record retained')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def source_seal(directory,expected):
    directory=scoped(directory);require(sha(directory/'SEAL.json')==expected,'Exact root-approved source seal')
    seal=read(directory/'SEAL.json')
    if 'files' in seal:
        for name,digest in seal['files'].items():require(sha(directory/name)==digest,'Source changed: '+name)
    else:
        require(sha(directory/'MANIFEST.json')==seal['manifest_sha256'],'Content manifest changed')
        files=read(directory/'MANIFEST.json')['files']
        records=files.items() if isinstance(files,dict) else ((r['path'],r) for r in files)
        for name,record in records:
            actual=item(directory/name)
            require(actual['bytes']==record['bytes'] and actual['sha256']==record['sha256'],'Source changed: '+name)

def load(name,receipt):
    path=checked(receipt);spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def freeze_recheck(value):
    for record in value['selected_states']+value['selected_archives']:
        checked({k:record[k] for k in ('path','bytes','sha256')})
    for key in ('core_complete','author_complete','cohort_owner_end'):checked(value[key])

def pipeline_gate(config_path,admission_path):
    route()
    cfg_path,adm_path=scoped(config_path),scoped(admission_path);cfg,adm=read(cfg_path),read(adm_path)
    here=scoped(Path(__file__).parent)
    require(adm.get('admitted') is True and adm.get('action')=='Physics_after_fit_one_shot'
            and adm.get('config_sha256')==sha(cfg_path) and adm.get('pipeline_seal_sha256')==sha(here/'SEAL.json')
            and adm.get('all_phases_pre_authorized') is True and adm.get('no_retries_or_recipe_changes') is True,
            'Root explicitly admits one whole fixed after-fit pipeline')
    source_seal(here,adm['pipeline_seal_sha256'])
    require(cfg.get('enabled') is True and cfg.get('root_directory')==str(H) and cfg.get('blocks')==list(BLOCKS)
            and cfg.get('core_arms')==list(CORE),'Disabled source template and fixed Physics root/roster')
    require(checked(cfg['protocol'])==H/'PROSPECTIVE_SCIENTIFIC_PROTOCOL_V1.json','Original root scientific protocol')
    require(read(checked(cfg['protocol']))['roles']['blocks']==list(BLOCKS),'Frozen ten paired masks')
    for value in cfg['source_receipts'].values():checked(value)
    for value in cfg['provider_modules'].values():checked(value)
    for directory,digest in cfg['stage_source_seals'].items():source_seal(directory,digest)
    limits=cfg['seconds']
    require(set(limits)=={'watch','export','join','calibrate','test'} and all(type(v) is int and v>0 for v in limits.values())
            and limits['watch']<=172800 and all(limits[k]<=3600 for k in ('export','join','calibrate','test')),
            'Root finite whole-watch/stage caps, chosen before outcomes')
    require(set(cfg['export_limits'])=={'wall_seconds','cuda_bytes','rss_bytes'}
            and cfg['export_limits']['wall_seconds']==limits['export']
            and all(type(v) is int and v>0 for v in cfg['export_limits'].values()),'Bound export wall/RSS/CUDA limits')
    require(cfg['cohort_owner_start']==adm['cohort_owner_start'],'Root exact fit owner start pin')
    checked(cfg['cohort_owner_start']);checked(cfg['prepared_receipt']);checked(cfg['core_integration_spec'])
    require(isinstance(cfg.get('prior_cost_receipts'),list),'Earlier preparation/qualification/failed-attempt costs retained')
    for value in cfg['prior_cost_receipts']:checked(value)
    require(sha(Path(cfg['readout_directory'])/'PROTOCOL.json')==cfg['protocol']['sha256'],
            'Same fixed statistical protocol in source-bound reader')
    require(cfg.get('cohort_owner_end')==str(H/'cohort_owner_v1/OWNER_END.json')
            and cfg.get('core_complete')==str(H/'core_acquisition_v1/COMPLETE_FAMILY.json')
            and cfg.get('author_complete')==str(H/'author_acquisition_v1/COMPLETE_AUTHOR_CONTEXT.json'),
            'Exact prospective complete paths, no live partial alternative')
    return cfg,adm,item(cfg_path),item(adm_path)
