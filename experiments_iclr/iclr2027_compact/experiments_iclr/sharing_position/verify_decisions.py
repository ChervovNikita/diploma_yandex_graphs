"""Recalculate retained decisions and traces in the anonymous lite stage."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=('tied','untied_propagation','private_first','private_last')
STUDIES=(('roman_study','roman',5),('crossgraph_study','wikics',3),('crossgraph_study','actor',3))


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''): h.update(c)
 return h.hexdigest()


def require(ok,msg):
 if not ok: raise RuntimeError(msg)


def tensor_sha(x):
 x=np.ascontiguousarray(x)
 h=hashlib.sha256()
 h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(x.tobytes())
 return h.hexdigest()


def trace_selected(path,row):
 with path.open(newline='') as f: lines=list(csv.DictReader(f))
 require(len(lines)==300,'trace length '+str(path))
 ba,bc,be=-1.,float('inf'),0
 for epoch,line in enumerate(lines,1):
  require(int(line['epoch'])==epoch,'trace order '+str(path))
  a,c=float(line['valid_accuracy']),float(line['valid_ce'])
  better=a>ba+1e-12 or (abs(a-ba)<=1e-12 and c<bc-1e-12)
  require(int(line['improved'])==int(better),'trace flag '+str(path))
  if better: ba,bc,be=a,c,epoch
 require(be==row['selected_epoch'] and abs(ba-row['valid_accuracy'])<1e-6,'selected epoch '+str(path))


def check_study(folder,dataset,nseeds,derivations):
 base=ROOT/folder
 results=base/'results'/dataset
 manifest=json.loads((results/'source_manifest.json').read_text())
 audit=json.loads((results/'completion_audit.json').read_text())
 require(audit['status']=='COMPLETE_CUDA_REPLAY_PASS','full audit status '+dataset)
 require(audit['source_manifest_sha256']==sha(results/'source_manifest.json'),'source manifest '+dataset)
 for name,digest in manifest['source_sha256'].items():
  if name.startswith('data/'):
   require(len(digest)==64,'public data hash syntax')
   continue
  require(sha(base/name)==digest,'source hash '+name)
 require(set(p.name for p in results.iterdir() if p.is_dir())=={f'seed{s}' for s in range(nseeds)},
         'seed directory set '+dataset)
 desc=manifest['data_descriptor']
 sizes=desc['split_sizes']; classes=desc['num_classes']
 scores={}
 for seed in range(nseeds):
  seed_dir=results/f'seed{seed}'
  require(set(p.name for p in seed_dir.iterdir() if p.is_dir())==set(ARMS),'arm set '+str(seed_dir))
  tied_init=json.loads((seed_dir/'tied/initialization.json').read_text())
  for arm in ARMS:
   run=seed_dir/arm
   row=json.loads((run/'result.json').read_text())
   require(row['dataset']==dataset and row['optimization_seed']==seed and row['arm']==arm and
           row['epochs_run']==300 and row['source_manifest_sha256']==sha(results/'source_manifest.json'),
           'run identity '+str(run))
   for name in ('validation_trace.csv','initialization.json'):
    require(sha(run/name)==row['artifact_sha256'][name],'retained artifact '+str(run/name))
   init=json.loads((run/'initialization.json').read_text())
   for name in ('canonical_tied_state_sha256','cpu_rng_sha256','cuda_rng_sha256'):
    require(init[name]==tied_init[name],'matched start '+str(run))
   require(init['paired_initial_logits_max_abs_diff']<=1e-5,'initial member equality '+str(run))
   trace_selected(run/'validation_trace.csv',row)
   rel=str(Path(folder)/'results'/dataset/f'seed{seed}'/arm)
   provenance=derivations[rel]
   require(provenance['source_selected_logits_sha256']==row['artifact_sha256']['selected_predictions.npz'],
           'source logit digest '+rel)
   require(sha(run/'selected_decisions.npz')==provenance['selected_decisions_sha256'],
           'decision digest '+rel)
   with np.load(run/'selected_decisions.npz',allow_pickle=False) as z:
    d={key:z[key].copy() for key in z.files}
   for part in ('valid','test'):
    n=sizes[part]
    require(d[f'{part}_member_pred'].shape==(4,n) and
            d[f'{part}_pooled_pred'].shape==(n,) and
            d[f'{part}_labels'].shape==(n,) and
            d[f'{part}_indices'].shape==(n,),
            'decision shape '+rel)
    require(int(d[f'{part}_labels'].max())<classes and
            int(d[f'{part}_member_pred'].max())<classes and
            int(d[f'{part}_pooled_pred'].max())<classes,
            'decision class range '+rel)
    require(tensor_sha(d[f'{part}_indices'].astype(np.int64))==
            desc['fingerprints_sha256'][f'{part}_indices'],
            'official IDs '+rel)
    accuracy=float(np.mean(d[f'{part}_pooled_pred']==d[f'{part}_labels']))
    require(abs(accuracy-row[f'{part}_accuracy'])<1e-6,
            'pooled accuracy '+rel)
    if part=='test':
     member=float(np.mean(d['test_member_pred']==d['test_labels'][None,:]))
     full=next(x for x in audit['records'] if x['seed']==seed and x['arm']==arm)
     require(abs(member-full['mean_member_test_accuracy'])<1e-6 and
             abs(accuracy-full['test_accuracy'])<1e-6,'audit score '+rel)
     scores[seed,arm]=(accuracy,row['parameter_count'])
 differences=[]
 for seed in range(nseeds):
  first,pf=scores[seed,'private_first']; last,pl=scores[seed,'private_last']
  require(pf==pl,'equal private parameter count '+dataset)
  differences.append(first-last)
 expected=audit['paired_differences']['private_first_minus_private_last']['differences']
 require(np.max(np.abs(np.array(differences)-expected))<1e-6,'paired contrast '+dataset)
 return {'dataset':dataset,'runs':4*nseeds,'private_first_minus_private_last_pp':[100*x for x in differences]}


def main():
 derivations=json.loads((ROOT/'decision_derivation_manifest.json').read_text())
 report=[check_study(*spec,derivations) for spec in STUDIES]
 manifest=json.loads((ROOT/'evidence_manifest.json').read_text())['sha256']
 for rel,digest in manifest.items():
  require(sha(ROOT/rel)==digest,'stage hash '+rel)
 found={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'}
 require(found==set(manifest),'stage manifest file set')
 print(json.dumps({'status':'LITE_DECISION_AUDIT_PASS','datasets':report},sort_keys=True))

if __name__=='__main__':main()
