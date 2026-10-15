"""Complete three-host-bundle intake and original scalar node diagnostics."""
import argparse,importlib.util,json,statistics,sys,time
from pathlib import Path
from binding import HERE,E,require,pin,digest,load_inventory,shard,relocated
from contracts import read,write,safe
def intake(directories,target_map,map_sha,root):
 require(len(directories)==3,'All three complete cohort bundles required');inventory=load_inventory();seen=set();accepted=[];global_ids=set()
 for directory in directories:
  folder=safe(directory,root);meta=read(folder/'BUNDLE.json');worker=meta['worker'];require(worker not in seen and worker in inventory['assignments'],'Duplicate/foreign worker bundle');seen.add(worker)
  rows=shard(inventory,worker);require(meta['schema']=='pubmed-portable-cohort-bundle-v1'and meta['status']=='complete'and meta['TEST_accessed']is False and meta['canonical_ids']==[r['id']for r in rows]and meta['seed']==inventory['assignments'][worker]and meta['target_map_sha256']==map_sha and meta['source_manifest']['sha256']==pin(HERE/'MANIFEST.json')['sha256']and meta['canonical_inventory_sha256']==pin(HERE/'CANONICAL.json')['sha256'],'Complete immutable matching cohort bundle required')
  artifacts={}
  for row in meta['files']:
   path=safe(folder/row['relative'],folder);now=pin(path);require(all(now[k]==row[k]==row['origin'][k]for k in('bytes','sha256')),'Copied evidence differs from preserved origin receipt');require(row['origin']['path']not in artifacts,'Duplicate evidence origin');artifacts[row['origin']['path']]=path
  predictions=safe(folder/meta['predictions']['relative'],folder);require(all(pin(predictions)[k]==meta['predictions'][k]for k in('bytes','sha256')),'Portable prediction payload changed')
  fits=read(folder/'FIT_OUTCOMES.json');end=read(folder/'FIT_END.json');collections=read(folder/'COLLECTION_OUTCOMES.json')
  require(end['status']=='complete'and end['outcomes']==fits and [v['id']for v in fits]==[r['id']for r in rows]and len(collections)==4 and all(v['status']=='complete'for v in fits)and [v['id']for v in collections]==[r['id']for r in rows if r['kind']!='native'],'Eight fit and four collection terminal identities required')
  host=target_map['workers'][worker];accepted_fits={};fit_outcomes={}
  for row,outcome in zip(rows,fits):
   expected,_=relocated(row,host,worker,target_map['queue_manifest_sha256'],map_sha)
   rp=artifacts[outcome['RESULT']['path']];result=read(rp);require(pin(rp)['sha256']==outcome['RESULT']['sha256'],'Fit result intake receipt differs')
   origin_job=str(Path(outcome['RESULT']['path']).with_name('JOB.json'));job=read(artifacts[origin_job])
   require(job==expected and result['job_hash']==digest(expected)and result['job_id']==row['id']and result['seed']==job['seed']and result['policy']==job['policy']and result['selected_epoch']is not None and result['status']=='complete'and result['TEST_accessed']is False and not result.get('failure')and not result.get('close_error'),'Copied fit canonical/execution/selection identity differs')
   for key in('selected_VALID_export',):
    receipt=result[key];require(receipt['path']in artifacts and all(pin(artifacts[receipt['path']])[k]==receipt[k]for k in('bytes','sha256')),'Own selected export bytes missing/changed')
   require(row['id']not in global_ids,'Duplicate physical fit across shards');global_ids.add(row['id'])
   accepted_fits[row['id']]=result;fit_outcomes[row['id']]=outcome
  for row,outcome in zip([r for r in rows if r['kind']!='native'],collections):
   collector_path=artifacts[outcome['RESULT']['path']];require(all(pin(collector_path)[k]==outcome['RESULT'][k]for k in('bytes','sha256')),'Collection terminal RESULT receipt differs')
   result=read(collector_path);expected,_=relocated(row,host,worker,target_map['queue_manifest_sha256'],map_sha);fit=accepted_fits[row['id']]
   require(result['job_id']==row['id']and result['job_hash']==digest(expected)and result['status']in('complete','complete_with_sanity_warning')and result['TEST_accessed']is False and result['actual_work']['TRAIN_callbacks']==result['actual_work']['optimizer_steps']==0 and result['actual_work']['native_forwards_completed']==4 and not result.get('close_error'),'Complete own-selected zero-training collection required')
   require(result['role']=='VALID'and result['selected_epoch']==fit['selected_epoch']and result['selected_checkpoint']==fit['selected_checkpoint']and result['saved_VALID_export']==fit['selected_VALID_export']and result['fit_RESULT']==fit_outcomes[row['id']]['RESULT']and result['members']==4 and result['VALID_nodes']==3943 and result['VALID_class_counts']==[72,1919,1952]and result['source']['sha256']==pin(E/'research/pubmed_cheb_transfer_driver_v1/collect.py')['sha256'],'Collection must bind exact accepted own fit/selected/source authority')
   receipt=result['raw_tensor_receipt'];require(receipt['path']in artifacts and all(pin(artifacts[receipt['path']])[k]==receipt[k]for k in('bytes','sha256')),'Selected member export bytes missing/changed')
  require(meta['collection_outcomes']==collections and [r['canonical_id']for r in meta['training_outcomes']]==[r['id']for r in rows]and all(r['result']==accepted_fits[r['canonical_id']]for r in meta['training_outcomes']),'Bundle cost/outcome records differ from accepted authorities')
  accepted.append((folder,meta,predictions,artifacts))
 require(seen==set(inventory['assignments'])and len(global_ids)==24,'All24 unique fits/12 collectors required before tensor/quality access')
 return accepted
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--target-map',required=True);p.add_argument('--target-map-sha256',required=True);p.add_argument('--bundle',action='append',required=True);p.add_argument('--output',required=True);args=p.parse_args();root=Path(args.root).resolve()
 ap=safe(args.target_map,root);require(pin(ap)['sha256']==args.target_map_sha256,'Exact approved map required');target=read(ap);require(target['approved']is True and target['TEST_access']is False,'Approved validation-only study map required');output=safe(args.output,root);require(not output.exists(),'Fresh analysis output required')
 accepted=intake(args.bundle,target,args.target_map_sha256,root);output.mkdir(parents=True);started=time.perf_counter()
 # Every24-fit/12-collection metadata and byte gate precedes tensor access.
 import torch
 torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('_pubmed_scalar_diagnostics',E/'research/pubmed_cheb_transfer_driver_v1/study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 banks=[];predictions={};labels_by_seed={};intake_records=[];warnings=[];reference_ids=None;reference_labels=None
 parent_inputs=read(HERE/'PARENT_INPUTS.json')
 for folder,meta,payload,artifacts in accepted:
  packet=torch.load(payload,map_location='cpu',weights_only=False);require(packet['schema']=='pubmed-portable-selected-six-banks-v1'and packet['worker']==meta['worker']and packet['seed']==meta['seed']and packet['TEST_accessed']is False and set(packet['banks'])==set(s.BANKS),'Portable six-bank schema differs')
  rows=shard(load_inventory(),meta['worker']);saved={};fit_by_id={};exported=read(folder/'FIT_OUTCOMES.json');collected=read(folder/'COLLECTION_OUTCOMES.json')
  for row,outcome in zip(rows,exported):
   fit=read(artifacts[outcome['RESULT']['path']]);job,_=relocated(row,target['workers'][meta['worker']],meta['worker'],target['queue_manifest_sha256'],args.target_map_sha256)
   raw=torch.load(artifacts[fit['selected_VALID_export']['path']],map_location='cpu',weights_only=False);s.accept_export(torch,raw,fit,job,parent_inputs['provider']);saved[row['id']]=raw;fit_by_id[row['id']]=fit
  native=sorted([r for r in rows if r['kind']=='native'],key=lambda r:r['job']['body']);body0=saved[native[0]['id']];i4=torch.load(artifacts[meta['I4_aggregation']['bank_receipt']['path']],map_location='cpu',weights_only=False);stack=torch.stack([saved[r['id']]['raw_logits']for r in native])
  require(torch.equal(packet['ids'],body0['ids'])and torch.equal(packet['labels'],body0['labels']),'Packet VALID IDs/labels must equal own authoritative body0 export')
  for name,values in packet['banks'].items():
   expected_rows=[native[0]]if name=='M1'else native if name=='I4'else[r for r in rows if r['kind']==name]
   require(values['canonical_ids']==[r['id']for r in expected_rows]and values['epochs']==[saved[r['id']]['selected_epoch']for r in expected_rows],'Packet canonical bank members/epochs must match own selected fit authorities')
  require(i4['M1_exact_member0_job_hash']==body0['job_hash']and i4['extra_training_fits']==0 and i4['pooled_epoch_reselection']is False and torch.equal(i4['member_raw_logits'],stack)and torch.equal(i4['raw_logits'],stack.mean(0)),'Portable I4 must reproduce its own four selected exports')
  require(torch.equal(packet['banks']['M1']['pool'],body0['raw_logits'])and torch.equal(packet['banks']['M1']['members'],body0['raw_logits'].unsqueeze(0))and torch.equal(packet['banks']['I4']['pool'],i4['raw_logits'])and torch.equal(packet['banks']['I4']['members'],i4['member_raw_logits']),'Portable native pools must equal saved scoring authorities')
  require(packet['banks']['M1']['saved_native_CE']==body0['validation']['loss']and packet['banks']['I4']['saved_native_CE']==i4['loss'],'Saved native/I4 CE authority differs')
  for row,outcome in zip([r for r in rows if r['kind']!='native'],collected):
   result=read(artifacts[outcome['RESULT']['path']]);raw=torch.load(artifacts[result['raw_tensor_receipt']['path']],map_location='cpu',weights_only=False);own=saved[row['id']];values=packet['banks'][row['kind']]
   require(result['selected_checkpoint']==fit_by_id[row['id']]['selected_checkpoint']and raw['job_hash']==own['job_hash']and raw['selected_epoch']==own['selected_epoch']and raw['selected_checkpoint']==own['selected_checkpoint']and torch.equal(raw['ids'],own['ids'])and torch.equal(raw['labels'],own['labels'])and torch.equal(values['pool'],own['raw_logits'])and torch.equal(values['members'],raw['member_raw_logits'])and values['saved_native_CE']==own['validation']['loss']and values['sanity']==result['sanity'],'Portable shared bank must preserve own selected pool and member authorities')
   if result['status']!='complete':warnings.append({'seed':meta['seed'],'bank':row['kind'],'sanity':result['sanity']})
  require(tuple(packet['ids'].shape)==(3943,)and torch.unique(packet['ids']).numel()==3943 and torch.bincount(packet['labels'],minlength=3).tolist()==[72,1919,1952],'Whole frozen VALID population required')
  if reference_ids is None:reference_ids,reference_labels=packet['ids'],packet['labels']
  require(torch.equal(packet['ids'],reference_ids)and torch.equal(packet['labels'],reference_labels),'Across-host VALID order/labels differ');seed=packet['seed'];labels=packet['labels'].tolist();labels_by_seed[seed]=labels
  for name,values in packet['banks'].items():
   require(tuple(values['pool'].shape)==(3943,3)and tuple(values['members'].shape)==((1 if name=='M1'else 4),3943,3)and torch.isfinite(values['pool']).all().item()and torch.isfinite(values['members']).all().item(),'Complete finite bank/member logits required')
   pool=values['pool'].tolist();members=values['members'].tolist();metrics=s.metrics(pool,labels);member=[s.metrics(z,labels)for z in members]
   row={'seed':seed,'bank':name,'pool':metrics,'members':member,'pool_accuracy':metrics['accuracy'],'pool_NLL':metrics['NLL'],'macro_accuracy':metrics['macro_accuracy'],'macro_NLL':metrics['macro_NLL'],'class0_accuracy':metrics['per_class'][0]['accuracy'],'class0_NLL':metrics['per_class'][0]['NLL'],'mean_member_accuracy':statistics.mean(v['accuracy']for v in member),'worst_member_accuracy':min(v['accuracy']for v in member),'mean_member_NLL':statistics.mean(v['NLL']for v in member),'worst_member_NLL':max(v['NLL']for v in member),'saved_native_CE':values['saved_native_CE'],'selected_epochs':values['epochs'],'canonical_ids':values['canonical_ids'],'host':meta['host'],**s.errors(members,pool,labels)}
   row['pool_minus_mean_member_accuracy_pp']=100*(row['pool_accuracy']-row['mean_member_accuracy'])
   if 'sanity'in values:row['sanity']=values['sanity']
   banks.append(row);predictions[(seed,name)]={'pool':pool,'members':members}
  intake_records.append({'bundle':pin(folder/'BUNDLE.json'),'predictions':pin(payload),'host':meta['host'],'training_outcomes':meta['training_outcomes'],'collection_outcomes':meta['collection_outcomes'],'I4_aggregation':meta['I4_aggregation']})
 require(len(banks)==18,'All18 banks required');pairs=[('G0',r)for r in('M1','I4')]+[(c,r)for c in('uniform','structured','permuted')for r in('M1','I4','G0')]+[('structured','uniform'),('structured','permuted'),('permuted','uniform')];contrasts=[]
 numeric=['pool_accuracy','pool_NLL','macro_accuracy','macro_NLL','class0_accuracy','class0_NLL','mean_member_accuracy','worst_member_accuracy','mean_member_NLL','worst_member_NLL','pool_minus_mean_member_accuracy_pp','no_correct_member','strict_common_false_rival']
 for seed in s.SEEDS:
  lookup={r['bank']:r for r in banks if r['seed']==seed};labels=labels_by_seed[seed]
  for candidate,reference in pairs:
   c,r=lookup[candidate],lookup[reference];value={'seed':seed,'comparison':candidate+'_minus_'+reference,'candidate':candidate,'reference':reference,**s.repairs(predictions[(seed,reference)],predictions[(seed,candidate)],labels)}
   for key in numeric:value['delta_'+key]=c[key]-r[key]
   value['delta_accuracy_pp']=100*(c['pool_accuracy']-r['pool_accuracy']);value['delta_NLL']=c['pool_NLL']-r['pool_NLL'];value['delta_saved_native_CE']=c['saved_native_CE']-r['saved_native_CE']
   value['per_class_signed']=[{'class':i,'support':c['pool']['per_class'][i]['support'],'delta_accuracy_pp':100*(c['pool']['per_class'][i]['accuracy']-r['pool']['per_class'][i]['accuracy']),'delta_NLL':c['pool']['per_class'][i]['NLL']-r['pool']['per_class'][i]['NLL']}for i in range(3)]
   require(abs(value['delta_accuracy_pp']-100*(value['repairs']-value['harms'])/3943)<1e-10,'Signed accuracy/repair accounting differs');contrasts.append(value)
 contrast_numeric=[k for k,v in contrasts[0].items()if isinstance(v,(int,float))and k!='seed'];result={'schema':'pubmed-three-GPU-complete-selected-analysis-v1','status':'complete_with_warnings'if warnings else'complete','warnings':warnings,'rows':banks,'summaries':s.summarize(banks,'bank',numeric),'contrasts':contrasts,'contrast_summaries':s.summarize(contrasts,'comparison',contrast_numeric),'intake':intake_records,'unique_training_fits':24,'selected_route_collections':12,'banks':18,'VALID_class_counts':[72,1919,1952],'TEST_accessed':False,'new_models':0,'new_optimizer_updates':0,'target_map':pin(ap),'source_manifest':pin(HERE/'MANIFEST.json'),'wall_seconds':time.perf_counter()-started,'scope':'Three paired optimizer cohorts on one exposed split across two hosts. Host and seed are coupled; retain each signed contrast. No protected confirmation, graph-level replication or broad superiority claim. Saved selected pools remain scoring authority; member reconstruction is diagnostic only.'}
 write(output/'RESULT.json',result)
 lines=['# PubMed/Cheb complete three-GPU study','',result['scope'],'','| Seed | Bank | Accuracy % | NLL | Macro accuracy % | Class0 accuracy % |','| --- | --- | ---: | ---: | ---: | ---: |']
 for r in banks:lines.append(f"| {r['seed']} | {r['bank']} | {100*r['pool_accuracy']:.4f} | {r['pool_NLL']:.5f} | {100*r['macro_accuracy']:.4f} | {100*r['class0_accuracy']:.4f} |")
 lines+=['',f"Collection sanity warnings retained: {len(warnings)}; see RESULT.json warnings and per-bank sanity."]
 (output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':result['status'],'fits':24,'banks':18,'output':str(output)}));return 0
if __name__=='__main__':raise SystemExit(main())
