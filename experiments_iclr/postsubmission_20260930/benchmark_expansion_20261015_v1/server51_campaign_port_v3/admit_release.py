"""Bind reviewed completed applicability witnesses; never launch or grant GO.

A binding map must explicitly cover every host-owned task. Historical dormant
qualification_blocks are not themselves failures; PASS is accepted only from a
pinned per-task evidence row with source/destination applicability stated.
"""
import argparse
import copy
import json
import importlib.util
import os
from pathlib import Path
from host_binding import config_path,load_config,guard,verify,pin,require
from descriptor_binding import digest

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for field in ('worker','release','bindings','output'):parser.add_argument('--'+field,required=True)
 args=parser.parse_args();config,config_pin=load_config(config_path(args.worker))
 os.environ['BENCHMARK_HOST_CONFIG']=config_pin['path'];os.environ['BENCHMARK_HOST_CONFIG_SHA256']=config_pin['sha256']
 guard(config);root=Path(config['project_root'])
 release_pin=pin(args.release);old=verify(release_pin,root);maps=verify(pin(args.bindings),root)
 require(old['host']==args.worker.split('/')[0] and old['all_tasks_qualified'] is False and old['human_GO'] is False,'Exact dormant host release required')
 require(maps['schema']=='allocation-server51-reviewed-task-bindings-v1' and maps['unadmitted_release']==release_pin,'Binding map must bind these exact staging bytes')
 require(set(maps['tasks'])=={t['id'] for t in old['tasks']},'Every host task needs one explicit reviewed applicability binding')
 release=copy.deepcopy(old);files={p['path']:p for p in old['source_files']}
 for evidence in maps['source_files']:
  verify(evidence,root,parse_json=False)
  require(evidence['path'] not in files or files[evidence['path']]==evidence,'Source/evidence receipt conflict');files[evidence['path']]=evidence
 contract_modules={}
 for worker in {t['worker'] for t in release['tasks']}:
  cfg,cp=load_config(config_path(worker));os.environ['BENCHMARK_HOST_CONFIG']=cp['path'];os.environ['BENCHMARK_HOST_CONFIG_SHA256']=cp['sha256']
  cs=importlib.util.spec_from_file_location('_server51_admission_'+worker.replace('/','_'),Path(__file__).with_name('contracts.py'))
  cm=importlib.util.module_from_spec(cs);cs.loader.exec_module(cm);contract_modules[worker]=cm
  _,legacy=cm.load_legacy(release['legacy_supervisor_sources'])
 for task in release['tasks']:
  binding=maps['tasks'][task['id']]
  cm=contract_modules[task['worker']]
  for gate in cm.GATES:
   evidence=verify(binding['gates'][gate],root)
   require(evidence.get('status')=='PASS' and evidence.get('task_id')==task['id'] and evidence.get('science_contract_hash')==task['science_contract_hash'],'Missing exact functional applicability: '+gate)
   require(evidence.get('applicability') and evidence.get('witnesses'),'Each gate must state its source/destination applicability and actual witnesses')
   for witness in evidence['witnesses']:verify(witness,root,parse_json=False)
  task['binding']=binding;task['execution_status']='qualified'
  cm.validate_binding(task,binding,{p:v['sha256'] for p,v in files.items()},legacy)
 release['all_tasks_qualified']=True;release['human_GO']=False;release['source_files']=list(files.values())
 release['source_bundle_sha256']=digest({p:v['sha256'] for p,v in files.items()});release['qualification_binding_map']=pin(args.bindings)
 output=Path(args.output).resolve();require(output.is_relative_to(root) and not output.exists(),'Fresh own qualified release output required')
 with output.open('x') as stream:json.dump(release,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps({'schema':'allocation-server51-qualified-release-preparation-v1','qualified_release':pin(output),'all_host_tasks_qualified':True,'human_GO':False,'claims_created':0,'children_started':0},sort_keys=True))
if __name__=='__main__':main()
