"""Verify all staged bytes after host synchronization; never start a child."""
import argparse
import json
from pathlib import Path
from host_binding import config_path,load_config,guard,verify,require,pin

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',required=True);parser.add_argument('--staging-receipts',required=True);parser.add_argument('--staging-receipts-sha256',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();config,_=load_config(config_path(args.worker));guard(config)
 root=Path(config['project_root']);stage_pin=pin(args.staging_receipts)
 require(stage_pin['sha256']==args.staging_receipts_sha256,'Exact reviewed staging receipt pin required')
 stage=verify(stage_pin,root)
 require(stage['host']['host']==args.worker.split('/')[0],'Wrong staged host')
 for row in stage['staged_files']:verify(row['execution'],root,parse_json=False)
 release=verify(stage['release'],root)
 for row in release['source_files']:verify(row,root,parse_json=False)
 from binding_rules import json_pin,validate_job_binding
 candidates=verify(stage['candidates'],root);by_id={t['id']:t for t in release['tasks']}
 fits=0
 for row in candidates:
  if row['job_descriptor'] is not None:
   job=json_pin(row['job_descriptor'],root);validate_job_binding(by_id[row['id']],job,root);fits+=1
 result={'schema':'allocation-server51-deployment-byte-verification-v1','status':'PASS','worker':args.worker,'host':stage['host'],
  'release':stage['release'],'source_files':len(release['source_files']),'execution_descriptors':fits,
  'destination_bytes_verified':True,'native_qualification_granted':False,'claims_created':0,'children_started':0,'human_GO':False}
 output=Path(args.output).resolve();require(output.is_relative_to(root) and not output.exists(),'Fresh own output required')
 with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
