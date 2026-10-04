"""Copy exact reviewed source dependencies into the authorized project only."""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent;PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
HELPER=PHASE/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'

def main():
 p=argparse.ArgumentParser();p.add_argument('kind',choices=('native','single'));a=p.parse_args()
 if a.kind=='native':
  source=PHASE/'native_fixed_view_independent_control_source_20261005_v1/control.py'
  spec=importlib.util.spec_from_file_location('bounded_source_descriptors',source)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  records,_,_=m.verify_source_bindings()
  rows=[dict(v,path=str(Path(v['path']).relative_to(PHASE))) for v in records.values()]
 else:
  records=json.loads((PHASE/'joint_pattern_capable_single_control_source_20261005_v1/SOURCE_BINDINGS.json').read_text())['records']
  wanted={'pilot_model.py','pattern_teacher.py','conditional_loss.py','prototype.py','graph_ops.py','native_reference.py','model.py','utils.py','design_spec.py','cardinality_model.py'}
  rows=[r for r in records if Path(r['path']).name in wanted]
 payload=[]
 for row in rows:
  raw=(PHASE/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  encoded=base64.b64encode(raw).decode()
  payload.append(dict(path=row['path'],bytes=len(raw),sha256=row['sha256'],chunks=[encoded[i:i+36000] for i in range(0,len(encoded),36000)]))
 batches=[];current=[];size=0
 for row in payload:
  if current and size+row['bytes']>45000:batches.append(current);current=[];size=0
  current.append(row);size+=row['bytes']
 if current:batches.append(current)
 spec=importlib.util.spec_from_file_location('source_only_dependency_transport',HELPER)
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.HERE=HERE
 for index,batch in enumerate(batches):
  code='from pathlib import Path\nimport base64,hashlib,json,subprocess\nrepo=Path('+repr(REPO)+');phase=repo/"experiments_iclr/postsubmission_20260930"\nassert Path.cwd().resolve()==repo.resolve()\nassert subprocess.run(["hostname"],capture_output=True,text=True,check=True).stdout.strip()=="peptide"\nrows='+repr(batch)+'\nout=[]\n'
  code+='''for row in rows:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase.resolve()) and not p.is_symlink()
 raw=base64.b64decode(''.join(row['chunks']),validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():
  assert p.is_file() and p.read_bytes()==raw,('existing_source_differs',row['path'])
  state='verified_existing'
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as h:h.write(raw)
  state='copied_absent_source'
 out.append(dict(path=row['path'],sha256=row['sha256'],bytes=row['bytes'],state=state))
print(json.dumps(dict(status='SOURCE_DEPENDENCIES_READY',files=out,training=False,dataset_checkpoint_outcome_reads=False)))
'''
  assert len(code.encode())<95000
  identity='quality_'+a.kind+'_source_dependencies_%02d_20261005_v1'%index
  value=helper.run(identity,code)
  with (HERE/(identity+'_RESULT.json')).open('x') as h:json.dump(value,h,indent=2);h.write('\n')
  print(json.dumps({'kind':a.kind,'batch':index,'files':len(value['files']),'status':value['status']}),flush=True)

if __name__=='__main__':main()
