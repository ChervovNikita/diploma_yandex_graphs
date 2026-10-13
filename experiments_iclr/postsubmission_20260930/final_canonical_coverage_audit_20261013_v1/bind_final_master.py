"""Bind a closed source snapshot to the root's final immutable master/index."""
import argparse,gzip,hashlib,json,shlex,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;sys.path.insert(0,str(P/'retained_prototype_batch_20261012_v1/preparation'));from read_allocation_metadata import PREFIX,GUARD
ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--index',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--plan-hash',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();snap=json.loads(gzip.decompress(Path(args.snapshot).read_bytes()));expected={r['entry']['id']:r['entry']['plan'] for r in snap['packets']};request={'expected':expected,'index':args.index,'plan':args.plan,'plan_hash':args.plan_hash}
code=GUARD+r'''
v=json.loads(sys.argv[1])
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def receipt(p):b=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
index=phase/v['index'];j=json.loads(index.read_text());actual={e['id']:e['plan'] for e in j['packets']};assert actual==v['expected'],'Final selected packet receipts differ from closed exact source snapshot'
pp=phase/v['plan'];plan=json.loads(pp.read_text());assert hashlib.sha256(canonical(plan).encode()).hexdigest()==v['plan_hash'];entries={r['id']:r['plan'] for r in plan.get('source_packet_entries',[])}
if entries:assert entries==actual
print(json.dumps({'schema':'final-master-closed-source-recipe-binding-v1','master_plan_hash':v['plan_hash'],'master_plan':receipt(pp),'input_index':receipt(index),'exact_selected_packet_receipts_match':True,'packet_count':len(actual),'source_coverage_complete':True,'route_guard':True,'tensor_reads':0,'scientific_fits':0,'GO':False}))
'''
out=Path(args.output);assert not out.exists();p=subprocess.run(PREFIX+[shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,json.dumps(request)])],capture_output=True,text=True,timeout=60);out.with_suffix('.transport.json').write_text(json.dumps({'exit_code':p.returncode,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr':p.stderr,'request':request,'tensor_reads':0,'GO':False},indent=2)+'\n');assert p.returncode==0,p.stdout;value=json.loads(next(x for x in p.stdout.splitlines() if x.startswith('{')));value['source_snapshot']={'path':str(Path(args.snapshot).resolve().relative_to(P)),'sha256':hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest()};out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
