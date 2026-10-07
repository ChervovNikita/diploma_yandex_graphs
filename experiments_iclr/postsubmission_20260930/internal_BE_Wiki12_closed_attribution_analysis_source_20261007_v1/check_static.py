"""Stdlib source/disabled-gate checks only; no numeric imports or data opening."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    parsed=[]
    for path in sorted(HERE.glob('*.py')):
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert not any(x.split('.')[0] in {'torch','numpy','torch_geometric','ogb','models','data','factors'} for x in names),path
        parsed.append(path.name)
    release=json.loads((HERE/'RELEASE_TEMPLATE_DISABLED.json').read_text())
    for key in ('enabled','root_execution_authorized','source_review_approved','entire12_complete','owner_lanes_and_children_terminal',
                'trusted_checkpoint_deserialization_authorized','runtime_resource_readiness_confirmed','collection_and_analysis_cost_charged','external_owned_bound_confirmed'):
        assert release[key] is False,key
    assert release['maximum_member_forwards']==48 and release['TEST_access'] is False and release['reselection'] is False and release['automatic_retry'] is False
    terminal=json.loads((HERE/'TERMINAL_EVIDENCE_TEMPLATE_DISABLED.json').read_text());assert terminal['owner_lanes_and_children_terminal'] is False and len(terminal['children'])==12
    collector=(HERE/'collect.py').read_text();gate=(HERE/'gate.py').read_text();analysis=(HERE/'analyse.py').read_text()
    for value in ('.backward(','.step(','torch.optim.','public.Session(','adapter.make_session(','selection.local_transition('): assert value not in collector,value
    for value in ("model.set_global(saved['global'])",'model.eval()','with torch.no_grad():','model.member_forward(batch,member)',
                  'member_probability=logits.softmax(-1); pooled=member_probability.mean(0)','member_representations',"baseline=[row for seed in SEEDS",
                  "consume(args.release,args.release_sha256)"):
        assert value in collector,value
    assert collector.index('consume(args.release,args.release_sha256)')<collector.index('import numpy as np; import torch')
    for value in ('Entire fixed12 roster complete before numerical opening','Root terminal custody','Original selected snapshot hash; no reselection','Exact whole completed lane'):
        assert value in gate,value
    for value in ('original.paired(','original.scalar_errors(','original.error_masks(','original.validate(','original.load(',
                  'C-P','C-A-R+P','4.302652729911275','common_wrong_competitor_flows','selection_population_caveat'):
        assert value in analysis,value
    pins=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in pins['references']:
        f=HERE.parent/row['path'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],str(f)
    if (HERE/'MANIFEST.json').exists():
        for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
            f=HERE/row['path'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],str(f)
    result=subprocess.run([sys.executable,'-B','-S',str(HERE/'collect.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    path=HERE/'RELEASE_TEMPLATE_DISABLED.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    rejected=subprocess.run([sys.executable,'-B','-S',str(HERE/'collect.py'),'--release',str(path),'--release-sha256',digest],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'Disabled pending root release: enabled' in rejected.stderr,rejected.stderr
    print(json.dumps(dict(AST=parsed,source_bindings_verified=len(pins['references']),stdlib_only_CLI_help_passed=True,
        disabled_release_rejected_before_numerical_import=True,numerical_or_remote_work=False,runtime_qualified=False),sort_keys=True))


if __name__=='__main__': main()
