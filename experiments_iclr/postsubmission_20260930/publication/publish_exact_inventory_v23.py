"""Retain guarded exact publication; accept exact copied small manuscript dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import types

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent

def main():
    inventory=PHASE/sys.argv[sys.argv.index('--inventory')+1]
    assert inventory.resolve().is_relative_to(PHASE)
    plan=json.loads(inventory.read_text())
    for row in plan['files']:
        path=PHASE/row['source']
        if path.suffix=='.png':
            assert path.resolve().is_relative_to(PHASE)
            assert path.stat().st_size<500000 and path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
            if path.parent.name == 'figures' and path.parent.parent.name in ('complete_anonymous_quality_draft_20261011_v1', 'complete_anonymous_quality_draft_20261011_v2'):
                donors = {
                    'architecture_sharing_compact.png': 'paper_dependency_recovery_root_20261011_v1/historical_round18_assets/architecture_sharing_compact.png',
                    'gnnm_vs_ens_heatmap.png': 'paper_dependency_recovery_root_20261011_v1/historical_round18_assets/gnnm_vs_ens_heatmap.png',
                    'F_member_coverage_v2.png': 'F_member_coverage_explanation_figure_root_20261011_v1/F_member_coverage_v2.png',
                }
                assert path.name in donors and path.read_bytes() == (PHASE / donors[path.name]).read_bytes()
            else:
                assert path.parent.name in ('historical_round18_assets','F_member_coverage_explanation_figure_root_20261011_v1')
    spec=importlib.util.spec_from_file_location('_v19_guarded_payload',HERE/'publish_exact_inventory_v19.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    original=prior.terminal_transport
    def guarded(argv,**kwargs):
        fields=shlex.split(argv[-1]);assert fields[:5]==['/usr/bin/python3','-I','-S','-B','-c']
        marker='repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];operation=sys.argv[3]'
        replacement="""import socket
assert socket.gethostname()=='anogena-2-0'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[sys.argv[2]]
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];operation=sys.argv[3]"""
        assert fields[5].count(marker)==1;fields[5]=fields[5].replace(marker,replacement)
        return original([*argv[:-1],shlex.join(fields)],**kwargs)
    basepath=HERE/'publish_exact_inventory_v18.py'
    assert hashlib.sha256(basepath.read_bytes()).hexdigest()==prior.BASE_SHA256
    spec=importlib.util.spec_from_file_location('_v18_exact_contract_with_paper_extensions',basepath)
    base=importlib.util.module_from_spec(spec)
    source=basepath.read_text();old="'.response') or source.name";new="'.response', '.bib', '.tex', '.sty', '.bst', '.png') or source.name"
    assert source.count(old)==1
    exec(compile(source.replace(old,new),str(basepath),'exec'),base.__dict__)
    base.subprocess=types.SimpleNamespace(run=guarded);base.__file__=__file__;base.main()

if __name__=='__main__':main()
