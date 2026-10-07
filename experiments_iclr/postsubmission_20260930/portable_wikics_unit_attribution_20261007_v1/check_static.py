"""AST/help/dependency-byte inspection only; never runs data or numerical code."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parsed=[]
    for path in sorted(HERE.glob('*.py')):
        tree=ast.parse(path.read_text(),filename=str(path)); parsed.append(path.name)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert not any(n.split('.')[0] in {'numpy','torch','torch_geometric','ogb'} for n in names),path
    origin=json.loads((HERE/'SOURCE_ORIGIN.json').read_text())
    assert sha(HERE/'recompute.py')==origin['recompute_origin']['sha256']
    dependency=HERE.parent/origin['public_dependency_manifest']['path']
    assert sha(dependency)==origin['public_dependency_manifest']['sha256']
    for row in json.loads(dependency.read_text())['files']:
        path=dependency.parent/row['path']; assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    assert origin['private_origin_files_required_at_runtime'] is False
    source=(HERE/'train.py').read_text(); tree=ast.parse(source)
    assert not any(isinstance(n,ast.For) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in {'backward','step'} for x in ast.walk(n)) for n in ast.walk(tree))
    for name in ('socket','subprocess','os'):
        assert not any(isinstance(n,ast.Import) and any(a.name==name for a in n.names) for n in tree.body),name
    for private in ('admit(',"--release",'CUDA_VISIBLE_DEVICES','nvidia-smi','sudo','/disk/','192.168.'):
        assert private not in source,private
    assert 'driver.main()' in source and 'recompute.install(session)' in source
    assert "torch.backends.cuda.matmul.allow_tf32=False" in source and "torch.backends.cudnn.benchmark=False" in source
    assert 'part_of_registered_author_Wiki12' in source and '--public-interface' in source
    help_result=subprocess.run([sys.executable,'-B','-S',str(HERE/'train.py'),'--help'],capture_output=True,text=True)
    assert help_result.returncode==0,help_result.stderr
    for flag in ('--condition','--seed','--device','--train','--valid','--polynormer','--output','--public-interface'):
        assert flag in help_result.stdout,flag
    if (HERE/'MANIFEST.json').exists():
        for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
            path=HERE/row['path']; assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    print(json.dumps(dict(AST=parsed,stdlib_only_help_passed=True,public_dependency_seal_verified=True,
        bundled_recompute_origin_bytes_verified=True,no_numerical_model_data_remote_or_training_work=True,
        no_author_host_GPU_release_or_evidence_runtime_dependencies=True,runtime_qualified=False),sort_keys=True))


if __name__=='__main__': main()
