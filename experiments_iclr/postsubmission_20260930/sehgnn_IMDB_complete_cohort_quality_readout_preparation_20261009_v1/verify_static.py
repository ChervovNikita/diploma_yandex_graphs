"""Parse/binding/default checks only; no numerical tests or outcome reads."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def check():
    sources=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in sources['files']:
        path=HERE.parent/row['path']
        assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    calls=[]
    for path in HERE.glob('*.py'):
        tree=ast.parse(path.read_text())
        if path.name=='RUN.py':
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                    calls.append(name)
                    assert name not in {'train','fit','fit_body','make_model','evaluate','collect_selected_logits','runtime','Popen','kill','sleep','check_output'}
                    if name=='load' and isinstance(node.func,ast.Attribute):
                        assert {item.arg:ast.literal_eval(item.value) for item in node.keywords}=={'map_location':'cpu','weights_only':True}
    assert calls.count('compare_independent_controls')==calls.count('compare_selected')==1
    release=json.loads((HERE/'RELEASE.disabled.json').read_text())
    assert release['enabled'] is release['root_source_review_approved'] is release['root_readout_approved'] is False
    for path in HERE.glob('*.json'):
        json.loads(path.read_text())
    default=subprocess.run([sys.executable,'-B',str(HERE/'RUN.py')],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout)=={'inactive':True,'outcomes_or_providers_read':False} and not default.stderr
    return dict(status='passed',parse_bindings_and_disabled_default_only=True,existing_analysis_functions_reused=True,
        no_model_fit_server_GPU_query_or_numerical_test_suite=True,scientific_outcomes_or_actual_receipts_read=False,
        source_verification_does_not_establish_quality=True)


if __name__=='__main__':
    result=check();target=HERE/'STATIC_VERIFICATION.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
