"""Narrow source review for lexical interpreter correction; no subprocesses."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'expanded_dispatcher_v1'


def check():
    assert (HERE/'engine.py').read_bytes()==(OLD/'engine.py').read_bytes()
    checks=['unchanged complete engine.py;38 predecessor fake checks reused without rerun']
    changed={'contracts.py':{'validate_binding','compile_release','authority'},'dispatcher.py':set()}
    # Backend.start is the only modified class method in the production interface.
    for name in ('contracts.py','dispatcher.py'):
        old=ast.parse((OLD/name).read_text());new=ast.parse((HERE/name).read_text())
        def funcs(tree):
            out={}
            for n in tree.body:
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):out[n.name]=n
                elif isinstance(n,ast.ClassDef):
                    for f in n.body:
                        if isinstance(f,ast.FunctionDef):out[n.name+'.'+f.name]=f
            return out
        a,b=funcs(old),funcs(new)
        for key,node in a.items():
            same=ast.dump(node,include_attributes=False)==ast.dump(b[key],include_attributes=False)
            expected=key in changed[name]or(name=='dispatcher.py'and key=='AllocationBackend.start')
            assert same != expected,(name,key,'unexpected source delta')
        checks.append('only declared function deltas:'+name)
    contracts=(HERE/'contracts.py').read_text();dispatch=(HERE/'dispatcher.py').read_text();entry=(HERE/'qualified_python_entry.py').read_text()
    assert "HERE=EXPANSION/'expanded_dispatcher_v2'"in contracts
    assert "validate_runtime_environment(spec)"in contracts
    assert "execution['python_executable'],'-B','-u',str(HERE/'qualified_python_entry.py')"in dispatch
    assert 'sys.prefix==spec[\'prefix\']'in entry and 'sys.base_prefix==spec[\'base_prefix\']'in entry
    assert entry[entry.index("def main():"):].index("socket.gethostname()=='anogena-2-0'")<entry[entry.index("def main():"):].index("data=path.read_bytes()")
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    return {'schema':'expanded-dispatcher-v2-narrow-source-check-v1','status':'PASS_source_only','checks':checks,
        'source_modifications':['contracts.py:lexical environment authority and successor closure','dispatcher.py:guarded lexical child entry wrapper'],
        'all38_predecessor_checks_reused':True,'new_fake_subprocess_campaign':False,'native_work':0,'TEST_accessed':False}


if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
