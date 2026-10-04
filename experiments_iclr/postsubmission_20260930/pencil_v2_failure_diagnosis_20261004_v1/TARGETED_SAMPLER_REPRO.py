"""Narrow stdlib reproduction of v2 RSS false-failure and proposed fix.

No live /proc, numerical imports, process creation or signals are used.
"""
import ast
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'pencil_collab_paired_predictive_preparation_20261004_v2/supervise.py'


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


class FakePath:
    def __init__(self,value):
        self.value=str(value)
    @property
    def name(self):
        return self.value.rsplit('/',1)[-1]
    def __truediv__(self,name):
        return FakePath(self.value+'/'+str(name))
    def iterdir(self):
        assert self.value=='/proc'
        return iter([FakePath('/proc/43')])
    def read_text(self):
        assert self.value=='/proc/43/status'
        # Linux get_task_mm may return None during an owned task's exit,
        # before task_state_to_char starts exposing Z.
        return 'Name:\texiting-loader\nState:\tR (running)\n'


def proc_text(state,rss_pages):
    fields=['0']*22
    fields[0]=state;fields[2]='42';fields[3]='42';fields[19]='100';fields[21]=str(rss_pages)
    return '43 (loader (comm)) '+ ' '.join(fields)


def main():
    tree=ast.parse(SOURCE.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='members_of_session')
    original_module=ast.Module(body=[function],type_ignores=[])
    identity=dict(pid=43,start_ticks=100,session=42,group=42,state='R')
    namespace={'Path':FakePath,'process_identity':lambda pid:dict(identity),'require':require}
    exec(compile(ast.fix_missing_locations(original_module),str(SOURCE),'exec'),namespace)
    try:
        namespace['members_of_session'](42)
    except RuntimeError as error:
        original_failure=str(error)
    else:
        raise AssertionError('Original v2 unexpectedly accepted exiting R/no-VmRSS fixture')
    assert original_failure=='Owned live RSS observation missing'
    spec=importlib.util.spec_from_file_location('proposed_rss_sampler',HERE/'MINIMUM_REPAIR_PROPOSAL.py')
    proposed=importlib.util.module_from_spec(spec);spec.loader.exec_module(proposed)
    cases=[]
    for state,pages in [('R',0),('S',17),('Z',0)]:
        row=proposed.parse_proc_stat(proc_text(state,pages),43,4096)
        assert row['RSS_bytes']==pages*4096 and row['session']==row['group']==42 and row['start_ticks']==100
        cases.append(dict(state=state,rss_pages=pages,RSS_bytes=row['RSS_bytes'],result='PASS'))
    rejects=[]
    for label,raw,pid in [('negative_rss',proc_text('R',-1),43),('wrong_pid',proc_text('R',0),44),('truncated_fields','43 (loader) R',43)]:
        try:
            proposed.parse_proc_stat(raw,pid,4096)
        except (RuntimeError,ValueError):
            rejects.append(label)
        else:
            raise AssertionError(label+' accepted')
    result=dict(status='PASS_NARROW_OBSERVER_REPRODUCTION',actual_v2_function_extracted_without_import=True,
        same_birth_session_R_without_VmRSS_original_failure=original_failure,proposed_stat_snapshot_cases=cases,
        malformed_or_mismatched_rejected=rejects,live_proc_read=False,processes_created=False,signals_sent=False,
        numerical_or_model_imports=False,fixture_note='Synthetic observer lifecycle only; does not identify the unlogged actual failing PID or prove scientific feasibility.')
    (HERE/'REPRO_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
