"""Serialize only the explicitly inventoried NumPy RNG state as plain data."""
import copy

def install_plain_numpy_rng(adapter):
    capture,restore=adapter.external_capture,adapter.external_restore
    if capture is None or restore is None:raise ValueError('Bound complete external state required')
    def convert(tree,to_plain):
        result=copy.deepcopy(tree)
        if 'source_controller' in result:
            result['source_controller']=convert(result['source_controller'],to_plain)
        if 'numpy_rng' in result:
            state=result['numpy_rng']
            if len(state)!=5 or state[0]!='MT19937':raise ValueError('Exact inventoried NumPy RNG required')
            if to_plain:
                result['numpy_rng']=[state[0],state[1].tolist(),*state[2:]]
            else:
                import numpy as np
                result['numpy_rng']=(state[0],np.array(state[1],dtype=np.uint32),*state[2:])
        return result
    adapter.external_capture=lambda:convert(capture(),True)
    def restored(value):
        restore(convert(value,False))
        session=getattr(adapter,'session',None)
        if session is not None and 'source_session' in session.__dict__:
            # The wrapper delegates reads through __getattr__; source-owned
            # assignments must reach that same source, rather than shadow it.
            for name in ('streams','steps','execution_totals','last_replay_diagnostics'):
                if name in session.__dict__:
                    setattr(session.source_session,name,session.__dict__.pop(name))
    adapter.external_restore=restored
    adapter.provenance['checkpoint_numpy_rng_encoding']='MT19937 five fields;uint32 state list;exact reconstruction before source restore'
    if getattr(adapter,'session',None) is not None and 'source_session' in adapter.session.__dict__:
        adapter.provenance['checkpoint_single8_source_forwarding']='Restore delegated source streams/clocks on source_session; no shadow state'
    return adapter
