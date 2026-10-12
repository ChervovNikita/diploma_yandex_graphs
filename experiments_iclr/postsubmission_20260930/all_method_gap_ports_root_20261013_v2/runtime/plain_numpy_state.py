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
    adapter.external_restore=lambda value:restore(convert(value,False))
    adapter.provenance['checkpoint_numpy_rng_encoding']='MT19937 five fields;uint32 state list;exact reconstruction before source restore'
    return adapter
