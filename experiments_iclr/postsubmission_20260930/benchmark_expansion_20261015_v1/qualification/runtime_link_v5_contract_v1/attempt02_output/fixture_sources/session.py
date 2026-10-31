import copy
def selected_mean_sidecar(session):
    if session.sidecar is None: raise ValueError("stale fixture sidecar")
    return copy.deepcopy(session.sidecar)
