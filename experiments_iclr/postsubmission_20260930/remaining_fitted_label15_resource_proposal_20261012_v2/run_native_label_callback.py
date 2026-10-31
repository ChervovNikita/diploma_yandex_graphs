"""Adapter callback for the existing root denial/state resource harness."""
def run_source_callback(adapter,step,*,expected_denial):
    try:
        adapter.train_step(step)
    except expected_denial:
        return {"complete_source_gradient_callback_reached_optimizer":True,"optimizer_commit":False}
    raise RuntimeError("Root source optimizer denial did not intercept the complete TRAIN callback")
