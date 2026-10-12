"""Apply a plain-state serialization repair to the unchanged source operator."""
def make_adapter(*,torch,project_root,study,run,device):
    from run_study import load_factory
    helper=load_factory(project_root,run['config']['plain_numpy_helper']+':install_plain_numpy_rng')
    factory=load_factory(project_root,run['config']['original_plain_state_adapter'])
    adapter=factory(torch=torch,project_root=project_root,study=study,run=run,device=device)
    return helper(adapter)
