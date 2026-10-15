import subprocess,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;started=time.time();state={'expected_commit':'0dd66efe12be9a851e5e8e56ad504677ed4a1b4c','commands':[]}
try:
 for argv in [ ['git','init'],['git','remote','add','origin','https://github.com/ChervovNikita/diploma_yandex_graphs.git'],['git','config','remote.origin.promisor','true'],['git','config','remote.origin.partialclonefilter','blob:none'],['git','fetch','--depth=1','--filter=blob:none','origin','codex/postsubmission-research-20260930'],['git','sparse-checkout','init','--cone'],['git','sparse-checkout','set','experiments_iclr/postsubmission_20260930'],['git','checkout','-b','codex/postsubmission-research-20260930','0dd66efe12be9a851e5e8e56ad504677ed4a1b4c'] ]:
  p=subprocess.run(argv,cwd=R,capture_output=True,text=True,timeout=1800);state['commands'].append({'argv':argv,'returncode':p.returncode,'stdout':p.stdout[-6000:],'stderr':p.stderr[-6000:]});assert p.returncode==0,p.stderr
 state['status']='complete';state['head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert state['head']==state['expected_commit']
except BaseException as e:state['status']='failed';state['error']=str(e)
state['elapsed_seconds']=time.time()-started;(R/'SYNC_20261015_RESULT.json').write_text(json.dumps(state,indent=2)+'\n')
