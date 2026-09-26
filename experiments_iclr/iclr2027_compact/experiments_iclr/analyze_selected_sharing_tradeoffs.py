"""Apply the stated validation choice to every completed 300-epoch position study."""
from pathlib import Path
import hashlib,json,statistics
S=Path(__file__).resolve().parent
sources=[('Roman Empire',S/'sharing_position/roman_study/results/roman',5,'retrospective'),('WikiCS',S/'sharing_position/crossgraph_study/results/wikics',3,'retrospective'),('Actor',S/'sharing_position/crossgraph_study/results/actor',3,'retrospective'),('ogbn-arxiv',S/'ogbn_arxiv_sharing/results/ogbn_arxiv',3,'retrospective'),('Cora',S/'new_graph_studies/position_results/cora',3,'frozen before outcomes'),('Legacy Chameleon',S/'new_graph_studies/position_results/chameleon',3,'frozen before outcomes'),('Filtered Chameleon',S/'filtered_chameleon_study/position_results/chameleon_filtered',3,'frozen before outcomes')]
rows=[]
for label,base,n,scope in sources:
 if not base.is_dir():
  alternatives=list((S/'sharing_position').rglob(base.name))
  alternatives=[q for q in alternatives if (q/'seed0/private_last/result.json').is_file()]
  if len(alternatives)!=1:raise RuntimeError((label,base,alternatives))
  base=alternatives[0]
 arms={}
 for arm in ['tied','private_first','private_last','untied_propagation']:
  files=[base/f'seed{seed}'/arm/'result.json' for seed in range(n)]
  rs=[json.loads(p.read_text()) for p in files]
  assert [r['optimization_seed'] for r in rs]==list(range(n))
  assert all(r['arm']==arm and r['epochs_run']==300 for r in rs)
  assert len({r['parameter_count'] for r in rs})==1
  arms[arm]={
   'mean_valid_accuracy':statistics.mean(r['valid_accuracy'] for r in rs),
   'mean_valid_ce':statistics.mean(r['valid_ce'] for r in rs),
   'mean_test_accuracy':statistics.mean(r['test_accuracy'] for r in rs),
   'test_accuracy_by_seed':[r['test_accuracy'] for r in rs],
   'parameter_count':rs[0]['parameter_count'],
   'result_sha256':{str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
 assert arms['private_first']['parameter_count']==arms['private_last']['parameter_count']
 chosen=min(['private_first','private_last'],key=lambda a:(-arms[a]['mean_valid_accuracy'],arms[a]['mean_valid_ce'],a!='private_last'))
 a=arms[chosen];u=arms['untied_propagation'];t=arms['tied']
 rows.append({'graph':label,'scope':scope,'seeds':n,'selected_partial':chosen,'selected_test_accuracy':a['mean_test_accuracy'],'selected_minus_untied_pp':100*(a['mean_test_accuracy']-u['mean_test_accuracy']),'selected_minus_tied_pp':100*(a['mean_test_accuracy']-t['mean_test_accuracy']),'parameter_saving_vs_untied_percent':100*(1-a['parameter_count']/u['parameter_count']),'test_regret_pp':100*(max(arms[x]['mean_test_accuracy'] for x in ['private_first','private_last'])-a['mean_test_accuracy']),'arms':arms})
out={'analysis':'Fixed validation choice of partial sharing across all seven completed graph settings','choice_rule':'maximize mean validation accuracy, minimize mean validation CE, prefer private_last on exact final tie','inference':'descriptive single-split results; legacy and filtered Chameleon are related graph variants','rows':rows}
(S/'SELECTED_SHARING_TRADEOFFS.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['graph'],r['selected_partial'],round(r['selected_minus_untied_pp'],3),round(r['parameter_saving_vs_untied_percent'],2),round(r['test_regret_pp'],3))
