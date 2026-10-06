
from pathlib import Path
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=repo/'experiments_iclr/postsubmission_20260930/shared_private_transfer_complete39_common_negative_owned_operation_20261006_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
exit=json.loads((root/'EXIT.json').read_text());assert exit['exit_code']==0 and exit['terminal_wait_observed'] is True and exit['reason'] is None and exit['signals_sent']==[]
result=json.loads((root/'child.stdout').read_text());assert result['schema']=='complete39_fixed12_common_negative_diagnostic_v1' and result['bindings_sha256']=='99981bad6baf3d6e1596d9064695fc07588917fc6f25eb74cabf9b1d2264524a'
routing=[]
for bank in result['bank_metrics']:
 rows=[r for r in result['per_query'] if r['block']==bank['block'] and r['cell']==bank['cell']];assert len(rows)==227
 minima=[min(r['member_strict_wrong_counts']) for r in rows]
 routing.append({'block':bank['block'],'cell':bank['cell'],'mean_minimum_member_strict_wrong_count':sum(minima)/227,
 'mean_minimum_member_minus_common_strict_count':sum(m-r['common_strict_count'] for m,r in zip(minima,rows))/227,
 'queries_minimum_member_has_zero_strict_errors':sum(m==0 for m in minima),
 'queries_pool_strict_error_but_common_empty':sum(r['pooled_strict_wrong_count']>0 and r['common_strict_count']==0 for r in rows),
 'queries_minimum_member_strict_count_exceeds_common':sum(m>r['common_strict_count'] for m,r in zip(minima,rows)),
 'queries_pool_strict_count_above_minimum_member':sum(r['pooled_strict_wrong_count']>m for m,r in zip(minima,rows)),
 'queries_pool_strict_count_below_minimum_member':sum(r['pooled_strict_wrong_count']<m for m,r in zip(minima,rows)),
 'scope':'Descriptive reductions of already emitted query counts; ties excluded from strict counts, no MRR/original score recomputation or deployable routing claim.'})
pid=exit['child_identity']['PID'];physical={'owned_child_PID_absent':not (Path('/proc')/str(pid)).exists(),'child_PID':pid,'child_start_ticks':exit['child_identity']['start_ticks']}
summary={k:result[k] for k in ['schema','scope','bindings_sha256','bank_metrics','equal_block_descriptive_summary','paired_block_comparisons','uncertainty','interpretation','TEST_access','model_execution','new_fits','checkpoint_reselection']}
summary['routing_count_reductions']=routing
summary['full_result_retained_only_server']={'path':str(root/'child.stdout'),'bytes':(root/'child.stdout').stat().st_size,'sha256':sha(root/'child.stdout')}
summary['terminal']=exit;summary['physical_terminal']=physical
summary['terminal_files']={name:{'bytes':(root/name).stat().st_size,'sha256':sha(root/name)} for name in ['EXIT.json','CHILD_STARTED.json','SUPERVISOR_STARTED.json','RESOURCES.json','CONFIG.json','child.stderr','supervisor.stderr']}
print(json.dumps(summary,allow_nan=False,sort_keys=True))
