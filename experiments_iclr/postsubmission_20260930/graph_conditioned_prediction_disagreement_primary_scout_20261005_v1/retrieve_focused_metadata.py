exec((__import__('pathlib').Path(__file__).parent/'retrieve_metadata.py').read_text().split('with concurrent.futures.ThreadPoolExecutor')[0])
queries=[('graph_exact_ncl','graph "negative correlation"'),('graph_exact_disagreement','graph "disagreement" "ensemble"'),('graph_diversity_regularization','"graph neural" "diversity" "ensemble"')]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:out=list(ex.map(run,queries))
(P/'discovery'/'FOCUSED_RETRIEVAL.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:
 print('QUERY',r['key'],'total',r.get('total'),'error',r.get('error'))
 for x in r.get('rows',[]):print(x['year'],x['title'],x['doi'],(x.get('best_oa_location')or{}).get('pdf_url'))
