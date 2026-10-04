exec((__import__('pathlib').Path(__file__).parent/'retrieve_geeni.py').read_text().split('with concurrent.futures.ThreadPoolExecutor')[0])
routes=[('geeni_s2','https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3489517.3530416?fields=title,authors,openAccessPdf,url,externalIds'),('geeni_arxiv','https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':'ti:"Efficient ensembles of graph neural networks"','max_results':5})),('geeni_search','https://www.google.com/search?'+urllib.parse.urlencode({'q':'"Efficient ensembles of graph neural networks" PDF','num':10}))]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:out=list(ex.map(get,routes))
(P/'discovery'/'GEENI_LOCATOR_RETRIEVAL.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:print(r)
