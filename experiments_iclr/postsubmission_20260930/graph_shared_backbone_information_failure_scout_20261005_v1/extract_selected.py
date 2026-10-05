import json,re
from pathlib import Path
from discover_primary import Parser,clean,Node
OUT=Path(__file__).resolve().parent

def scoped_text(n):
 if not isinstance(n,Node):return n
 if n.tag=='math':
  if n.attrs.get('alttext'):return ' $'+n.attrs['alttext']+'$ '
  a=n.find('annotation');return ' $'+clean(a[0].text())+'$ ' if a else n.text()
 if n.tag in ['script','style']:return ''
 return ' '.join(scoped_text(c) for c in n.children)

choices={'hopper':['2.3 Linearized Graph Sequence Models','3.1 Learnable Hop Extraction','3.2 Hypernetwork-Conditioned Structural-State Recurrence'], 'nba':['3.1 Motivation from sensitivity analysis','3.2 Method description']}
rs=[]
for key,titles in choices.items():
 p=Parser();p.feed((OUT/('_'+key+'.html')).read_text());sections=[]
 for section in p.root.find('section'):
  if not section.children:continue
  direct=[c for c in section.children if isinstance(c,Node) and c.tag in ['h2','h3','h4']]
  if not direct:continue
  title=clean(direct[0].text())
  if title in titles:
   paragraphs=[{'id':n.attrs.get('id'),'text':clean(scoped_text(n))} for n in section.find('div','ltx_para')]
   eqs=[{'id':n.attrs.get('id'),'text':clean(scoped_text(n))} for n in section.find('table','ltx_equation')]
   figures=[{'id':n.attrs.get('id'),'text':clean(scoped_text(n))} for n in section.find('figure')]
   sections.append({'title':title,'html_id':section.attrs.get('id'),'text':clean(scoped_text(section)),'paragraph_ids':[a['id'] for a in paragraphs],'equation_ids':[a['id'] for a in eqs],'figure_ids':[a['id'] for a in figures]})
 assert set(a['title'] for a in sections)==set(titles),(key,[a['title'] for a in sections])
 rs.append({'key':key,'selected_complete_subsections':sections,'full_paper_read':False,'section_scope_note':'Only these complete method/protocol subsections semantically inspected. Other content, experiment scores, proofs and references not read.'})
(OUT/'SCOPE_EXCERPTS.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2)+'\n')
for r in rs:print(r['key'],[(a['title'],a['html_id'],len(a['text']),a['equation_ids'],a['figure_ids']) for a in r['selected_complete_subsections']])
