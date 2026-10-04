from html.parser import HTMLParser
from pathlib import Path
import json
class Node:
 def __init__(self,tag,attrs): self.tag=tag;self.attrs=dict(attrs);self.children=[]
class Parser(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.root=Node('root',[]);self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  n=Node(tag,attrs);self.stack[-1].children.append(n)
  if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:self.stack.append(n)
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:self.stack=self.stack[:i];break
 def handle_startendtag(self,tag,attrs):self.stack[-1].children.append(Node(tag,attrs))
 def handle_data(self,data):self.stack[-1].children.append(data)
def text(n):
 if isinstance(n,str):return n
 if n.tag=='math':return ' '+n.attrs.get('alttext','')+' '
 if n.tag in {'script','style','annotation'}:return ''
 return ''.join(text(c) for c in n.children)
def walk(n):
 if isinstance(n,str):return
 yield n
 for c in n.children:yield from walk(c)
b=Path(__file__).parent;p=Parser();p.feed((b/'primary/amgcn_html.html').read_text())
blocks=[]
for n in walk(p.root):
 cls=n.attrs.get('class','').split()
 if n.tag in {'h1','h2','h3','h4'} or (n.tag=='p' and 'ltx_p' in cls) or 'ltx_equation' in cls:
  blocks.append({'index':len(blocks),'tag':n.tag,'id':n.attrs.get('id'),'classes':cls,'text':' '.join(text(n).split())})
(b/'primary/amgcn_blocks.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2))
for r in blocks:
 if r['tag'].startswith('h') or any(k in r['text'].lower() for k in ['shared','sharing','ablation','variant','consistency','disparity']):print(json.dumps(r,ensure_ascii=False))
q=Parser();q.feed((b/'primary/amgcn_abs.html').read_text());print('METADATA')
for n in walk(q.root):
 if n.tag=='meta' and n.attrs.get('name','').startswith('citation'):print(n.attrs)
 if 'submission-history' in n.attrs.get('class','').split():print(text(n))
