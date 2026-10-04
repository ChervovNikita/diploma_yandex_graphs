from pathlib import Path
from html.parser import HTMLParser
import json
P=Path(__file__).resolve().parent
class Node:
 def __init__(self,tag,attrs=None):self.tag=tag;self.attrs=dict(attrs or []);self.children=[]
 def text(self):
  if self.tag=='math' and 'alttext' in self.attrs:return ' $'+self.attrs['alttext']+'$ '
  return ' '.join(c if isinstance(c,str) else c.text() for c in self.children)
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.root=Node('root');self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  n=Node(tag,attrs);self.stack[-1].children.append(n)
  if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:self.stack.append(n)
 def handle_startendtag(self,tag,attrs):self.stack[-1].children.append(Node(tag,attrs))
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:self.stack=self.stack[:i];break
 def handle_data(self,d):self.stack[-1].children.append(d)
classes={'ltx_para','ltx_equation','ltx_equationgroup','ltx_table','ltx_figure'}
def collect(n,out):
 cs=set(n.attrs.get('class','').split())
 if n.tag in {'h1','h2','h3','h4','h5'} or classes&cs:
  out.append({'i':len(out),'tag':n.tag,'id':n.attrs.get('id'),'classes':list(cs),'text':' '.join(n.text().split())});return
 for c in n.children:
  if isinstance(c,Node):collect(c,out)
for key in ['emrgnn','gre']:
 p=Parser();p.feed((P/'primary'/f'{key}.html').read_text());blocks=[];collect(p.root,blocks)
 (P/'primary'/f'{key}.blocks.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n')
 print('\nPAPER',key)
 for b in blocks:
  if b['tag'].startswith('h') or b['i']<10:print(b['i'],b['id'],b['text'])
 print('BLOCKS',len(blocks))
