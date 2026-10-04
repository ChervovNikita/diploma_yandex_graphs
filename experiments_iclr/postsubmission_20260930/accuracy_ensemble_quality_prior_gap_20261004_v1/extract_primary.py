from html.parser import HTMLParser
from pathlib import Path
import json,hashlib,re
P=Path(__file__).resolve().parent
class N:
 def __init__(self,tag='root',attrs=None,parent=None): self.tag=tag;self.attrs=dict(attrs or []);self.children=[];self.parent=parent
class Parser(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.root=N();self.cur=self.root
 def handle_starttag(self,t,a):
  n=N(t,a,self.cur);self.cur.children.append(n)
  if t not in ['meta','link','img','br','hr','input','source','wbr']:self.cur=n
 def handle_endtag(self,t):
  c=self.cur
  while c.parent and c.tag!=t:c=c.parent
  if c.tag==t and c.parent:self.cur=c.parent
 def handle_data(self,d):self.cur.children.append(d)
def walk(n):
 if isinstance(n,str):return
 yield n
 for x in n.children:yield from walk(x)
def txt(n):
 if isinstance(n,str):return n
 if n.tag=='math':return ' '+n.attrs.get('alttext',''.join(txt(x) for x in n.children))+' '
 if n.tag in ['script','style']:return ''
 return ''.join(txt(x) for x in n.children)
for key in ['shallow','camero']:
 source=P/'sources'/f'{key}_html.html';a=Parser();a.feed(source.read_text());blocks=[]
 for n in walk(a.root):
  if n.tag not in ['p','h1','h2','h3','h4','h5','h6','figure','math']:continue
  if n.tag=='math' and n.attrs.get('display')!='block':continue
  anc=n.parent;skip=False
  while anc:
   if anc.tag in ['table','figure'] or (anc.tag=='p' and n.tag=='math'):skip=True
   anc=anc.parent
  if skip:continue
  text=re.sub(r'\s+',' ',txt(n)).strip()
  if text:blocks.append({'index':len(blocks),'tag':n.tag,'id':n.attrs.get('id'),'class':n.attrs.get('class'),'text':text})
 out=P/'sources'/f'{key}_blocks.json';out.write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n')
 print(key,len(blocks))
 for b in blocks:
  if b['tag'].startswith('h'):print(b['index'],b['text'])
