from html.parser import HTMLParser
from pathlib import Path
import json,re
class Parser(HTMLParser):
 def __init__(self):
  super().__init__(convert_charrefs=True);self.stack=[];self.active=[];self.blocks=[];self.n=0;self.skip=0
 def handle_starttag(self,t,a):
  a=dict(a)
  if self.skip:self.skip+=t not in ['br','img','hr','meta','link','input'];return
  if t in ['script','style','math']:
   if t=='math':
    for x in self.active:x['parts'].append(' $'+a.get('alttext','')+'$ ')
   self.skip=1;return
  if t in ['br','img','hr','meta','link','input']:return
  self.stack.append((t,a.get('id')))
  if t in ['h1','h2','h3','h4','h5','h6','p','figcaption'] or (t=='table' and ('ltx_equation' in a.get('class',''))):
   self.active.append(dict(order=self.n,id=a.get('id') or next((x[1] for x in reversed(self.stack[:-1]) if x[1]),None),tag=t,depth=len(self.stack),parts=[]));self.n+=1
 def handle_endtag(self,t):
  if self.skip:self.skip-=1;return
  for x in self.active[:]:
   if x['depth']==len(self.stack):
    x['text']=re.sub(r'\s+',' ',' '.join(x.pop('parts'))).strip();x.pop('depth');self.blocks.append(x);self.active.remove(x)
  if self.stack:self.stack.pop()
 def handle_data(self,d):
  if not self.skip:
   for x in self.active:x['parts'].append(d)
root=Path(__file__).resolve().parent
for name in ['ride','sade']:
 p=Parser();p.feed((root/'sources'/f'{name}_v4.html').read_text());blocks=sorted(p.blocks,key=lambda x:x['order']);(root/'sources'/f'{name}_v4.blocks.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n');(root/'sources'/f'{name}_v4.readable.txt').write_text('\n\n'.join(f"[{b['order']} | {b['id']} | {b['tag']}] {b['text']}" for b in blocks)+'\n')
 print(name,len(blocks));print('\n'.join(f"{b['order']} | {b['id']} | {b['text']}" for b in blocks if b['tag'].startswith('h')))
