"""Deterministic stdlib HTML block/MathML-alttext extraction, no content execution."""
from html.parser import HTMLParser
from pathlib import Path
import json,sys,re
class Node:
 def __init__(self,tag,attrs,parent=None):self.tag=tag;self.attrs=dict(attrs);self.parent=parent;self.children=[]
 def walk(self):
  yield self
  for child in self.children:
   if isinstance(child,Node):yield from child.walk()
 def text(self):
  if self.tag in ('script','style','nav'):return ''
  if self.tag=='math':
   latex=self.attrs.get('alttext')
   if latex is None:
    annotation=next((n for n in self.walk() if n.tag=='annotation' and 'tex' in n.attrs.get('encoding','')),None)
    latex=annotation.text() if annotation is not None else None
   if latex is not None:return ' '+latex+' '
  return ' '.join(child.text() if isinstance(child,Node) else child for child in self.children)
class Tree(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.root=Node('root',[]);self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  node=Node(tag,attrs,self.stack[-1]);self.stack[-1].children.append(node)
  if tag not in ('meta','link','img','br','hr','source','input','area','embed','param','wbr'):self.stack.append(node)
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:del self.stack[i:];break
 def handle_startendtag(self,tag,attrs):self.handle_starttag(tag,attrs);self.handle_endtag(tag)
 def handle_data(self,data):self.stack[-1].children.append(data)
def clean(text):return re.sub(r'\s+',' ',text).strip()
p=Path(__file__).resolve().parent
for key in sys.argv[1:]:
 tree=Tree();tree.feed((p/'primary'/f'{key}.html').read_text())
 nodes=list(tree.root.walk());maths=[n for n in nodes if n.tag=='math'];math_ids={id(n):i for i,n in enumerate(maths)}
 equations=[{'ordinal':i,'id':n.attrs.get('id'),'display':n.attrs.get('display'),'latex_or_text':clean(n.text())} for i,n in enumerate(maths)]
 blocks=[];heading=''
 for node in nodes:
  is_equation=node.tag=='table' and 'ltx_equation' in node.attrs.get('class','')
  if node.tag not in ('p','h1','h2','h3','h4','h5','h6') and not is_equation:continue
  text=clean(node.text())
  if not text:continue
  if node.tag.startswith('h') and node.tag[1:].isdigit():heading=text
  blocks.append({'index':len(blocks),'tag':node.tag,'id':node.attrs.get('id'),'class':node.attrs.get('class'),'heading_locator':heading,'text':text,'math_ordinals':[math_ids[id(n)] for n in node.walk() if n.tag=='math']})
 (p/'primary'/f'{key}.blocks.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n')
 (p/'primary'/f'{key}.equations.json').write_text(json.dumps(equations,ensure_ascii=False,indent=2)+'\n')
 print(key,'blocks',len(blocks),'maths',len(equations))
 for b in blocks:
  if b['tag'].startswith('h') or b['tag']=='table':print(b['index'],b['tag'],b['text'][:130])
