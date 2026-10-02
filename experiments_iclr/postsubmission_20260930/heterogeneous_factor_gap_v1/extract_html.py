from pathlib import Path
from html.parser import HTMLParser
import re,json,hashlib
ROOT=Path(__file__).resolve().parent
class Extract(HTMLParser):
 def __init__(self):
  super().__init__();self.parts=[];self.rows=[];self.skip=0;self.math_depth=0
 def flush(self):
  t=re.sub(r'\s+',' ',''.join(self.parts)).strip();self.parts=[]
  if t:self.rows.append(t)
 def handle_starttag(self,t,a):
  if self.math_depth:self.math_depth+=1;return
  d=dict(a)
  if t in ('script','style','nav'):self.skip+=1;return
  if self.skip:return
  if t=='math':
   self.parts.append(' '+d.get('alttext','[math without alttext]')+' ');self.math_depth=1;return
  if t in ('p','h1','h2','h3','h4','tr','figcaption','section','article'):self.flush()
  if t in ('td','th'):self.parts.append(' | ')
 def handle_endtag(self,t):
  if self.math_depth:self.math_depth-=1;return
  if t in ('script','style','nav'):
   if self.skip:self.skip-=1
   return
  if self.skip:return
  if t in ('p','h1','h2','h3','h4','tr','figcaption','section','article'):self.flush()
 def handle_data(self,d):
  if not self.skip and not self.math_depth:self.parts.append(d)

if __name__=='__main__':
 records=[]
 for name in ['hgadapter_v1.html','thgp_v1.html','thgp_abs.html']:
  p=ROOT/'sources'/name;x=Extract();x.feed(p.read_text());x.flush();out=p.with_suffix('.txt');out.write_text('\n'.join(x.rows)+'\n')
  records.append({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'derived':str(out.relative_to(ROOT)),'derived_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'method':'stdlib HTMLParser blocks, MathML alttext kept; navigation/script/style dropped; flattened whitespace','line_count':len(x.rows)})
 (ROOT/'DERIVATIONS.json').write_text(json.dumps(records,indent=2))
 for r in records:print(r)
