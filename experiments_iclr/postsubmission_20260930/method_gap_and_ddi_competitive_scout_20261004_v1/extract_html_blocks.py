from html.parser import HTMLParser
from pathlib import Path
import sys,json
class Blocks(HTMLParser):
 def __init__(self):
  super().__init__();self.blocks=[];self.active=None;self.depth=0;self.mathdepth=0;self.tabledepth=0;self.heading="";self.links=[]
 def handle_starttag(self,t,attrs):
  a=dict(attrs)
  if t=="a" and a.get("href"):self.links.append(a["href"])
  if t=="table" and "ltx_equation" not in a.get("class",""):self.tabledepth+=1
  if self.mathdepth:
   self.mathdepth+=1;return
  if t=="math":
   tex=a.get("alttext","");self.mathdepth=1
   if self.active is not None:self.active["text"].append(" $"+tex+"$ ")
   elif not self.tabledepth:self.blocks.append({"kind":"math","section":self.heading,"text":tex})
   return
  if self.active is not None:self.depth+=1;return
  if not self.tabledepth and (t in ["h1","h2","h3","h4","h5","h6","p"]):
   self.active={"kind":t,"id":a.get("id"),"section":self.heading,"text":[]};self.depth=1
 def handle_endtag(self,t):
  if self.mathdepth:
   self.mathdepth-=1;return
  if self.active is not None:
   self.depth-=1
   if self.depth==0:
    b=self.active;b["text"]=" ".join("".join(b["text"]).split());self.active=None
    if b["kind"].startswith("h"):self.heading=b["text"];b["section"]=self.heading
    if b["text"]:self.blocks.append(b)
  if t=="table" and self.tabledepth:self.tabledepth-=1
 def handle_data(self,d):
  if self.active is not None and not self.mathdepth:self.active["text"].append(d)
for arg in sys.argv[1:]:
 f=Path(arg);h=Blocks();h.feed(f.read_text());out=f.with_suffix(".blocks.json");out.write_text(json.dumps(h.blocks,indent=2)+"\n");f.with_suffix(".links.json").write_text(json.dumps(sorted(set(h.links)),indent=2)+"\n")
 print(f.name,len(h.blocks),"blocks",len(h.links),"links")
