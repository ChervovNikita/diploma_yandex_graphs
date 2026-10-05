from html.parser import HTMLParser
from pathlib import Path
import json
class PaperBlocks(HTMLParser):
 def __init__(self):
  super().__init__();self.active=None;self.balance=0;self.blocks=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs);classes=a.get("class","").split()
  if self.active:
   if tag==self.active["tag"]:self.balance+=1
   if tag=="math" and a.get("alttext"):self.active["parts"].append("[MATH "+a["alttext"]+"]")
  elif any(x in classes for x in ["ltx_title","ltx_para","ltx_equation","ltx_algorithm","ltx_table"]):
   self.active={"index":len(self.blocks),"id":a.get("id"),"tag":tag,"classes":classes,"parts":[]};self.balance=1
 def handle_endtag(self,tag):
  if self.active and tag==self.active["tag"]:
   self.balance-=1
   if self.balance==0:
    b=self.active;b["text"]=" ".join(" ".join(b.pop("parts")).split());self.blocks.append(b);self.active=None
 def handle_startendtag(self,tag,attrs):
  if self.active and tag=="math":
   a=dict(attrs)
   if a.get("alttext"):self.active["parts"].append("[MATH "+a["alttext"]+"]")
 def handle_data(self,data):
  if self.active:self.active["parts"].append(data)
def parse(path):
 h=PaperBlocks();h.feed(Path(path).read_text());return h.blocks
