"""Extract declared article sections for bounded reading, with math alttext."""
from html.parser import HTMLParser
import json,pathlib,sys,re
class SectionText(HTMLParser):
    def __init__(self,target):
        super().__init__(convert_charrefs=True);self.target=target;self.depth=0;self.active=False;self.out=[];self.math_depth=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='section':
            if not self.active and d.get('id')==self.target:self.active=True;self.depth=1
            elif self.active:self.depth+=1
        if not self.active:return
        if self.math_depth:self.math_depth+=1;return
        if tag=='math':
            self.out.append(' ['+d.get('alttext','')+'] ');self.math_depth=1
        elif tag in {'p','div','h1','h2','h3','h4','li','figure','figcaption','table','tr'}:self.out.append('\n')
    def handle_startendtag(self,tag,attrs):pass
    def handle_endtag(self,tag):
        if not self.active:return
        if self.math_depth:
            self.math_depth-=1
        elif tag in {'p','h1','h2','h3','h4','li','figure','figcaption','tr'}:self.out.append('\n')
        if tag=='section':
            self.depth-=1
            if self.depth==0:self.active=False
    def handle_data(self,data):
        if self.active and not self.math_depth:self.out.append(data)
    def text(self):
        return '\n'.join(re.sub(r'[ \t\r\f\v]+',' ',x).strip() for x in ''.join(self.out).split('\n') if x.strip())
if __name__=='__main__':
    file=pathlib.Path(sys.argv[1]);result=[]
    for section in sys.argv[2:]:
        p=SectionText(section);p.feed(file.read_text());result.append({'file':str(file),'section':section,'text':p.text()})
    print(json.dumps(result,ensure_ascii=False,indent=2))
