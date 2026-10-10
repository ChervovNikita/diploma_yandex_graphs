"""Mechanical primary-HTML passage/heading extraction, not semantic reading."""
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
import json, re

ROOT = Path(__file__).resolve().parent

class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.active = []
        self.blocks = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.stack.append(tag)
        is_block = tag in ['h1','h2','h3','h4','h5','p'] or 'ltx_equation' in attrs.get('class','').split()
        if is_block:
            self.active.append({'tag':tag, 'depth':len(self.stack), 'parts':[]})
        if tag in ['br','img']:
            if tag == 'img' and attrs.get('alt'):
                self.handle_data(attrs['alt'])
            if tag in ['br','img']:
                self.handle_endtag(tag)
    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        pos = len(self.stack) - 1 - self.stack[::-1].index(tag)
        remaining=[]
        for item in self.active:
            if item['depth'] > pos:
                text=' '.join(''.join(item['parts']).split())
                if text:
                    self.blocks.append({'tag':item['tag'], 'text':text})
            else:
                remaining.append(item)
        self.active=remaining
        del self.stack[pos:]
    def handle_data(self, data):
        for item in self.active:
            item['parts'].append(data)

if __name__ == '__main__':
    for name in ['anytime_hierarchical_ensemble','bcnn']:
        p=ROOT/'sources'/f'{name}_ar5iv.html'
        source=p.read_text()
        def alt(match):
            start=match.group(1)
            hit=re.search(r'alttext="([^"]*)"',start)
            return ' '+unescape(hit.group(1))+' ' if hit else match.group(0)
        source=re.sub(r'<math\b([^>]*)>.*?</math>',alt,source,flags=re.S)
        parser=Blocks();parser.feed(source)
        blocks=[dict(i=i,**b) for i,b in enumerate(parser.blocks)]
        (ROOT/'sources'/f'{name}_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
        print(name,'HEADINGS',[(b['i'],b['text']) for b in blocks if b['tag'].startswith('h')])
    for p in sorted((ROOT/'sources').glob('*.json')):
        if p.name.endswith('_blocks.json'):continue
        v=json.loads(p.read_text());print(p.name)
        for x in v.get('results',[v]):
            print(json.dumps({'title':x.get('display_name'),'doi':x.get('doi'),'year':x.get('publication_year'),'locations':[{'url':y.get('landing_page_url'),'pdf':y.get('pdf_url')} for y in x.get('locations',[])]},ensure_ascii=False))
