"""Text extraction only for the two scoped primary HTML documents."""
import json
from html.parser import HTMLParser
from pathlib import Path

class Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks, self.current, self.depth, self.hide = [], None, 0, 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'):
            return
        if self.current is None and tag in ('h1','h2','h3','h4','h5','h6','p','math'):
            self.current = dict(tag=tag, id=attrs.get('id'), text=[])
            self.depth = 1
        elif self.current is not None:
            self.depth += 1
        if self.current is not None and tag == 'math' and attrs.get('alttext'):
            self.current['text'].append(' [MATH '+attrs['alttext']+'] ')
            self.hide = self.depth
    def handle_data(self, data):
        if self.current is not None and not self.hide:
            self.current['text'].append(data)
    def handle_endtag(self, tag):
        if self.current is None:
            return
        if self.hide == self.depth:
            self.hide = 0
        self.depth -= 1
        if self.depth == 0:
            self.current['text'] = ' '.join(''.join(self.current['text']).split())
            self.blocks.append(self.current)
            self.current = None

if __name__ == '__main__':
    sources = Path(__file__).resolve().parent / 'sources'
    for name, output in (('nash_mtl_v1', 'nash_blocks'), ('pathmlp_v2', 'pathmlp_blocks')):
        parser = Blocks()
        parser.feed((sources/(name+'.html')).read_text())
        blocks = [dict(block=i, **row) for i, row in enumerate(parser.blocks)]
        (sources/(output+'.json')).write_text(json.dumps(blocks, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(dict(source=name, headings=[b for b in blocks if b['tag'].startswith('h')]), ensure_ascii=False))
