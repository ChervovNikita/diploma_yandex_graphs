from html.parser import HTMLParser
from pathlib import Path
import json


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = []
        self.captures = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'section':
            self.sections.append(a.get('id'))
        cls = a.get('class', '').split()
        kind = ('paragraph' if tag == 'p' and 'ltx_p' in cls else
                'heading' if tag in ('h1', 'h2', 'h3', 'h4', 'h5') else
                'equation' if tag == 'table' and any(c.startswith('ltx_equation') for c in cls) else None)
        if kind:
            self.captures.append({'kind': kind, 'tag': tag, 'depth': 1,
                'id': a.get('id'), 'sections': list(self.sections), 'parts': []})
        else:
            for c in self.captures:
                if c['tag'] == tag:
                    c['depth'] += 1

    def handle_data(self, data):
        for c in self.captures:
            c['parts'].append(data)

    def handle_endtag(self, tag):
        done = []
        for c in self.captures:
            if c['tag'] == tag:
                c['depth'] -= 1
                if c['depth'] == 0:
                    done.append(c)
        for c in done:
            self.captures.remove(c)
            self.blocks.append({'kind': c['kind'], 'id': c['id'],
                'sections': c['sections'], 'text': ' '.join(''.join(c['parts']).split())})
        if tag == 'section' and self.sections:
            self.sections.pop()


if __name__ == '__main__':
    for source in sorted((Path(__file__).parent / 'primary').glob('*.source')):
        parser = Blocks()
        parser.feed(source.read_text())
        counts = {'paragraph': 0, 'equation': 0, 'heading': 0}
        for block in parser.blocks:
            block['index'] = counts[block['kind']]
            counts[block['kind']] += 1
        target = source.with_suffix('.blocks.json')
        target.write_text(json.dumps(parser.blocks, indent=2, ensure_ascii=False) + '\n')
        heads = [{k: v for k, v in b.items() if k != 'kind'} for b in parser.blocks if b['kind'] == 'heading']
        print(json.dumps({'id': source.stem, 'counts': counts, 'headings': heads}, ensure_ascii=False))
