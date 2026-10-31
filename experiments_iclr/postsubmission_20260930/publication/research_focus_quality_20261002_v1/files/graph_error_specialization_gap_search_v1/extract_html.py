"""Stdlib static HTML block extraction; no content execution."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.article = False
        self.sections = []
        self.capture = None
        self.blocks = []
        self.title = False
        self.title_parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'title':
            self.title = True
        if tag == 'article':
            self.article = True
        if tag == 'section':
            self.sections.append(attrs.get('id'))
        if not self.article:
            return
        if tag in ['h1', 'h2', 'h3', 'h4', 'p', 'figcaption', 'table'] and self.capture is None:
            self.capture = {'index': len(self.blocks), 'tag': tag, 'id': attrs.get('id'),
                            'section': self.sections[-1] if self.sections else None,
                            'parts': [], 'math_alttext': []}
        if tag == 'math' and self.capture is not None and attrs.get('alttext'):
            self.capture['math_alttext'].append(attrs['alttext'])

    def handle_data(self, data):
        if self.title:
            self.title_parts.append(data)
        if self.capture is not None:
            self.capture['parts'].append(data)

    def handle_endtag(self, tag):
        if tag == 'title':
            self.title = False
        if self.capture is not None and tag == self.capture['tag']:
            self.capture['text'] = re.sub(r'\s+', ' ', ' '.join(self.capture.pop('parts'))).strip()
            self.blocks.append(self.capture)
            self.capture = None
        if tag == 'section' and self.sections:
            self.sections.pop()
        if tag == 'article':
            self.article = False


def main():
    path = Path(sys.argv[1])
    parser = Blocks()
    parser.feed(path.read_text())
    target = Path(sys.argv[2])
    target.write_text(json.dumps(parser.blocks, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'title': ''.join(parser.title_parts), 'blocks': len(parser.blocks),
                      'headings': [{k: b[k] for k in ['index', 'id', 'text']} for b in parser.blocks if b['tag'].startswith('h')]},ensure_ascii=False))


if __name__ == '__main__':
    main()
