"""Reproduce the bounded primary HTML paragraph/equation extraction with stdlib."""
from html.parser import HTMLParser
from pathlib import Path
import argparse
import json


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.parts, self.out = [], [], []
        self.active = None

    def handle_starttag(self, tag, attrs):
        if tag in {'br', 'hr', 'input', 'img', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr'}:
            return
        self.stack.append(tag)
        if self.active is None and tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table'}:
            self.active = tag, len(self.stack), dict(attrs).get('id')
            self.parts = []

    def handle_endtag(self, tag):
        if self.active and tag == self.active[0] and len(self.stack) == self.active[1]:
            self.out.append(dict(index=len(self.out), tag=tag, id=self.active[2], text=' '.join(' '.join(self.parts).split())))
            self.active = None
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - self.stack[::-1].index(tag) - 1]

    def handle_data(self, data):
        if self.active and not set(self.stack) & {'script', 'style'}:
            self.parts.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    blocks = Blocks()
    blocks.feed(args.source.read_text())
    with args.output.open('x') as stream:
        stream.write(json.dumps(blocks.out, indent=2) + '\n')


if __name__ == '__main__':
    main()
