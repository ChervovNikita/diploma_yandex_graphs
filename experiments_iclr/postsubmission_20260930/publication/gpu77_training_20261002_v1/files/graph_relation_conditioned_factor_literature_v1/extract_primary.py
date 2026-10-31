"""Extract local primary HTML as text; no network or scientific execution."""
from html.parser import HTMLParser
from pathlib import Path
import json
import re


class Node:
    def __init__(self, tag="", attrs=None, parent=None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.parent = parent
        self.children = []

    def text(self):
        if self.tag == "math" and self.attrs.get("alttext"):
            return " [MATH " + self.attrs["alttext"] + "] "
        return " ".join(c if isinstance(c, str) else c.text() for c in self.children)


class DOM(HTMLParser):
    void = set("area base br col embed hr img input link meta param source track wbr".split())

    def __init__(self):
        super().__init__()
        self.root = Node()
        self.stack = [self.root]
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        self.nodes.append(node)
        if tag not in self.void:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        self.nodes.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def extract(path):
    dom = DOM()
    dom.feed(path.read_text())
    blocks, listings = [], []
    for node in dom.nodes:
        classes = node.attrs.get("class", "").split()
        selected = (node.tag in ("h1", "h2", "h3", "h4")
                    or (node.tag == "p" and "ltx_p" in classes)
                    or "ltx_equation" in classes)
        listing = "ltx_listingline" in classes
        if not selected and not listing:
            continue
        ancestors = []
        parent = node.parent
        while parent:
            if parent.attrs.get("id"):
                ancestors.append(parent.attrs["id"])
            parent = parent.parent
        record = dict(tag=node.tag, id=node.attrs.get("id"), ancestors=ancestors,
                      class_names=classes, text=re.sub(r"\s+", " ", node.text()).strip())
        (listings if listing else blocks).append(record)
    path.with_name(path.stem + "_blocks.json").write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2) + "\n")
    path.with_name(path.stem + "_listings.json").write_text(
        json.dumps(listings, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for name in ("rgcn_v4", "hgt_v1", "sehgnn_v3"):
        extract(root / "primary" / (name + ".html"))
