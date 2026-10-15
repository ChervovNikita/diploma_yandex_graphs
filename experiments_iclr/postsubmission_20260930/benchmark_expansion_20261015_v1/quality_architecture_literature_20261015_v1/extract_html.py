"""Local text extraction only; retain TeX alttext for arXiv MathML."""
from html.parser import HTMLParser
from pathlib import Path
import re


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []


class Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.ids = {}

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if node.attrs.get("id"):
            self.ids[node.attrs["id"]] = node
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def text(node):
    if isinstance(node, str):
        return node
    if node.tag in {"script", "style", "nav", "svg"}:
        return ""
    if node.tag == "math" and node.attrs.get("alttext"):
        return " " + node.attrs["alttext"] + " "
    value = "".join(text(child) for child in node.children)
    if node.tag in {"section", "p", "div", "li", "table", "tr", "figcaption", "h1", "h2", "h3", "h4"}:
        return "\n" + value + "\n"
    return value


def extract(path):
    document = Document()
    document.feed(Path(path).read_text())
    return document


def cleaned(node):
    lines = [re.sub(r"\s+", " ", line).strip() for line in text(node).splitlines()]
    return "\n".join(line for line in lines if line) + "\n"


if __name__ == "__main__":
    root = Path(__file__).parent / "primary"
    for name in ("graph_experts_v1", "tabm_v3"):
        path = root / (name + ".html")
        if not path.is_file():
            continue
        document = extract(path)
        (root/(name+".txt")).write_text(cleaned(document.root))
        for identity, node in document.ids.items():
            if node.tag == "section":
                heading = next((child for child in node.children if isinstance(child, Node) and child.tag in {"h1", "h2", "h3", "h4"}), None)
                if heading:
                    print(name, identity, cleaned(heading).strip())
