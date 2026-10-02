"""Stdlib mechanical extraction of already saved public arXiv HTML."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path


class Node:
    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs or []), parent
        self.children = []

    def nodes(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.nodes()

    def text(self):
        parts = []
        for child in self.children:
            parts.append(child.text() if isinstance(child, Node) else child)
        return " ".join(" ".join(parts).split())


class Parser(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = self.current = Node("document")

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in self.VOID:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.current.children.append(Node(tag, attrs, self.current))

    def handle_endtag(self, tag):
        node = self.current
        while node.parent and node.tag != tag:
            node = node.parent
        if node.parent:
            self.current = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


PACKET = Path(__file__).resolve().parent
WORKSPACE = PACKET.parent.parent


def extract():
    receipts = json.loads((PACKET / "PRIMARY_RETRIEVAL.json").read_text())
    previews = []
    for row in receipts:
        if row.get("status") != 200:
            continue
        path = WORKSPACE / row["path"]
        parser = Parser()
        parser.feed(path.read_text())
        nodes = list(parser.root.nodes())
        block_nodes = [n for n in nodes if n.tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6"}]
        blocks = [{"index": i, "tag": n.tag, "id": n.attrs.get("id"), "text": n.text(),
                   "links": [{"href": a.attrs["href"], "text": a.text()} for a in n.nodes()
                             if a.tag == "a" and a.attrs.get("href")]}
                  for i, n in enumerate(block_nodes)]
        all_math = [{"index": i, "id": n.attrs.get("id"), "display": n.attrs.get("display"),
                     "alttext": n.attrs.get("alttext")}
                    for i, n in enumerate(n for n in nodes if n.tag == "math")]
        display_math = [{"index": i, "id": n["id"], "alttext": n["alttext"], "all_math_index": n["index"]}
                        for i, n in enumerate(n for n in all_math if n["display"] == "block")]
        algorithms = [{"index": i, "id": n.attrs.get("id"), "text": n.text()}
                      for i, n in enumerate(n for n in nodes if "ltx_float_algorithm" in n.attrs.get("class", "").split())]
        metadata = {}
        for n in nodes:
            key = n.attrs.get("name", "")
            if n.tag == "meta" and key.startswith("citation_"):
                metadata.setdefault(key, []).append(n.attrs.get("content"))
            if n.tag == "blockquote" and "abstract" in n.attrs.get("class", "").split():
                metadata["abstract"] = [n.text()]
        row["mechanical_extracts"] = {}
        for key, value in [("blocks", blocks), ("all_math", all_math), ("display_math", display_math),
                           ("algorithms", algorithms), ("metadata", metadata)]:
            output = path.with_name(path.stem + "_" + key + ".json")
            output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            row["mechanical_extracts"][key] = {"path": str(output.relative_to(WORKSPACE)),
                                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                                "items": len(value)}
        if row["kind"] == "html":
            previews.append({"canonical_id": row["canonical_id"], "headings": [
                {"index": b["index"], "text": b["text"]} for b in blocks if b["tag"].startswith("h")],
                "display_math_nodes": len(display_math), "algorithm_ids": [a["id"] for a in algorithms]})
        else:
            previews.append({"canonical_id": row["canonical_id"], "metadata_and_abstract": metadata})
    (PACKET / "PRIMARY_RETRIEVAL.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(previews, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    extract()
