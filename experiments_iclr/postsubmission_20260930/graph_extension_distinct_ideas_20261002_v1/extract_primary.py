"""Versioned primary paragraph extraction; stdlib only, no model execution."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.stack = []
        self.blocks = []
        self.math_depth = None

    def handle_starttag(self, tag, attrs):
        self.depth += 1
        attrs = dict(attrs)
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.stack.append({"tag": tag, "id": attrs.get("id"), "depth": self.depth, "text": []})
        if tag == "math":
            self.math_depth = self.depth
            if self.stack:
                self.stack[-1]["text"].append(attrs.get("alttext", ""))

    def handle_endtag(self, tag):
        if self.math_depth == self.depth:
            self.math_depth = None
        if self.stack and self.stack[-1]["tag"] == tag and self.stack[-1]["depth"] == self.depth:
            b = self.stack.pop()
            b["text"] = re.sub(r"\s+", " ", " ".join(b["text"])).strip()
            del b["depth"]
            if b["text"]:
                b["index"] = len(self.blocks)
                self.blocks.append(b)
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data):
        if self.stack and self.math_depth is None:
            self.stack[-1]["text"].append(data)


root = Path(__file__).resolve().parent
for ident in ("2608.02128v1", "2609.32929v1"):
    for kind in ("abs", "html"):
        parser = Blocks()
        parser.feed((root / "primary" / f"{ident}_{kind}.html").read_text())
        (root / "primary" / f"{ident}_{kind}_blocks.json").write_text(json.dumps(parser.blocks, indent=2) + "\n")
        (root / "primary" / f"{ident}_{kind}_blocks.txt").write_text("\n".join(f"[{b['index']}] {b['tag']} {b['text']}" for b in parser.blocks) + "\n")
        print(ident, kind, "blocks", len(parser.blocks))
        for b in parser.blocks:
            if kind == "abs" or b["tag"].startswith("h"):
                print(f"[{b['index']}] {b['tag']} {b['text'][:600]}")
