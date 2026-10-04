"""Text/TeX block extraction for the retained version-pinned primary HTML.

Standard library only. MathML is replaced by its source TeX alttext so a
quotation does not duplicate visual MathML and its annotation. Indices are
specific to this retained response, not universal arXiv paragraph numbers.
"""
import json
from html.parser import HTMLParser
from pathlib import Path


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.out, self.current, self.skip_math = [], [], None, 0

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.stack.append((tag, attrs))
        target = tag in ("p", "h1", "h2", "h3", "h4", "h5") or (
            tag in ("div", "table") and "ltx_equation" in attrs.get("class", "")
        )
        if self.current is None and target:
            self.current = dict(tag=tag, html_id=attrs.get("id"),
                                **{"class": attrs.get("class")},
                                parts=[], depth=len(self.stack))
        if self.current is not None:
            if tag == "math":
                self.current["parts"].append(" " + attrs.get("alttext", "") + " ")
                self.skip_math += 1
            elif tag == "br" and not self.skip_math:
                self.current["parts"].append(" ")

    def handle_endtag(self, tag):
        if self.current is not None and tag == "math":
            self.skip_math = max(0, self.skip_math - 1)
        if (self.current is not None and len(self.stack) == self.current["depth"]
                and self.stack[-1][0] == tag):
            current = self.current
            current["text"] = " ".join("".join(current.pop("parts")).split())
            current.pop("depth")
            current["index"] = len(self.out)
            self.out.append(current)
            self.current = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.stack = self.stack[:index]
                break

    def handle_data(self, text):
        if self.current is not None and not self.skip_math:
            self.current["parts"].append(text)


def extract(path):
    parser = Blocks()
    parser.feed(Path(path).read_text())
    return parser.out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    output = extract(here / "primary/2602_15747v1_html.html")
    (here / "primary/2602_15747v1_blocks.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
