"""Public pinned paper retrieval and mechanical HTML extraction; no model imports."""
from pathlib import Path
from html.parser import HTMLParser
import concurrent.futures
import datetime
import hashlib
import json
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent


class Extract(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.active = None
        self.mathdepth = None
        self.blocks = []
        self.maths = []
        self.heading = ""

    def handle_starttag(self, tag, attributes):
        a = dict(attributes)
        self.stack.append((tag, a))
        depth = len(self.stack)
        if tag == "math" and a.get("alttext"):
            self.maths.append({"index": len(self.maths), "section": self.heading,
                               "id": a.get("id"), "display": a.get("display"),
                               "latex": a["alttext"]})
        skip = any(t in ("table", "figure", "script", "style") or
                   "ltx_bibliography" in aa.get("class", "") for t, aa in self.stack)
        if skip:
            return
        if tag in ("h1", "h2", "h3", "h4") or tag == "p" and "ltx_p" in a.get("class", ""):
            self.active = {"index": len(self.blocks), "tag": tag, "id": a.get("id"),
                           "text": "", "depth": depth}
        if tag == "math" and self.active and a.get("alttext"):
            self.active["text"] += " $" + a["alttext"] + "$ "
            self.mathdepth = depth

    def handle_data(self, data):
        if self.active and not self.mathdepth and not any(
                t in ("table", "figure", "script", "style") for t, _ in self.stack):
            self.active["text"] += data

    def handle_endtag(self, tag):
        if self.mathdepth and tag == "math":
            self.mathdepth = None
        if self.active and tag == self.active["tag"]:
            x = self.active
            self.active = None
            x.pop("depth")
            x["text"] = " ".join(x["text"].split())
            if x["text"]:
                self.blocks.append(x)
                if x["tag"] != "p":
                    self.heading = x["text"]
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                self.stack = self.stack[:i]
                break


def retrieve(ident):
    result = {"id": ident, "requests": []}
    for kind in ("abs", "html"):
        url = f"https://arxiv.org/{kind}/{ident}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "research-source-audit/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read()
                status = response.status
            name = f"primary/{ident}_{kind}.html"
            (ROOT / name).write_bytes(data)
            result["requests"].append({"url": url, "status": status, "bytes": len(data),
                                       "sha256": hashlib.sha256(data).hexdigest(), "saved_path": name})
            raw = data.decode()
            if kind == "abs":
                fields = re.findall(r'<meta name="(citation_[^"]+)" content="([^"]*)"', raw)
                result["metadata"] = {"fields": dict((k, v) for k, v in fields if k != "citation_abstract"),
                                      "authors": [v for k, v in fields if k == "citation_author"]}
            else:
                parser = Extract()
                parser.feed(raw)
                for suffix, values in (("blocks", parser.blocks), ("math", parser.maths)):
                    name = f"primary/{ident}_{suffix}.json"
                    (ROOT / name).write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")
                    result[suffix + "_path"] = name
                result["headings"] = [{"index": x["index"], "text": x["text"]}
                                      for x in parser.blocks if x["tag"] != "p"]
        except Exception as error:
            result["requests"].append({"url": url, "failure": str(error)})
    return result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        papers = list(pool.map(retrieve, sys.argv[1:]))
    receipt = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "papers": papers}
    (ROOT / "PRIMARY_RECENT_RETRIEVAL.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(papers, indent=2, ensure_ascii=False))
