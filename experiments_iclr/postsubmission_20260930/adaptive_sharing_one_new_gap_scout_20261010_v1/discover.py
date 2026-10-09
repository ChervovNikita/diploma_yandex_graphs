"""Bounded public metadata discovery; no model/data/outcome access."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
QUERIES={
    "recent_graph_complementarity": 'all:"graph contrastive" AND (all:complementary OR all:ensemble) AND submittedDate:[202501010000 TO 202610102359]',
    "contrastive_shared_ensemble": 'all:ensemble AND all:contrastive AND (all:"weight sharing" OR all:"shared backbone") AND submittedDate:[202401010000 TO 202610102359]',
    "graph_factor_contrastive": 'all:"graph contrastive" AND (all:"fast weights" OR all:"low rank" OR all:"model diversity") AND submittedDate:[202401010000 TO 202610102359]',
}


def fetch(item):
    name,query=item
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":query,"start":0,"max_results":12,"sortBy":"submittedDate","sortOrder":"descending"})
    record={"name":name,"query":query,"url":url,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request=urllib.request.Request(url,headers={"User-Agent":"GNNM-public-method-scout/1.0"})
        with urllib.request.urlopen(request,timeout=45) as response:
            payload=response.read(2_000_001)
            record.update(http_status=response.status,resolved_url=response.url)
        if len(payload)>2_000_000: raise ValueError("metadata response too large")
        path=HERE/(name+".xml"); path.write_bytes(payload)
        ns={"a":"http://www.w3.org/2005/Atom"}; root=ET.fromstring(payload)
        entries=[]
        for entry in root.findall("a:entry",ns):
            entries.append({"id":entry.findtext("a:id",default="",namespaces=ns),
                            "title":" ".join(entry.findtext("a:title",default="",namespaces=ns).split()),
                            "abstract":" ".join(entry.findtext("a:summary",default="",namespaces=ns).split()),
                            "published":entry.findtext("a:published",default="",namespaces=ns),
                            "updated":entry.findtext("a:updated",default="",namespaces=ns)})
        record.update(success=True,bytes=len(payload),response_sha256=hashlib.sha256(payload).hexdigest(),entries=entries)
    except Exception as error:
        record.update(success=False,error_type=type(error).__name__,error=str(error))
    return record


def main():
    current=json.loads((BASE/"literature_memory/CURRENT_SUPPLEMENT.json").read_text())
    chain=[]; path=BASE/current["current"]["path"]; seen=set()
    texts=[]
    while path:
        if str(path) in seen: raise ValueError("memory predecessor cycle")
        seen.add(str(path)); text=path.read_text(); texts.append(text); data=json.loads(text)
        chain.append({"path":str(path.relative_to(BASE)),"sha256":hashlib.sha256(text.encode()).hexdigest(),"status":data.get("status")})
        prior=data.get("prior_supplements"); path=BASE/prior["path"] if isinstance(prior,dict) and "path" in prior else None
    index_path=BASE/"literature_memory/index_v72/LITERATURE_INDEX.json"; index_text=index_path.read_text(); texts.append(index_text)
    # Scope receipts are reading metadata only. Exclude active comparative
    # packets and duplicate publication mirrors; do not read outcome files.
    receipt_bindings=[]
    for path in sorted(BASE.glob("*/READ_SCOPES.json")):
        if any(tag in path.parent.name.lower() for tag in ("qk36","pre_sigmoid","wiki15","relation18","attention18")):
            continue
        text=path.read_text(); texts.append(text)
        receipt_bindings.append({"path":str(path.relative_to(BASE)),"sha256":hashlib.sha256(text.encode()).hexdigest()})
    known=set(re.findall(r"\b(?:[0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?\b","\n".join(texts)))
    known={re.sub(r"v[0-9]+$","",identity) for identity in known}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        records=list(executor.map(fetch,QUERIES.items()))
    summaries=[]
    for record in records:
        for entry in record.get("entries",[]):
            identity=re.search(r"([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?$",entry["id"])
            entry["normalized_arxiv_id"]=identity.group(1) if identity else None
            entry["identifier_seen_in_saved_index_chain_or_scope_receipts"]=bool(identity and identity.group(1) in known)
            summaries.append({key:entry[key] for key in ("id","title","identifier_seen_in_saved_index_chain_or_scope_receipts","abstract")})
    (HERE/"DISCOVERY.json").write_text(json.dumps({"schema":"bounded-graph-contrastive-complementarity-discovery-v1","queries":records,
        "new_method_read_credit":0,"new_full_paper_read_credit":0,"source_outcomes_or_models_accessed":False},indent=2)+"\n")
    (HERE/"REUSED_MEMORY_BINDINGS.json").write_text(json.dumps({"schema":"saved-memory-first-dedup-v1","pointer":current,
        "pointer_sha256":hashlib.sha256((BASE/"literature_memory/CURRENT_SUPPLEMENT.json").read_bytes()).hexdigest(),
        "base_index_path":str(index_path.relative_to(BASE)),"base_index_sha256":hashlib.sha256(index_text.encode()).hexdigest(),
        "chain":chain,"method_scope_receipts":receipt_bindings,"known_arxiv_ids":sorted(known),
        "all_prior_scope_credit_reused_only":True},indent=2)+"\n")
    for summary in summaries: print(json.dumps(summary,ensure_ascii=False))
    print(json.dumps({"queries":len(records),"known_identifiers":len(known),"scope_receipts_metadata_only":len(receipt_bindings),"new_primary_method_reads":0}))


if __name__=="__main__": main()
