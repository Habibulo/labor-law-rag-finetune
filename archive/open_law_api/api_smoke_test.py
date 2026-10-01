"""Phase 0: one test call per API target to discover the real response schema.

Usage:
    set LAW_OC=<your OC value>          (PowerShell: $env:LAW_OC="<your OC value>")
    python src/api_smoke_test.py

Raw responses are saved to data/raw/_smoke/ so we can design the parser from real data.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
BASE = CFG["api"]["base_url"]
OUT = ROOT / "data" / "raw" / "_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def call(endpoint, params, name):
    params = {"OC": os.environ["LAW_OC"], "type": "JSON", **params}
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    print(f"\n=== {name} | HTTP {r.status_code} | {r.url.replace(params['OC'], '<OC>')}")
    (OUT / f"{name}.raw.txt").write_text(r.text, encoding="utf-8")
    try:
        data = r.json()
    except ValueError:
        print("Response is not JSON. First 800 chars:")
        print(r.text[:800])
        return None
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(CFG["api"]["sleep_seconds"])
    return data


def outline(obj, depth=0, max_depth=3):
    """Print the key structure of a JSON object so we can see the schema."""
    pad = "  " * depth
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kind = type(v).__name__
            size = f" len={len(v)}" if isinstance(v, (list, dict)) else ""
            preview = "" if isinstance(v, (list, dict)) else f" = {str(v)[:60]!r}"
            print(f"{pad}{k}: {kind}{size}{preview}")
            outline(v, depth + 1, max_depth)
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0]:")
        outline(obj[0], depth + 1, max_depth)


def first_list_of_dicts(obj):
    """Find the first list of result records anywhere in the response."""
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
            if isinstance(v, dict):
                found = first_list_of_dicts(v)
                if found:
                    return found
    return []


if "LAW_OC" not in os.environ:
    sys.exit("Set the LAW_OC environment variable first.")

# 1. Law search: 근로기준법
law_search = call("lawSearch.do", {"target": "law", "query": "근로기준법", "display": 20}, "1_law_search")
if law_search:
    outline(law_search, max_depth=2)
    for rec in first_list_of_dicts(law_search):
        print({k: rec.get(k) for k in ("법령명한글", "법령ID", "법령일련번호", "법령구분명", "시행일자", "현행연혁코드")})

# 2. Law body: pick the exact 근로기준법 record and fetch its full text
law_body = None
if law_search:
    recs = [r for r in first_list_of_dicts(law_search) if r.get("법령명한글") == "근로기준법"]
    if recs:
        law_body = call("lawService.do", {"target": "law", "MST": recs[0]["법령일련번호"]}, "2_law_body")
if law_body:
    outline(law_body, max_depth=3)

# 3. Precedent search: cases referencing 근로기준법
prec_search = call("lawSearch.do", {"target": "prec", "JO": "근로기준법", "display": 5}, "3_prec_search")
if prec_search:
    outline(prec_search, max_depth=2)
    prec_recs = first_list_of_dicts(prec_search)
    for rec in prec_recs:
        print({k: rec.get(k) for k in ("판례일련번호", "사건명", "사건번호", "선고일자", "법원명")})
    # 4. Precedent body for the first result
    if prec_recs:
        prec_body = call("lawService.do", {"target": "prec", "ID": prec_recs[0]["판례일련번호"]}, "4_prec_body")
        if prec_body:
            outline(prec_body, max_depth=2)

# 5. Legal interpretation search
expc_search = call("lawSearch.do", {"target": "expc", "query": "근로기준법", "display": 5}, "5_expc_search")
if expc_search:
    outline(expc_search, max_depth=2)

print(f"\nRaw responses saved to {OUT}")
