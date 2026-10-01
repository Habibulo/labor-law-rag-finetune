"""Phase 1: parse data/raw Markdown into data/processed/documents.jsonl.

Usage:
    python src/parse_clean.py [--samples 5]

Laws: law name > 장 > 절 > 관 > 조 (with title) > 항 > 호 > 목. One document per 조.
Precedents: frontmatter + 판시사항, 판결요지, 참조조문 (parsed into referenced articles).
"""
import argparse
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter

from common import ROOT, load_config, split_frontmatter

sys.stdout.reconfigure(encoding="utf-8")

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "documents.jsonl"
STATS = ROOT / "data" / "processed" / "stats.json"

ARTICLE_RE = re.compile(r"^##### 제(\d+)조(?:의(\d+))?(?: \((.*)\))?\s*$")
CHAPTER_RE = re.compile(r"^## (제\d+장.*)$")
SECTION_RE = re.compile(r"^### (제\d+절.*)$")
SUBSECTION_RE = re.compile(r"^#### (제\d+관.*)$")
PARA_RE = re.compile(r"^\*\*([①-⑳㉑-㉟])\*\*\s*(.*)$")
ITEM_RE = re.compile(r"^\s*(\d+(?:의\d+)?)\\?\.\s*(.*)$")
SUBITEM_RE = re.compile(r"^\s*([가-힣](?:의\d+)?)\\?\.\s*(.*)$")
DATE_TAG_RE = re.compile(r"\s*<[^<>]*\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}[^<>]*>")
HANJA_RE = re.compile(r"\(([㐀-䶿一-鿿豈-﫿]+)\)")
MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")
DELETED_RE = re.compile(r"^삭제\s*$")
# Formulas/tables shown as images on law.go.kr arrive as <img ...> plus a box-drawing text
# rendering whose columns are interleaved. Keep the words, drop the tag and box characters.
IMG_TAG_RE = re.compile(r"<img[^>]*>|</img>")
BOX_RE = re.compile(r"[─-╿]+")


def has_formula_table(s):
    return bool(IMG_TAG_RE.search(s) or BOX_RE.search(s))


def clean(s, strip_hanja):
    s = unicodedata.normalize("NFC", s)
    s = BOX_RE.sub(" ", IMG_TAG_RE.sub(" ", s))
    s = MD_ESCAPE_RE.sub(r"\1", s).replace("**", "")
    s = DATE_TAG_RE.sub("", s)
    if strip_hanja:
        s = HANJA_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_article_body(lines, strip_hanja):
    """Return (paragraphs, notes, deleted_whole_article). Paragraph/item/subitem nodes carry
    'deleted' flags; notes are bracket lines like [본조신설 ...] or [시행일] blocks."""
    paras, notes = [], []
    in_note = False
    for raw in lines:
        if not raw.strip():
            continue
        s = raw.strip()
        if s.startswith("[") and (s.endswith("]") or s.startswith("[시행일")):
            notes.append(clean(s, strip_hanja))
            in_note = s.startswith("[시행일")
            continue
        m = PARA_RE.match(s)
        if m:
            in_note = False
            text = clean(m.group(2), strip_hanja)
            paras.append({"no": m.group(1), "text": text, "deleted": bool(DELETED_RE.match(text)), "items": []})
            continue
        if in_note:
            notes[-1] += " " + clean(s, strip_hanja)
            continue
        if not paras:
            paras.append({"no": None, "text": "", "deleted": False, "items": []})
        para = paras[-1]
        m = ITEM_RE.match(raw)
        if m and raw.startswith(" "):
            text = clean(m.group(2), strip_hanja)
            para["items"].append({"no": m.group(1), "text": text, "deleted": bool(DELETED_RE.match(text)),
                                  "subitems": []})
            continue
        m = SUBITEM_RE.match(raw)
        if m and raw.startswith(" ") and para["items"]:
            text = clean(m.group(2), strip_hanja)
            para["items"][-1]["subitems"].append({"no": m.group(1), "text": text,
                                                  "deleted": bool(DELETED_RE.match(text))})
            continue
        # Continuation line: append to the deepest open node.
        target = para
        if para["items"]:
            target = para["items"][-1]
            if target["subitems"]:
                target = target["subitems"][-1]
        target["text"] = (target["text"] + " " + clean(s, strip_hanja)).strip()
        target["deleted"] = bool(DELETED_RE.match(target["text"]))
    live = [p for p in paras if not p["deleted"] and (p["text"] or p["items"])]
    return paras, notes, not live


def article_text(paras):
    """Flat text without deleted 항/호/목."""
    out = []
    for p in paras:
        if p["deleted"]:
            continue
        head = f"{p['no']} {p['text']}".strip() if p["no"] else p["text"]
        if head:
            out.append(head)
        for it in p["items"]:
            if it["deleted"]:
                continue
            out.append(f"{it['no']}. {it['text']}")
            out += [f"{si['no']}. {si['text']}" for si in it["subitems"] if not si["deleted"]]
    return "\n".join(out)


def parse_law_file(path, law_name, doc_type, pcfg, counters):
    fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
    statute = unicodedata.normalize("NFC", fm["제목"])
    chapter = section = subsection = None
    in_addenda = False
    articles, cur = [], None

    def flush():
        if cur is None:
            return
        paras, notes, deleted = parse_article_body(cur["lines"], pcfg["strip_hanja_annotations"])
        if deleted and not pcfg["include_deleted"]:
            counters["law_articles_deleted_excluded"] += 1
            return
        key = cur["no"] + (f"-{cur['sub']}" if cur["sub"] else "")
        label = f"제{cur['no']}조" + (f"의{cur['sub']}" if cur["sub"] else "")
        title = clean(cur["title"], pcfg["strip_hanja_annotations"]) if cur["title"] else ""
        if title.startswith("(") and title.endswith(")"):  # old-format headings: 제1조 ((목적))
            title = title[1:-1]
        heading = f"{label}({title})" if title else label
        articles.append({
            "doc_id": f"{doc_type}:{fm['법령ID']}:{key}",
            "doc_type": doc_type,
            "title": f"{statute} {heading}",
            "source_url": fm["출처"],
            "text": article_text(paras),
            "metadata": {
                "law_name": law_name, "statute_title": statute, "law_id": str(fm["법령ID"]),
                "law_mst": str(fm["법령MST"]), "statute_kind": fm["법령구분"],
                "chapter": cur["chapter"], "section": cur["section"], "subsection": cur["subsection"],
                "article_no": label, "article_key": key, "article_title": title,
                "promulgation_date": str(fm["공포일자"]), "effective_date": str(fm["시행일자"]),
                "notes": notes, "paragraphs": paras,
                "has_formula_table": any(has_formula_table(l) for l in cur["lines"]),
                "annex_titles": [a["제목"] for a in fm.get("첨부파일") or []],
            },
        })

    for line in body.split("\n"):
        if line.startswith("## 부칙"):
            flush()
            cur, in_addenda = None, True
            counters["law_addenda_blocks_excluded"] += 1
            continue
        if in_addenda and not pcfg["include_addenda"]:
            continue
        m = CHAPTER_RE.match(line)
        if m:
            flush(); cur = None
            chapter, section, subsection = clean(m.group(1), False), None, None
            continue
        m = SECTION_RE.match(line)
        if m:
            flush(); cur = None
            section, subsection = clean(m.group(1), False), None
            continue
        m = SUBSECTION_RE.match(line)
        if m:
            flush(); cur = None
            subsection = clean(m.group(1), False)
            continue
        m = ARTICLE_RE.match(line)
        if m:
            flush()
            cur = {"no": m.group(1), "sub": m.group(2), "title": m.group(3) or "", "lines": [],
                   "chapter": chapter, "section": section, "subsection": subsection}
            continue
        if line.startswith("#"):
            if not line.startswith("# "):
                counters["law_unrecognized_headings"] += 1
                print(f"  unrecognized heading in {path.name}: {line[:80]}")
            continue
        if cur is not None:
            cur["lines"].append(line)
    flush()
    return articles


def norm_law(name):
    return unicodedata.normalize("NFC", re.sub(r"\s+", "", name)).replace("·", "ㆍ")


def parse_refs(raw):
    """Parse 참조조문 into [{law, article_key, old_version, current_equivalent}].
    '(현행 X 참조)' blocks name the current article that replaced an old one; they are parsed
    separately so they don't change which law the following tokens inherit. Other
    parentheticals (e.g. '(2007. 4. 11. 법률 제8372호로 전부 개정되기 전의 것)') are dropped."""
    current = []
    for inner in re.findall(r"\(현행\s*([^()]*?)\s*참조\)", raw):
        current += [{**r, "current_equivalent": True} for r in _parse_ref_tokens(inner)]
    s = re.sub(r"\(현행\s*[^()]*?\s*참조\)", "", raw)
    return [{**r, "current_equivalent": False} for r in _parse_ref_tokens(s)] + current


def _parse_ref_tokens(s):
    while True:
        new = re.sub(r"\([^()]*\)", "", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\[\d+\]", ",", s)
    refs, cur_law, cur_old = [], None, False
    for tok in re.split(r"[,/\n]", s):
        tok = tok.strip()
        m = re.match(r"^(구\s*)?(.*?)\s*제(\d+)조(?:의\s*(\d+))?", tok)
        if not m:
            continue
        old, law, no, sub = m.groups()
        law = law.strip()
        if law:
            if law.replace(" ", "").startswith("같은법") and cur_law:
                base = re.sub(r"\s*(시행령|시행규칙)$", "", cur_law)
                rest = law.replace(" ", "")[3:]
                law = f"{base} {rest}".strip()
            cur_law, cur_old = law, bool(old)
        elif cur_law is None:
            continue
        key = no + (f"-{sub}" if sub else "")
        refs.append({"law": cur_law, "article_key": key, "old_version": cur_old})
    return refs


def parse_precedent(path, pcfg, law_index):
    fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
    sec, cur = {}, None
    for line in body.split("\n"):
        if line.startswith("## "):
            cur = line[3:].strip()
            sec[cur] = []
        elif cur:
            sec[cur].append(line)
    sec = {k: "\n".join(v).strip() for k, v in sec.items()}
    strip = pcfg["strip_hanja_annotations"]
    holding = "\n".join(clean(l, strip) for l in sec.get("판시사항", "").split("\n") if l.strip())
    summary = "\n".join(clean(l, strip) for l in sec.get("판결요지", "").split("\n") if l.strip())
    ref_raw = clean(sec.get("참조조문", ""), False)
    refs = parse_refs(ref_raw)
    linked = []
    for r in refs:
        doc_id = law_index.get((norm_law(r["law"]), r["article_key"]))
        r["doc_id"] = doc_id if doc_id and not r["old_version"] else None
        if r["doc_id"] and r["doc_id"] not in linked:
            linked.append(r["doc_id"])
    date = str(fm.get("선고일자", ""))
    y, mo, d = (date.split("-") + ["", "", ""])[:3]
    text = holding + ("\n" + summary if summary else "")
    if pcfg["precedent_full_text"]:
        text += "\n" + clean(sec.get("판례내용", ""), strip)
    return {
        "doc_id": f"precedent:{fm['판례일련번호']}",
        "doc_type": "precedent",
        "title": f"{fm['법원명']} {int(y)}. {int(mo)}. {int(d)}. 선고 {fm['사건번호']} 판결 [{fm['사건명']}]",
        "source_url": fm["출처"],
        "text": text,
        "metadata": {
            "case_number": str(fm["사건번호"]), "case_name": fm["사건명"], "court": fm["법원명"],
            "case_type": fm["사건종류"], "decision_date": date,
            "holding": holding, "summary": summary,
            "referenced_articles_raw": ref_raw, "referenced_articles": refs,
            "linked_doc_ids": linked, "referenced_cases_raw": clean(sec.get("참조판례", ""), False),
        },
    }


def length_stats(docs):
    lens = sorted(len(d["text"]) for d in docs)
    if not lens:
        return {}
    return {"n": len(lens), "min": lens[0], "median": statistics.median(lens),
            "p95": lens[int(0.95 * (len(lens) - 1))], "max": lens[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    pcfg = cfg["parse"]
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    counters = Counter()

    docs = []
    for rec in manifest["laws"]:
        doc_type = cfg["law_kinds"][rec["kind"]]
        docs += parse_law_file(ROOT / rec["raw_path"], rec["law"], doc_type, pcfg, counters)
    counters["law_articles_empty_text"] = sum(1 for d in docs if not d["text"])
    counters["law_articles_with_formula_table"] = sum(d["metadata"]["has_formula_table"] for d in docs)

    law_index = {(norm_law(d["metadata"]["statute_title"]), d["metadata"]["article_key"]): d["doc_id"]
                 for d in docs}
    precs = [parse_precedent(p, pcfg, law_index) for p in sorted((RAW / "precedents").glob("*/*.md"))]
    docs += precs

    ids = Counter(d["doc_id"] for d in docs)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        sys.exit(f"Duplicate doc_ids: {dupes[:10]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    all_refs = [r for p in precs for r in p["metadata"]["referenced_articles"]]
    stats = {
        "per_doc_type_chars": {t: length_stats([d for d in docs if d["doc_type"] == t])
                               for t in ["law", "decree", "rule", "precedent"]},
        "counters": dict(counters),
        "precedent_refs_total": len(all_refs),
        "precedent_refs_old_version": sum(r["old_version"] for r in all_refs),
        "precedent_refs_linked_to_corpus": sum(bool(r["doc_id"]) for r in all_refs),
        "precedents_with_at_least_one_link": sum(bool(p["metadata"]["linked_doc_ids"]) for p in precs),
        "precedents_without_summary": sum(not p["metadata"]["summary"] for p in precs),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(docs)} documents to {OUT}")

    rng = random.Random(42)
    for t in ["law", "decree", "rule", "precedent"]:
        pool = [d for d in docs if d["doc_type"] == t]
        for d in rng.sample(pool, min(args.samples, len(pool))):
            print(f"\n----- [{t}] {d['doc_id']} | {d['title']}")
            print(d["text"][:700] + (" ..." if len(d["text"]) > 700 else ""))
            md = d["metadata"]
            if t == "precedent":
                print(f"  참조조문: {md['referenced_articles_raw'][:200]}")
                print(f"  linked: {md['linked_doc_ids']}")
            else:
                print(f"  장/절: {md['chapter']} / {md['section']} | notes: {md['notes'][:2]}")


if __name__ == "__main__":
    main()
