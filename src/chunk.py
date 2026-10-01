"""Phase 2: split documents into retrieval chunks.

Usage:
    python src/chunk.py [--samples 3]

Laws: one chunk per 조, prefixed "[법령명 제N조(제목)]". Articles over max_tokens are split at
항 boundaries (then 호/목 lines), repeating the prefix and, for a split 항, its lead sentence.
Precedents: prefix "[법원 날짜 선고 사건번호 판결: 사건명]". When 판시사항 and 판결요지 are both
numbered [1], [2], ... each issue is paired with its holding as one unit. Units over
max_tokens are split into sentence windows with ~12% overlap.

Output: data/processed/chunks.jsonl, results/phase2_chunks/{stats.json, chunk_length_hist.png}
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter

import matplotlib
import yaml
from tokenizers import Tokenizer

from common import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DOCS = ROOT / "data" / "processed" / "documents.jsonl"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"
RESULTS = ROOT / "results" / "phase2_chunks"
ISSUE_RE = re.compile(r"\[(\d+)\]\s*")
SENT_RE = re.compile(r"(?<=[.?!])\s+")


class Counter_:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids)


def pack(units, budget, n_tokens, overlap=0):
    """Greedily pack text units into bodies of <= budget tokens. With overlap > 0, each new
    body starts with the trailing units of the previous one (up to `overlap` tokens)."""
    bodies, cur = [], []
    for u in units:
        if cur and n_tokens("\n".join(cur + [u])) > budget:
            bodies.append(cur)
            carry = []
            if overlap:
                for prev in reversed(cur):
                    if n_tokens("\n".join([prev] + carry)) > overlap:
                        break
                    carry.insert(0, prev)
            cur = carry if n_tokens("\n".join(carry + [u])) <= budget else []
        cur.append(u)
    if cur:
        bodies.append(cur)
    return ["\n".join(b) for b in bodies]


def hard_split(text, budget, tok, overlap):
    """Token-window fallback for a single unit longer than the budget. Windows are cut from
    the original text via token offsets; decoding would apply the tokenizer's NFKC
    normalization (e.g. '②' -> '2')."""
    offsets = tok.encode(text, add_special_tokens=False).offsets
    pieces, i = [], 0
    while i < len(offsets):
        end = min(i + budget, len(offsets))
        # A cut-out piece can re-tokenize slightly longer (word-start marker); shrink until it fits.
        while True:
            piece = text[offsets[i][0]:offsets[end - 1][1]].strip()
            if len(tok.encode(piece, add_special_tokens=False).ids) <= budget:
                break
            end -= 1
        pieces.append(piece)
        if end >= len(offsets):
            break
        i = max(end - overlap, i + 1)
    return pieces


def law_units(doc):
    """Return list of (paragraph_head, [lines]) blocks, one per non-deleted 항."""
    blocks = []
    for p in doc["metadata"]["paragraphs"]:
        if p["deleted"]:
            continue
        head = f"{p['no']} {p['text']}".strip() if p["no"] else p["text"]
        lines = []
        for it in p["items"]:
            if it["deleted"]:
                continue
            lines.append(f"{it['no']}. {it['text']}")
            lines += [f"{si['no']}. {si['text']}" for si in it["subitems"] if not si["deleted"]]
        blocks.append((head, lines))
    return blocks


def chunk_law(doc, cfg, n_tokens, tok):
    prefix = f"[{doc['title']}]"
    budget = cfg["max_tokens"] - n_tokens(prefix + "\n")
    if n_tokens(doc["text"]) <= budget:
        return [doc["text"]], prefix
    units = []
    for head, lines in law_units(doc):
        block = "\n".join([head] + lines) if head else "\n".join(lines)
        if n_tokens(block) <= budget:
            units.append(block)
            continue
        # Split this 항 by its 호/목 lines, repeating the 항 lead sentence when short enough.
        repeat = head if head and n_tokens(head) <= cfg["repeat_paragraph_head_max_tokens"] else None
        sub_budget = budget - (n_tokens(repeat + "\n") if repeat else 0)
        pieces = []
        for line in ([head] if head and not repeat else []) + lines:
            pieces += [line] if n_tokens(line) <= sub_budget else hard_split(line, sub_budget, tok, 0)
        for i, body in enumerate(pack(pieces, sub_budget, n_tokens)):
            units.append(f"{repeat}\n{body}" if repeat and (i > 0 or not body.startswith(repeat)) else body)
    return pack(units, budget, n_tokens), prefix


def split_issues(text):
    """'[1] a [2] b' -> {1: 'a', 2: 'b'}; returns {} when the text is not numbered."""
    parts = ISSUE_RE.split(text)
    if len(parts) < 3 or parts[0].strip():
        return {}
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def chunk_precedent(doc, cfg, n_tokens, tok, counters):
    md = doc["metadata"]
    y, m, d = md["decision_date"].split("-")
    prefix = f"[{md['court']} {int(y)}. {int(m)}. {int(d)}. 선고 {md['case_number']} 판결: {md['case_name']}]"
    budget = cfg["max_tokens"] - n_tokens(prefix + "\n")
    overlap = int(budget * cfg["precedent_overlap_ratio"])
    hi, si = split_issues(md["holding"]), split_issues(md["summary"])
    if len(hi) >= 2 and hi.keys() == si.keys():
        counters["precedent_issue_paired"] += 1
        units = [(f"판시사항: {hi[k]}", f"판결요지: {si[k]}") for k in sorted(hi)]
    else:
        counters["precedent_single_unit"] += 1
        units = [(f"판시사항: {md['holding']}", f"판결요지: {md['summary']}")]
    bodies = []
    for issue, holding in units:
        whole = f"{issue}\n{holding}"
        if n_tokens(whole) <= budget:
            bodies.append(whole)
            continue
        counters["precedent_units_window_split"] += 1
        sents = []
        for s in SENT_RE.split(whole):
            sents += [s] if n_tokens(s) <= budget else hard_split(s, budget, tok, overlap)
        bodies += pack(sents, budget, n_tokens, overlap=overlap)
    return bodies, prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "chunk.yaml").read_text(encoding="utf-8"))
    tok = Tokenizer.from_pretrained(cfg["tokenizer"])
    n_tokens = Counter_(tok)
    counters = Counter()

    chunks = []
    for line in DOCS.open(encoding="utf-8"):
        doc = json.loads(line)
        md = doc["metadata"]
        if doc["doc_type"] == "precedent":
            bodies, prefix = chunk_precedent(doc, cfg, n_tokens, tok, counters)
            meta = {k: md[k] for k in ("case_number", "case_name", "court", "decision_date",
                                       "linked_doc_ids", "referenced_articles")}
        else:
            bodies, prefix = chunk_law(doc, cfg, n_tokens, tok)
            meta = {k: md[k] for k in ("law_name", "statute_title", "law_id", "article_no",
                                       "article_key", "article_title", "effective_date",
                                       "has_formula_table")}
        for i, body in enumerate(bodies):
            text = f"{prefix}\n{body}"
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{i}",
                "doc_id": doc["doc_id"],
                "group_id": doc["doc_id"],   # Phase 3 splits train/val/test by this key
                "doc_type": doc["doc_type"],
                "title": doc["title"],
                "source_url": doc["source_url"],
                "part": i + 1,
                "n_parts": len(bodies),
                "text": text,
                "n_tokens": n_tokens(text),
                "metadata": meta,
            })

    over = [c for c in chunks if c["n_tokens"] > cfg["max_tokens"]]
    if over:
        sys.exit(f"{len(over)} chunks exceed max_tokens, e.g. {over[0]['chunk_id']} ({over[0]['n_tokens']})")

    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    types = ["law", "decree", "rule", "precedent"]
    stats = {"config": cfg, "counters": dict(counters), "per_doc_type": {}}
    for t in types:
        cs = [c for c in chunks if c["doc_type"] == t]
        lens = sorted(c["n_tokens"] for c in cs)
        docs_split = len({c["doc_id"] for c in cs if c["n_parts"] > 1})
        stats["per_doc_type"][t] = {
            "documents": len({c["doc_id"] for c in cs}), "chunks": len(cs), "documents_split": docs_split,
            "tokens_min": lens[0], "tokens_median": statistics.median(lens),
            "tokens_p95": lens[int(0.95 * (len(lens) - 1))], "tokens_max": lens[-1],
        }
    stats["total_chunks"] = len(chunks)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, t in zip(axes.flat, types):
        ax.hist([c["n_tokens"] for c in chunks if c["doc_type"] == t], bins=40, range=(0, cfg["max_tokens"]))
        ax.set_title(f"{t} (n={stats['per_doc_type'][t]['chunks']})")
        ax.set_xlabel("tokens (multilingual-e5 tokenizer)")
        ax.set_ylabel("chunks")
    fig.tight_layout()
    fig.savefig(RESULTS / "chunk_length_hist.png", dpi=120)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(chunks)} chunks to {OUT}")
    print(f"Histogram: {RESULTS / 'chunk_length_hist.png'}")

    rng = random.Random(42)
    split_docs = [c["doc_id"] for c in chunks if c["n_parts"] > 1 and c["part"] == 1]
    picks = rng.sample(split_docs, min(args.samples, len(split_docs)))
    for doc_id in picks:
        print(f"\n===== split document {doc_id}")
        for c in chunks:
            if c["doc_id"] == doc_id:
                print(f"--- {c['chunk_id']} ({c['n_tokens']} tokens)\n{c['text'][:500]}{' ...' if len(c['text']) > 500 else ''}")


if __name__ == "__main__":
    main()
